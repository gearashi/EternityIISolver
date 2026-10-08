"""Shared GPU search engine; CPU initializes, validates and checkpoints only."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
import numpy as np
from gpu_backends import make_backend

class GPU:
    def __init__(self,bundle,n=4096,seed=20261007,backend="auto",cache_dir=None,*,allow_opencl_cpu=False,device_name=None):
        if not isinstance(n,int) or not 1<=n<=1048576:raise ValueError("replicas must be an integer in 1..1048576")
        self.runtime=make_backend(backend,cache_dir,allow_opencl_cpu=allow_opencl_cpu,device_name=device_name)
        self.xp=self.runtime.xp;self.backend=self.runtime.name;self.device=self.runtime.device
        self.backend_details=self.runtime.details
        self.n=n;self.bundle=bundle;self.host_rng=np.random.default_rng(seed)
        self.faces=np.asarray(bundle.oriented_edges,dtype=np.uint8)
        self.neighbors=np.full((256,4),-1,np.int16)
        self.fixed=np.zeros(256,np.uint8)
        for s in bundle.fixed_clues:self.fixed[s]=1
        self.types=np.zeros(256,np.uint8);self.allowed=np.zeros((256,1024),np.uint8)
        for s in range(256):
            row,col=divmod(s,16);bounds=np.array([row==0,col==15,row==15,col==0])
            self.types[s]=2-int(sum(bounds))
            for k,t in enumerate([s-16,s+1,s+16,s-1]):
                if not bounds[k]:self.neighbors[s,k]=t
            self.allowed[s]=np.all((self.faces==0)==bounds,axis=1)
        self.groups=[np.flatnonzero((self.types==t)&(self.fixed==0)).astype(np.int16) for t in range(3)]
        members=np.full((3,256),-1,np.int16)
        for t,g in enumerate(self.groups):members[t,:len(g)]=g
        self.tables=[self.xp.asarray(x) for x in [self.faces,self.neighbors,self.fixed,members,np.array([len(g) for g in self.groups],np.int32),self.types,self.allowed]]
        pairs=[(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)];buckets=[[] for _ in range(6*23*23)]
        for code,f in enumerate(self.faces):
            for k,(x,y) in enumerate(pairs):buckets[(k*23+int(f[x]))*23+int(f[y])].append(code)
        offsets=[0];values=[]
        for bucket in buckets:values.extend(bucket);offsets.append(len(values))
        self.pair_offsets=self.xp.asarray(offsets,dtype=np.int32);self.pair_codes=self.xp.asarray(values,dtype=np.int16)
        self.arms=np.arange(n)%8;self.arm_trials=np.zeros(8,np.int64);self.arm_rewards=np.zeros(8,np.float64)
        self.guide_levels=np.array([0,.25,.5,.75,.9,1,.9,.6],np.float32)
        self.guide_prob=self.xp.asarray(self.guide_levels[self.arms])
        self.module=self.runtime.module
        self.search_kernel=self.runtime.search_kernel;self.delta_kernel=self.runtime.delta_kernel;self.score_kernel=self.runtime.score_kernel
        self.rng=self.xp.asarray(self.host_rng.integers(1,2**32,size=n,dtype=np.uint32))
        self.counters=self.xp.zeros(2*n,np.uint64);self.temps=self.xp.ones(n,np.float32)

    def score_cpu(self,boards):
        b=np.asarray(boards);f=self.faces[b]
        return ((f[:,np.arange(256)%16<15,1]==f[:,np.flatnonzero(np.arange(256)%16<15)+1,3]).sum(axis=1)+(f[:,:240,2]==f[:,16:,0]).sum(axis=1)).astype(np.int32)

    def perturb(self,boards,strengths):
        b=boards.copy()
        for idx,num in enumerate(strengths):
            for _ in range(int(num)):
                t=2 if self.host_rng.random()<.9 else int(self.host_rng.integers(0,2))
                a,z=self.host_rng.choice(self.groups[t],2,replace=False)
                pa,pz=int(b[idx,a])//4,int(b[idx,z])//4
                ra=self.host_rng.choice(np.flatnonzero(self.allowed[a,pz*4:pz*4+4]))
                rz=self.host_rng.choice(np.flatnonzero(self.allowed[z,pa*4:pa*4+4]))
                b[idx,a]=pz*4+ra;b[idx,z]=pa*4+rz
        return b

    def initialize(self,seeds):
        self.seeds=np.asarray(seeds,np.int16)
        base=self.seeds[np.arange(self.n)%len(seeds)]
        strengths=np.array([0,2,4,8,12,24,40,80])[np.arange(self.n)%8]
        b=self.perturb(base,strengths)
        self.boards=self.xp.asarray(b.T.copy());self.bestboards=self.xp.asarray(base.T.copy())
        self.scores=self.xp.asarray(self.score_cpu(b));self.bestscores=self.xp.asarray(self.score_cpu(base))
        self.positions=self.xp.asarray(np.argsort(b//4,axis=1).astype(np.int16).T.copy())

    def step(self,steps=32):
        if not isinstance(steps,int) or not 1<=steps<=4096:raise ValueError('steps must be an integer in 1..4096')
        return self.runtime.timed(self.search_kernel,((self.n+127)//128,),(128,),(self.boards,self.bestboards,*self.tables,self.positions,self.pair_offsets,self.pair_codes,self.guide_prob,self.scores,self.bestscores,self.rng,self.temps,self.counters,np.int32(self.n),np.int32(steps)))

    def cool(self,elapsed):
        # Staggered 45-second heat/cool cycles and a range of peak temperatures.
        i=np.arange(self.n);phase=(elapsed/45+i/self.n)%1
        peak=np.array([.35,.5,.7,.9,1.2,1.6,2.0,2.5])[self.arms]
        self.temps.set((.10+peak*(1-phase)**2).astype(np.float32))

    def reseed(self):
        ids=self.host_rng.choice(self.n,max(1,self.n//8),replace=False)
        base=self.seeds[self.host_rng.integers(0,len(self.seeds),size=len(ids))]
        b=self.perturb(base,self.host_rng.choice([2,4,8,12,24,40,80],len(ids)))
        self.boards[:,ids]=self.xp.asarray(b.T.copy());self.scores[ids]=self.xp.asarray(self.score_cpu(b))
        self.positions[:,ids]=self.xp.asarray(np.argsort(b//4,axis=1).astype(np.int16).T.copy())
        self.bestboards[:,ids]=self.xp.asarray(base.T.copy());self.bestscores[ids]=self.xp.asarray(self.score_cpu(base))
        return ids

    def adapt(self):
        # Reward independently validated, locally new boards absent from the library.
        # Equal evidence keeps equal allocation; at least50% remains exploration.
        self.arm_trials+=np.bincount(self.arms,minlength=8)
        rates=self.arm_rewards/np.maximum(1,self.arm_trials)
        weights=np.ones(8)/8 if rates.max()==0 else .5/8+.5*rates/rates.sum()
        self.arms=self.host_rng.choice(8,self.n,p=weights)
        self.guide_prob.set(self.guide_levels[self.arms])

    def strategy_status(self):
        return [{'id':i,'color_guidance':float(self.guide_levels[i]),'replicas':int((self.arms==i).sum()),'reward':float(self.arm_rewards[i]),'replica_epochs':int(self.arm_trials[i])} for i in range(8)]

    def checkpoint(self,path):
        path=Path(path);tmp=path.with_suffix('.tmp.npz')
        np.savez_compressed(tmp,boards=self.boards.get(),bestboards=self.bestboards.get(),scores=self.scores.get(),bestscores=self.bestscores.get(),rng=self.rng.get(),counters=self.counters.get(),seed_pool=self.seeds,arms=self.arms,arm_trials=self.arm_trials,arm_rewards=self.arm_rewards)
        os.replace(tmp,path)

    def resume(self,path):
        """Validate first, then restore/migrate replicas without discarding the source."""
        import shutil
        from validator import validate_board
        path=Path(path)
        with np.load(path,allow_pickle=False) as archive:
            required=('boards','bestboards','scores','bestscores','rng','counters','seed_pool')
            if any(key not in archive for key in required):raise ValueError('Checkpoint is missing required arrays')
            saved={key:archive[key].copy() for key in required}
            if saved['boards'].ndim!=2 or saved['boards'].shape[0]!=256:raise ValueError('Malformed checkpoint boards')
            old_n=saved['boards'].shape[1]
            if not 1<=old_n<=1048576:raise ValueError('Invalid checkpoint replica count')
            expected={'boards':((256,old_n),np.int16),'bestboards':((256,old_n),np.int16),'scores':((old_n,),np.int32),'bestscores':((old_n,),np.int32),'rng':((old_n,),np.uint32),'counters':((2*old_n,),np.uint64)}
            for key,(shape,dtype) in expected.items():
                if saved[key].shape!=shape or saved[key].dtype!=dtype:raise ValueError('Malformed checkpoint array: '+key)
            seeds=saved['seed_pool']
            if seeds.dtype!=np.int16 or seeds.ndim!=2 or seeds.shape[1]!=256 or not 1<=len(seeds)<=1024:raise ValueError('Invalid seed pool')
            for board in seeds:
                if not validate_board(board,self.bundle)['valid']:raise ValueError('Invalid checkpoint seed')
            strategy_keys=('arms','arm_trials','arm_rewards')
            present=[key in archive for key in strategy_keys]
            if any(present) and not all(present):raise ValueError('Incomplete checkpoint strategy state')
            if all(present):
                saved.update({key:archive[key].copy() for key in strategy_keys})
                if saved['arms'].shape!=(old_n,) or saved['arms'].dtype.kind not in 'iu' or np.any((saved['arms']<0)|(saved['arms']>7)):raise ValueError('Invalid strategy array')
                if saved['arm_trials'].shape!=(8,) or saved['arm_trials'].dtype!=np.int64 or np.any(saved['arm_trials']<0):raise ValueError('Invalid strategy trials')
                if saved['arm_rewards'].shape!=(8,) or saved['arm_rewards'].dtype!=np.float64 or not np.all(np.isfinite(saved['arm_rewards'])) or np.any(saved['arm_rewards']<0):raise ValueError('Invalid strategy rewards')
            for label,scorelabel in [('boards','scores'),('bestboards','bestscores')]:
                for board,reported in zip(saved[label].T,saved[scorelabel]):
                    validation=validate_board(board,self.bundle)
                    if not validation['valid']:raise ValueError('Illegal checkpoint board')
                    if validation['score']!=int(reported):raise ValueError('Checkpoint scores failed validation')
        # A changed population size requires a fresh initialized population for
        # added slots. Validate everything above before creating any backup.
        keep=min(old_n,self.n)
        selection=np.argsort(-saved['bestscores'],kind='stable')[:keep] if old_n>self.n else np.arange(keep)
        fields=('boards','bestboards','scores','bestscores','rng','counters')
        if old_n==self.n:
            merged={key:saved[key] for key in fields}
        else:
            if not hasattr(self,'boards'):raise ValueError('Initialize the new replica population before migrating a checkpoint')
            merged={key:getattr(self,key).get() for key in fields}
            for key in ('boards','bestboards'):merged[key][:,:keep]=saved[key][:,selection]
            for key in ('scores','bestscores','rng'):merged[key][:keep]=saved[key][selection]
            merged['counters'][:keep]=saved['counters'][selection]
            merged['counters'][self.n:self.n+keep]=saved['counters'][old_n+selection]
        arms=self.arms.copy()
        if 'arms' in saved:arms[:keep]=saved['arms'][selection]
        backup=None
        if old_n!=self.n:
            backup=path.with_name(f'{path.stem}.replicas-{old_n}{path.suffix}')
            shutil.copy2(path,backup)
        for field in fields:setattr(self,field,self.xp.asarray(merged[field]))
        self.seeds=saved['seed_pool'];self.arms=arms
        self.positions=self.xp.asarray(np.argsort(merged['boards'].T//4,axis=1).astype(np.int16).T.copy())
        if 'arm_trials' in saved:
            self.arm_trials=saved['arm_trials'];self.arm_rewards=saved['arm_rewards']
        self.guide_prob.set(self.guide_levels[self.arms])
        self.resume_info={'saved_replicas':old_n,'restored_replicas':keep,'new_replicas':max(0,self.n-old_n),'backup':str(backup) if backup else None}
        return True

    def verify_device_scores(self,indices):
        from validator import validate_board
        idx=np.asarray(indices,dtype=np.int32);boards=self.boards[:,idx].get().T;reported=self.scores[idx].get()
        for b,s in zip(boards,reported):
            v=validate_board(b,self.bundle)
            if not v['valid'] or v['score']!=int(s):raise RuntimeError('GPU state failed independent CPU validation')
        return len(idx)


def create_engine(bundle,n=4096,seed=20261007,backend='auto',cache_dir=None,*,allow_opencl_cpu=False,device_name=None):
    """Select a real GPU by default; CPU OpenCL is explicitly diagnostic only."""
    return GPU(bundle,n,seed,backend,cache_dir,allow_opencl_cpu=allow_opencl_cpu,device_name=device_name)
