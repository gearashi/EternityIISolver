"""Validated offline adapter for the optional C++17 repair-radius helper."""
from pathlib import Path
import os
import subprocess
import time

from app_paths import resource_root
from exact_model import grid_edges, initial_domains
from exact_optimize import evaluate_board
from repair_transfer_bound import certificate
from repair_triples import search as search_triples


def helper_path():
    name='eternity-radius.exe' if os.name=='nt' else 'eternity-radius'
    return resource_root()/'native'/name


def _decode(stdout,problem,reference,radius,cap,*,conditional=False,expected_rows=None):
    if expected_rows is not None and (not conditional or type(expected_rows) is not int or expected_rows<0):
        raise ValueError('Invalid expected component count')
    rows=stdout.splitlines()
    if not rows:
        raise RuntimeError('Radius helper returned empty output')
    head=rows[0].split()
    # V1 research helpers can ignore the extra selector. A v2 scoped
    # negative requires a receipt from the requested mode.
    allowed = ('family_complete','complete','work_limit') if conditional else ('complete','work_limit')
    if len(head)!=9 or head[0] not in allowed:
        raise RuntimeError('Unrecognized radius-helper protocol')
    names=('gain','work','partial_sets','tested_sets','exterior_pruned','allocations','tuples','setup_state_visits')
    try:
        counts=dict(zip(names,map(int,head[1:])))
    except ValueError as exc:
        raise RuntimeError('Invalid radius-helper counters') from exc
    spent=sum(counts[k] for k in ('partial_sets','tested_sets','allocations','tuples','setup_state_visits'))
    if (any(v<0 for v in counts.values()) or counts['work']!=spent or spent>cap
            or counts['exterior_pruned']>counts['tested_sets'] or counts['gain']>4*radius):
        raise RuntimeError('Inconsistent radius-helper counters')
    if head[0]=='work_limit' and (counts['gain'] or counts['work']!=cap):
        raise RuntimeError('Incomplete native run reported a gain or an inconsistent limit')
    if expected_rows is not None and head[0] in ('work_limit','family_complete'):
        # Work-limited unrestricted helpers may lack the optional diagnostic.
        # A scoped negative proof always needs a complete v2 row receipt.
        receipt=rows[-1].split()
        if receipt[:1]==['component_rows']:
            try:
                started,finished=map(int,receipt[1:])
            except ValueError as exc:
                raise RuntimeError('Invalid component-row receipt') from exc
            if not (0<=finished<=started<=expected_rows and started-finished<=1 and finished<=counts['partial_sets']):
                raise RuntimeError('Inconsistent component-row coverage')
            if head[0]=='family_complete':
                if counts['gain'] and started!=finished+1:
                    raise RuntimeError('Invalid early component witness')
                if not counts['gain'] and finished!=expected_rows:
                    raise RuntimeError('Incomplete component-row coverage')
            counts.update(component_rows_started=started,component_rows_finished=finished)
            rows.pop()
        elif head[0]=='family_complete':
            raise RuntimeError('Missing component-row coverage')
    if len(rows)!=1+int(counts['gain']>0):
        raise RuntimeError('Inconsistent radius-helper witness count')
    board=list(reference)
    initial=evaluate_board(problem,reference)
    if counts['gain']:
        try:
            board=list(map(int,rows[1].split()))
        except ValueError as exc:
            raise RuntimeError('Noninteger state in native radius witness') from exc
        actual=evaluate_board(problem,board)
        if (not actual['valid'] or actual['score']!=initial['score']+counts['gain']
                or sum(a!=b for a,b in zip(reference,board))!=radius):
            raise RuntimeError('Native radius witness failed independent score, distance or legality validation')
    if expected_rows is not None and head[0]=='complete' and not counts['gain']:
        # Returning an unrestricted header cannot establish that the helper
        # accepted or processed the serialized component family.
        return 'native_helper_incompatible',counts,board
    return head[0],counts,board


def search(problem,reference,*,radius,seconds=120,max_work=None,helper=None):
    if type(radius) is not int or radius not in (4,5,6,7,8):
        raise ValueError('native radius must be between 4 and 8')
    if type(seconds) not in (int,float) or not 0<=seconds<float('inf'):
        raise ValueError('seconds must be finite and nonnegative')
    if max_work is not None and (type(max_work) is not int or not 0<=max_work<2**63):
        raise ValueError('max_work must be a nonnegative signed 64-bit integer')
    reference=tuple(reference)
    budget=(2**63-1) if max_work is None else max_work
    deadline=time.monotonic()+seconds
    prefix=search_triples(problem,reference,max_work=budget,should_stop=lambda:time.monotonic()>=deadline)
    used=prefix['work']
    completed=prefix['max_exhausted_radius']
    phases=[]
    def finish(why,board=None):
        b=list(reference if board is None else board)
        points=evaluate_board(problem,b)['score']
        return dict(termination=why,score=points,reference_score=prefix['reference_score'],
                    radius=radius,max_exhausted_radius=completed,board=b,
                    improved=points>prefix['reference_score'],work=used,
                    proof_scope='strict improvement from supplied board within changed-cell radius only',
                    lower_radius_counts=prefix['counts'],native_phases=phases)
    if prefix['termination']=='already_perfect':
        completed=radius
        return finish('already_perfect')
    if prefix['termination']!='exhausted':
        return finish('timeout' if prefix['termination']=='stopped' else prefix['termination'],prefix['board'])
    if time.monotonic()>=deadline:
        return finish('timeout')
    prices=certificate(problem,reference,max_work=budget-used,should_stop=lambda:time.monotonic()>=deadline)
    used+=prices['work']
    phases.append(dict(phase='transfer_prices',certified=prices['certified'],work=prices['work']))
    if prices['status']=='positive_cycle':
        # A failed relaxation sometimes supplies a physically improving
        # permutation. It qualifies only after rescoring its real edges and
        # checking its changed-cell distance against the requested radius.
        candidate=prices['board']
        replay=evaluate_board(problem,candidate)
        distance=sum(a!=b for a,b in zip(reference,candidate))
        if not replay['valid']:
            raise RuntimeError('Transfer cycle failed independent board validation')
        phases[-1].update(witness_score=replay['score'],witness_distance=distance)
        if replay['score']>prefix['reference_score'] and distance<=radius:
            if distance<=completed:
                raise RuntimeError('Transfer witness contradicts the exhausted radius')
            return finish('improved',candidate)
    if not prices['certified']:
        status={'stopped':'timeout','work_limit':'work_limit','positive_cycle':'transfer_bound_unavailable'}
        return finish(status[prices['status']])
    exe=helper_path() if helper is None else Path(helper).resolve()
    if not exe.is_file():
        return finish('native_helper_unavailable')
    # Every input-domain scan and serialized face, price, parent and state is
    # charged as setup, separately from the helper's internal search counters.
    setup=len(problem.faces)+2*len(reference)+2*sum(map(len,problem.domains))
    if setup>budget-used:
        return finish('work_limit')
    ds=initial_domains(problem)
    used+=setup
    phases.append(dict(phase='serialize',work=setup))
    body=[' '.join(map(str,f)) for f in problem.faces]
    body+=[' '.join(map(str,reference)),' '.join(map(str,prices['potentials']))]
    body+=[' '.join(map(str,(len(d),*d))) for d in ds]
    body='\n'.join(body)+'\n'
    from repair_component_family import prepare as positive_cover
    from repair_signed_cover import decode as decode_signed, prepare as signed_cover
    flags=getattr(subprocess,'CREATE_NO_WINDOW',0)
    for k in range(4,radius+1):
        methods=['unrestricted'] if k==4 else ['signed']
        if k in (5,6):
            # An older trusted helper can still run its original method when
            # it explicitly rejects the new signed selector before searching.
            methods.append('one_positive')
        if k==5:methods.append('unrestricted')
        for index,method in enumerate(methods):
            command=[str(exe)]
            suffix=''
            cover=None
            if method!='unrestricted':
                if method=='signed':
                    cover,suffix=signed_cover(problem,reference,prices['potentials'],k,
                                              max_work=budget-used,deadline=deadline)
                    command.append('--signed-family')
                else:
                    cover,suffix=positive_cover(problem,reference,prices['potentials'],k,budget-used,deadline)
                    command.append('--component-family-v2')
                used+=cover['work']
                phases.append(dict(phase='component_cover',method=method,radius=k,**cover))
                if cover['status']!='complete':
                    if cover['status']=='timeout' or index==len(methods)-1:
                        return finish(cover['status'])
                    continue
            remaining=deadline-time.monotonic()
            if remaining<=0:return finish('timeout')
            cap=budget-used
            if cap<1:return finish('work_limit')
            try:
                run=subprocess.run(command,input=f'{problem.size} {k} {cap}\n'+body+suffix,
                                   text=True,capture_output=True,timeout=remaining,check=True,creationflags=flags)
            except subprocess.CalledProcessError as exc:
                if cover is None or exc.returncode!=2:raise
                phases.append(dict(phase='native_radius',radius=k,method=method,
                                   status='native_helper_incompatible',work=0))
                if method=='signed' and index+1<len(methods):continue
                return finish('native_helper_incompatible')
            except subprocess.TimeoutExpired:
                phases.append(dict(phase='native_radius',radius=k,method=method,status='timeout',work=None))
                return finish('timeout')
            if method=='signed' and run.stdout.startswith('complete '):
                # Standalone legacy helpers may ignore unknown switches. A
                # lower proof remains available, positive boards can be checked.
                status,counts,board=_decode(run.stdout,problem,reference,k,cap)
                if not counts['gain']:status='native_helper_incompatible'
            elif method=='signed':
                status,counts,board=decode_signed(run.stdout,problem,reference,k,cap,cover)
            else:
                expected=cover['positive_shapes'] if cover is not None else None
                status,counts,board=_decode(run.stdout,problem,reference,k,cap,
                                           conditional=cover is not None,expected_rows=expected)
            used+=counts['work']
            phases.append(dict(phase='native_radius',radius=k,method=method,status=status,counts=counts))
            if counts['gain']:return finish('improved',board)
            if status in ('work_limit','native_helper_incompatible'):return finish(status)
            completed=k
            break
    return finish('exhausted')
