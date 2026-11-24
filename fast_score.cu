// Per-lane persistent row-major score DFS. All state has a lane dimension.
typedef unsigned long long U64;
typedef unsigned short U16;
typedef unsigned char U8;

__device__ U64 rotl64(U64 x, int k) { return (x << k) | (x >> (64-k)); }
__device__ U64 splitmix(U64 &x) {
    U64 z = (x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
__device__ U64 next_rng(U64 *s) {
    U64 result = rotl64(s[1]*5, 7)*9, t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
    s[2] ^= t; s[3] = rotl64(s[3],45); return result;
}

extern "C" __global__ void make_orders(int lanes, int first, int last, U64 seed,
    const int *restart, const int *active, U64 *rng, const int *starts, const int *ends,
    const U16 *base, const U8 *depletion, U16 *order) {
    int lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= lanes || !active[lane]) return;
    U64 s[4];
    if (first == 0) {
        U64 x = seed * 0x100000001B3ULL ^ ((U64)lane + 0x632BE59BD9B4E019ULL) * 0xD6E8FEB86659FD93ULL
              ^ rotl64((U64)restart[lane],32);
        for (int k=0;k<4;k++) s[k]=splitmix(x);
    } else {
        for (int k=0;k<4;k++) s[k]=rng[k*lanes+lane];
    }
    for (int key=first; key<last; key++) {
        int a=starts[key], b=ends[key];
        for (int j=a;j<b;j++) order[j*lanes+lane]=base[j];
        for (int j=b-1;j>a;j--) {
            int k=a+(int)(next_rng(s)%(U64)(j-a+1));
            U16 swap=order[j*lanes+lane];
            order[j*lanes+lane]=order[k*lanes+lane];
            order[k*lanes+lane]=swap;
        }
        // Stable descending depletion count, as in the CPU helper's hsort.
        for (int j=a+1;j<b;j++) {
            U16 value=order[j*lanes+lane];
            int k=j, h=depletion[value];
            while (k>a && depletion[order[(k-1)*lanes+lane]]<h) {
                order[k*lanes+lane]=order[(k-1)*lanes+lane]; --k;
            }
            order[k*lanes+lane]=value;
        }
    }
    for (int k=0;k<4;k++) rng[k*lanes+lane]=s[k];
}

// Status per lane: 0 searching, 1 exhausted, 2 emitted a board, 3 restart due.
extern "C" __global__ void advance(int lanes, int n, int steps, U64 restart_limit,
    const int *sig, const int *offsets, const U16 *order, const unsigned *faces,
    const U8 *depletion, const U16 *need, const U8 *allowed,
    int *depth, U8 *seg, int *cursor, int *end, U16 *board, U8 *breaks, U16 *hsum,
    unsigned *used, U64 *nodes, U64 *restart_nodes, int *deepest, int *status) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=lanes || status[i]) return;
    int d=depth[i], maxd=deepest[i], cells=n*n;
    U64 placed=0, internal=restart_nodes[i];
    unsigned mask[8];
    #pragma unroll
    for (int k=0;k<8;k++) mask[k]=used[k*lanes+i];
    for (int tick=0;tick<steps;tick++) {
        int p=d*lanes+i;
        int left=(d%n) ? (faces[board[p-lanes]] >> 5)&31 : 0;
        int top=(d>=n) ? (faces[board[p-n*lanes]] >> 10)&31 : 0;
        int s=seg[p];
        if (s==255 || cursor[p]==end[p]) {
            if (s==255) s=0;
            else if (s<2 && breaks[p]<allowed[d]) {
                ++s;
                if (s==1 && d<n) ++s;
                if (s==2 && d%n==0) ++s;
            } else s=3;
            if (s==3) {
                if (!d) { status[i]=1; break; }
                --d; int piece=board[d*lanes+i]/4;
                mask[piece>>5] &= ~(1u<<(piece&31));
                continue;
            }
            int key=sig[d]+(s==0 ? left*32+top : s==1 ? 1024+left : 1056+top);
            seg[p]=(U8)s; cursor[p]=offsets[key]; end[p]=offsets[key+1];
            if (cursor[p]==end[p]) continue;
        }
        U16 state=order[(cursor[p]++)*lanes+i];
        int piece=state/4; unsigned bit=1u<<(piece&31);
        if (mask[piece>>5]&bit) continue;
        unsigned f=faces[state];
        if ((s==1 && (f&31)==top) || (s==2 && (int)(f>>15)==left)) continue;
        int h=hsum[p]+depletion[state];
        if (h<need[d+1]) continue;
        ++placed;
        board[p]=state; hsum[p+lanes]=h; breaks[p+lanes]=breaks[p]+(s!=0);
        maxd=max(maxd,d+1);
        if (d+1==cells) { status[i]=2; break; }
        mask[piece>>5]|=bit; ++d; seg[d*lanes+i]=255;
        ++internal;
        if (restart_limit && internal>=restart_limit) {status[i]=3;break;}
    }
    #pragma unroll
    for(int k=0;k<8;k++) used[k*lanes+i]=mask[k];
    nodes[i]+=placed; restart_nodes[i]=internal; depth[i]=d; deepest[i]=maxd;
}
