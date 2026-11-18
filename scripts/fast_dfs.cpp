/* eternity-fast: compiled edge-matching backtracker for Eternity II style puzzles.
 *
 * Scan order is row-major. Each cell looks up its candidates in a table keyed by
 * (left color, top color); a 256-byte "used" array enforces each physical piece
 * once. Border classes, clue cells and the cells left of / above a clue get
 * their own filtered tables, so gray stays exactly on the frame and clue edges
 * are enforced when the neighbor is placed.
 *
 * Modes
 *   strict   (no --breaks): every shared edge must match. A complete board is a
 *            full solution.
 *   breaks   (--breaks d1,d2,...): the k-th mismatching edge may be taken only at
 *            cell index >= dk. A complete board then scores EDGES - breaks.
 * Randomized restarts (--restart-nodes) reshuffle the candidate order from the
 * seed. --partition makes all threads share one order and split the depth-P
 * prefixes between them, so no two threads search the same branch.
 * Depletion colors (--hcolors with --hsched or --hdepth) are a score-search
 * heuristic: a minimum number of sides in those colors must be used by each
 * cell index, which leaves an easier piece set for the last rows. It removes
 * branches, so it is never combined with an exhaustive strict search by the app.
 *
 * The solver never judges its own boards: it writes them as state lists
 * (state = 4*piece + clockwise rotation, row-major cells) for an independent
 * checker. Standard C++17 only (std::thread), no other dependency, no network,
 * no I/O outside --out.
 */
#define _CRT_SECURE_NO_WARNINGS
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>
#ifdef _WIN32
#include <filesystem>
#include <io.h>
#include <system_error>
#define LOW_READ _read
#define FAST_REPLACE_RETRY 1
#else
#include <unistd.h>
#define LOW_READ read
#endif

#define MAXN 16
#define MAXCELLS 256
#define NCOL 32
#define MAXSIG 64
#define MAXTHREADS 64
#define VERSION "eternity-fast-1"

typedef struct { uint8_t piece, rot, top, right, bottom, left, h, pad1; } Cand;

typedef struct {
    int n;
    Cand *ex;  int ex_start[NCOL * NCOL + 1];
    Cand *byL; int l_start[NCOL + 1];
    Cand *byT; int t_start[NCOL + 1];
} Sig;

typedef struct { int gray[4]; int reqR, reqB, clue; } SigKey;

/* ---- problem ---- */
static int N, CELLS, PIECES, EDGES;
static uint8_t face[MAXCELLS * 4][4];          /* [state][T,R,B,L] */
static int clue_at[MAXCELLS];                  /* state or -1 */
static uint8_t is_clue_piece[MAXCELLS];
static int cellsig[MAXCELLS];
static SigKey sigkeys[MAXSIG];
static Sig base_sig[MAXSIG];
static int nsig;
static uint8_t allowed[MAXCELLS + 1];          /* max breaks usable up to and including cell d */
static int nbreaks;
static uint64_t problem_hash;
static uint8_t is_h[NCOL];                     /* depletion colors */
static int need[MAXCELLS + 1];                 /* min depletion-color sides among the first d cells */
static int h_total, opt_hsort = 1;

/* ---- options ---- */
static int opt_threads = 1, opt_count = 0, opt_partition = 0, opt_prefix = 4, opt_keep = 0;
static uint64_t opt_seed = 1, opt_restart_nodes = 0, opt_max_nodes = 0;
static double opt_seconds = 0, opt_report = 10, opt_ckpt = 60;
static const char *opt_out = ".";
static const char *opt_stop_file = NULL;
static int opt_stdin_guard = 0;
static int opt_resume = 0, opt_shuffle = 1, opt_min_save = -1;

/* ---- shared state ---- */
static volatile std::sig_atomic_t stop_flag = 0;
static std::mutex best_mu;
static std::atomic<int> best_score(-1);
static std::atomic<int> solved(0);
static double t0;

typedef struct {
    int id;
    Sig sig[MAXSIG];
    uint64_t restart;
    std::atomic<uint64_t> nodes, complete, perfect;
    uint64_t nodes_restart, prefix_id;
    std::atomic<int> maxdepth;
    int depth;
    int done;
    /* stack */
    const Cand *p[MAXCELLS], *e[MAXCELLS];
    uint8_t seg[MAXCELLS], L[MAXCELLS], T[MAXCELLS];
    uint8_t pc[MAXCELLS], rotv[MAXCELLS], rt[MAXCELLS], bt[MAXCELLS];
    uint8_t brk[MAXCELLS + 1];
    uint16_t hs[MAXCELLS + 1];
    uint8_t used[MAXCELLS];
    double last_ckpt;
    uint64_t hist[MAXCELLS + 1];                 /* session placements per cell index (not checkpointed) */
} Thread;

static Thread *threads[MAXTHREADS];

static double now(void) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static void die(const char *msg) { fprintf(stderr, "eternity-fast: %s\n", msg); exit(2); }

/* splitmix64 / xoshiro256** */
typedef struct { uint64_t s[4]; } Rng;
static uint64_t splitmix(uint64_t *x) {
    uint64_t z = (*x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static void rng_seed(Rng *r, uint64_t a, uint64_t b, uint64_t c) {
    uint64_t x = a * 0x100000001B3ULL ^ (b + 0x632BE59BD9B4E019ULL) * 0xD6E8FEB86659FD93ULL ^ (c << 32 | c >> 32);
    for (int i = 0; i < 4; i++) r->s[i] = splitmix(&x);
}
static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
static uint64_t rng_next(Rng *r) {
    uint64_t *s = r->s, res = rotl(s[1] * 5, 7) * 9, t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = rotl(s[3], 45);
    return res;
}
static void shuffle(Rng *r, Cand *a, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = (int)(rng_next(r) % (uint64_t)(i + 1));
        Cand t = a[i]; a[i] = a[j]; a[j] = t;
    }
}

/* ---- table construction ---- */
static int cmp_ex(const void *a, const void *b) {
    const Cand *x = (const Cand *)a, *y = (const Cand *)b;
    int kx = x->left * NCOL + x->top, ky = y->left * NCOL + y->top;
    if (kx != ky) return kx - ky;
    return (x->piece * 4 + x->rot) - (y->piece * 4 + y->rot);
}
static int cmp_l(const void *a, const void *b) {
    const Cand *x = (const Cand *)a, *y = (const Cand *)b;
    if (x->left != y->left) return x->left - y->left;
    return (x->piece * 4 + x->rot) - (y->piece * 4 + y->rot);
}
static int cmp_t(const void *a, const void *b) {
    const Cand *x = (const Cand *)a, *y = (const Cand *)b;
    if (x->top != y->top) return x->top - y->top;
    return (x->piece * 4 + x->rot) - (y->piece * 4 + y->rot);
}

static void build_sig(Sig *s, const SigKey *k) {
    Cand tmp[MAXCELLS * 4];
    int n = 0;
    for (int st = 0; st < PIECES * 4; st++) {
        if (k->clue >= 0 ? st != k->clue : is_clue_piece[st / 4]) continue;
        const uint8_t *f = face[st];
        int ok = 1;
        for (int side = 0; side < 4; side++) if ((f[side] == 0) != k->gray[side]) ok = 0;
        if (k->reqR >= 0 && f[1] != k->reqR) ok = 0;
        if (k->reqB >= 0 && f[2] != k->reqB) ok = 0;
        if (!ok) continue;
        Cand c = { (uint8_t)(st / 4), (uint8_t)(st % 4), f[0], f[1], f[2], f[3], (uint8_t)(is_h[f[0]] + is_h[f[1]] + is_h[f[2]] + is_h[f[3]]), 0 };
        tmp[n++] = c;
    }
    s->n = n;
    s->ex = (Cand *)malloc(sizeof(Cand) * (n + 1)); s->byL = (Cand *)malloc(sizeof(Cand) * (n + 1)); s->byT = (Cand *)malloc(sizeof(Cand) * (n + 1));
    memcpy(s->ex, tmp, sizeof(Cand) * n); memcpy(s->byL, tmp, sizeof(Cand) * n); memcpy(s->byT, tmp, sizeof(Cand) * n);
    qsort(s->ex, n, sizeof(Cand), cmp_ex); qsort(s->byL, n, sizeof(Cand), cmp_l); qsort(s->byT, n, sizeof(Cand), cmp_t);
    int i = 0;
    for (int key = 0; key <= NCOL * NCOL; key++) {
        while (i < n && s->ex[i].left * NCOL + s->ex[i].top < key) i++;
        s->ex_start[key] = i;
    }
    i = 0;
    for (int key = 0; key <= NCOL; key++) { while (i < n && s->byL[i].left < key) i++; s->l_start[key] = i; }
    i = 0;
    for (int key = 0; key <= NCOL; key++) { while (i < n && s->byT[i].top < key) i++; s->t_start[key] = i; }
}

static void build_tables(void) {
    nsig = 0;
    for (int c = 0; c < CELLS; c++) {
        int r = c / N, col = c % N;
        SigKey k; memset(&k, 0, sizeof k);
        k.gray[0] = r == 0; k.gray[1] = col == N - 1; k.gray[2] = r == N - 1; k.gray[3] = col == 0;
        k.reqR = -1; k.reqB = -1; k.clue = clue_at[c];
        if (col + 1 < N && clue_at[c + 1] >= 0) k.reqR = face[clue_at[c + 1]][3];
        if (r + 1 < N && clue_at[c + N] >= 0) k.reqB = face[clue_at[c + N]][0];
        int found = -1;
        for (int i = 0; i < nsig; i++) if (!memcmp(&sigkeys[i], &k, sizeof k)) { found = i; break; }
        if (found < 0) {
            if (nsig == MAXSIG) die("too many table signatures");
            sigkeys[nsig] = k; build_sig(&base_sig[nsig], &k); found = nsig++;
        }
        cellsig[c] = found;
    }
}

static void hsort(Cand *a, int n) { /* stable, most depletion-color sides first */
    for (int i = 1; i < n; i++) {
        Cand x = a[i]; int j = i - 1;
        while (j >= 0 && a[j].h < x.h) { a[j + 1] = a[j]; j--; }
        a[j + 1] = x;
    }
}

static void thread_tables(Thread *t) {
    Rng r;
    rng_seed(&r, opt_seed, opt_partition ? 0 : (uint64_t)t->id, t->restart);
    for (int i = 0; i < nsig; i++) {
        Sig *d = &t->sig[i]; const Sig *b = &base_sig[i];
        if (!d->ex) {
            *d = *b;
            d->ex = (Cand *)malloc(sizeof(Cand) * (b->n + 1)); d->byL = (Cand *)malloc(sizeof(Cand) * (b->n + 1)); d->byT = (Cand *)malloc(sizeof(Cand) * (b->n + 1));
        }
        memcpy(d->ex, b->ex, sizeof(Cand) * b->n); memcpy(d->byL, b->byL, sizeof(Cand) * b->n); memcpy(d->byT, b->byT, sizeof(Cand) * b->n);
        if (opt_shuffle) {
            for (int k = 0; k < NCOL * NCOL; k++) shuffle(&r, d->ex + d->ex_start[k], d->ex_start[k + 1] - d->ex_start[k]);
            for (int k = 0; k < NCOL; k++) shuffle(&r, d->byL + d->l_start[k], d->l_start[k + 1] - d->l_start[k]);
            for (int k = 0; k < NCOL; k++) shuffle(&r, d->byT + d->t_start[k], d->t_start[k + 1] - d->t_start[k]);
        }
        if (h_total && opt_hsort) {
            for (int k = 0; k < NCOL * NCOL; k++) hsort(d->ex + d->ex_start[k], d->ex_start[k + 1] - d->ex_start[k]);
            for (int k = 0; k < NCOL; k++) hsort(d->byL + d->l_start[k], d->l_start[k + 1] - d->l_start[k]);
            for (int k = 0; k < NCOL; k++) hsort(d->byT + d->t_start[k], d->t_start[k + 1] - d->t_start[k]);
        }
    }
}

/* ---- output ---- */
static void write_atomic(const char *path, const char *text) {
    char tmp[1200];
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
    FILE *f = fopen(tmp, "wb");
    if (!f) { fprintf(stderr, "eternity-fast: cannot write %s\n", tmp); return; }
    fputs(text, f); fclose(f);
#ifdef FAST_REPLACE_RETRY
    /* Windows refuses to replace a file while a reader has it open, so a report
     * could be lost when the app polled at the same moment. Try again until the
     * reader has closed it; a reader holds the file for well under a millisecond. */
    for (int attempt = 0; attempt < 600; attempt++) {
        std::error_code failure;
        std::filesystem::rename(tmp, path, failure);
        if (!failure) return;
        std::this_thread::sleep_for(std::chrono::milliseconds(attempt < 100 ? 1 : 10));
    }
    fprintf(stderr, "eternity-fast: cannot replace %s\n", path);
#else
    if (rename(tmp, path)) fprintf(stderr, "eternity-fast: cannot rename %s\n", tmp);
#endif
}

static void save_board(Thread *t, int score, uint64_t nodes, int is_best) {
    char buf[MAXCELLS * 6 + 600], path[1100];
    int o = snprintf(buf, sizeof buf, "{\"engine\":\"%s\",\"claimed_score\":%d,\"edges\":%d,\"seed\":%" PRIu64 ",\"thread\":%d,\"restart\":%" PRIu64
                     ",\"thread_nodes\":%" PRIu64 ",\"problem\":\"%016" PRIx64 "\",\"board\":[", VERSION, score, EDGES, opt_seed, t->id, t->restart, nodes, problem_hash);
    for (int c = 0; c < CELLS; c++) o += snprintf(buf + o, sizeof buf - o, "%s%d", c ? "," : "", t->pc[c] * 4 + t->rotv[c]);
    snprintf(buf + o, sizeof buf - o, "]}\n");
    if (is_best) { snprintf(path, sizeof path, "%s/best.json", opt_out); write_atomic(path, buf); }
    if (opt_keep && score >= opt_min_save) {
        snprintf(path, sizeof path, "%s/board_%d_t%d_r%" PRIu64 "_n%" PRIu64 ".json", opt_out, score, t->id, t->restart, nodes);
        write_atomic(path, buf);
    }
}

/* ---- checkpoint ---- */
static void ckpt_path(char *path, size_t n, int id) { snprintf(path, n, "%s/ckpt_t%d.txt", opt_out, id); }

static const Cand *seg_base(Thread *t, int d, int seg, const Cand **end) {
    const Sig *s = &t->sig[cellsig[d]];
    if (seg == 0) { int k = t->L[d] * NCOL + t->T[d]; *end = s->ex + s->ex_start[k + 1]; return s->ex + s->ex_start[k]; }
    if (seg == 1) { *end = s->byL + s->l_start[t->L[d] + 1]; return s->byL + s->l_start[t->L[d]]; }
    *end = s->byT + s->t_start[t->T[d] + 1]; return s->byT + s->t_start[t->T[d]];
}

static void save_ckpt(Thread *t) {
    char local[MAXCELLS * 16 + 800], path[1100];
    int o = snprintf(local, sizeof local, "%s %016" PRIx64 " %" PRIu64 " %d %" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 " %d %d %d\n",
                     VERSION, problem_hash, opt_seed, t->id, t->restart, t->nodes.load(), t->nodes_restart, t->prefix_id,
                     t->complete.load(), t->perfect.load(), t->maxdepth.load(), t->done, t->depth);
    for (int d = 0; d <= t->depth && d < CELLS && !t->done; d++) {
        const Cand *end; const Cand *base = seg_base(t, d, t->seg[d], &end);
        o += snprintf(local + o, sizeof local - o, "%d %d\n", t->seg[d], (int)(t->p[d] - base));
    }
    ckpt_path(path, sizeof path, t->id);
    write_atomic(path, local);
}

static int load_ckpt(Thread *t) {
    char path[1100], ver[64];
    ckpt_path(path, sizeof path, t->id);
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    uint64_t ph, seed, restart, nodes, nr, pid, complete, perfect; int id, maxd, done, depth;
    if (fscanf(f, "%63s %" SCNx64 " %" SCNu64 " %d %" SCNu64 " %" SCNu64 " %" SCNu64 " %" SCNu64 " %" SCNu64 " %" SCNu64 " %d %d %d",
               ver, &ph, &seed, &id, &restart, &nodes, &nr, &pid, &complete, &perfect, &maxd, &done, &depth) != 13) die("unreadable checkpoint");
    if (strcmp(ver, VERSION) || ph != problem_hash || seed != opt_seed || id != t->id) die("checkpoint belongs to a different problem, seed, options or thread");
    t->restart = restart; t->nodes = nodes; t->nodes_restart = nr; t->prefix_id = pid; t->complete = complete; t->perfect = perfect;
    t->maxdepth = maxd; t->done = done;
    thread_tables(t);
    memset(t->used, 0, sizeof t->used);
    t->brk[0] = 0; t->hs[0] = 0;
    if (done) { fclose(f); return 1; }
    for (int d = 0; d <= depth; d++) {
        int seg, off;
        if (fscanf(f, "%d %d", &seg, &off) != 2 || seg < 0 || seg > 2) die("truncated checkpoint");
        t->L[d] = (d % N) ? t->rt[d - 1] : 0;
        t->T[d] = (d >= N) ? t->bt[d - N] : 0;
        const Cand *end; const Cand *base = seg_base(t, d, seg, &end);
        if (off < 0 || base + off > end) die("checkpoint offset out of range");
        t->seg[d] = (uint8_t)seg; t->p[d] = base + off; t->e[d] = end;
        if (d == depth) break;
        if (off == 0) die("checkpoint has an unplaced ancestor");
        const Cand *c = base + off - 1;
        if (t->used[c->piece] || (seg == 1 && c->top == t->T[d]) || (seg == 2 && c->left == t->L[d])) die("checkpoint replays an illegal placement");
        if (seg && !(t->brk[d] < allowed[d])) die("checkpoint exceeds the break schedule");
        t->hs[d + 1] = (uint16_t)(t->hs[d] + c->h);
        if (t->hs[d + 1] < need[d + 1]) die("checkpoint violates the depletion schedule");
        t->used[c->piece] = 1; t->pc[d] = c->piece; t->rotv[d] = c->rot; t->rt[d] = c->right; t->bt[d] = c->bottom;
        t->brk[d + 1] = (uint8_t)(t->brk[d] + (seg ? 1 : 0));
    }
    t->depth = depth;
    fclose(f);
    return 1;
}

/* ---- search ---- */
static inline void setup(Thread *t, int d) {
    int L = (d % N) ? t->rt[d - 1] : 0, T = (d >= N) ? t->bt[d - N] : 0;
    const Sig *s = &t->sig[cellsig[d]];
    int k = L * NCOL + T;
    t->L[d] = (uint8_t)L; t->T[d] = (uint8_t)T; t->seg[d] = 0;
    t->p[d] = s->ex + s->ex_start[k]; t->e[d] = s->ex + s->ex_start[k + 1];
}

static void run(Thread *t) {
    int d = t->depth;
    uint64_t nodes = t->nodes, local = 0;
    int maxd = t->maxdepth;
    const int parts = opt_threads, pdepth = opt_prefix;
    if (t->done) return;
    for (;;) {
        if (t->p[d] == t->e[d]) {
            int sg = t->seg[d];
            if (sg < 2 && t->brk[d] < allowed[d]) {
                const Sig *s = &t->sig[cellsig[d]];
                if (sg == 0) {
                    t->seg[d] = 1;
                    if (d >= N) { t->p[d] = s->byL + s->l_start[t->L[d]]; t->e[d] = s->byL + s->l_start[t->L[d] + 1]; }
                    else t->p[d] = t->e[d] = s->byL + s->l_start[t->L[d] + 1];
                    continue;
                }
                t->seg[d] = 2;
                if (d % N) { t->p[d] = s->byT + s->t_start[t->T[d]]; t->e[d] = s->byT + s->t_start[t->T[d] + 1]; }
                else t->p[d] = t->e[d] = s->byT + s->t_start[t->T[d] + 1];
                continue;
            }
            if (d == 0) { t->done = 1; break; }
            d--;
            t->used[t->pc[d]] = 0;
            continue;
        }
        const Cand *c = t->p[d]++;
        if (t->used[c->piece]) continue;
        int sg = t->seg[d];
        if (sg) { if (sg == 1 ? c->top == t->T[d] : c->left == t->L[d]) continue; }
        int hsum = t->hs[d] + c->h;
        if (hsum < need[d + 1]) continue;
        t->hs[d + 1] = (uint16_t)hsum;
        if (opt_partition && d == pdepth - 1) { if ((int)(t->prefix_id++ % (uint64_t)parts) != t->id) continue; }
        nodes++;
        t->hist[d]++;
        t->pc[d] = c->piece; t->rotv[d] = c->rot; t->rt[d] = c->right; t->bt[d] = c->bottom;
        t->brk[d + 1] = (uint8_t)(t->brk[d] + (sg ? 1 : 0));
        if (d + 1 == CELLS) {
            int score = EDGES - t->brk[CELLS];
            t->complete++;
            if (score == EDGES) t->perfect++;
            t->nodes = nodes; maxd = CELLS; t->maxdepth = CELLS;
            if (score > best_score || (opt_keep && score >= opt_min_save)) {
                std::lock_guard<std::mutex> lock(best_mu);
                int is_best = score > best_score;
                if (is_best) best_score = score;
                save_board(t, score, nodes, is_best);
            }
            if (score == EDGES && !opt_count) { solved = 1; stop_flag = 1; }
            continue; /* the last piece stays unmarked; try its siblings */
        }
        t->used[c->piece] = 1;
        d++;
        if (d > maxd) maxd = d;
        setup(t, d);
        if ((++local & 0xFFFFF) == 0) {
            t->nodes = nodes; t->nodes_restart += 0x100000; t->maxdepth = maxd;
            if (stop_flag || (opt_max_nodes && nodes >= opt_max_nodes)) break;
            double tn = now();
            if (opt_seconds > 0 && tn - t0 >= opt_seconds) break;
            if (opt_restart_nodes && t->nodes_restart >= opt_restart_nodes) {
                t->restart++; t->nodes_restart = 0;
                thread_tables(t);
                memset(t->used, 0, sizeof t->used);
                d = 0; setup(t, 0);
            }
            if (opt_ckpt > 0 && tn - t->last_ckpt >= opt_ckpt) { t->depth = d; save_ckpt(t); t->last_ckpt = tn; }
        }
    }
    t->nodes = nodes; t->depth = d; t->maxdepth = maxd;
    save_ckpt(t);
}

/* ---- setup ---- */
static void on_signal(int s) { (void)s; stop_flag = 1; }

/* The launching app keeps our stdin pipe open. End of input means the app is
 * gone (even if it was killed), so the search stops and saves its checkpoints
 * instead of running on unattended. Unbuffered read: no stdio lock is held. */
static void stdin_guard(void) {
    char buf[64];
    while (LOW_READ(0, buf, sizeof buf) > 0) {}
    stop_flag = 1;
}

static void load_pieces(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) die("cannot open piece file");
    int u, dn, l, r, n = 0;
    while (fscanf(f, "%d %d %d %d", &u, &dn, &l, &r) == 4) {
        if (n == MAXCELLS) die("too many pieces");
        int base[4] = { u, r, dn, l };
        for (int k = 0; k < 4; k++) if (base[k] < 0 || base[k] >= NCOL) die("color out of range");
        for (int rot = 0; rot < 4; rot++)
            for (int side = 0; side < 4; side++) face[n * 4 + rot][side] = (uint8_t)base[(side - rot + 4) % 4];
        n++;
    }
    fclose(f);
    PIECES = n;
}

static uint64_t fnv(uint64_t h, const void *data, size_t n) {
    const uint8_t *p = (const uint8_t *)data;
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 0x100000001B3ULL; }
    return h;
}

static void usage(void) {
    puts("usage: eternity-fast --pieces FILE --size N [--clue CELL:STATE]... [options]\n"
         "  --threads K          worker threads (default 1)\n"
         "  --seed S             base seed (default 1)\n"
         "  --breaks d1,d2,...   k-th mismatching edge allowed at cell index >= dk (default: none = strict)\n"
         "  --hcolors c1,c2,..   depletion colors: by cell index D (--hdepth) at least F (--hfrac, default 1) of all their\n"
         "                       sides must be placed, ramping as (d/D)^G (--hpow, default 1); --no-hsort keeps random order\n"
         "  --hsched d:n,d:n,..  explicit depletion schedule instead of --hdepth/--hfrac/--hpow (piecewise linear)\n"
         "  --restart-nodes M    reshuffle and restart each thread every M placements (0 = never)\n"
         "  --partition          threads share one order and split depth-P prefixes (use with --restart-nodes 0)\n"
         "  --prefix P           prefix depth for --partition (default 4)\n"
         "  --no-shuffle         keep the canonical candidate order\n"
         "  --seconds T          stop after T seconds (0 = until done)\n"
         "  --max-nodes M        stop each thread after M placements\n"
         "  --count              keep going after a perfect board and count all complete boards\n"
         "  --keep MIN           also save every complete board scoring >= MIN\n"
         "  --out DIR            output directory (best.json, status.json, ckpt_t*.txt)\n"
         "  --stop-file PATH     stop cleanly (checkpoint saved) once this file exists\n"
         "  --floor S            only save complete boards scoring above S\n"
         "  --stdin-guard        stop cleanly when standard input reaches end of file (the launching app exited)\n"
         "  --resume             continue from the checkpoints in --out\n"
         "  --report SEC         status interval (default 10)    --checkpoint SEC (default 60)");
}

int main(int argc, char **argv) {
    const char *pieces = NULL, *breaks = NULL, *hcolors = NULL;
    int hdepth = 0; double hfrac = 1.0, hpow = 1.0; const char *hsched = NULL;
    int clues[MAXCELLS][2], nclue = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
#define ARG (i + 1 < argc ? argv[++i] : (die("missing value"), (char *)""))
        if (!strcmp(a, "--pieces")) pieces = ARG;
        else if (!strcmp(a, "--size")) N = atoi(ARG);
        else if (!strcmp(a, "--clue")) { if (sscanf(ARG, "%d:%d", &clues[nclue][0], &clues[nclue][1]) != 2) die("bad --clue"); nclue++; }
        else if (!strcmp(a, "--threads")) opt_threads = atoi(ARG);
        else if (!strcmp(a, "--seed")) opt_seed = strtoull(ARG, NULL, 10);
        else if (!strcmp(a, "--breaks")) breaks = ARG;
        else if (!strcmp(a, "--hcolors")) hcolors = ARG;
        else if (!strcmp(a, "--hdepth")) hdepth = atoi(ARG);
        else if (!strcmp(a, "--hsched")) hsched = ARG;
        else if (!strcmp(a, "--hfrac")) hfrac = atof(ARG);
        else if (!strcmp(a, "--hpow")) hpow = atof(ARG);
        else if (!strcmp(a, "--no-hsort")) opt_hsort = 0;
        else if (!strcmp(a, "--restart-nodes")) opt_restart_nodes = strtoull(ARG, NULL, 10);
        else if (!strcmp(a, "--partition")) opt_partition = 1;
        else if (!strcmp(a, "--prefix")) opt_prefix = atoi(ARG);
        else if (!strcmp(a, "--no-shuffle")) opt_shuffle = 0;
        else if (!strcmp(a, "--seconds")) opt_seconds = atof(ARG);
        else if (!strcmp(a, "--max-nodes")) opt_max_nodes = strtoull(ARG, NULL, 10);
        else if (!strcmp(a, "--count")) opt_count = 1;
        else if (!strcmp(a, "--keep")) { opt_keep = 1; opt_min_save = atoi(ARG); }
        else if (!strcmp(a, "--out")) opt_out = ARG;
        else if (!strcmp(a, "--stop-file")) opt_stop_file = ARG;
        else if (!strcmp(a, "--floor")) best_score = atoi(ARG);
        else if (!strcmp(a, "--stdin-guard")) opt_stdin_guard = 1;
        else if (!strcmp(a, "--resume")) opt_resume = 1;
        else if (!strcmp(a, "--report")) opt_report = atof(ARG);
        else if (!strcmp(a, "--checkpoint")) opt_ckpt = atof(ARG);
        else { usage(); return 2; }
    }
    if (!pieces || N < 1 || N > MAXN) { usage(); return 2; }
    if (opt_threads < 1 || opt_threads > MAXTHREADS) die("threads must be 1..64");
    CELLS = N * N; EDGES = 2 * N * (N - 1);
    load_pieces(pieces);
    if (PIECES != CELLS) die("piece count must equal size*size");
    if (opt_partition && (opt_prefix < 1 || opt_prefix > CELLS)) die("bad --prefix");
    for (int c = 0; c < CELLS; c++) clue_at[c] = -1;
    for (int i = 0; i < nclue; i++) {
        int c = clues[i][0], s = clues[i][1];
        if (c < 0 || c >= CELLS || s < 0 || s >= 4 * PIECES || clue_at[c] >= 0 || is_clue_piece[s / 4]) die("invalid or duplicate clue");
        clue_at[c] = s; is_clue_piece[s / 4] = 1;
    }
    if (breaks) {
        const char *q = breaks;
        while (*q) {
            char *end; long v = strtol(q, &end, 10);
            if (end == q || v < 0 || v > CELLS) die("bad --breaks list");
            for (int d = (int)v; d <= CELLS; d++) allowed[d]++;
            nbreaks++;
            q = (*end == ',') ? end + 1 : end;
            if (*end && *end != ',') die("bad --breaks list");
        }
    }
    if (hcolors) {
        const char *q = hcolors;
        while (*q) {
            char *end; long v = strtol(q, &end, 10);
            if (end == q || v < 1 || v >= NCOL) die("bad --hcolors list");
            is_h[v] = 1;
            q = (*end == ',') ? end + 1 : end;
            if (*end && *end != ',') die("bad --hcolors list");
        }
        for (int p = 0; p < PIECES; p++) for (int side = 0; side < 4; side++) h_total += is_h[face[p * 4][side]];
        if (hsched) { /* piecewise linear "cell:count,cell:count,..." (0 before the first point, flat after the last) */
            int pd = 0; double pv = 0; const char *q2 = hsched;
            while (*q2) {
                char *end; long dd = strtol(q2, &end, 10);
                if (end == q2 || *end != ':' || dd < pd || dd > CELLS) die("bad --hsched");
                q2 = end + 1; double vv = strtod(q2, &end);
                if (end == q2 || vv < pv || vv > h_total) die("bad --hsched");
                for (int d = pd; d <= dd; d++) need[d] = (int)(dd == pd ? vv : pv + (vv - pv) * (d - pd) / (double)(dd - pd));
                pd = (int)dd; pv = vv;
                q2 = (*end == ',') ? end + 1 : end;
                if (*end && *end != ',') die("bad --hsched");
            }
            for (int d = pd; d <= CELLS; d++) need[d] = (int)pv;
        } else {
            if (hdepth < 1 || hdepth > CELLS || hfrac < 0 || hfrac > 1 || hpow <= 0) die("bad depletion schedule");
            for (int d = 0; d <= CELLS; d++) {
                double x = d >= hdepth ? 1.0 : (double)d / hdepth;
                need[d] = (int)(hfrac * h_total * pow(x, hpow));
            }
        }
    }
    problem_hash = fnv(0xCBF29CE484222325ULL, face, (size_t)PIECES * 16);
    problem_hash = fnv(problem_hash, need, sizeof need);
    problem_hash = fnv(problem_hash, is_h, sizeof is_h);
    problem_hash = fnv(problem_hash, clue_at, sizeof(int) * (size_t)CELLS);
    problem_hash = fnv(problem_hash, allowed, (size_t)CELLS + 1);
    { int o[5] = { N, opt_partition, opt_partition ? opt_prefix : 0, opt_shuffle, opt_hsort }; problem_hash = fnv(problem_hash, o, sizeof o); }
    { uint64_t o[2] = { opt_restart_nodes, opt_partition ? (uint64_t)opt_threads : 0 }; problem_hash = fnv(problem_hash, o, sizeof o); }
    build_tables();
    signal(SIGINT, on_signal); signal(SIGTERM, on_signal);
    t0 = now();
    uint64_t resumed_nodes = 0;
    for (int i = 0; i < opt_threads; i++) {
        Thread *t = new Thread();
        if (!t) die("out of memory");
        t->id = i; threads[i] = t;
        if (!(opt_resume && load_ckpt(t))) { thread_tables(t); t->depth = 0; t->brk[0] = 0; t->hs[0] = 0; setup(t, 0); }
        resumed_nodes += t->nodes;
        t->last_ckpt = t0;
    }
    fprintf(stderr, "eternity-fast %s | %dx%d | clues %d | tables %d | breaks %d | threads %d | seed %" PRIu64 " | problem %016" PRIx64 "%s\n",
            VERSION, N, N, nclue, nsig, nbreaks, opt_threads, opt_seed, problem_hash, opt_resume ? " | resumed" : "");
    if (opt_stdin_guard) std::thread(stdin_guard).detach();
    std::vector<std::thread> th;
    for (int i = 0; i < opt_threads; i++) th.emplace_back(run, threads[i]);
    /* reporter */
    double last = t0;
    for (;;) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        if (opt_stop_file) { FILE *sf = fopen(opt_stop_file, "rb"); if (sf) { fclose(sf); stop_flag = 1; } }
        double tn = now();
        int alive = 0;
        uint64_t nodes = 0, complete = 0, perfect = 0, restarts = 0; int maxd = 0;
        for (int i = 0; i < opt_threads; i++) {
            Thread *t = threads[i];
            nodes += t->nodes; complete += t->complete; perfect += t->perfect; restarts += t->restart;
            if (t->maxdepth > maxd) maxd = t->maxdepth;
        }
        /* a thread is finished when pthread_tryjoin is unavailable portably: use done/stop conditions */
        int all_done = 1;
        for (int i = 0; i < opt_threads; i++) if (!threads[i]->done) all_done = 0;
        int timeup = (opt_seconds > 0 && tn - t0 >= opt_seconds) || stop_flag;
        int capped = opt_max_nodes != 0;
        if (capped) { capped = 1; for (int i = 0; i < opt_threads; i++) if (threads[i]->nodes < opt_max_nodes && !threads[i]->done) capped = 0; }
        alive = !(all_done || timeup || capped);
        if (!alive || tn - last >= opt_report) {
            char buf[900], path[1100];
            snprintf(buf, sizeof buf, "{\"engine\":\"%s\",\"elapsed\":%.2f,\"nodes\":%" PRIu64 ",\"session_nodes\":%" PRIu64 ",\"nodes_per_second\":%.0f,\"best_score\":%d,\"edges\":%d,"
                     "\"complete_boards\":%" PRIu64 ",\"perfect_boards\":%" PRIu64 ",\"max_depth\":%d,\"restarts\":%" PRIu64 ",\"threads\":%d,\"seed\":%" PRIu64 ",\"state\":\"%s\"}\n",
                     VERSION, tn - t0, nodes, nodes - resumed_nodes, (nodes - resumed_nodes) / (tn - t0 + 1e-9), best_score.load(), EDGES, complete, perfect, maxd, restarts,
                     opt_threads, opt_seed, alive ? "running" : (solved ? "solved" : all_done ? "exhausted" : "stopped"));
            fputs(buf, stdout); fflush(stdout);
            snprintf(path, sizeof path, "%s/status.json", opt_out);
            write_atomic(path, buf);
            last = tn;
        }
        if (!alive) break;
    }
    stop_flag = 1;
    for (auto &x : th) x.join();
    /* final status after the threads have flushed their counters */
    {
        uint64_t nodes = 0, complete = 0, perfect = 0, restarts = 0; int maxd = 0, all_done = 1;
        for (int i = 0; i < opt_threads; i++) {
            Thread *t = threads[i];
            nodes += t->nodes; complete += t->complete; perfect += t->perfect; restarts += t->restart;
            if (t->maxdepth > maxd) maxd = t->maxdepth;
            if (!t->done) all_done = 0;
        }
        double el = now() - t0;
        char buf[900], path[1100];
        snprintf(buf, sizeof buf, "{\"engine\":\"%s\",\"final\":true,\"elapsed\":%.2f,\"nodes\":%" PRIu64 ",\"session_nodes\":%" PRIu64 ",\"nodes_per_second\":%.0f,\"best_score\":%d,\"edges\":%d,"
                 "\"complete_boards\":%" PRIu64 ",\"perfect_boards\":%" PRIu64 ",\"max_depth\":%d,\"restarts\":%" PRIu64 ",\"threads\":%d,\"seed\":%" PRIu64 ",\"state\":\"%s\"}\n",
                 VERSION, el, nodes, nodes - resumed_nodes, (nodes - resumed_nodes) / (el + 1e-9), best_score.load(), EDGES, complete, perfect, maxd, restarts,
                 opt_threads, opt_seed, solved ? "solved" : all_done ? "exhausted" : "stopped");
        fputs(buf, stdout);
        snprintf(path, sizeof path, "%s/status.json", opt_out);
        write_atomic(path, buf);
        {
            static char hb[MAXCELLS * 24 + 64]; int o = snprintf(hb, sizeof hb, "[");
            for (int d = 0; d < CELLS; d++) {
                uint64_t v = 0;
                for (int i = 0; i < opt_threads; i++) v += threads[i]->hist[d];
                o += snprintf(hb + o, sizeof hb - o, "%s%" PRIu64, d ? "," : "", v);
            }
            snprintf(hb + o, sizeof hb - o, "]\n");
            snprintf(path, sizeof path, "%s/depth_hist.json", opt_out);
            write_atomic(path, hb);
        }
        /* Leave without stdio teardown: the guard thread may still be blocked on stdin. */
        fflush(stdout); fflush(stderr);
        std::_Exit(solved ? 0 : all_done ? 3 : 4);
    }
}
