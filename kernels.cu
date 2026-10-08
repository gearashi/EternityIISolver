// Eternity II: one independent search replica per CUDA thread.
// No state is shared between replicas. The host validates all input arrays.
// Board storage is SoA: boards[cell * N + replica], int16 codes 0..1023.
// Code = 4 * zero_based_piece_id + clockwise_rotation. faces is NESW.
// neighbors[cell*4+side] is the adjacent cell or -1 at the exterior.
// allowed[cell*1024+code] is nonzero iff that placement has the exact frame.
// positions[piece*N+replica] is the inverse current-board piece placement.
// pair_offsets[6*23*23+1] indexes buckets in pair_codes. Side pairs are
// (N,E),(N,S),(N,W),(E,S),(E,W),(S,W), and bucket=(pair*23+c1)*23+c2.
// Every oriented code belongs to one bucket for each pair. guide_prob[N]
// controls how often a swap partner is drawn by matching two neighbor colors.
// groupmembers[type*256+k] and groupsizes[type] enumerate same-frame-type
// cells. The host should exclude fixed cells; the kernel also checks fixed.
// counters[replica] counts valid proposals; counters[N+replica] accepted
// proposals. Every counted proposal changes at least one orientation code.

__device__ __forceinline__ unsigned int next_u32(unsigned int *state) {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    return *state;
}

__device__ __forceinline__ int opposite(int side) {
    return (side + 2) & 3;
}

__device__ __forceinline__ int face(const unsigned char *faces, int code, int side) {
    return (int)faces[code * 4 + side];
}

// Exact difference in the number of matched INTERIOR edges. Each edge
// incident to a is visited once; each edge incident to b is then visited
// once except the shared a--b edge, already counted by the first loop.
// b == -1 denotes a one-cell change; new_b is ignored in that case.
__device__ __forceinline__ int placement_delta(
    const short *boards, const unsigned char *faces, const short *neighbors,
    int N, int replica, int a, int b, int new_a, int new_b
) {
    const int old_a = (int)boards[a * N + replica];
    int delta = 0;
    #pragma unroll
    for (int side = 0; side < 4; ++side) {
        const int other = (int)neighbors[a * 4 + side];
        if (other < 0) continue;
        const int old_other = (int)boards[other * N + replica];
        const int new_other = (other == b) ? new_b : old_other;
        const int os = opposite(side);
        delta += (face(faces, new_a, side) == face(faces, new_other, os))
               - (face(faces, old_a, side) == face(faces, old_other, os));
    }
    if (b >= 0) {
        const int old_b = (int)boards[b * N + replica];
        #pragma unroll
        for (int side = 0; side < 4; ++side) {
            const int other = (int)neighbors[b * 4 + side];
            if (other < 0 || other == a) continue;
            const int neighbor_code = (int)boards[other * N + replica];
            const int os = opposite(side);
            delta += (face(faces, new_b, side) == face(faces, neighbor_code, os))
                   - (face(faces, old_b, side) == face(faces, neighbor_code, os));
        }
    }
    return delta;
}

__device__ __forceinline__ bool has_mismatch(
    const short *boards, const unsigned char *faces, const short *neighbors,
    int N, int replica, int cell
) {
    const int code = (int)boards[cell * N + replica];
    #pragma unroll
    for (int side = 0; side < 4; ++side) {
        const int other = (int)neighbors[cell * 4 + side];
        if (other >= 0 && face(faces, code, side) !=
            face(faces, (int)boards[other * N + replica], opposite(side))) {
            return true;
        }
    }
    return false;
}

// Diagnostic kernel: independently score all 480 edges from complete boards.
// It does not modify boards or trust the search's accumulated delta scores.
extern "C" __global__ void score_boards(
    const short *boards,
    const unsigned char *faces,
    const short *neighbors,
    int *out_scores,
    int N
) {
    const int replica = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (replica >= N) return;
    int score = 0;
    for (int cell = 0; cell < 256; ++cell) {
        const int code = (int)boards[cell * N + replica];
        // East and South count each physical interior edge exactly once.
        for (int side = 1; side <= 2; ++side) {
            const int other = (int)neighbors[cell * 4 + side];
            if (other >= 0) {
                score += face(faces, code, side) ==
                    face(faces, (int)boards[other * N + replica], opposite(side));
            }
        }
    }
    out_scores[replica] = score;
}

// Diagnostic kernel: one supplied move per replica; no mutation or legality
// filtering. This intentionally permits arbitrary valid codes for CPU delta
// comparison. a,b,new_a,new_b have length N and dtype int16. b == -1 means
// rotation. Invalid indices/codes yield INT_MIN rather than unsafe reads.
extern "C" __global__ void test_delta(
    const short *boards,
    const unsigned char *faces,
    const short *neighbors,
    const short *a,
    const short *b,
    const short *new_a,
    const short *new_b,
    int *out_delta,
    int N
) {
    const int replica = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (replica >= N) return;
    const int ca = (int)a[replica];
    const int cb = (int)b[replica];
    const int na = (int)new_a[replica];
    const int nb = (int)new_b[replica];
    if (ca < 0 || ca >= 256 || cb < -1 || cb >= 256 || ca == cb ||
        na < 0 || na >= 1024 || (cb >= 0 && (nb < 0 || nb >= 1024))) {
        out_delta[replica] = (-2147483647 - 1);
        return;
    }
    out_delta[replica] = placement_delta(boards, faces, neighbors, N, replica, ca, cb, na, nb);
}

// Bounded annealed local search. Each step first tries up to eight random
// cells, taking the first unfixed cell with a mismatch. If none is found,
// its first sampled unfixed cell is the uniform fallback. A 1/8 proposal
// rotates that cell; otherwise it swaps with a random cell of the same type.
// Every legal resulting rotation (or rotation pair) is evaluated. Ties among
// maximum-delta candidates are sampled by reservoir sampling. Metropolis
// acceptance uses exp(delta/temperature) for negative delta. This heuristic
// is not claimed to sample a Boltzmann distribution: proposals are biased.
extern "C" __global__ void search_moves(
    short *boards,
    short *bestboards,
    const unsigned char *faces,
    const short *neighbors,
    const unsigned char *fixed,
    const short *groupmembers,
    const int *groupsizes,
    const unsigned char *celltype,
    const unsigned char *allowed,
    short *positions,
    const int *pair_offsets,
    const short *pair_codes,
    const float *guide_prob,
    int *scores,
    int *bestscores,
    unsigned int *rng,
    const float *temperatures,
    unsigned long long *counters,
    int N,
    int steps
) {
    const int replica = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (replica >= N) return;
    unsigned int state = rng[replica];
    if (state == 0u) {
        state = 0x9e3779b9u ^ ((unsigned int)replica + 1u);
        if (state == 0u) state = 1u;
    }
    int current_score = scores[replica];
    int best_score = bestscores[replica];
    const float temperature = temperatures[replica];
    unsigned long long proposed = 0;
    unsigned long long accepted = 0;
    for (int step = 0; step < steps; ++step) {
        int a = -1;
        #pragma unroll
        for (int probe = 0; probe < 8; ++probe) {
            const int candidate = (int)(next_u32(&state) & 255u);
            if (fixed[candidate]) continue;
            if (a < 0) a = candidate;
            if (has_mismatch(boards, faces, neighbors, N, replica, candidate)) {
                a = candidate;
                break;
            }
        }
        if (a < 0) continue;
        const int old_a = (int)boards[a * N + replica];
        const bool rotate_only = ((next_u32(&state) & 7u) == 0u);
        int b = -1;
        if (!rotate_only) {
            const int type = (int)celltype[a];
            if (type < 0 || type > 2) continue;
            const int count = groupsizes[type];
            if (count < 2 || count > 256) continue;
            const float guide_draw = (float)(next_u32(&state) >> 8) * (1.0f / 16777216.0f);
            if (guide_draw < guide_prob[replica]) {
                int sides[4];
                int side_count = 0;
                #pragma unroll
                for (int side = 0; side < 4; ++side) {
                    if (neighbors[a * 4 + side] >= 0) sides[side_count++] = side;
                }
                if (side_count >= 2) {
                    int first = (int)(next_u32(&state) % (unsigned int)side_count);
                    int second = (int)(next_u32(&state) % (unsigned int)(side_count - 1));
                    if (second >= first) ++second;
                    if (first > second) {
                        const int temporary = first; first = second; second = temporary;
                    }
                    const int side1 = sides[first];
                    const int side2 = sides[second];
                    const int neighbor1 = (int)neighbors[a * 4 + side1];
                    const int neighbor2 = (int)neighbors[a * 4 + side2];
                    const int color1 = face(faces, (int)boards[neighbor1 * N + replica], opposite(side1));
                    const int color2 = face(faces, (int)boards[neighbor2 * N + replica], opposite(side2));
                    const int pair = (side1 == 0) ? side2 - 1 : ((side1 == 1) ? side2 + 1 : 5);
                    const int bucket = (pair * 23 + color1) * 23 + color2;
                    const int start = pair_offsets[bucket];
                    const int end = pair_offsets[bucket + 1];
                    if (end > start) {
                        const int code = (int)pair_codes[start + (next_u32(&state) % (unsigned int)(end - start))];
                        const int candidate = (int)positions[(code >> 2) * N + replica];
                        // The lookup is a heuristic proposal, not a constraint.
                        // Reject incompatible frame orientations and then use
                        // the ordinary uniform proposal if guidance fails.
                        if (candidate >= 0 && candidate < 256 && candidate != a &&
                            !fixed[candidate] && celltype[candidate] == celltype[a] &&
                            allowed[a * 1024 + code]) {
                            b = candidate;
                        }
                    }
                }
            }
            // The host's member lists exclude clues. Retrying also safely
            // handles a supplied list that contains the selected cell/clues.
            for (int attempt = 0; b < 0 && attempt < 8; ++attempt) {
                const int candidate = (int)groupmembers[type * 256 + (next_u32(&state) % (unsigned int)count)];
                if (candidate >= 0 && candidate < 256 && candidate != a && !fixed[candidate]) {
                    b = candidate;
                    break;
                }
            }
            if (b < 0) continue;
        }
        int chosen_a = -1;
        int chosen_b = -1;
        int maximum_delta = -100000;
        unsigned int ties = 0u;
        if (rotate_only) {
            const int base = old_a & ~3;
            #pragma unroll
            for (int rotation = 0; rotation < 4; ++rotation) {
                const int code = base + rotation;
                if (code == old_a || !allowed[a * 1024 + code]) continue;
                const int delta = placement_delta(boards, faces, neighbors, N, replica, a, -1, code, 0);
                if (delta > maximum_delta) {
                    maximum_delta = delta;
                    chosen_a = code;
                    chosen_b = -1;
                    ties = 1u;
                } else if (delta == maximum_delta) {
                    ++ties;
                    if (next_u32(&state) % ties == 0u) chosen_a = code;
                }
            }
        } else {
            const int old_b = (int)boards[b * N + replica];
            const int base_a = old_b & ~3;
            const int base_b = old_a & ~3;
            #pragma unroll
            for (int rotation_a = 0; rotation_a < 4; ++rotation_a) {
                const int code_a = base_a + rotation_a;
                if (!allowed[a * 1024 + code_a]) continue;
                #pragma unroll
                for (int rotation_b = 0; rotation_b < 4; ++rotation_b) {
                    const int code_b = base_b + rotation_b;
                    if (!allowed[b * 1024 + code_b]) continue;
                    if (code_a == old_a && code_b == old_b) continue;
                    const int delta = placement_delta(boards, faces, neighbors, N, replica, a, b, code_a, code_b);
                    if (delta > maximum_delta) {
                        maximum_delta = delta;
                        chosen_a = code_a;
                        chosen_b = code_b;
                        ties = 1u;
                    } else if (delta == maximum_delta) {
                        ++ties;
                        if (next_u32(&state) % ties == 0u) {
                            chosen_a = code_a;
                            chosen_b = code_b;
                        }
                    }
                }
            }
        }
        if (chosen_a < 0) continue;
        ++proposed;
        bool take = (maximum_delta >= 0);
        if (!take && temperature > 0.0f) {
            // Exactly representable 24-bit U in [0,1), including zero.
            const float uniform = (float)(next_u32(&state) >> 8) * (1.0f / 16777216.0f);
            take = uniform < expf((float)maximum_delta / temperature);
        }
        if (!take) continue;
        boards[a * N + replica] = (short)chosen_a;
        if (b >= 0) {
            boards[b * N + replica] = (short)chosen_b;
            positions[(chosen_a >> 2) * N + replica] = (short)a;
            positions[(chosen_b >> 2) * N + replica] = (short)b;
        }
        current_score += maximum_delta;
        ++accepted;
        if (current_score > best_score) {
            best_score = current_score;
            for (int cell = 0; cell < 256; ++cell) {
                bestboards[cell * N + replica] = boards[cell * N + replica];
            }
        }
    }
    scores[replica] = current_score;
    bestscores[replica] = best_score;
    rng[replica] = state;
    counters[replica] += proposed;
    counters[N + replica] += accepted;
}
