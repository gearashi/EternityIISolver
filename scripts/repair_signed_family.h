#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

// A complete family of component-size/gain patterns supplied by the validated
// Python screen. Adjacent or overlapping shapes cannot be distinct components.
class SignedFamilies {
    using U64 = std::uint64_t;
    using Mask = std::array<U64, 4>;
    struct Shape {
        std::array<int, 8> cells{};
        Mask occupied{}, blocked{};
    };
    struct Group {
        int size = 0, gain = 0;
        std::vector<Shape> shapes;
    };
    int radius = 0;
    std::vector<Group> groups;
    std::vector<std::vector<int>> patterns;
    std::array<int, 8> members{};

    static U64 multiply(U64 a, U64 b) {
        if (b && a > std::numeric_limits<U64>::max() / b)
            throw std::invalid_argument("signed-family count overflow");
        return a * b;
    }

    static U64 choose(int n, int r) {
        if (n < r) return 0;
        U64 total = 1;
        for (int j = 1; j <= r; ++j) {
            int a = n - r + j, div = std::gcd(a, j);
            total = multiply(total / (j / div), a / div);
        }
        return total;
    }

    static void set(Mask &mask, int c) { mask[c / 64] |= U64(1) << (c % 64); }

    template<class Charge, class Evaluate>
    bool visit(const std::vector<int> &pattern, const std::array<int, 8> &same,
               const std::array<U64, 9> &suffix, int depth, int start, int placed,
               const Mask &blocked, Charge &charge, Evaluate &evaluate) {
        if (depth == static_cast<int>(pattern.size())) {
            const bool winner = evaluate(members);
            ++covered;
            return winner;
        }
        const auto &group = groups[pattern[depth]];
        int remaining = same[depth] - 1;
        int last = static_cast<int>(group.shapes.size()) - remaining;
        for (int i = start; i < last; ++i) {
            charge();
            const Shape &shape = group.shapes[i];
            bool collision = false;
            for (int z = 0; z < 4; ++z)
                if (blocked[z] & shape.occupied[z]) collision = true;
            if (collision) {
                U64 choices = choose(static_cast<int>(group.shapes.size()) - i - 1, remaining);
                covered += multiply(choices, suffix[depth + same[depth]]);
                continue;
            }
            Mask next = blocked;
            for (int z = 0; z < 4; ++z) next[z] |= shape.blocked[z];
            std::copy_n(shape.cells.begin(), group.size, members.begin() + placed);
            if (visit(pattern, same, suffix, depth + 1, remaining ? i + 1 : 0, placed + group.size,
                      next, charge, evaluate)) return true;
        }
        return false;
    }

public:
    U64 covered = 0, started = 0, finished = 0, rows = 0;

    template<class Charge> void read(int k, const std::vector<int> &mutable_cells,
              const std::vector<std::array<int, 4>> &neighbors, Charge charge) {
        radius = k;
        int gcount;
        if (!(std::cin >> gcount) || gcount < 0 || gcount > 4096)
            throw std::invalid_argument("invalid number of signed shape groups");
        std::vector<bool> allowed(neighbors.size());
        for (int c : mutable_cells) allowed[c] = true;
        groups.resize(gcount);
        for (auto &g : groups) {
            int count;
            if (!(std::cin >> g.size >> g.gain >> count) || g.size < 1 || g.size > k ||
                g.gain < -4*k || g.gain > 4*k || count < 1 || rows + count > 850000)
                throw std::invalid_argument("invalid signed shape group");
            g.shapes.resize(count);
            for (auto &shape : g.shapes) {
                for (int i = 0; i < g.size; ++i) {
                    charge();
                    int c;
                    if (!(std::cin >> c) || c < 0 || c >= static_cast<int>(allowed.size()) ||
                        !allowed[c] || (i && c <= shape.cells[i-1]))
                        throw std::invalid_argument("invalid signed shape cell");
                    shape.cells[i] = c;
                    set(shape.occupied, c);
                    set(shape.blocked, c);
                    for (int d : neighbors[c]) if (d >= 0) set(shape.blocked, d);
                }
                ++rows;
            }
        }
        int count;
        if (!(std::cin >> count) || count < 0 || count > 100000)
            throw std::invalid_argument("invalid signed pattern count");
        patterns.resize(count);
        for (auto &pattern : patterns) {
            int n, size = 0, gain = 0;
            if (!(std::cin >> n) || n < 1 || n > k)
                throw std::invalid_argument("invalid signed pattern length");
            pattern.resize(n);
            std::vector<bool> seen(gcount);
            int prior = -1;
            for (int &i : pattern) {
                charge();
                if (!(std::cin >> i) || i < 0 || i >= gcount || (seen[i] && prior != i))
                    throw std::invalid_argument("invalid signed pattern group");
                size += groups[i].size; gain += groups[i].gain; seen[i] = true; prior = i;
            }
            if (size != k || gain <= 0) throw std::invalid_argument("incorrect signed pattern");
        }
    }

    template<class Charge, class Evaluate> bool search(Charge charge, Evaluate evaluate) {
        for (const auto &pattern : patterns) {
            ++started;
            std::array<int, 8> same{};
            std::array<U64, 9> suffix{};
            suffix[pattern.size()] = 1;
            for (int j = static_cast<int>(pattern.size()) - 1; j >= 0; --j) {
                same[j] = 1;
                if (j + 1 < static_cast<int>(pattern.size()) && pattern[j+1] == pattern[j])
                    same[j] += same[j+1];
                U64 options = choose(static_cast<int>(groups[pattern[j]].shapes.size()), same[j]);
                suffix[j] = multiply(options, suffix[j + same[j]]);
            }
            U64 before = covered;
            if (visit(pattern, same, suffix, 0, 0, 0, Mask{}, charge, evaluate)) return true;
            if (covered - before != suffix[0])
                throw std::runtime_error("signed pattern traversal coverage disagrees with binomial count");
            ++finished;
        }
        return false;
    }
};
