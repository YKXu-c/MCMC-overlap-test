#pragma once
/**
 * @file union_find.hpp
 * @brief Union-Find (Disjoint Set Union) with path compression and union by rank.
 *
 * Used by cluster algorithms (Swendsen-Wang, Wolff) to identify connected
 * components of activated FK bonds in O(α(N)) amortized time.
 *
 * Supports reset() for reuse across sweeps without reallocation.
 *
 * @ref  Tarjan, "Data Structures and Network Algorithms", SIAM (1983)
 * @complexity O(α(N)) amortized per find/unite, where α is inverse Ackermann.
 */

#include <cassert>
#include <vector>

namespace mc {

class UnionFind {
public:
    /**
     * @brief Construct a Union-Find structure for n elements.
     * @param n  Number of elements (each initially in its own set).
     */
    explicit UnionFind(int n)
        : parent_(n), rank_(n, 0), size_(n, 1), num_sets_(n) {
        for (int i = 0; i < n; ++i) {
            parent_[i] = i;
        }
    }

    /**
     * @brief Find the root representative of element x with path compression.
     * @param x  Element index.
     * @return Root of the set containing x.
     *
     * Iterative implementation to avoid stack overflow for large lattices.
     */
    [[nodiscard]] int find(int x) {
        assert(x >= 0 && x < static_cast<int>(parent_.size()));
        int root = x;
        while (parent_[root] != root) {
            root = parent_[root];
        }
        // Path compression: point every node on the path directly to root
        while (parent_[x] != root) {
            int next = parent_[x];
            parent_[x] = root;
            x = next;
        }
        return root;
    }

    /**
     * @brief Merge the sets containing x and y (union by rank).
     * @param x  First element.
     * @param y  Second element.
     * @return true if the sets were actually merged (were different).
     */
    bool unite(int x, int y) {
        int rx = find(x);
        int ry = find(y);
        if (rx == ry) return false;

        // Union by rank: attach smaller tree under larger
        if (rank_[rx] < rank_[ry]) {
            parent_[rx] = ry;
            size_[ry] += size_[rx];
        } else if (rank_[rx] > rank_[ry]) {
            parent_[ry] = rx;
            size_[rx] += size_[ry];
        } else {
            parent_[ry] = rx;
            size_[rx] += size_[ry];
            ++rank_[rx];
        }
        --num_sets_;
        return true;
    }

    /**
     * @brief Size of the set containing element x.
     * @param x  Element index.
     * @return Number of elements in the set containing x.
     */
    [[nodiscard]] int setSize(int x) {
        return size_[find(x)];
    }

    /** @brief Current number of disjoint sets. */
    [[nodiscard]] int numSets() const { return num_sets_; }

    /**
     * @brief Reset to initial state (each element in its own set).
     *
     * Reuses the already-allocated vectors — no memory allocation.
     */
    void reset() {
        const int n = static_cast<int>(parent_.size());
        for (int i = 0; i < n; ++i) {
            parent_[i] = i;
            rank_[i] = 0;
            size_[i] = 1;
        }
        num_sets_ = n;
    }

private:
    std::vector<int> parent_;   ///< Parent pointer (root points to itself).
    std::vector<int> rank_;     ///< Approximate tree depth for union-by-rank.
    std::vector<int> size_;     ///< Size of each set (valid only at roots).
    int num_sets_;              ///< Current number of disjoint sets.
};

}  // namespace mc
