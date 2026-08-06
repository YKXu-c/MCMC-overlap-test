#pragma once
/**
 * @file potts_model.hpp
 * @brief q-state Potts model on an L×L square lattice with periodic boundaries.
 *
 *   H = -J Σ_{<ij>} δ_{s_i, s_j},   s_i ∈ {0, 1, …, q-1},  J > 0 ferromagnetic.
 *
 * q = 2 reproduces the Ising model (S_i = ±1 ↔ s_i ∈ {0,1}), and the q-state
 * magnetization m = (q·n_max − N)/((q−1)·N) reduces to |m_Ising| for q = 2.
 *
 * This is a self-contained companion to ising_model.hpp (the Ising binaries are
 * left byte-untouched — they drive the oakk chaos pipeline). The overlap
 * machinery (config overlap U_n, Wolff cluster overlap U_n^(W)) is copied from
 * IsingModel; it is color-agnostic (spin equality / cluster-membership
 * intersection), so it works unchanged for integer Potts colors.
 *
 * @ref  Potts, Proc. Camb. Phil. Soc. 48, 106 (1952)
 *       FK representation: Fortuin & Kasteleyn; Swendsen & Wang, PRL 58, 86 (1987)
 * @complexity O(N) energy, O(1) single-site energy change.
 */

#include "mc_base.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace mc {

/**
 * @brief q-state Potts configuration with NN ferromagnetic coupling.
 *
 * Stores spins as a flat vector of integers in {0,…,q-1} (row-major,
 * index = y * L + x). Provides neighbor lookup, energy computation,
 * the overlap observables, and observable registration.
 */
class PottsModel {
public:
    /**
     * @brief Construct an L×L q-state Potts model.
     * @param L              Linear lattice size.
     * @param J              NN coupling (J > 0: ferromagnetic).
     * @param q              Number of Potts states (q ≥ 2; q = 2 ≡ Ising).
     * @param seed           RNG seed (0 = std::random_device).
     * @param init_all_same  If true, start all spins = 0 (ordered ground state);
     *                       else random uniform in {0,…,q-1}.
     */
    PottsModel(int L, double J, int q,
               uint64_t seed = 0, bool init_all_same = true)
        : L_(L), N_(L * L), J_(J), q_(q),
          rng_seed_(seed),
          spins_(N_, 0),  // default: all color 0
          rng_(seed == 0 ? std::random_device{}() : seed) {

        assert(L > 0 && "Lattice size must be positive");
        assert(J != 0.0 && "Coupling constant J must be nonzero");
        assert(q >= 2 && "Potts q must be >= 2");

        if (!init_all_same) {
            std::uniform_int_distribution<int> dist(0, q_ - 1);
            for (auto& s : spins_) {
                s = dist(rng_);
            }
        }

        registerDefaultObservables();
    }

    // --- Accessors ---

    [[nodiscard]] int L() const { return L_; }
    [[nodiscard]] int N() const { return N_; }
    [[nodiscard]] double J() const { return J_; }
    [[nodiscard]] int q() const { return q_; }
    [[nodiscard]] uint64_t seed() const { return rng_seed_; }
    [[nodiscard]] double beta() const { return beta_; }

    /** @brief Set inverse temperature β = 1/(k_B T). */
    void setBeta(double beta) { beta_ = beta; }

    /** @brief Set temperature T (sets β = 1/T in natural units). */
    void setTemperature(double T) { beta_ = 1.0 / T; }

    /** @brief Access spin at site i (0-indexed, row-major). */
    [[nodiscard]] int spin(int i) const { return spins_[i]; }
    [[nodiscard]] int& spin(int i) { return spins_[i]; }

    /** @brief Set spin at site i to color s. */
    void setSpin(int i, int s) { spins_[i] = s; }

    /** @brief Direct access to the spin array. */
    [[nodiscard]] const std::vector<int>& spins() const { return spins_; }
    [[nodiscard]] std::vector<int>& spins() { return spins_; }

    /** @brief Access the observable registry. */
    [[nodiscard]] ObservableRegistry& observables() { return observables_; }
    [[nodiscard]] const ObservableRegistry& observables() const { return observables_; }

    /** @brief Access the RNG (for sweep implementations). */
    [[nodiscard]] std::mt19937_64& rng() { return rng_; }

    // --- Neighbor queries ---

    /**
     * @brief Get the 4 nearest-neighbor site indices (periodic BC).
     * @return Array {right, left, up, down}.
     */
    [[nodiscard]] std::array<int, 4> nearestNeighbors(int site) const {
        const int x = site % L_;
        const int y = site / L_;
        const int xr = wrap(x + 1);
        const int xl = wrap(x - 1);
        const int yu = wrap(y + 1);
        const int yd = wrap(y - 1);
        return {y * L_ + xr, y * L_ + xl, yu * L_ + x, yd * L_ + x};
    }

    // --- Energy computation ---

    /**
     * @brief Total energy E = -J Σ_{<ij>} δ_{s_i,s_j} (each NN bond counted once).
     * @complexity O(N).
     */
    [[nodiscard]] double computeEnergy() const {
        double energy = 0.0;
        for (int i = 0; i < N_; ++i) {
            const int si = spins_[i];
            // Count right and up only (avoid double-counting)
            const auto [right, left, up, down] = nearestNeighbors(i);
            energy -= J_ * (si == spins_[right] ? 1.0 : 0.0);
            energy -= J_ * (si == spins_[up]    ? 1.0 : 0.0);
        }
        return energy;
    }

    /**
     * @brief Energy change ΔE from recoloring site to a specific new color.
     * @param site       Site to recolor.
     * @param new_color  Proposed new color (must differ from current).
     * @return ΔE = J·(n_same_current − n_same_new); > 0 means the move costs energy.
     *
     * For Potts the move is s_i → new_color (not a flip), so ΔE depends on the
     * proposed color. ΔE = -J·n_same_new + J·n_same_current.
     *
     * @complexity O(1) (4 neighbors).
     */
    [[nodiscard]] double energyChangeTo(int site, int new_color) const {
        const int cur = spins_[site];
        int n_same_cur = 0;
        int n_same_new = 0;
        for (int nb : nearestNeighbors(site)) {
            if (spins_[nb] == cur) ++n_same_cur;
            if (spins_[nb] == new_color) ++n_same_new;
        }
        // E_cur contribution = -J·n_same_cur ; E_new = -J·n_same_new
        return J_ * static_cast<double>(n_same_cur - n_same_new);
    }

    /**
     * @brief Propose a new color uniformly from {0,…,q-1} \ {current}.
     * @return The proposed color.
     */
    int proposeNewColor(int site, std::mt19937_64& rng) {
        const int cur = spins_[site];
        std::uniform_int_distribution<int> dist(0, q_ - 2);  // q-1 allowed colors
        int c = dist(rng);
        if (c >= cur) ++c;  // skip the current color
        return c;
    }

    // --- Observables ---

    /**
     * @brief q-state magnetization per spin.
     * @return m = (q·n_max − N) / ((q−1)·N), range [0, 1].
     *
     * n_max = population of the most populous color. For q = 2 this equals
     * |m_Ising|. Always ≥ 0 (n_max ≥ N/q).
     */
    [[nodiscard]] double magnetization() const {
        std::vector<int> counts(q_, 0);
        for (int s : spins_) counts[s]++;
        const int n_max = *std::max_element(counts.begin(), counts.end());
        return static_cast<double>(q_ * n_max - N_) /
               static_cast<double>((q_ - 1) * N_);
    }

    /** @brief |m| — same as magnetization() (already ≥ 0 for q-state). */
    [[nodiscard]] double absMagnetization() const {
        return magnetization();
    }

    /** @brief Energy per spin. */
    [[nodiscard]] double energyPerSpin() const {
        return computeEnergy() / static_cast<double>(N_);
    }

    // --- Overlap observables (U_N) ---
    //
    // Identical logic to IsingModel: config overlap U_n compares spin equality
    // (works for integer colors); cluster overlap is a set-intersection of
    // cluster membership (color-agnostic). Copied verbatim in spirit.

    /** @brief Enable overlap measurement at a single step n (1-element-list alias). */
    void enableOverlap(int step = 2, bool cluster_only = false) {
        enableOverlap(std::vector<int>{step}, cluster_only);
    }

    /**
     * @brief Enable overlap measurement at a list of separations {n_1, n_2, …}.
     *
     * Registers one observable per separation, named "overlap_Nt{n}", returning
     * the overlap at that step. History depth = max(n_k)+1 so every separation
     * is measured from a single trajectory (extra N_t is then ~free). For Wolff
     * the primary "overlap_Nt{n}" is the *cluster* overlap 𝓡^W
     * (computeClusterOverlap overwrites the config value); for Metropolis/SW it
     * is the config overlap P_coin. A per-step config overlap is registered by
     * the Wolff main as "config_overlap_Nt{n}".
     * @param steps        The n_k in U_{n_k} (sweeps between compared configs/clusters).
     * @param cluster_only If true, do NOT allocate spin_history_ (the config-overlap
     *                     buffer, vector<int>, ~67 GB at L=4096/Nt=1000). Only the
     *                     cluster overlap (vector<bool>, packed) is measured; the
     *                     config-overlap observables read 0. Use for large-L Wolff
     *                     where the FSS figures only need 𝓡^W.
     * @ref  Pile, Deng, Shchur, arXiv:2604.10254 — algorithmic overlaps.
     */
    void enableOverlap(std::vector<int> steps, bool cluster_only = false) {
        assert(!steps.empty() && "overlap step list must be non-empty");
        std::sort(steps.begin(), steps.end());
        steps.erase(std::unique(steps.begin(), steps.end()), steps.end());
        overlap_steps_ = std::move(steps);
        overlap_step_max_ = overlap_steps_.back();
        overlap_enabled_ = true;

        const int depth = overlap_step_max_ + 1;
        // cluster_only: skip the config-overlap buffer (saves ~depth*N*4 B at large L).
        if (cluster_only) {
            spin_history_.clear();
        } else {
            assert(q_ <= 4 && "bit-packed spin_history_ supports q<=4 only");
            n_spin_words_ = (N_ + 31) / 32;
            spin_history_.assign(depth, std::vector<uint64_t>(n_spin_words_, 0));
        }
        spin_history_head_ = 0;
        spin_history_count_ = 0;

        n_words_ = (N_ + 63) / 64;
        cluster_history_.assign(depth, std::vector<uint64_t>(n_words_, 0));
        cluster_history_head_ = 0;
        cluster_history_count_ = 0;

        overlap_values_.assign(overlap_steps_.size(), 0.0);
        config_overlap_values_.assign(overlap_steps_.size(), 0.0);

        // One primary observable per Nt; capture the step index kk by value.
        for (int k = 0; k < static_cast<int>(overlap_steps_.size()); ++k) {
            const int kk = k;
            observables_.registerObservable(
                "overlap_Nt" + std::to_string(overlap_steps_[k]),
                [kk](const void* ptr) -> double {
                    return static_cast<const PottsModel*>(ptr)->overlapValueAt(kk);
                });
        }
    }

    [[nodiscard]] bool overlapEnabled() const { return overlap_enabled_; }
    [[nodiscard]] const std::vector<int>& overlapSteps() const { return overlap_steps_; }
    [[nodiscard]] int overlapStepMax() const { return overlap_step_max_; }
    [[nodiscard]] double overlapValue() const {
        return overlap_values_.empty() ? 0.0 : overlap_values_.back();
    }
    /** @brief Primary overlap at enabled-step index k (cluster for Wolff, config for Met/SW). */
    [[nodiscard]] double overlapValueAt(int k) const {
        return (k >= 0 && k < static_cast<int>(overlap_values_.size()))
                   ? overlap_values_[k] : 0.0;
    }

    /** @brief Save current spin configuration into the circular buffer (bit-packed). */
    void saveSpinSnapshot() {
        if (spin_history_.empty()) return;   // cluster_only mode: no config-overlap buffer
        // Pack spins_ (0..q-1, 2 bits each for q<=4) into uint64_t words for fast
        // SWAR equality in computeConfigOverlap (~16x less memory than vector<int>).
        std::vector<uint64_t>& snap = spin_history_[spin_history_head_];
        for (int w = 0; w < n_spin_words_; ++w) {
            uint64_t bits = 0;
            const int base = w * 32;
            const int lim = std::min(base + 32, N_);
            for (int i = base; i < lim; ++i) {
                bits |= (static_cast<uint64_t>(spins_[i] & 0x3) << (2 * (i - base)));
            }
            snap[w] = bits;
        }
        spin_history_head_ = (spin_history_head_ + 1) %
                             static_cast<int>(spin_history_.size());
        if (spin_history_count_ < static_cast<int>(spin_history_.size())) {
            ++spin_history_count_;
        }
    }

    /**
     * @brief Configuration overlap U_n = (1/N) Σ_i δ(s_i(now), s_i(n sweeps ago)),
     *        computed for every enabled separation n in one pass.
     *
     * Fills config_overlap_values_[k] for each step and also writes the primary
     * overlap_values_[k] (which computeClusterOverlap overwrites for Wolff).
     * @return Overlap at the largest enabled step, in [0,1] (0.0 if too little history).
     * @complexity O(|steps| · N).
     */
    double computeConfigOverlap() {
        if (spin_history_count_ < overlap_step_max_ + 1) {
            std::fill(overlap_values_.begin(), overlap_values_.end(), 0.0);
            std::fill(config_overlap_values_.begin(), config_overlap_values_.end(), 0.0);
            return 0.0;
        }
        const int cur_idx = (spin_history_head_ - 1 +
                             static_cast<int>(spin_history_.size())) %
                            static_cast<int>(spin_history_.size());
        const auto& cur = spin_history_[cur_idx];
        for (size_t k = 0; k < overlap_steps_.size(); ++k) {
            const int n = overlap_steps_[k];
            const int past_idx = (spin_history_head_ - n - 1 +
                                  static_cast<int>(spin_history_.size())) %
                                 static_cast<int>(spin_history_.size());
            const auto& past = spin_history_[past_idx];
            long total_differs = 0;
            for (int w = 0; w < n_spin_words_; ++w) {
                const uint64_t x = cur[w] ^ past[w];   // equal 2-bit slots -> 00
                total_differs += __builtin_popcountll((x | (x >> 1)) & 0x5555555555555555ULL);
            }
            const long match = static_cast<long>(N_) - total_differs;
            const double val = static_cast<double>(match) / static_cast<double>(N_);
            config_overlap_values_[k] = val;
            overlap_values_[k] = val;
        }
        return overlap_values_.back();
    }

    /** @brief Save cluster membership into the circular buffer (Wolff), bit-packed. */
    void saveClusterSnapshot(const std::vector<bool>& in_cluster, int cluster_size) {
        // Pack the bool membership into uint64_t words for fast popcount
        // intersection in computeClusterOverlap (~100x faster than element-wise
        // vector<bool> at large L; identical intersection counts).
        std::vector<uint64_t>& snap = cluster_history_[cluster_history_head_];
        for (int w = 0; w < n_words_; ++w) {
            uint64_t bits = 0;
            const int base = w * 64;
            const int lim = std::min(base + 64, N_);
            for (int i = base; i < lim; ++i) {
                if (in_cluster[i]) bits |= (uint64_t{1} << (i - base));
            }
            snap[w] = bits;
        }
        last_cluster_size_ = cluster_size;
        cluster_history_head_ = (cluster_history_head_ + 1) %
                                static_cast<int>(cluster_history_.size());
        if (cluster_history_count_ < static_cast<int>(cluster_history_.size())) {
            ++cluster_history_count_;
        }
    }

    /**
     * @brief Wolff cluster geometric overlap U_n^(W) = (1/N)|C_t ∩ C_{t-n}|,
     *        computed for every enabled separation n in one pass.
     *
     * Overwrites the primary overlap_values_[k] with the cluster value, so for
     * Wolff the "overlap_Nt{n}" observable is the cluster overlap 𝓡^W.
     * @return Cluster overlap at the largest enabled step, in [0,1] (0.0 if too little history).
     * @complexity O(|steps| · N).
     */
    double computeClusterOverlap() {
        if (cluster_history_count_ < overlap_step_max_ + 1) {
            std::fill(overlap_values_.begin(), overlap_values_.end(), 0.0);
            return 0.0;
        }
        const int cur_idx = (cluster_history_head_ - 1 +
                             static_cast<int>(cluster_history_.size())) %
                            static_cast<int>(cluster_history_.size());
        const auto& cur = cluster_history_[cur_idx];
        for (size_t k = 0; k < overlap_steps_.size(); ++k) {
            const int n = overlap_steps_[k];
            const int past_idx = (cluster_history_head_ - n - 1 +
                                  static_cast<int>(cluster_history_.size())) %
                                 static_cast<int>(cluster_history_.size());
            const auto& past = cluster_history_[past_idx];
            long intersection = 0;
            for (int w = 0; w < n_words_; ++w) {
                intersection += __builtin_popcountll(cur[w] & past[w]);
            }
            overlap_values_[k] =
                static_cast<double>(intersection) / static_cast<double>(N_);
        }
        return overlap_values_.back();
    }

    [[nodiscard]] int lastClusterSize() const { return last_cluster_size_; }

    /** @brief Config-overlap value at the largest enabled step (compat). */
    [[nodiscard]] double configOverlapValue() const {
        return config_overlap_values_.empty() ? 0.0 : config_overlap_values_.back();
    }
    /** @brief Config-overlap value at enabled-step index k. */
    [[nodiscard]] double configOverlapValueAt(int k) const {
        return (k >= 0 && k < static_cast<int>(config_overlap_values_.size()))
                   ? config_overlap_values_[k] : 0.0;
    }

    // --- Output helpers ---

    /** @brief Serialize model parameters as a JSON header string. */
    [[nodiscard]] std::string parameterJson() const {
        std::ostringstream oss;
        oss << "{\"L\":" << L_
            << ",\"J\":" << J_
            << ",\"q\":" << q_
            << ",\"beta\":" << beta_
            << ",\"N\":" << N_
            << "}";
        return oss.str();
    }

private:
    int L_;
    int N_;
    double J_;
    int q_;
    double beta_ = 1.0;
    uint64_t rng_seed_;
    std::vector<int> spins_;       ///< colors in {0,…,q-1}
    std::mt19937_64 rng_;
    ObservableRegistry observables_;

    // --- Overlap measurement state ---
    bool overlap_enabled_ = false;
    std::vector<int> overlap_steps_;                 ///< enabled separations n_k (sorted, de-duped)
    int overlap_step_max_ = 0;                       ///< max(overlap_steps_) — history depth-1
    std::vector<double> overlap_values_;             ///< per-step primary overlap (cluster for Wolff, config for Met/SW)
    std::vector<double> config_overlap_values_;      ///< per-step config overlap (Wolff secondary)

    std::vector<std::vector<uint64_t>> spin_history_;  ///< bit-packed spin history (2 bits/spin, q<=4) for config-overlap
    int n_spin_words_ = 0;                             ///< (N_ + 31) / 32
    int spin_history_head_ = 0;
    int spin_history_count_ = 0;

    std::vector<std::vector<uint64_t>> cluster_history_;  ///< bit-packed cluster membership (fast popcount intersection)
    int n_words_ = 0;                                     ///< (N_ + 63) / 64
    int cluster_history_head_ = 0;
    int cluster_history_count_ = 0;
    int last_cluster_size_ = 0;

    /** @brief Wrap coordinate into [0, L_) with periodic BC. */
    [[nodiscard]] int wrap(int coord) const {
        int r = coord % L_;
        return r < 0 ? r + L_ : r;
    }

    /** @brief Register the default observables (m, |m|, e). */
    void registerDefaultObservables() {
        observables_.registerObservable("magnetization",
            [](const void* ptr) -> double {
                return static_cast<const PottsModel*>(ptr)->magnetization();
            });
        observables_.registerObservable("abs_magnetization",
            [](const void* ptr) -> double {
                return static_cast<const PottsModel*>(ptr)->absMagnetization();
            });
        observables_.registerObservable("energy_per_spin",
            [](const void* ptr) -> double {
                return static_cast<const PottsModel*>(ptr)->energyPerSpin();
            });
    }
};

}  // namespace mc
