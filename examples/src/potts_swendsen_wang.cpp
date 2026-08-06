/**
 * @file potts_swendsen_wang.cpp
 * @brief Swendsen-Wang cluster algorithm for the 2D q-state ferromagnetic Potts model.
 *
 * One "sweep":
 *   1. For each pair of same-colored nearest neighbors, activate an FK bond
 *      with probability p = 1 - exp(-β J).   [Potts factor 1; Ising used 2.]
 *   2. Identify connected clusters via union-find.
 *   3. Recolor each cluster independently, uniformly in {0,…,q-1}.
 *
 * Exact spin-overlap result: P_coin^SW = 1/q  (independent of T, L, N_t).
 *
 * @ref  Swendsen & Wang, Phys. Rev. Lett. 58, 86 (1987)
 * @complexity O(N α(N)) per sweep.
 *
 * Usage:
 *   potts_swendsen_wang --L 32 --J 1.0 --T 0.995 --q 3 --sweeps 50000 [--overlap-step 1]
 */

#include "../include/potts_model.hpp"
#include "../include/mc_base.hpp"
#include "../include/union_find.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

using namespace mc;

// ---------------------------------------------------------------------------
// SwendsenWangPottsSweep — CRTP update rule
// ---------------------------------------------------------------------------

/**
 * @brief Swendsen-Wang multi-cluster sweep for PottsModel.
 * @complexity O(N α(N)) per sweep.
 */
class SwendsenWangPottsSweep
    : public MCSamplerCRTP<SwendsenWangPottsSweep> {
public:
    explicit SwendsenWangPottsSweep(PottsModel& model)
        : model_(model),
          uf_(model.N()),
          cluster_id_(model.N(), -1),
          new_color_(),
          bond_prob_cache_(0.0) {
        updateBondProbability();
    }

    /**
     * @brief One SW sweep: bond activation → cluster IDs → recolor.
     *
     * Key identity (Potts):  e^{βJ}(1-p) = 1  ⟹  p = 1 - e^{-βJ}.
     */
    void sweep_impl() {
        const int N = model_.N();
        const int L = model_.L();
        const int q = model_.q();
        updateBondProbability();

        uf_.reset();

        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        auto& rng = model_.rng();

        // Step 1: FK bond activation on same-color NN pairs (right + up)
        for (int site = 0; site < N; ++site) {
            const int x = site % L;
            const int y = site / L;
            const int si = model_.spin(site);

            const int right = y * L + (x + 1 == L ? 0 : x + 1);
            if (si == model_.spin(right)) {
                if (uniform(rng) < bond_prob_cache_) {
                    uf_.unite(site, right);
                }
            }

            const int up = ((y + 1 == L ? 0 : y + 1)) * L + x;
            if (si == model_.spin(up)) {
                if (uniform(rng) < bond_prob_cache_) {
                    uf_.unite(site, up);
                }
            }
        }

        // Step 2: Assign sequential cluster IDs from roots
        int num_clusters = 0;
        std::fill(cluster_id_.begin(), cluster_id_.end(), -1);
        for (int site = 0; site < N; ++site) {
            int root = uf_.find(site);
            if (cluster_id_[root] == -1) {
                cluster_id_[root] = num_clusters++;
            }
        }
        for (int site = 0; site < N; ++site) {
            cluster_id_[site] = cluster_id_[uf_.find(site)];
        }

        // Step 3: Recolor each cluster uniformly in {0,…,q-1}
        new_color_.assign(num_clusters, 0);
        std::uniform_int_distribution<int> color_dist(0, q - 1);
        for (int c = 0; c < num_clusters; ++c) {
            new_color_[c] = color_dist(rng);
        }

        auto& spins = model_.spins();
        for (int site = 0; site < N; ++site) {
            spins[site] = new_color_[cluster_id_[site]];
        }

        // Diagnostics
        num_clusters_last_ = num_clusters;
        std::vector<int> cluster_sizes(num_clusters, 0);
        for (int site = 0; site < N; ++site) {
            ++cluster_sizes[cluster_id_[site]];
        }
        largest_cluster_size_last_ =
            *std::max_element(cluster_sizes.begin(), cluster_sizes.end());

        // Config overlap U_n (computed AFTER the sweep).
        if (model_.overlapEnabled()) {
            model_.saveSpinSnapshot();
            model_.computeConfigOverlap();
        }

        ++total_sweeps_;
    }

    [[nodiscard]] static std::string name_impl() { return "SwendsenWangPotts"; }

    [[nodiscard]] int numClusters() const { return num_clusters_last_; }
    [[nodiscard]] int largestClusterSize() const { return largest_cluster_size_last_; }
    [[nodiscard]] int totalSweeps() const { return total_sweeps_; }

private:
    PottsModel& model_;
    UnionFind uf_;
    std::vector<int> cluster_id_;
    std::vector<int> new_color_;
    double bond_prob_cache_;
    int num_clusters_last_ = 0;
    int largest_cluster_size_last_ = 0;
    int total_sweeps_ = 0;

    /** @brief FK bond activation probability p = 1 - exp(-βJ) (Potts factor 1). */
    void updateBondProbability() {
        bond_prob_cache_ = 1.0 - std::exp(-1.0 * model_.beta() * model_.J());
    }
};

// ---------------------------------------------------------------------------
// CLI argument parsing
// ---------------------------------------------------------------------------

struct CLIArgs {
    int L = 16;
    double J = 1.0;
    int q = 3;
    double T = 0.994972;   // default: 3-state Potts Tc = 1/ln(1+√3)
    int therm_sweeps = 1000;
    int measure_sweeps = 10000;
    uint64_t seed = 0;
    bool init_random = true;
    bool auto_therm = false;
    bool time_series = false;
    std::vector<int> overlap_steps;   // empty = overlap disabled
};

CLIArgs parseArgs(int argc, char* argv[]) {
    CLIArgs args;
    for (int i = 1; i < argc; ++i) {
        std::string_view arg(argv[i]);
        if (arg == "--L" && i + 1 < argc)       args.L = std::atoi(argv[++i]);
        else if (arg == "--J" && i + 1 < argc)   args.J = std::atof(argv[++i]);
        else if (arg == "--q" && i + 1 < argc)   args.q = std::atoi(argv[++i]);
        else if (arg == "--T" && i + 1 < argc)   args.T = std::atof(argv[++i]);
        else if (arg == "--therm" && i + 1 < argc) args.therm_sweeps = std::atoi(argv[++i]);
        else if (arg == "--sweeps" && i + 1 < argc) args.measure_sweeps = std::atoi(argv[++i]);
        else if (arg == "--seed" && i + 1 < argc) args.seed = std::strtoull(argv[++i], nullptr, 10);
        else if (arg == "--all-up")              args.init_random = false;
        else if (arg == "--auto-therm")          args.auto_therm = true;
        else if (arg == "--ts")                  args.time_series = true;
        else if (arg == "--overlap-step" && i + 1 < argc) {
            args.overlap_steps = {std::atoi(argv[++i])};   // single-step (legacy alias)
        } else if (arg == "--overlap-steps" && i + 1 < argc) {
            std::stringstream ss(argv[++i]);
            std::string tok;
            while (std::getline(ss, tok, ',')) {
                if (!tok.empty()) args.overlap_steps.push_back(std::atoi(tok.c_str()));
            }
        }
    }
    return args;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    auto args = parseArgs(argc, argv);

    PottsModel model(args.L, args.J, args.q, args.seed, !args.init_random);
    model.setTemperature(args.T);

    SwendsenWangPottsSweep sweep(model);
    MCSimulation<PottsModel, SwendsenWangPottsSweep> sim(model, sweep);

    if (!args.overlap_steps.empty()) {
        model.enableOverlap(args.overlap_steps);
    }

    int actual_therm = 0;
    if (args.auto_therm) {
        ThermConfig cfg;
        auto therm = sim.autoThermalize("abs_magnetization", cfg);
        actual_therm = therm.sweeps_used;
        std::cerr << "# auto_therm: converged=" << (therm.converged ? "true" : "false")
                  << " sweeps=" << therm.sweeps_used
                  << " mean_1st=" << therm.mean_first_half
                  << " mean_2nd=" << therm.mean_second_half << "\n";
    } else {
        for (int i = 0; i < args.therm_sweeps; ++i) {
            sweep.sweep();
        }
        actual_therm = args.therm_sweeps;
    }

    auto result = sim.run(0, args.measure_sweeps,
                          /*measure_interval=*/1, args.time_series);

    sim.writeResults(result, std::cout);
    std::cerr << "# thermalization_sweeps_used: " << actual_therm << "\n";
    std::cerr << "# clusters_last: " << sweep.numClusters()
              << " largest_cluster: " << sweep.largestClusterSize()
              << " / " << model.N() << " ("
              << 100.0 * sweep.largestClusterSize() / model.N() << "%)\n";
    if (args.time_series) {
        sim.writeTimeSeries(result, std::cout);
    }

    return 0;
}
