/**
 * @file potts_wolff.cpp
 * @brief Wolff single-cluster algorithm for the 2D q-state ferromagnetic Potts model.
 *
 * One "sweep":
 *   1. Pick a random seed site i0 (color c0).
 *   2. Grow a single cluster via BFS: for each unvisited neighbor j of the
 *      frontier, if s_j = c0, add j with probability p = 1 - exp(-β J).
 *   3. Recolor the entire cluster to a single uniform new color in {0,…,q-1}\{c0}.
 *
 * @ref  U. Wolff, Phys. Rev. Lett. 62, 361 (1989)
 * @complexity O(<|C|>) per sweep.
 *
 * Usage:
 *   potts_wolff --L 40 --J 1.0 --T 0.995 --q 3 --sweeps 50000 [--overlap-step 2]
 */

#include "../include/potts_model.hpp"
#include "../include/mc_base.hpp"

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
// WolffPottsSweep — CRTP update rule
// ---------------------------------------------------------------------------

/**
 * @brief Wolff single-cluster sweep for PottsModel.
 *
 * Grows one cluster per sweep via BFS with FK bond probability p = 1 - exp(-βJ)
 * (Potts factor 1), then recolors the whole cluster to one new color.
 * @complexity O(<|C|>) per sweep.
 */
class WolffPottsSweep : public MCSamplerCRTP<WolffPottsSweep> {
public:
    explicit WolffPottsSweep(PottsModel& model)
        : model_(model),
          in_cluster_(model.N(), false),
          stack_(),
          bond_prob_cache_(0.0) {
        stack_.reserve(model.N());
        updateBondProbability();
    }

    /**
     * @brief One Wolff sweep: seed → BFS grow → recolor cluster.
     *
     * Key identity (Potts): (1-p) = exp(-βJ).
     */
    void sweep_impl() {
        const int N = model_.N();
        updateBondProbability();

        std::fill(in_cluster_.begin(), in_cluster_.end(), false);
        stack_.clear();

        std::uniform_int_distribution<int> site_dist(0, N - 1);
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        auto& rng = model_.rng();

        // Step 1: Pick random seed
        const int seed = site_dist(rng);
        const int seed_spin = model_.spin(seed);
        in_cluster_[seed] = true;
        stack_.push_back(seed);

        // Step 2: BFS cluster growth on same-color neighbors
        int idx = 0;
        while (idx < static_cast<int>(stack_.size())) {
            const int site = stack_[idx];
            ++idx;

            auto neighbors = model_.nearestNeighbors(site);
            for (int nb : neighbors) {
                if (in_cluster_[nb]) continue;
                if (model_.spin(nb) != seed_spin) continue;
                if (uniform(rng) < bond_prob_cache_) {
                    in_cluster_[nb] = true;
                    stack_.push_back(nb);
                }
            }
        }

        // Step 3: Recolor the entire cluster to one uniform new color ≠ seed_spin
        const int new_color = model_.proposeNewColor(seed, rng);
        for (int site : stack_) {
            model_.setSpin(site, new_color);
        }

        cluster_size_last_ = static_cast<int>(stack_.size());

        if (model_.overlapEnabled()) {
            model_.saveSpinSnapshot();
            model_.computeConfigOverlap();
            model_.saveClusterSnapshot(in_cluster_, cluster_size_last_);
            model_.computeClusterOverlap();
        }

        ++total_sweeps_;
    }

    [[nodiscard]] static std::string name_impl() { return "WolffPotts"; }
    [[nodiscard]] int clusterSize() const { return cluster_size_last_; }
    [[nodiscard]] int totalSweeps() const { return total_sweeps_; }

private:
    PottsModel& model_;
    std::vector<bool> in_cluster_;
    std::vector<int> stack_;
    double bond_prob_cache_;
    int cluster_size_last_ = 0;
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
    double T = 0.994972;   // 3-state Potts Tc = 1/ln(1+√3)
    int therm_sweeps = 1000;
    int measure_sweeps = 10000;
    uint64_t seed = 0;
    bool init_random = true;
    bool auto_therm = false;
    bool time_series = false;
    std::vector<int> overlap_steps;   // empty = overlap disabled
    bool cluster_only = false;        // skip config overlap (large-L memory: spin_history ~67 GB at L=4096)
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
        else if (arg == "--cluster-only")        args.cluster_only = true;
        else if (arg == "--overlap-step" && i + 1 < argc) {
            args.overlap_steps = {std::atoi(argv[++i])};   // single-step (legacy alias)
        } else if (arg == "--overlap-steps" && i + 1 < argc) {
            // comma-separated list, e.g. "1,2,5,10,100"
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

    WolffPottsSweep sweep(model);
    MCSimulation<PottsModel, WolffPottsSweep> sim(model, sweep);

    // χ = <|C|> (FK identity), always available for Wolff
    model.observables().registerObservable("cluster_size",
        [](const void* ptr) -> double {
            return static_cast<double>(static_cast<const PottsModel*>(ptr)->lastClusterSize());
        });

    if (!args.overlap_steps.empty()) {
        model.enableOverlap(args.overlap_steps, args.cluster_only);

        // Per-Nt config overlap (secondary for Wolff; the primary overlap_Nt{n}
        // is the cluster overlap 𝓡^W, registered inside enableOverlap). Skipped
        // in --cluster-only mode (no spin_history_ buffer).
        if (!args.cluster_only) {
            for (int k = 0; k < static_cast<int>(args.overlap_steps.size()); ++k) {
                const int kk = k;
                model.observables().registerObservable(
                    "config_overlap_Nt" + std::to_string(args.overlap_steps[k]),
                    [kk](const void* ptr) -> double {
                        return static_cast<const PottsModel*>(ptr)->configOverlapValueAt(kk);
                    });
            }
        }
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
    std::cerr << "# cluster_size_last: " << sweep.clusterSize()
              << " / " << model.N() << " ("
              << 100.0 * sweep.clusterSize() / model.N() << "%)\n";
    if (args.time_series) {
        sim.writeTimeSeries(result, std::cout);
    }

    return 0;
}
