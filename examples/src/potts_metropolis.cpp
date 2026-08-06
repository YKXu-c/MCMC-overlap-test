/**
 * @file potts_metropolis.cpp
 * @brief Metropolis single-site update for the 2D q-state ferromagnetic Potts model.
 *
 * One "sweep" = N random single-site update attempts. For each attempt:
 *   1. Pick a random site i.
 *   2. Propose a new color c' ∈ {0,…,q-1} \ {s_i} (uniform).
 *   3. ΔE = J·(n_same(s_i) − n_same(c')); accept with probability min(1, exp(-β ΔE)).
 *
 * @ref  Metropolis et al., J. Chem. Phys. 21, 1087 (1953)
 * @complexity O(N) per sweep.
 *
 * Usage:
 *   potts_metropolis --L 32 --J 1.0 --T 0.995 --q 3 --sweeps 50000 [--overlap-step 2]
 */

#include "../include/potts_model.hpp"
#include "../include/mc_base.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

using namespace mc;

// ---------------------------------------------------------------------------
// MetropolisPottsSweep — CRTP update rule
// ---------------------------------------------------------------------------

/**
 * @brief Metropolis single-site sweep for PottsModel.
 * @complexity O(N) per sweep.
 */
class MetropolisPottsSweep : public MCSamplerCRTP<MetropolisPottsSweep> {
public:
    explicit MetropolisPottsSweep(PottsModel& model) : model_(model) {}

    /**
     * @brief One full sweep: N random single-site recolor attempts.
     *
     * Detailed balance: W(S→S')/W(S'→S) = e^{-βΔE} = π(S')/π(S).
     * If overlap is enabled, saves spin snapshot before the sweep and computes
     * configuration overlap U_n with the snapshot from n sweeps ago.
     */
    void sweep_impl() {
        std::uniform_int_distribution<int> site_dist(0, model_.N() - 1);
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        const double beta = model_.beta();
        auto& rng = model_.rng();

        for (int step = 0; step < model_.N(); ++step) {
            int site = site_dist(rng);

            // Propose a new color (uniform in the q-1 other colors)
            int new_color = model_.proposeNewColor(site, rng);

            // ΔE = J·(n_same_current − n_same_new)
            double deltaE = model_.energyChangeTo(site, new_color);

            if (deltaE <= 0.0 || uniform(rng) < std::exp(-beta * deltaE)) {
                model_.setSpin(site, new_color);
                ++accepted_;
            }
            ++attempted_;
        }

        // Config overlap U_n (computed AFTER the sweep).
        if (model_.overlapEnabled()) {
            model_.saveSpinSnapshot();
            model_.computeConfigOverlap();
        }
    }

    [[nodiscard]] static std::string name_impl() { return "MetropolisPotts"; }

    [[nodiscard]] double acceptanceRatio() const {
        return attempted_ > 0
            ? static_cast<double>(accepted_) / static_cast<double>(attempted_)
            : 0.0;
    }

    void resetCounters() { accepted_ = 0; attempted_ = 0; }

private:
    PottsModel& model_;
    long long accepted_ = 0;
    long long attempted_ = 0;
};

// ---------------------------------------------------------------------------
// CLI argument parsing
// ---------------------------------------------------------------------------

struct CLIArgs {
    int L = 16;
    double J = 1.0;
    int q = 3;
    double T = 0.994972;
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

    MetropolisPottsSweep sweep(model);
    MCSimulation<PottsModel, MetropolisPottsSweep> sim(model, sweep);

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
    if (args.time_series) {
        sim.writeTimeSeries(result, std::cout);
    }

    return 0;
}
