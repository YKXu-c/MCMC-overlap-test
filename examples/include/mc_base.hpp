#pragma once
/**
 * @file mc_base.hpp
 * @brief Core framework for Monte Carlo sampling using CRTP.
 *
 * Provides the simulation loop (burn-in → measurement) with zero-overhead
 * polymorphism via CRTP.  Designed so that the same MCSimulation<>
 * driver works for Ising, Heisenberg, and future QMC config types
 * by specializing ModelType and UpdateRule.
 *
 * @complexity Depends on the UpdateRule; the framework itself is O(1)
 *             overhead per sweep on top of the rule cost.
 */

#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace mc {

// ---------------------------------------------------------------------------
// Observable Registry — extensible measurement interface
// ---------------------------------------------------------------------------

/**
 * @brief Registry of named observable measurements.
 *
 * Each observable is a function that takes a const reference to the model
 * and returns a double.  Part 1 registers magnetization and energy;
 * Part 2 can add "cluster_overlap" without touching core code.
 *
 * @complexity O(K) per measurement, where K = number of registered observables.
 */
class ObservableRegistry {
public:
    using MeasureFn = std::function<double(const void*)>;

    /**
     * @brief Register a named observable.
     * @param name  Unique observable name (e.g. "magnetization").
     * @param fn    Measurement function bound to the concrete model.
     */
    void registerObservable(std::string_view name, MeasureFn fn) {
        observables_[std::string(name)] = std::move(fn);
    }

    /** @brief Measure all registered observables, return name→value map. */
    std::map<std::string, double> measureAll(const void* model) const {
        std::map<std::string, double> results;
        for (const auto& [name, fn] : observables_) {
            results[name] = fn(model);
        }
        return results;
    }

    /** @brief List registered observable names. */
    std::vector<std::string> names() const {
        std::vector<std::string> ns;
        for (const auto& [name, _] : observables_) ns.push_back(name);
        return ns;
    }

private:
    std::map<std::string, MeasureFn> observables_;
};

// ---------------------------------------------------------------------------
// MCSamplerCRTP — zero-overhead base for MC update rules
// ---------------------------------------------------------------------------

/**
 * @brief CRTP base for Monte Carlo update rules (sweeps).
 *
 * Derived classes must implement:
 *   - `void sweep_impl()`            — perform one full sweep
 *   - `std::string name() const`     — human-readable algorithm name
 *
 * The static_cast dispatch is resolved at compile time, so there is no
 * virtual-function overhead compared to a hand-written loop.
 *
 * @tparam Derived  Concrete sweep class (e.g. MetropolisSweep).
 */
template <typename Derived>
class MCSamplerCRTP {
public:
    /** @brief Perform one full MC sweep. */
    void sweep() { static_cast<Derived*>(this)->sweep_impl(); }

    /** @brief Algorithm name for logging / output headers. */
    [[nodiscard]] std::string name() const {
        return static_cast<const Derived*>(this)->name_impl();
    }
};

// ---------------------------------------------------------------------------
// MCResult — accumulated measurement statistics
// ---------------------------------------------------------------------------

/**
 * @brief Accumulated results from an MC simulation run.
 *
 * Stores per-observable: mean, mean-of-squares (for variance/error),
 * and the raw time series if retention is requested.
 */
struct MCResult {
    /** Per-observable statistics. */
    struct Stats {
        double mean  = 0.0;
        double mean2 = 0.0;  // <O^2> for variance / susceptibility
        std::vector<double> time_series;  // optional: kept if requested
    };

    std::map<std::string, Stats> observables;
    int total_sweeps = 0;
    int thermalization_sweeps = 0;
    uint64_t seed_used = 0;
};

// ---------------------------------------------------------------------------
// Thermalization convergence detection
// ---------------------------------------------------------------------------

/**
 * @brief Configuration for automatic thermalization convergence detection.
 *
 * Criterion: the observable has plateaued when the relative change between
 * the first and second halves of a sliding window falls below tolerance.
 */
struct ThermConfig {
    int check_interval = 100;      ///< Evaluate convergence every N sweeps.
    int window_size = 500;         ///< Sliding window length for plateau test.
    double rel_tolerance = 0.01;   ///< |Δ⟨O⟩| / |⟨O⟩| < tolerance ⇒ converged.
    int max_therm_sweeps = 100000; ///< Safety limit — abort if not converged.
};

/**
 * @brief Result of an automatic thermalization check.
 */
struct ThermResult {
    bool converged = false;        ///< Whether the observable plateaued.
    int sweeps_used = 0;           ///< Sweeps actually performed.
    double mean_first_half = 0.0;  ///< Mean of first half of the final window.
    double mean_second_half = 0.0; ///< Mean of second half of the final window.
};

// ---------------------------------------------------------------------------
// MCSimulation — main simulation driver (model–algorithm decoupled)
// ---------------------------------------------------------------------------

/**
 * @brief Top-level MC simulation driver.
 *
 * Owns the model and the update rule.  Handles:
 *   - thermalization (burn-in)
 *   - measurement sweeps with observable recording
 *   - JSON-header output
 *
 * @tparam ModelType  e.g. IsingModel — stores the configuration.
 * @tparam UpdateRule e.g. MetropolisSweep<IsingModel> — performs sweeps.
 *
 * @ref  General MC framework design following the strategy pattern.
 */
template <typename ModelType, typename UpdateRule>
class MCSimulation {
public:
    /**
     * @brief Construct a simulation referencing external model and updater.
     * @param model   The spin / quantum configuration (caller-owned).
     * @param updater The MC sweep rule (caller-owned, must reference model).
     *
     * @note The simulation does NOT take ownership.  Both model and updater
     *       must remain alive for the lifetime of this MCSimulation.
     *       This avoids dangling-reference issues when the updater holds
     *       a reference/pointer back to the model.
     */
    MCSimulation(ModelType& model, UpdateRule& updater)
        : model_(model), updater_(updater) {}

    /** @brief Access the model (mutable, for initialization). */
    [[nodiscard]] ModelType& model() { return model_; }
    [[nodiscard]] const ModelType& model() const { return model_; }

    /** @brief Access the update rule. */
    [[nodiscard]] UpdateRule& updater() { return updater_; }

    /**
     * @brief Run the full simulation: thermalize → measure.
     *
     * @param therm_sweeps    Number of burn-in sweeps (discarded).
     * @param measure_sweeps  Number of measurement sweeps.
     * @param measure_interval  Record observables every N sweeps (default 1).
     * @param keep_time_series  If true, store full time series in result.
     * @return MCResult with accumulated statistics.
     *
     * @complexity O((therm_sweeps + measure_sweeps) * cost_of_one_sweep).
     */
    MCResult run(int therm_sweeps, int measure_sweeps,
                 int measure_interval = 1, bool keep_time_series = false) {
        assert(therm_sweeps >= 0);
        assert(measure_sweeps > 0);
        assert(measure_interval >= 1);

        MCResult result;
        result.thermalization_sweeps = therm_sweeps;
        result.seed_used = model_.seed();

        // --- Thermalization (burn-in) ---
        for (int i = 0; i < therm_sweeps; ++i) {
            updater_.sweep();
        }

        // --- Measurement phase ---
        const auto obs_names = model_.observables().names();
        for (auto& name : obs_names) {
            result.observables[name] = {};
        }

        int measurements = 0;
        for (int i = 0; i < measure_sweeps; ++i) {
            updater_.sweep();
            if ((i + 1) % measure_interval == 0) {
                auto values = model_.observables().measureAll(&model_);
                for (auto& [name, val] : values) {
                    auto& st = result.observables[name];
                    st.mean += val;
                    st.mean2 += val * val;
                    if (keep_time_series) {
                        st.time_series.push_back(val);
                    }
                }
                ++measurements;
            }
        }
        result.total_sweeps = measure_sweeps;

        // --- Finalize means ---
        if (measurements > 0) {
            for (auto& [name, st] : result.observables) {
                st.mean /= static_cast<double>(measurements);
                st.mean2 /= static_cast<double>(measurements);
            }
        }
        return result;
    }

    /**
     * @brief Write results to stream in CSV format with JSON parameter header.
     *
     * Output format:
     *   # {"algorithm": "...", "L": ..., "J": ..., ...}
     *   # observable  mean  std  mean2
     *   magnetization  0.52  0.03  0.271
     *   energy         -1.93 0.01  3.725
     */
    void writeResults(const MCResult& result, std::ostream& os) const {
        // JSON parameter header
        os << "# " << model_.parameterJson() << "\n";
        os << "# algorithm: " << updater_.name() << "\n";
        os << "# thermalization_sweeps: " << result.thermalization_sweeps << "\n";
        os << "# measurement_sweeps: " << result.total_sweeps << "\n";
        os << "# seed: " << result.seed_used << "\n";
        os << "observable\tmean\tvariance\tmean2\n";

        for (const auto& [name, st] : result.observables) {
            double variance = st.mean2 - st.mean * st.mean;
            os << name << "\t" << st.mean << "\t"
               << variance << "\t" << st.mean2 << "\n";
        }
        os.flush();
    }

    /**
     * @brief Write time-series data for each observable.
     *
     * Output blocks delimited by `# time_series_begin/end` markers,
     * one value per line.  Only outputs observables with non-empty series.
     */
    void writeTimeSeries(const MCResult& result, std::ostream& os) const {
        for (const auto& [name, st] : result.observables) {
            if (!st.time_series.empty()) {
                os << "# time_series_begin: " << name << "\n";
                for (double val : st.time_series) {
                    os << val << "\n";
                }
                os << "# time_series_end\n";
            }
        }
        os.flush();
    }

    /**
     * @brief Auto-thermalize: sweep until observable plateaus or max sweeps reached.
     *
     * Runs sweeps in batches, tracking the named observable.  Convergence is
     * detected when the relative change between the first and second halves of
     * a sliding window falls below the configured tolerance.
     *
     * @param obs_name  Observable to monitor (e.g. "abs_magnetization").
     * @param cfg       Convergence parameters.
     * @return ThermResult with convergence status and sweep count.
     *
     * @complexity O(sweeps_used * cost_of_one_sweep).
     */
    ThermResult autoThermalize(const std::string& obs_name,
                               const ThermConfig& cfg = {}) {
        assert(cfg.check_interval > 0);
        assert(cfg.window_size > 0);
        assert(cfg.max_therm_sweeps > 0);

        ThermResult result;
        std::vector<double> history;
        history.reserve(cfg.max_therm_sweeps / cfg.check_interval + 1);

        int total = 0;
        while (total < cfg.max_therm_sweeps) {
            // Run a batch of sweeps
            int batch = std::min(cfg.check_interval, cfg.max_therm_sweeps - total);
            for (int i = 0; i < batch; ++i) {
                updater_.sweep();
            }
            total += batch;

            // Measure the observable
            auto values = model_.observables().measureAll(&model_);
            if (values.count(obs_name)) {
                history.push_back(values[obs_name]);
            } else {
                // Observable not found — can't check convergence
                result.converged = false;
                result.sweeps_used = total;
                return result;
            }

            // Check plateau once we have enough data points
            if (static_cast<int>(history.size()) >= 2) {
                int n = static_cast<int>(history.size());
                int win_start = std::max(0, n - cfg.window_size);
                int win_len = n - win_start;
                if (win_len >= 2) {
                    int half = win_len / 2;
                    double m1 = 0.0, m2 = 0.0;
                    for (int i = win_start; i < win_start + half; ++i)
                        m1 += history[i];
                    for (int i = win_start + half; i < win_start + win_len; ++i)
                        m2 += history[i];
                    m1 /= static_cast<double>(half);
                    m2 /= static_cast<double>(win_len - half);

                    double denom = std::max(std::abs(m1), 1e-10);
                    double rel_change = std::abs(m2 - m1) / denom;

                    if (rel_change < cfg.rel_tolerance) {
                        result.converged = true;
                        result.sweeps_used = total;
                        result.mean_first_half = m1;
                        result.mean_second_half = m2;
                        return result;
                    }
                }
            }
        }

        // Reached max sweeps without converging
        result.converged = false;
        result.sweeps_used = total;
        int n = static_cast<int>(history.size());
        int win_start = std::max(0, n - cfg.window_size);
        int win_len = n - win_start;
        if (win_len >= 2) {
            int half = win_len / 2;
            double m1 = 0.0, m2 = 0.0;
            for (int i = win_start; i < win_start + half; ++i)
                m1 += history[i];
            for (int i = win_start + half; i < win_start + win_len; ++i)
                m2 += history[i];
            m1 /= static_cast<double>(half);
            m2 /= static_cast<double>(win_len - half);
            result.mean_first_half = m1;
            result.mean_second_half = m2;
        }
        return result;
    }

private:
    ModelType& model_;
    UpdateRule& updater_;
};

}  // namespace mc
