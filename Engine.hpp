//
// Engine.hpp -- Multi-threaded Monte Carlo simulation engine.
//
// Returns a Result struct with the mean, standard error, 95% confidence
// interval, and total paths. Each thread maintains its own RNG state
// (via thread_local in the Model) to avoid contention.
//
#ifndef DISSERTATION_ENGINE_HPP
#define DISSERTATION_ENGINE_HPP

#include <vector>      // std::vector
#include <future>      // std::async, std::future
#include <thread>      // std::thread::hardware_concurrency
#include <cmath>       // std::sqrt
#include <iostream>    // std::cout
#include <utility>     // std::pair

namespace Dissertation {

    // Result of one MC run -- what the engine returns to caller
    struct Result {
        double mean;              // sample mean of payoffs (the price estimate)
        double std_error;         // standard error of the mean = stdev / sqrt(N)
        double ci95_half_width;   // 1.96 * std_error
        long long num_paths;      // total paths actually run
    };

    template <typename Model>
    class Engine {
    public:
        // Constructor: number of simulations to target
        explicit Engine(int num_simulations) : num_sims(num_simulations) {}

        Result run(const Model& model) {
            // 1. Detect hardware threads. On M2 Pro this is typically 10 or 12.
            unsigned int num_threads = std::thread::hardware_concurrency();
            if (num_threads == 0) num_threads = 1;  // fallback if detection fails
            std::cout << "Number of Threads: " << num_threads << std::endl;

            // Divide work evenly. Integer division -- some paths may be dropped.
            int sims_per_thread = num_sims / num_threads;
            long long total_paths = static_cast<long long>(sims_per_thread) * num_threads;
            std::cout << "Sims per Thread:   " << sims_per_thread << std::endl;
            std::cout << "Total paths:       " << total_paths << std::endl;

            // 2. Launch threads. Each returns (sum, sum_of_squares) of payoffs.
            std::vector<std::future<std::pair<double, double>>> futures;
            futures.reserve(num_threads);
            for (unsigned int i = 0; i < num_threads; ++i) {
                futures.push_back(std::async(std::launch::async, [=, &model]() {
                    double sum = 0.0;
                    double sum_sq = 0.0;
                    for (int j = 0; j < sims_per_thread; ++j) {
                        double x = model();        // call the model to produce one payoff
                        sum    += x;               // accumulate for mean
                        sum_sq += x * x;           // accumulate for variance
                    }
                    return std::pair<double, double>{sum, sum_sq};
                }));
            }

            // 3. Wait for all threads and aggregate
            double total_sum = 0.0;
            double total_sum_sq = 0.0;
            for (auto& f : futures) {
                auto [s, sq] = f.get();
                total_sum    += s;
                total_sum_sq += sq;
            }

            // 4. Compute mean, sample variance (Bessel-corrected), standard error
            double N    = static_cast<double>(total_paths);
            double mean = total_sum / N;
            // Sample variance = (sum_sq - N*mean^2) / (N-1)
            double var  = (total_sum_sq - N * mean * mean) / (N - 1.0);
            if (var < 0.0) var = 0.0;                       // guard against FP cancellation
            double se   = std::sqrt(var / N);               // SE of the mean

            return Result{mean, se, 1.96 * se, total_paths};
        }

    private:
        int num_sims;
    };

} // namespace Dissertation

#endif
