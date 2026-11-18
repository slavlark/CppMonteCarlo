// Created by Skylark
//
// main_calibrated.cpp
// -------------------
// Stage 2 of the vertical stack: price the 2Y x 2Y GBP payer swaption using
// Vasicek parameters extracted from the Fourier notebook, then compare to
// Stage 1 (hand-picked parameters).
//
// Reads three numbers from vasicek_params.txt:
//   line 1: a       (mean-reversion speed)
//   line 2: b       (long-term mean)
//   line 3: sigma   (annualised volatility)
//
// Uses the same Engine and SwaptionModel as Stage 1 -- no other changes.
//
// Build target name: DissertationStage2  (add to CMakeLists.txt)
//

#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <stdexcept>

#include "Engine.hpp"
#include "../Models/SwaptionModel.hpp"
#include "../Models/JamshidianSwaption.hpp"

using namespace Dissertation;
using namespace Dissertation::Models;

// Read the three parameters from vasicek_params.txt.
// Plain text, one number per line, in the order a, b, sigma.
struct VasicekParams { double a, b, sigma; };

VasicekParams loadParams(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Cannot open " + path +
            " -- run the Fourier notebook first to produce it.");
    }
    VasicekParams p;
    in >> p.a >> p.b >> p.sigma;
    if (!in) {
        throw std::runtime_error("Failed to parse three numbers from " + path);
    }
    return p;
}



int main() {
    // ---- Fixed instrument parameters (2Y x 2Y GBP ATM payer swaption) ------
    double r0       = 0.03;                            // initial short rate
    double K        = 0.03;                            // strike
    double T_expiry = 2.0;                             // option expiry (years)
    double T_tenor  = 2.0;                             // swap tenor after expiry
    double freq     = 0.5;                             // semi-annual coupons
    double notional = 1'000'000.0;                     // GBP 1m
    int    num_sims = 1'000'000;                      // total MC paths

    // ---- Stage 1: hand-picked parameters -----------------------------------
    double a1 = 0.10, b1 = 0.03, sigma1 = 0.01;

    // ---- Stage 2: read calibrated parameters from disk ---------------------
    VasicekParams fp = loadParams("/Users/skylark/Cprojects/Dissertation/src/vasicek_params.txt");

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "================================================================\n";
    std::cout << " Stage 2 -- Fourier-calibrated Vasicek vs Stage 1 baseline\n";
    std::cout << " Instrument: 2Y x 2Y GBP ATM Payer Swaption\n";
    std::cout << "================================================================\n\n";

    // Print both parameter sets so the reader can see what's being compared
    std::cout << "Stage 1 (hand-picked):  a=" << a1 << "  b=" << b1
              << "  sigma=" << sigma1 << "\n";
    std::cout << "Stage 2 (Fourier):       a=" << fp.a << "  b=" << fp.b
              << "  sigma=" << fp.sigma << "\n\n";

    // ===== STAGE 1 RUN ======================================================
    std::cout << "--- Stage 1 MC run ----------------------------------------------\n";
    SwaptionModel model1(r0, a1, b1, sigma1, K, T_expiry, T_tenor, freq, notional);
    Engine<SwaptionModel> engine1(num_sims);

    auto t0 = std::chrono::high_resolution_clock::now();
    auto r1 = engine1.run(model1);
    auto t1 = std::chrono::high_resolution_clock::now();
    double time1 = std::chrono::duration<double>(t1 - t0).count();

    // Analytical benchmark for the Stage 1 parameters
    double jam1 = jamshidianPayerSwaption(r0, a1, b1, sigma1, K,
                                           T_expiry, T_tenor, freq, notional);

    std::cout << std::setprecision(4);
    std::cout << "  MC price       = " << r1.mean << "\n";
    std::cout << "  SE             = " << r1.std_error << "\n";
    std::cout << "  95% CI         = [" << r1.mean - r1.ci95_half_width
              << ", " << r1.mean + r1.ci95_half_width << "]\n";
    std::cout << "  Jamshidian     = " << jam1
              << "   (delta " << r1.mean - jam1
              << " = " << (r1.mean - jam1) / r1.std_error << " SE)\n";
    std::cout << "  time           = " << time1 << " s\n\n";

    // ===== STAGE 2 RUN ======================================================
    std::cout << "--- Stage 2 MC run (Fourier-calibrated) -------------------------\n";
    SwaptionModel model2(r0, fp.a, fp.b, fp.sigma, K, T_expiry, T_tenor, freq, notional);
    Engine<SwaptionModel> engine2(num_sims);

    auto t2 = std::chrono::high_resolution_clock::now();
    auto r2 = engine2.run(model2);
    auto t3 = std::chrono::high_resolution_clock::now();
    double time2 = std::chrono::duration<double>(t3 - t2).count();

    double jam2 = jamshidianPayerSwaption(r0, fp.a, fp.b, fp.sigma, K,
                                           T_expiry, T_tenor, freq, notional);

    std::cout << "  MC price       = " << r2.mean << "\n";
    std::cout << "  SE             = " << r2.std_error << "\n";
    std::cout << "  95% CI         = [" << r2.mean - r2.ci95_half_width
              << ", " << r2.mean + r2.ci95_half_width << "]\n";
    std::cout << "  Jamshidian     = " << jam2
              << "   (delta " << r2.mean - jam2
              << " = " << (r2.mean - jam2) / r2.std_error << " SE)\n";
    std::cout << "  time           = " << time2 << " s\n\n";

    // ===== Comparison =======================================================
    std::cout << "--- Comparison --------------------------------------------------\n";
    std::cout << "  Stage 1 MC price:        " << r1.mean << "\n";
    std::cout << "  Stage 2 MC price:        " << r2.mean << "\n";
    std::cout << "  Difference (S2 - S1):    " << r2.mean - r1.mean
              << "  (" << 100.0 * (r2.mean - r1.mean) / r1.mean << "%)\n";
    std::cout << "\n  Compare against the market price computed via Bachelier\n";
    std::cout << "  from the BPV_OCM_CA quote on your chosen valuation date.\n";

    return 0;
}
