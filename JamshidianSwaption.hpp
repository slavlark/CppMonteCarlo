#ifndef DISSERTATION_JAMSHIDIAN_HPP
#define DISSERTATION_JAMSHIDIAN_HPP

#include <vector>
#include <cmath>

namespace Dissertation::Models {

    inline double jamshidianPayerSwaption(
            double r0, double a, double b, double sigma,
            double K, double T_expiry, double T_tenor,
            double freq, double notional)
    {
        // ---- Vasicek ZCB building blocks ------------------------------------
        // P(t,T) = A(t,T) * exp(-B(t,T) * r)
        auto Bfn = [&](double t, double T) {
            return (1.0 - std::exp(-a * (T - t))) / a;
        };
        auto Afn = [&](double t, double T) {
            double Bv  = Bfn(t, T);
            double tau = T - t;
            double term1 = (b - (sigma * sigma) / (2.0 * a * a)) * (Bv - tau);
            double term2 = (sigma * sigma * Bv * Bv) / (4.0 * a);
            return std::exp(term1 - term2);
        };
        auto Pfn = [&](double t, double T, double r) {
            return Afn(t, T) * std::exp(-Bfn(t, T) * r);
        };

        // ---- Build the fixed-leg coupon schedule ----------------------------
        int n = static_cast<int>(T_tenor / freq);     // number of coupon dates
        std::vector<double> T_i(n), c_i(n);
        for (int i = 0; i < n; ++i) {
            T_i[i] = T_expiry + (i + 1) * freq;       // coupon date
            c_i[i] = K * freq;                         // coupon cashflow
        }
        c_i[n - 1] += 1.0;                             // principal at the last date

        // ---- Step 1: Newton solve for the critical short rate r* -----------
        // At r*, the coupon bond hits par at expiry: sum c_i * P(T_exp, T_i; r*) = 1
        double r_star = K;                             // initial guess
        for (int iter = 0; iter < 100; ++iter) {
            double f = -1.0, fprime = 0.0;
            for (int i = 0; i < n; ++i) {
                double Pi = Pfn(T_expiry, T_i[i], r_star);
                f      += c_i[i] * Pi;
                fprime -= c_i[i] * Bfn(T_expiry, T_i[i]) * Pi;
            }
            double delta = f / fprime;
            r_star -= delta;
            if (std::abs(delta) < 1e-12) break;        // converged
        }

        // ---- Step 2: derive each ZCB-put strike X_i ------------------------
        std::vector<double> X(n);
        for (int i = 0; i < n; ++i) {
            X[i] = Pfn(T_expiry, T_i[i], r_star);
        }

        // ---- Step 3: sum the ZCB-put portfolio (Black-style formula) -------
        // Normal CDF via complementary error function
        auto Phi = [](double x) {
            return 0.5 * std::erfc(-x / std::sqrt(2.0));
        };

        double P0Texp = Pfn(0.0, T_expiry, r0);        // today's price of expiry ZCB
        double price  = 0.0;
        for (int i = 0; i < n; ++i) {
            double P0Ti = Pfn(0.0, T_i[i], r0);        // today's price of coupon-date ZCB
            // Volatility of log P(T_exp, T_i) over [0, T_exp]
            double sigma_p = (sigma / a) *
                (1.0 - std::exp(-a * (T_i[i] - T_expiry))) *
                std::sqrt((1.0 - std::exp(-2.0 * a * T_expiry)) / (2.0 * a));
            double h = std::log(P0Ti / (P0Texp * X[i])) / sigma_p + 0.5 * sigma_p;
            // Vasicek ZCB put: X * P(0,T_exp) * Phi(-h+sigma_p) - P(0,T_i) * Phi(-h)
            double ZBP = X[i] * P0Texp * Phi(-h + sigma_p) - P0Ti * Phi(-h);
            price += c_i[i] * ZBP;
        }

        return notional * price;
    }

} // namespace Dissertation::Models

#endif
