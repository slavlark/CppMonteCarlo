#ifndef DISSERTATION_SWAPTIONMODEL_HPP
#define DISSERTATION_SWAPTIONMODEL_HPP

#include <cmath>       // std::exp, std::sqrt
#include <random>      // std::mt19937, std::normal_distribution
#include <algorithm>   // std::max

namespace Dissertation::Models {

    class SwaptionModel {
    public:
        // Constructor: store all instrument and model parameters
        SwaptionModel(double r0, double a, double b, double sigma,
                      double K, double T_expiry, double T_tenor,
                      double freq, double notional, int steps = 252)
            : r0(r0), a(a), b(b), sigma(sigma),
              K(K), T_expiry(T_expiry), T_tenor(T_tenor),
              freq(freq), notional(notional),
              dt(T_expiry / steps), steps(steps) {}

        // operator() runs ONE Monte Carlo path and returns its discounted payoff.
        // Engine calls this N times in parallel and averages.
        double operator()() const {
            // thread_local: each thread gets its own independent generator state.
            // Avoids contention; each is seeded from the system RNG once per thread.
            static thread_local std::mt19937 generator(std::random_device{}());
            static thread_local std::normal_distribution<double> distribution(0.0, 1.0);

            // 1. Simulate the short rate path to expiry, accumulating discount integral
            double r_t        = r0;
            double integral_r = 0.0;
            double sqrt_dt    = std::sqrt(dt);   // precompute for speed

            for (int i = 0; i < steps; ++i) {
                double z = distribution(generator);             // standard Gaussian
                double r_prev = r_t;                            // save before update

                // Vasicek Euler step: dr = a(b - r_t)dt + sigma*sqrt(dt)*Z
                double dr = a * (b - r_t) * dt + sigma * sqrt_dt * z;
                r_t += dr;                                       // update rate

                // Trapezoidal: average the start-of-step and end-of-step rate.
                // Removes a small first-order bias vs right-Riemann.
                integral_r += 0.5 * (r_prev + r_t) * dt;
            }

            // 2. Price the underlying swap at expiry using Vasicek ZCB formula.
            //    The payer swap value equals (floating leg) - (fixed leg).
            double T_end    = T_expiry + T_tenor;
            // Standard identity: floating-leg value at par = 1 - P(T_exp, T_end)
            double P_end    = vasicekBondPrice(r_t, T_expiry, T_end);
            double val_float = 1.0 - P_end;

            // Fixed leg: K * Sum( freq * P(T_exp, T_i) ) over coupon dates
            double val_fixed = 0.0;
            int num_payments = static_cast<int>(T_tenor / freq);
            for (int i = 1; i <= num_payments; ++i) {
                double T_payment = T_expiry + i * freq;
                double P_payment = vasicekBondPrice(r_t, T_expiry, T_payment);
                val_fixed += K * freq * P_payment;
            }

            // Payer swap value, scaled to GBP
            double swap_value = (val_float - val_fixed) * notional;

            // 3. Option payoff: exercise only if swap value is positive
            double payoff = std::max(swap_value, 0.0);

            // 4. Discount back to today using the accumulated integral
            return payoff * std::exp(-integral_r);
        }

    private:
        // Vasicek zero-coupon bond price P(t, T) given short rate r_t at time t.
        // P(t,T) = A(t,T) * exp(-B(t,T) * r_t).
        double vasicekBondPrice(double r_t, double t, double T) const {
            double tau = T - t;
            if (tau <= 0.0) return 1.0;                          // ZCB at maturity = 1

            // B(t,T) = (1 - exp(-a*tau)) / a
            double B = (1.0 - std::exp(-a * tau)) / a;

            // A(t,T) = exp( (b - sigma^2/(2a^2)) * (B - tau)  -  sigma^2 * B^2 / (4a) )
            double term1 = (b - (sigma * sigma) / (2.0 * a * a)) * (B - tau);
            double term2 = (sigma * sigma * B * B) / (4.0 * a);
            double A     = std::exp(term1 - term2);

            return A * std::exp(-B * r_t);
        }

        // Instrument and model parameters
        double r0, a, b, sigma;
        double K, T_expiry, T_tenor, freq, notional;
        double dt;
        int steps;
    };

} // namespace Dissertation::Models

#endif