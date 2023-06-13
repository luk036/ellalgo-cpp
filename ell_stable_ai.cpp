#include <iostream>
#include <vector>
#include <cmath>
#include <functional>

using Matrix = std::vector<std::vector<double>>;
using ArrayType = std::vector<double>;
using CutChoice = std::variant<double, ArrayType>;
using Cut = std::pair<ArrayType, CutChoice>;

class EllCalc {
    public:
        static double sum(const ArrayType& arr) {
            double sum = 0;
            for (const auto& elem : arr) {
                sum += elem;
            }
            return sum;
        }
};

enum class CutStatus {
    Success,
    Infeasible,
    Unbounded
};

class EllStable {
    public:
        bool no_defer_trick = false;
        Matrix mq;
        ArrayType xc;
        double kappa;
        double tsq;
        int n;
        EllCalc helper;

        CutStatus update_core(Cut cut, std::function<CutStatus(double, double)> cut_strategy) {
            ArrayType g = cut.first;
            CutChoice beta = cut.second;
            ArrayType invLg = g;  
            for (int j = 0; j < n - 1; j++) {
                for (int i = j + 1; i < n; i++) {
                    mq[j][i] = mq[i][j] * invLg[j];
                    invLg[i] -= mq[j][i];
                }
            }
            ArrayType invDinvLg = invLg;  
            for (int i = 0; i < n; i++) {
                invDinvLg[i] *= mq[i][i];
            }
            ArrayType gg_t(n);
            for (int i = 0; i < n; i++) {
                gg_t[i] = invLg[i] * invDinvLg[i];
            }
            double omega = helper.sum(gg_t);
            tsq = kappa * omega;  
            CutStatus status;
            double rho, sigma, delta;
            std::tie(status, rho, sigma, delta) = cut_strategy(std::get<double>(beta), tsq);
            ArrayType g_t = invDinvLg;  
            for (int i = n - 1; i > 0; i--) {
                for (int j = i; j < n; j++) {
                    g_t[i - 1] -= mq[j][i - 1] * g_t[j];  
                }
            }
            double mu = sigma / (1.0 - sigma);
            double oldt = omega / mu;  
            ArrayType v = g;
            kappa *= delta;
            if (no_defer_trick) {
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        mq[i][j] *= kappa;
                    }
                }
                kappa = 1.0;
            }
            return status;
        }
};

