#include <cmath>
#include <optional>
#include <tuple>
#include <vector>

using Arr = std::vector<double>;
using Cut = std::tuple<Arr, double>;

/**
 * @brief
 *
 */
class ProfitOracle {
    double log_pA;
    double log_k;
    Arr price_out;
    Arr elasticities;

  public:
    /**
     * @brief Construct a new Profit Oracle object
     *
     * @param params
     * @param elasticities
     * @param price_out
     */
    ProfitOracle(std::tuple<double, double, double> params, Arr elasticities, Arr price_out) {
        double unit_price, scale, limit;
        std::tie(unit_price, scale, limit) = params;
        log_pA = std::log(unit_price * scale);
        log_k = std::log(limit);
        this->price_out = price_out;
        this->elasticities = elasticities;
    }

    /**
     * @brief
     *
     * @param y
     * @param gamma
     * @return std::tuple<Cut, std::optional<double>>
     */
    std::tuple<Cut, std::optional<double>> assess_optim(Arr y, double gamma) {
        double fj;
        if ((fj = y[0] - log_k) > 0.0) {
            Arr g = {1.0, 0.0};
            return std::make_tuple(g, fj), std::nullopt;
        }
        double log_Cobb = log_pA;
        for (int i = 0; i < elasticities.size(); i++) {
            log_Cobb += elasticities[i] * y[i];
        }
        Arr q;
        for (int i = 0; i < price_out.size(); i++) {
            q.push_back(price_out[i] * std::exp(y[i]));
        }
        double vx = q[0] + q[1];
        if ((fj = std::log(gamma + vx) - log_Cobb) >= 0.0) {
            Arr g;
            for (int i = 0; i < q.size(); i++) {
                g.push_back(q[i] / (gamma + vx) - elasticities[i]);
            }
            return std::make_tuple(g, fj), std::nullopt;
        }
        gamma = std::exp(log_Cobb) - vx;
        Arr g;
        for (int i = 0; i < q.size(); i++) {
            g.push_back(q[i] / (gamma + vx) - elasticities[i]);
        }
        return std::make_tuple(g, 0.0), gamma;
    }
};
