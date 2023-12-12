#include <ellalgo/oracles/profit_oracle.hpp>

using Vec = std::valarray<double>;
using Cut = std::pair<Vec, double>;

/**
 * The function assess_feas assesses the feasibility of a given solution based on certain conditions
 * and returns a tuple containing a cut and a boolean value.
 *
 * @param[in] y The parameter `y` is a vector of values. It is used to calculate various values in
 * the function. The specific meaning of each element in the vector depends on the context and the
 * specific implementation of the `ProfitOracle` class.
 * @param[in,out] gamma The `gamma` parameter is a reference to a `double` variable. It is used to
 * store the best-so-far value for the feasibility process. The function `assess_feas` assesses
 * the feasibility of a given solution and updates the `gamma` value if necessary.
 *
 * @return The function `assess_feas` returns a tuple containing two elements. The first element is
 * of type `Cut`, which is a struct or class that contains a vector `g` and a double `fj`. The
 * second element is of type `bool`.
 */
auto ProfitOracle::assess_feas(const Vec &y, const double &gamma) -> Cut * {
    static auto cut1 = Cut{Vec{1.0, 0.0}, 0.0};
    static auto cut2 = Cut{Vec{-1.0, 1.0}, 0.0};

    const Vec x = std::exp(y);
    auto te = 0.0;

    for (int i = 0; i < 2; i++) {
        this->idx++;
        if (this->idx == 2) {
            this->idx = 0;  // round robin
        }
        double fj = 0.0;
        switch (this->idx) {
            case 0:  // y0 <= log k
                fj = y[0] - this->_log_k;
                break;
            case 1:
                this->_log_Cobb
                    = this->_log_pA + this->_elasticities[0] * y[0] + this->_elasticities[1] * y[1];
                this->_vx = this->_price_out[0] * x[0] + this->_price_out[1] * x[1];
                te = gamma + this->_vx;
                fj = std::log(te) - this->_log_Cobb;
                break;
            default:
                exit(0);
        }
        if (fj > 0.0) {
            switch (this->idx) {
                case 0:
                    cut1.second = fj;
                    return &cut1;
                case 1:
                    cut2.first = (this->_price_out * x) / te - this->_elasticities;
                    cut2.second = fj;
                    return &cut2;
                default:
                    exit(0);
            }
        }
    }

    return nullptr;
}

/**
 * The function assess_optim assesses the optimality of a given solution based on certain conditions
 * and returns a tuple containing a cut and a boolean value.
 *
 * @param[in] y The parameter `y` is a vector of values. It is used to calculate various values in
 * the function. The specific meaning of each element in the vector depends on the context and the
 * specific implementation of the `ProfitOracle` class.
 * @param[in,out] gamma The `gamma` parameter is a reference to a `double` variable. It is used to
 * store the best-so-far value for the optimization process. The function `assess_optim` assesses
 * the optimality of a given solution and updates the `gamma` value if necessary.
 *
 * @return The function `assess_optim` returns a tuple containing two elements. The first element is
 * of type `Cut`, which is a struct or class that contains a vector `g` and a double `fj`. The
 * second element is of type `bool`.
 */
auto ProfitOracle::assess_optim(const Vec &y, double &gamma) -> std::tuple<Cut, bool> {
    auto cut = this->assess_feas(y, gamma);
    if (cut) {
        return {*cut, false};
    }

    const Vec x = std::exp(y);
    auto te = std::exp(this->_log_Cobb);
    gamma = te - this->_vx;
    Vec grad = (this->_price_out * x) / te - this->_elasticities;
    return {{std::move(grad), 0.0}, true};
}

/**
 * The function assess_optim_q assesses the optimization of a given gamma value using a set of
 * input parameters.
 *
 * @param[in] y A vector containing the input values.
 * @param[in,out] gamma The "gamma" parameter is a reference to a double value. It is used to
 * store the best-so-far value for optimization.
 * @param[in] retry A boolean flag indicating whether the function should retry the assessment or not.
 *
 * @return The function `assess_optim_q` returns a tuple containing the following elements:
 */
auto ProfitOracleQ::assess_optim_q(const Vec &y, double &gamma, bool retry)
    -> std::tuple<Cut, bool, Vec, bool> {
    if (!retry) {
        auto cut = this->_P.assess_feas(y, gamma);
        if (cut) {
            return {*cut, false, y, true};
        }

        Vec x = std::exp(y);
        x = x.apply([](double n) { return std::round(n); });
        if (x[0] == 0.0) {
            x[0] = 1.0;  // nearest integer than 0
        }
        if (x[1] == 0.0) {
            x[1] = 1.0;
        }
        this->_yd = std::log(x);
    }
    auto result1 = this->_P.assess_optim(this->_yd, gamma);
    auto &cut = std::get<0>(result1);
    auto &shrunk = std::get<1>(result1);
    auto &grad = std::get<0>(cut);
    auto &beta = std::get<1>(cut);
    auto diff = this->_yd - y;
    beta += grad[0] * diff[0] + grad[1] * diff[1];
    return {std::move(cut), shrunk, this->_yd, !retry};
}
