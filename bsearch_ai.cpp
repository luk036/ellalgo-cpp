// Define the cutting_plane_feas function
std::pair<std::vector<double>, int> cutting_plane_feas(OracleFeas& omega, SearchSpace& space,
                                                       Options options) {
    for (int niter = 0; niter < options.max_iters; niter++) {
        auto cut = omega.assess_feas(space.xc());  // query the oracle at space.xc()
        if (cut.empty()) {                         // feasible sol'n obtained
            return {space.xc(), niter};
        }
        auto status = space.update_bias_cut(cut);  // update space
        if (status != CutStatus::Success || space.tsq() < options.tolerance) {
            return {{}, niter};
        }
    }
    return {{}, options.max_iters};
}

// Define the BSearchAdaptor class
class BSearchAdaptor : public OracleBS {
  public:
    BSearchAdaptor(OracleFeas2& omega, SearchSpace2& space, Options options)
        : omega(omega), space(space), options(options) {}

    bool assess_bs(double gamma) {
        SearchSpace2 space_copy = space;
        omega.update(gamma);
        auto [x_feas, _] = cutting_plane_feas(omega, space_copy, options);
        if (!x_feas.empty()) {
            space.set_xc(x_feas);
            return true;
        }
        return false;
    }

  private:
    OracleFeas2& omega;
    SearchSpace2& space;
    Options options;
};

// Define the bsearch function
std::pair<double, int> bsearch(OracleBS& omega, std::pair<double, double> intrvl, Options options) {
    double lower = intrvl.first;
    double upper = intrvl.second;
    for (int niter = 0; niter < options.max_iters; niter++) {
        double tau = (upper - lower) / 2;
        if (tau < options.tolerance) {
            return {upper, niter};
        }
        double gamma = lower + tau;
        if (omega.assess_bs(gamma)) {  // feasible sol'n obtained
            upper = gamma;
        } else {
            lower = gamma;
        }
    }
    return {upper, options.max_iters};
}

// Define the MyOracle3 class
class MyOracle3 : public OracleFeas2 {
  public:
    int idx = 0;
    double target = -1e100;

    double fn1(double x, double) { return -x - 1; }

    double fn2(double, double y) { return -y - 2; }

    double fn3(double x, double y) { return x + y - 1; }

    double fn4(double x, double y) { return 2 * x - 3 * y - target; }

    std::vector<double> grad1() { return {-1.0, 0.0}; }

    std::vector<double> grad2() { return {0.0, -1.0}; }

    std::vector<double> grad3() { return {1.0, 1.0}; }

    std::vector<double> grad4() { return {2.0, -3.0}; }

    MyOracle3() {
        fns = {&MyOracle3::fn1, &MyOracle3::fn2, &MyOracle3::fn3, &MyOracle3::fn4};
        grads = {&MyOracle3::grad1, &MyOracle3::grad2, &MyOracle3::grad3, &MyOracle3::grad4};
    }

    std::pair<std::vector<double>, double> assess_feas(std::vector<double> xc) {
        double x = xc[0];
        double y = xc[1];

        for (int i = 0; i < 4; i++) {
            idx = (idx + 1) % 4;  // round robin
            double fj = (this->*fns[idx])(x, y);
            if (fj > 0) {
                return (this->*grads[idx])();
            }
        }
        return {{}, 0};
    }

    void update(double gamma) { target = gamma; }
};

// Define the test_case_feasible function
void test_case_feasible() {
    std::vector<double> xinit = {0.0, 0.0};  // initial xinit
    Ell ellip(100.0, xinit);
    Options options;
    options.tolerance = 1e-8;
    BSearchAdaptor adaptor(MyOracle3(), ellip, options);
    auto [xbest, num_iters] = bsearch(adaptor, {-100.0, 100.0}, options);
    assert(!xbest.empty());
    assert(num_iters == 34);
}
