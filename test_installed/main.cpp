#include <ellalgo/ell.hpp>
#include <iostream>
#include <utility>
#include <vector>

auto main() -> int {
    auto x = std::vector<double>{0.0, 0.0};
    auto ellip = Ell(10.0, x);

    auto grad = std::vector<double>{1.0, 1.0};
    const auto status = ellip.update_bias_cut(std::make_pair(grad, 0.0));

    std::cout << "ellalgo-cpp installed test: status=" << static_cast<int>(status) << " xc=("
              << ellip.xc()[0] << ", " << ellip.xc()[1] << ")\n";

    return status == CutStatus::Success ? 0 : 1;
}
