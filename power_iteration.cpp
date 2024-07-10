#include <algorithm>
#include <cmath>
#include <iostream>
#include <utility>  // for pair
#include <vector>

class Vector {
  public:
    std::vector<double> data;

    Vector(std::initializer_list<double> list) : data(list) {}
    Vector(size_t size, double value = 0.0) : data(size, value) {}

    double& operator[](size_t i) { return data[i]; }
    const double& operator[](size_t i) const { return data[i]; }
    size_t size() const { return data.size(); }

    Vector& operator/=(double scalar) {
        for (auto& val : data) val /= scalar;
        return *this;
    }

    double dot(const Vector& other) const {
        double sum = 0.0;
        for (size_t i = 0; i < size(); ++i) sum += data[i] * other.data[i];
        return sum;
    }

    double norm() const { return std::sqrt(dot(*this)); }

    void normalize() { *this /= norm(); }

    double inf_norm() const {
        return std::abs(*std::max_element(data.begin(), data.end(), [](double a, double b) {
            return std::abs(a) < std::abs(b);
        }));
    }

    double l1_norm() const {
        double sum = 0.0;
        for (size_t i = 0; i < size(); ++i) sum += std::abs(data[i]);
        return sum;
    }
};

Vector operator-(const Vector& a, const Vector& b) {
    Vector result(a.size());
    for (size_t i = 0; i < a.size(); ++i) result[i] = a[i] - b[i];
    return result;
}

Vector operator+(const Vector& a, const Vector& b) {
    Vector result(a.size());
    for (size_t i = 0; i < a.size(); ++i) result[i] = a[i] + b[i];
    return result;
}

class Matrix {
  public:
    std::vector<std::vector<double>> data;

    Matrix(std::initializer_list<std::initializer_list<double>> list)
        : data(list.begin(), list.end()) {}

    Vector operator*(const Vector& v) const {
        Vector result(data.size());
        for (size_t i = 0; i < data.size(); ++i) {
            for (size_t j = 0; j < v.size(); ++j) {
                result[i] += data[i][j] * v[j];
            }
        }
        return result;
    }
};

struct Options {
    int max_iters = 2000;
    double tolerance = 1e-9;
};

std::pair<double, int> power_iteration(const Matrix& A, Vector& x, const Options& options) {
    x.normalize();
    for (int niter = 0; niter < options.max_iters; ++niter) {
        Vector x1 = x;
        x = A * x1;
        x.normalize();
        if ((x - x1).l1_norm() <= options.tolerance || (x + x1).l1_norm() <= options.tolerance) {
            double ld = (A * x1).dot(x1);
            return {ld, niter};
        }
    }
    double ld = (A * x).dot(x);
    return {ld, options.max_iters};
}

std::pair<double, int> power_iteration4(const Matrix& A, Vector& x, const Options& options) {
    x /= x.l1_norm();
    for (int niter = 0; niter < options.max_iters; ++niter) {
        Vector x1 = x;
        x = A * x1;
        x /= x.l1_norm();
        if ((x - x1).l1_norm() <= options.tolerance || (x + x1).l1_norm() <= options.tolerance) {
            x.normalize();
            double ld = (A * x).dot(x);
            return {ld, niter};
        }
    }
    x.normalize();
    double ld = (A * x).dot(x);
    return {ld, options.max_iters};
}

std::pair<double, int> power_iteration2(const Matrix& A, Vector& x, const Options& options) {
    x.normalize();
    Vector new_x = A * x;
    double ld = x.dot(new_x);
    for (int niter = 0; niter < options.max_iters; ++niter) {
        double ld1 = ld;
        x = new_x;
        x.normalize();
        new_x = A * x;
        ld = x.dot(new_x);
        if (std::abs(ld1 - ld) <= options.tolerance) {
            return {ld, niter};
        }
    }
    return {ld, options.max_iters};
}

std::pair<double, int> power_iteration3(const Matrix& A, Vector& x, const Options& options) {
    Vector new_x = A * x;
    double dot = x.dot(x);
    double ld = x.dot(new_x) / dot;
    for (int niter = 0; niter < options.max_iters; ++niter) {
        double ld1 = ld;
        x = new_x;
        dot = x.dot(x);
        if (dot >= 1e150) {
            x /= std::sqrt(dot);
            new_x = A * x;
            ld = x.dot(new_x);
            if (std::abs(ld1 - ld) <= options.tolerance) {
                return {ld, niter};
            }
        } else {
            new_x = A * x;
            ld = x.dot(new_x) / dot;
            if (std::abs(ld1 - ld) <= options.tolerance) {
                x /= std::sqrt(dot);
                return {ld, niter};
            }
        }
    }
    x /= std::sqrt(dot);
    return {ld, options.max_iters};
}

int main() {
    Matrix A{{3.7, -3.6, 0.7}, {-3.6, 4.3, -2.8}, {0.7, -2.8, 5.4}};
    Options options;

    options.tolerance = 1e-7;
    std::cout << "1-----------------------------\n";
    Vector x1{0.3, 0.5, 0.4};
    auto [ld1, niter1] = power_iteration(A, x1, options);
    std::cout << x1[0] << " " << x1[1] << " " << x1[2] << std::endl;
    std::cout << ld1 << std::endl;
    std::cout << niter1 << std::endl;

    std::cout << "4-----------------------------\n";
    Vector x4{0.3, 0.5, 0.4};
    auto [ld4, niter4] = power_iteration4(A, x4, options);
    std::cout << x4[0] << " " << x4[1] << " " << x4[2] << std::endl;
    std::cout << ld4 << std::endl;
    std::cout << niter4 << std::endl;

    options.tolerance = 1e-14;
    std::cout << "2-----------------------------\n";
    Vector x2{0.3, 0.5, 0.4};
    auto [ld2, niter2] = power_iteration2(A, x2, options);
    std::cout << x2[0] << " " << x2[1] << " " << x2[2] << std::endl;
    std::cout << ld2 << std::endl;
    std::cout << niter2 << std::endl;

    std::cout << "3-----------------------------\n";
    Vector x3{0.3, 0.5, 0.4};
    auto [ld3, niter3] = power_iteration3(A, x3, options);
    std::cout << x3[0] << " " << x3[1] << " " << x3[2] << std::endl;
    std::cout << ld3 << std::endl;
    std::cout << niter3 << std::endl;

    return 0;
}
