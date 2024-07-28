#include <algorithm>
#include <cmath>
#include <iostream>
#include <utility>  // for pair
#include <vector>

class Vector {
  public:
    std::vector<double> data;

    /**
     * The function `Vector` initializes a vector with the values provided in the `initializer_list`
     * `list`.
     *
     * @param[in] list The `list` parameter in the code snippet
     * `Vector(std::initializer_list<double> list) : data(list) {}` is of type
     * `std::initializer_list<double>`. This parameter allows you to initialize the `Vector` object
     * with a list of `double` values.
     */
    Vector(std::initializer_list<double> list) : data(list) {}

    /**
     * The Vector constructor initializes a vector of a specified size with an optional default
     * value for each element.
     *
     * @param[in] size The `size` parameter in the `Vector` constructor represents the number of
     * elements that the vector will initially contain.
     * @param[in] value The `value` parameter in the constructor of the `Vector` class is a default
     * parameter with a default value of `0.0`. This means that if no value is provided for the
     * `value` parameter when creating a `Vector` object, it will default to `0.0`.
     */
    Vector(size_t size, double value = 0.0) : data(size, value) {}

    /**
     * The function overloads the subscript operator to allow access to elements in an array-like
     * data structure.
     *
     * @return A reference to a double value at index `i` in the `data` array is being returned.
     */
    double& operator[](size_t i) { return data[i]; }

    /**
     * The function overloads the subscript operator to allow read-only access to elements in a data
     * array.
     *
     * @return A reference to a constant double value at index `i` in the `data` array is being
     * returned.
     */
    const double& operator[](size_t i) const { return data[i]; }

    /**
     * The `size()` function in C++ returns the size of the data container.
     *
     * @return The `size()` function is returning the size of the `data` container, which is likely
     * a `std::vector` or a similar container type. The return type of the function is `size_t`,
     * which is an unsigned integer type used for representing sizes of objects.
     */
    size_t size() const { return data.size(); }

    /**
     * The above function overloads the division assignment operator for a Vector class, dividing
     * each element in the vector by a scalar value.
     *
     * @return The `Vector` object that the `operator/=` function is being called on is being
     * returned.
     */
    Vector& operator/=(double scalar) {
        for (auto& val : data) val /= scalar;
        return *this;
    }

    /**
     * The `dot` function calculates the dot product of two vectors.
     *
     * @param[in] other The `other` parameter in the `dot` function is of type `Vector&`, which
     * means it is a reference to another `Vector` object. This parameter is used to calculate the
     * dot product of the current `Vector` object with another `Vector` object passed as an argument
     * to the function
     *
     * @return The `dot` product of the current `Vector` object and the `other` `Vector` object is
     * being returned.
     */
    double dot(const Vector& other) const {
        double sum = 0.0;
        for (size_t i = 0; i < size(); ++i) sum += data[i] * other.data[i];
        return sum;
    }

    /**
     * The `norm` function calculates the magnitude of a vector by taking the square root of the dot
     * product of the vector with itself.
     *
     * @return The `norm()` function is returning the square root of the dot product of the current
     * object with itself.
     */
    double norm() const { return std::sqrt(dot(*this)); }

    /**
     * The normalize function divides the current object by its norm to make it a unit vector.
     */
    void normalize() { *this /= norm(); }

    /**
     * The function calculates the infinity norm of a vector by finding the maximum absolute value
     * element.
     *
     * @return The function `inf_norm` is returning the infinity norm (also known as the maximum
     * norm) of a vector `data`. It calculates the maximum absolute value of the elements in the
     * vector using `std::max_element` and a lambda function that compares the absolute values of
     * the elements.
     */
    double inf_norm() const {
        return std::abs(*std::max_element(data.begin(), data.end(), [](double a, double b) {
            return std::abs(a) < std::abs(b);
        }));
    }

    /**
     * The function calculates the L1 norm of a vector by summing the absolute values of its
     * elements.
     *
     * @return The function `l1_norm` is returning the L1 norm of a vector, which is the sum of the
     * absolute values of all elements in the vector.
     */
    double l1_norm() const {
        double sum = 0.0;
        for (size_t i = 0; i < size(); ++i) sum += std::abs(data[i]);
        return sum;
    }
};

/**
 * This function overloads the subtraction operator for two vectors, returning a new vector with the
 * element-wise subtraction of the corresponding elements.
 *
 * @param[in] a The parameter `a` is a constant reference to a `Vector` object.
 * @param[in] b The parameter `b` in the `operator-` function represents a `Vector` object that is
 * being subtracted from another `Vector` object `a`.
 *
 * @return a new `Vector` object that contains the result of subtracting each element of vector `b`
 * from the corresponding element of vector `a`.
 */
Vector operator-(const Vector& a, const Vector& b) {
    Vector result(a.size());
    for (size_t i = 0; i < a.size(); ++i) result[i] = a[i] - b[i];
    return result;
}

/**
 * The function overloads the addition operator for two Vector objects to perform element-wise
 * addition and return the result.
 *
 * @param[in] a The parameters `a` and `b` are both of type `Vector`, which is a custom class
 * representing a mathematical vector. The `operator+` function overloads the `+` operator for
 * adding two vectors element-wise.
 * @param[in] b The parameter `b` in the `operator+` function represents a `Vector` object that is
 * being added to another `Vector` object `a`.
 *
 * @return The function `operator+` is returning a new `Vector` object that contains the
 * element-wise sum of the elements of two input `Vector` objects `a` and `b`.
 */
Vector operator+(const Vector& a, const Vector& b) {
    Vector result(a.size());
    for (size_t i = 0; i < a.size(); ++i) result[i] = a[i] + b[i];
    return result;
}

class Matrix {
  public:
    std::vector<std::vector<double>> data;

    /**
     * The Matrix constructor initializes the data using a list of lists of doubles.
     *
     * @param[in] list A list of lists of double values that is used to initialize the Matrix
     * object.
     */
    Matrix(std::initializer_list<std::initializer_list<double>> list)
        : data(list.begin(), list.end()) {}

    /**
     * The function overloads the * operator to perform matrix-vector multiplication.
     *
     * @param[in] v The `operator*` function you provided is for multiplying a matrix (represented
     * as a vector of vectors) by a vector. The `v` parameter in this context represents the vector
     * that you want to multiply the matrix with.
     *
     * @return The `operator*` function is returning a new `Vector` object that is the result of
     * multiplying the current `Vector` object (`this`) with another `Vector` object `v`.
     */
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

/**
 * The struct Options defines parameters for an algorithm with default values for maximum iterations
 * and tolerance.
 * @property {int} max_iters - The `max_iters` property in the `Options` struct represents the
 * maximum number of iterations that a particular algorithm or process will perform before stopping.
 * In this case, the default value for `max_iters` is set to 2000.
 * @property {double} tolerance - The `tolerance` property in the `Options` struct represents the
 * acceptable level of error or difference that is allowed in a calculation or algorithm. In this
 * case, the tolerance is set to `1e-9`, which means that the algorithm will stop iterating once the
 * difference between consecutive iterations falls
 */
struct Options {
    int max_iters = 2000;
    double tolerance = 1e-9;
};

/**
 * The function `power_iteration` implements the power iteration method to find the dominant
 * eigenvector of a matrix.
 *
 * @param[in] A A matrix representing the linear transformation
 * @param[in,out] x The parameter `x` in the `power_iteration` function represents the initial
 * vector for the power iteration algorithm. It is assumed to be a non-zero vector. The function
 * iteratively applies the matrix `A` to this vector and normalizes it at each step to find the
 * dominant eigenvector of
 * @param[in] options Options struct contains the following parameters:
 *
 * @return A pair containing the dominant eigenvalue of the matrix A and the number of iterations
 * performed by the power iteration algorithm.
 */
std::pair<double, int> power_iteration(const Matrix& A, Vector& x, const Options& options) {
    x.normalize();
    for (int niter = 0; niter < options.max_iters; ++niter) {
        const Vector x1 = x;
        x = A * x1;
        x.normalize();
        if ((x - x1).l1_norm() <= options.tolerance || (x + x1).l1_norm() <= options.tolerance) {
            return {(A * x).dot(x), niter};
        }
    }
    return {(A * x).dot(x), options.max_iters};
}

/**
 * The function `power_iteration4` implements the power iteration method to find the dominant
 * eigenvalue and eigenvector of a matrix.
 *
 * @param[in] A Matrix A is a matrix used in the power iteration algorithm.
 * @param[in,out] x The `x` parameter in the `power_iteration4` function represents the initial
 * vector used in the power iteration algorithm. It is a vector that will be iteratively transformed
 * by the matrix `A` to find the dominant eigenvector of `A`.
 * @param[in] options The `options` parameter in the `power_iteration4` function likely contains the
 * following fields:
 *
 * @return A std::pair<double, int> is being returned, where the first element is the dominant
 * eigenvalue of the matrix A and the second element is the number of iterations performed.
 */
std::pair<double, int> power_iteration4(const Matrix& A, Vector& x, const Options& options) {
    x /= x.l1_norm();
    for (int niter = 0; niter < options.max_iters; ++niter) {
        const Vector x1 = x;
        x = A * x1;
        x /= x.l1_norm();
        if ((x - x1).l1_norm() <= options.tolerance || (x + x1).l1_norm() <= options.tolerance) {
            x.normalize();
            return {(A * x).dot(x), niter};
        }
    }
    x.normalize();
    return {(A * x).dot(x), options.max_iters};
}

/**
 * The function `power_iteration2` implements the power iteration method to find the dominant
 * eigenvalue and eigenvector of a matrix.
 *
 * @param[in] A The parameter `A` in the `power_iteration2` function represents a matrix. It is used
 * for matrix-vector multiplication and eigenvalue calculations within the power iteration
 * algorithm.
 * @param[in,out] x The parameter `x` is a vector that represents the initial guess for the dominant
 * eigenvector of the matrix `A`. It is normalized at the beginning of the `power_iteration2`
 * function to ensure that the calculations are stable and converge properly.
 * @param[in] options The `Options` struct likely contains parameters for controlling the power
 * iteration algorithm. These parameters could include:
 *
 * @return A std::pair<double, int> is being returned, where the first element of the pair is the
 * dominant eigenvalue (ld) and the second element is the number of iterations (niter) taken to
 * converge or reach the maximum number of iterations specified in the options.
 */
std::pair<double, int> power_iteration2(const Matrix& A, Vector& x, const Options& options) {
    x.normalize();
    Vector new_x = A * x;
    double ld = x.dot(new_x);
    for (int niter = 0; niter < options.max_iters; ++niter) {
        const double ld1 = ld;
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

/**
 * The function `power_iteration3` implements the power iteration method to find the dominant
 * eigenvalue and eigenvector of a matrix.
 *
 * @param[in] A The parameter `A` in the `power_iteration3` function represents a matrix. It is used
 * in the power iteration algorithm to find the dominant eigenvalue and eigenvector of the matrix.
 * @param[in,out] x The `x` parameter in the `power_iteration3` function represents the initial
 * vector used in the power iteration algorithm. It is the starting vector that is iteratively
 * transformed and normalized to find the dominant eigenvalue and eigenvector of the given matrix
 * `A`.
 * @param[in] options The `Options` struct likely contains parameters for controlling the behavior
 * of the power iteration algorithm. These parameters could include:
 *
 * @return A std::pair<double, int> is being returned, where the first element of the pair is the
 * dominant eigenvalue (ld) and the second element is the number of iterations (niter) performed.
 */
std::pair<double, int> power_iteration3(const Matrix& A, Vector& x, const Options& options) {
    Vector new_x = A * x;
    double dot = x.dot(x);
    double ld = x.dot(new_x) / dot;
    for (int niter = 0; niter < options.max_iters; ++niter) {
        const double ld1 = ld;
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
