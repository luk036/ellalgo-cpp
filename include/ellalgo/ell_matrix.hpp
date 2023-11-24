#pragma once

#include <valarray>

/** Square matrix */
class Matrix {
    size_t ndim;
    std::valarray<double> data;

  public:
    /**
     * @brief Construct a new Matrix object
     *
     * The function is a constructor for a Matrix object that takes in the number of dimensions and an
     * optional initialization value.
     * 
     * @param[in] ndim The parameter `ndim` represents the dimension of the matrix. It specifies the number
     * of rows and columns in the matrix.
     * @param[in] init The `init` parameter is used to specify the initial value for all elements of the
     * matrix. By default, it is set to 0.0, which means that if no value is provided for `init` when
     * creating a `Matrix` object, all elements of the matrix will be initialized to
     */
    explicit Matrix(size_t ndim, double init = 0.0) : ndim{ndim}, data(init, ndim * ndim) {}

    /**
     * The clear function sets the value of a variable to 0.0.
     * 
     * @param[in] value The value parameter is a double type and it has a default value of 0.0.
     */
    void clear(double value = 0.0) { data = value; }

    /**
     * The function is an overloaded operator that allows accessing and modifying elements of a 2D
     * array using the () operator.
     * 
     * @param[in] row The parameter "row" is of type std::size_t and represents the index of the row that
     * you want to access in the data array.
     * @param[in] col The parameter "col" is of type std::size_t and represents the index of the column that
     * you want to access in the data array.
     * 
     * @return a reference to a double value.
     */
    double &operator()(size_t row, size_t col) { return this->data[row * ndim + col]; }

    /**
     * The function is an overloaded operator that returns a constant reference to a double value in a
     * 2D array-like structure.
     * 
     * @param[in] row The parameter "row" is of type std::size_t and represents the index of the row that
     * you want to access in the data array.
     * @param[in] col The parameter "col" is of type std::size_t and represents the index of the column that
     * you want to access in the data array.
     * 
     * @return a constant reference to a double value.
     */
    const double &operator()(size_t row, size_t col) const { return this->data[row * ndim + col]; }

    /**
     * The function multiplies each element of the matrix by a scalar value.
     * 
     * @param[in] alpha The parameter "alpha" is a double value that represents the scalar value by which
     * each element of the matrix is multiplied.
     * 
     * @return a reference to the current object, which is of type Matrix.
     */
    Matrix &operator*=(double alpha) {
        this->data *= alpha;
        return *this;
    }

    /**
     * The function overloads the multiplication operator to multiply a matrix by a scalar value.
     * 
     * @param[in] alpha The parameter "alpha" is a double value that represents the scalar value by which
     * each element of the matrix is multiplied.
     * 
     * @return a Matrix object.
     */
    Matrix operator*(double alpha) const {
        Matrix res(*this);
        return res *= alpha;
    }

    /**
     * The identity function sets the matrix to be a diagonal matrix with all diagonal elements equal
     * to 1.
     */
    void identity() {
        this->clear();
        this->diagonal() = 1;
    }

    /**
     * The function "diagonal" returns a slice of the "data" array containing the diagonal elements.
     * 
     * @return a `std::slice_array<double>`.
     */
    std::slice_array<double> diagonal() { return this->data[std::slice(0, ndim, ndim + 1)]; }

    /**
     * The function "secondary_diagonal" returns a slice of the "data" array containing the elements on
     * the secondary diagonal.
     * 
     * @return a `std::slice_array<double>`.
     */
    std::slice_array<double> secondary_diagonal() {
        return this->data[std::slice(ndim - 1, ndim, ndim - 1)];
    }

    /**
     * The function "row" returns a slice of a 2D array, representing a specific row.
     * 
     * @param[in] row The parameter "row" is of type std::size_t and represents the index of the row that
     * you want to access in the data array.
     * 
     * @return a `std::slice_array<double>`.
     */
    std::slice_array<double> row(std::size_t row) {
        return this->data[std::slice(ndim * row, ndim, 1)];
    }

    /**
     * The function "column" returns a slice of a 2D array, representing a specific column.
     * 
     * @param[in] col The parameter "col" is of type std::size_t and represents the index of the column that
     * you want to access in the data array.
     * 
     * @return a `std::slice_array<double>`.
     */
    std::slice_array<double> column(std::size_t col) {
        return this->data[std::slice(col, ndim, ndim)];
    }

    /**
     * The function calculates the trace of a matrix.
     * 
     * @return a double value.
     */
    double trace() const { return this->data[std::slice(0, ndim, ndim + 1)].sum(); }
};
