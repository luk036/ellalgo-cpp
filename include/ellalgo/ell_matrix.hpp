#pragma once

#include <valarray>

/** Square matrix */
class Matrix {
    size_t ndim;
    std::valarray<double> data;

  public:
    /**
     * Constructor to create a new Matrix object.
     *
     * Example:
     * @code{.cpp}
     * Matrix A(2, 2);
     * @endcode
     *
     * @param[in] ndim - The dimension of the matrix (number of rows/columns)
     * @param[in] init - Optional initial value for all elements. Default is 0.0.
     */
    explicit Matrix(size_t ndim, double init = 0.0) : ndim{ndim}, data(init, ndim * ndim) {}

    /**
     * Sets all elements of the matrix to the given value.
     *
     * Example:
     * @code{.cpp}
     * Matrix A(2, 2);
     * A.clear(1.0);
     * @endcode
     *
     * @param[in] value - The value to set all elements to. Defaults to 0.0.
     */
    void clear(double value = 0.0) { data = value; }

    /**
     * Operator overload to access elements using 2D array syntax.
     *
     * Allows modifying elements using matrix(row, col) = value.
     *
     * Example:
     * @code{.cpp}
     * Matrix A(2, 2);
     * A(0, 0) = 1.0;
     * A(0, 1) = 2.0;
     * A(1, 0) = 3.0;
     * A(1, 1) = 4.0;
     * @endcode
     *
     * @param[in] row - Row index to access
     * @param[in] col - Column index to access
     * @return Reference to the element at the given row and column.
     */
    double& operator()(size_t row, size_t col) { return this->data[row * ndim + col]; }

    /**
     * Read-only accessor for matrix elements using 2D array syntax.
     *
     * Allows accessing elements using matrix(row, col).
     *
     * Example:
     * @code{.cpp}
     * Matrix A(2, 2);
     * A(0, 0) = 1.0;
     * A(0, 1) = 2.0;
     * A(1, 0) = 3.0;
     * A(1, 1) = 4.0;
     * @endcode
     *
     * @param[in] row - Row index
     * @param[in] col - Column index
     * @return Constant reference to element at (row, col)
     */
    const double& operator()(size_t row, size_t col) const { return this->data[row * ndim + col]; }

    /**
     * Multiplies each element of the matrix by the given scalar value.
     *
     * Example:
     * @code{.cpp}
     * Matrix A(2, 2);
     * A *= 2.0;
     * @endcode
     *
     * @param[in] alpha - The scalar value to multiply each element by.
     * @return Reference to this matrix after multiplication.
     */
    Matrix& operator*=(double alpha) {
        this->data *= alpha;
        return *this;
    }

    /**
     * Overloads the multiplication operator (*) to multiply the matrix by a scalar value.
     *
     * This performs an element-wise multiplication of the matrix by the scalar value alpha.
     *
     * Example:
     * >>> matrix = Matrix(3, 0.0)
     * >>> matrix *= 2.0
     * >>> matrix
     * valarray([0., 0., 0., 0., 0., 0., 0., 0., 0.],
     *          [3])
     *
     * @param[in] alpha - The scalar value to multiply the matrix by.
     * @return A new Matrix object containing the result of the multiplication.
     */
    Matrix operator*(double alpha) const {
        Matrix res(*this);
        return res *= alpha;
    }

    /**
     * Sets the matrix to be an identity matrix by clearing it and then setting
     * all diagonal elements to 1.
     *
     * Example:
     * >>> matrix = Matrix(3, 0.0)
     * >>> matrix.identity()
     * >>> matrix
     * valarray([1., 0., 0., 0., 1., 0., 0., 0., 1.],
     *          [3])
     */
    void identity() {
        this->clear();
        this->diagonal() = 1;
    }

    /**
     * Returns a view representing the diagonal elements of the matrix.
     *
     * This slices the underlying data array to extract the elements
     * where row index equals column index, corresponding to the diagonal.
     *
     * Example:
     * >>> matrix = Matrix(3, 0.0)
     * >>> matrix(0, 0) = 1.0
     * >>> matrix(1, 1) = 2.0
     * >>> matrix(2, 2) = 3.0
     * >>> matrix.diagonal()
     * valarray([1., 2., 3.], 3)
     *
     * @return View of the diagonal elements as a slice_array.
     */
    std::slice_array<double> diagonal() { return this->data[std::slice(0, ndim, ndim + 1)]; }

    /**
     * Returns a view representing the secondary diagonal elements of the matrix.
     *
     * This slices the underlying data array to extract the elements
     * where row index + column index equals ndim-1, corresponding to the secondary diagonal.
     *
     * Example:
     * >>> matrix = Matrix(3, 0.0)
     * >>> matrix(0, 0) = 1.0
     * >>> matrix(1, 1) = 2.0
     * >>> matrix(2, 2) = 3.0
     * >>> matrix.secondary_diagonal()
     * valarray([0., 0.], 2)
     *
     * @return View of the secondary diagonal elements as a slice_array.
     */
    std::slice_array<double> secondary_diagonal() {
        return this->data[std::slice(ndim - 1, ndim, ndim - 1)];
    }

    /**
     * Returns a view representing a row of the matrix.
     *
     * This slices the underlying data array to extract the elements
     * of the specified row index.
     *
     * Example:
     * >>> matrix = Matrix(3, 0.0)
     * >>> matrix(0, 0) = 1.0
     * >>> matrix(1, 1) = 2.0
     * >>> matrix(2, 2) = 3.0
     * >>> matrix.row(0)
     * valarray([1., 0., 0.], 3)
     * >>> matrix.row(1)
     * valarray([0., 2., 0.], 3)
     * >>> matrix.row(2)
     * valarray([0., 0., 3.], 3)
     *
     * @param[in] row - The index of the row to extract.
     * @return View of the row elements as a slice_array.
     */
    std::slice_array<double> row(std::size_t row) {
        return this->data[std::slice(ndim * row, ndim, 1)];
    }

    /**
     * Returns a view representing a column of the matrix.
     *
     * This slices the underlying data array to extract the elements
     * of the specified column index.
     *
     * Example:
     * >>> matrix = Matrix(3, 0.0)
     * >>> matrix(0, 0) = 1.0
     * >>> matrix(1, 1) = 2.0
     * >>> matrix(2, 2) = 3.0
     * >>> matrix.column(0)
     * valarray([1., 0., 0.], 3)
     * >>> matrix.column(1)
     * valarray([0., 2., 0.], 3)
     * >>> matrix.column(2)
     * valarray([0., 0., 3.], 3)
     *
     * @param[in] col - The index of the column to extract.
     * @return View of the column elements as a slice_array.
     */
    std::slice_array<double> column(std::size_t col) {
        return this->data[std::slice(col, ndim, ndim)];
    }

    /**
     * Calculates the trace of the matrix, which is the sum of the diagonal elements.
     *
     * The trace is computed by slicing the data array to extract just the diagonal
     * elements, and then summing those elements.
     *
     * Example:
     * >>> matrix = Matrix(3, 0.0)
     * >>> matrix(0, 0) = 1.0
     * >>> matrix(1, 1) = 2.0
     * >>> matrix(2, 2) = 3.0
     * >>> matrix.trace()
     * 6.0
     *
     * @return The trace of the matrix as a double.
     */
    double trace() const { return this->data[std::slice(0, ndim, ndim + 1)].sum(); }
};
