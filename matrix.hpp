//
// Created by cartercpp on 10/2/26.
//

#pragma once

#include <stdexcept>
#include <initializer_list>
#include <vector>
#include <memory_resource>
#include <algorithm>
#include <concepts>
#include <cstddef>
#include "math_vector.hpp"

template <typename ValueType> requires (std::integral<ValueType> || std::floating_point<ValueType>)
class matrix
{
public:

    // CONSTRUCTORS

    explicit matrix(
        std::size_t rows,
        std::size_t columns,
        ValueType value = 0,
        std::pmr::memory_resource* resource = std::pmr::get_default_resource()
    )
        : m_data{resource},
        m_rows{rows},
        m_columns{columns},
        m_resource{resource}
    {
        m_data.resize(rows, math_vector<ValueType>(columns, value, resource));
    }

    explicit matrix(
        std::initializer_list<math_vector<ValueType>> data,
        std::pmr::memory_resource* resource = std::pmr::get_default_resource()
    )
        : m_data{resource},
        m_rows{data.size()},
        m_columns{(data.size() > 0) ? data.begin()->size() : 0},
        m_resource{resource}
    {
        for (const math_vector<ValueType>& rowVec : data)
            if (rowVec.size() != m_columns)
                throw std::invalid_argument{"Each row in a matrix must have the same # of columns"};

        m_data.resize(m_rows, math_vector<ValueType>(m_columns, 0, m_resource));
        for (std::size_t row = 0; row < m_rows; ++row)
        {
            const math_vector<ValueType>& rowVec{*(data.begin() + row)};

            for (std::size_t column = 0; column < m_columns; ++column)
                m_data[row][column] = rowVec[column];
        }
    }

    // METHODS

    [[nodiscard]] std::size_t rows() const noexcept
    {
        return m_rows;
    }

    [[nodiscard]] std::size_t columns() const noexcept
    {
        return m_columns;
    }

    [[nodiscard]] math_vector<ValueType>& operator[](std::size_t row)
    {
        if (row >= m_rows)
            throw std::out_of_range{"Out of bounds"};

        return m_data[row];
    }

    [[nodiscard]] const math_vector<ValueType>& operator[](std::size_t row) const
    {
        if (row >= m_rows)
            throw std::out_of_range{"Out of bounds"};

        return m_data[row];
    }

    [[nodiscard]] math_vector<ValueType> get_column(std::size_t column) const
    {
        if (column >= m_columns)
            throw std::out_of_range{"Out of bounds"};

        math_vector<ValueType> output(m_rows, 0, m_resource);
        for (std::size_t row = 0; row < m_rows; ++row)
            output[row] = m_data[row][column];

        return output;
    }

    [[nodiscard]] matrix transpose() const
    {
        matrix output(m_columns, m_rows, 0, m_resource);

        for (std::size_t column = 0; column < m_columns; ++column)
            output[column] = get_column(column);

        return output;
    }

    matrix& operator+=(const matrix& other)
    {
        if ((m_rows != other.rows()) || (m_columns != other.columns()))
            throw std::invalid_argument{"Cannot add 2 matrices with different dimensions"};

        for (std::size_t row = 0; row < m_rows; ++row)
            m_data[row] += other[row];

        return *this;
    }

    matrix& operator-=(const matrix& other)
    {
        if ((m_rows != other.rows()) || (m_columns != other.columns()))
            throw std::invalid_argument{"Cannot subtract 2 matrices with different dimensions"};

        for (std::size_t row = 0; row < m_rows; ++row)
            m_data[row] -= other[row];

        return *this;
    }

    matrix& operator*=(ValueType scalar)
    {
        std::for_each(m_data.begin(), m_data.end(), [scalar](math_vector<ValueType>& vecRef){vecRef *= scalar;});
        return *this;
    }

    matrix& operator/=(ValueType scalar)
    {
        std::for_each(m_data.begin(), m_data.end(), [scalar](math_vector<ValueType>& vecRef){vecRef /= scalar;});
        return *this;
    }

    [[nodiscard]] std::pmr::memory_resource* resource() const noexcept
    {
        return m_resource;
    }

private:

    std::pmr::vector<math_vector<ValueType>> m_data;
    std::size_t m_rows,
                m_columns;
    mutable std::pmr::memory_resource* m_resource;
};

template <typename ValueType>
bool operator==(const matrix<ValueType>& lArg, const matrix<ValueType>& rArg)
{
    if ((lArg.rows() != rArg.rows()) || (lArg.columns() != rArg.columns()))
        return false;

    for (std::size_t row = 0; row < lArg.rows(); ++row)
        if (lArg[row] != rArg[row])
            return false;

    return true;
}

template <typename ValueType>
bool operator!=(const matrix<ValueType>& lArg, const matrix<ValueType>& rArg)
{
    return !(lArg == rArg);
}

template <typename ValueType>
auto outer_product(const math_vector<ValueType>& lArg, const math_vector<ValueType>& rArg)
{
    matrix<ValueType> output(lArg.size(), rArg.size(), 0, lArg.resource());

    for (std::size_t i = 0; i < lArg.size(); ++i)
        for (std::size_t i2 = 0; i2 < rArg.size(); ++i2)
            output[i][i2] = lArg[i] * rArg[i2];

    return output;
}

template <typename ValueType>
auto operator*(const matrix<ValueType>& mat, const math_vector<ValueType>& vec)
{
    if (mat.columns() != vec.size())
        throw std::invalid_argument{
            "To do matrix-vector multiplication, the matrix must have as many columns as the vector has elements"
        };

    math_vector<ValueType> output(mat.rows(), 0, mat.resource());
    for (std::size_t i = 0; i < mat.rows(); ++i)
        output[i] = mat[i] * vec;

    return output;
}

template <typename ValueType>
auto operator*(const math_vector<ValueType>& vec, const matrix<ValueType>& mat)
{
    if (vec.size() != mat.rows())
        throw std::invalid_argument{
            "To do vector-matrix multiplication, the vector must have as many elements as the matrix has rows"
        };

    math_vector<ValueType> output(mat.columns(), 0, vec.resource());
    for (std::size_t i = 0; i < mat.columns(); ++i)
        output[i] = vec * mat.get_column(i);

    return output;
}

template <typename ValueType>
auto operator*(const matrix<ValueType>& lArg, const matrix<ValueType>& rArg)
{
    if (lArg.columns() != rArg.rows())
        throw std::invalid_argument{
            "In order to multiply matrices A and B, A must have as many columns as B has rows"
        };

    matrix<ValueType> output(lArg.rows(), rArg.columns(), 0, lArg.resource());
    for (std::size_t row = 0; row < lArg.rows(); ++row)
        for (std::size_t column = 0; column < rArg.columns(); ++column)
            output[row][column] = lArg[row] * rArg.get_column(column);

    return output;
}

template <typename ValueType>
auto operator*(matrix<ValueType> mat, ValueType scalar)
{
    return mat *= scalar;
}

template <typename ValueType>
auto operator*(ValueType scalar, matrix<ValueType> mat)
{
    return mat *= scalar;
}

template <typename ValueType>
auto operator/(matrix<ValueType> mat, ValueType scalar)
{
    return mat /= scalar;
}

template <typename ValueType>
auto operator+(matrix<ValueType> lArg, const matrix<ValueType>& rArg)
{
    return lArg += rArg;
}

template <typename ValueType>
auto operator-(matrix<ValueType> lArg, const matrix<ValueType>& rArg)
{
    return lArg -= rArg;
}