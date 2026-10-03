//
// Created by cartercpp on 10/2/26.
//

#pragma once

#include <stdexcept>
#include <initializer_list>
#include <vector>
#include <algorithm>
#include <memory_resource>
#include <concepts>
#include <cstddef>

template <typename ValueType> requires (std::integral<ValueType> || std::floating_point<ValueType>)
class math_vector
{
public:

    // CONSTRUCTORS

    explicit math_vector(
        std::size_t size,
        ValueType value = 0,
        std::pmr::memory_resource* resource = std::pmr::get_default_resource()
    )
        : m_data{resource},
        m_resource{resource}
    {
        m_data.resize(size, value);
    }

    explicit math_vector(
        std::initializer_list<ValueType> data,
        std::pmr::memory_resource* resource = std::pmr::get_default_resource()
    )
        : m_data{resource},
        m_resource{resource}
    {
        m_data.resize(data.size());
        for (std::size_t i = 0; i < data.size(); ++i)
            m_data[i] = *(data.begin() + i);
    }

    // METHODS

    [[nodiscard]] std::size_t size() const noexcept
    {
        return m_data.size();
    }

    [[nodiscard]] ValueType& operator[](std::size_t index)
    {
        if (index >= m_data.size())
            throw std::out_of_range{"Out of bounds"};

        return m_data[index];
    }

    [[nodiscard]] const ValueType& operator[](std::size_t index) const
    {
        if (index >= m_data.size())
            throw std::out_of_range{"Out of bounds"};

        return m_data[index];
    }

    [[nodiscard]] math_vector multiply(const math_vector<ValueType>& other) const
    {
        if (m_data.size() != other.size())
            throw std::invalid_argument{"Cannot do element-wise multiplication of 2 vectors with different sizes"};

        math_vector<ValueType> output(m_data.size(), 0, m_resource);
        for (std::size_t i = 0; i < m_data.size(); ++i)
            output[i] = m_data[i] * other[i];

        return output;
    }

    math_vector& operator*=(ValueType scalar)
    {
        std::for_each(m_data.begin(), m_data.end(), [scalar](ValueType& valueRef){valueRef *= scalar;});
        return *this;
    }

    math_vector& operator/=(ValueType scalar)
    {
        std::for_each(m_data.begin(), m_data.end(), [scalar](ValueType& valueRef){valueRef /= scalar;});
        return *this;
    }

    math_vector& operator+=(const math_vector& other)
    {
        if (m_data.size() != other.size())
            throw std::invalid_argument{"Cannot add vectors with different sizes"};

        for (std::size_t i = 0; i < m_data.size(); ++i)
            m_data[i] += other[i];

        return *this;
    }

    math_vector& operator-=(const math_vector& other)
    {
        if (m_data.size() != other.size())
            throw std::invalid_argument{"Cannot subtract vectors with different sizes"};

        for (std::size_t i = 0; i < m_data.size(); ++i)
            m_data[i] -= other[i];

        return *this;
    }

    [[nodiscard]] std::pmr::memory_resource* resource() const noexcept
    {
        return m_resource;
    }

private:

    std::pmr::vector<ValueType> m_data;
    mutable std::pmr::memory_resource* m_resource;
};

template <typename ValueType>
bool operator==(const math_vector<ValueType>& lArg, const math_vector<ValueType>& rArg)
{
    if (lArg.size() != rArg.size())
        return false;

    for (std::size_t i = 0; i < lArg.size(); ++i)
        if (lArg[i] != rArg[i])
            return false;

    return true;
}

template <typename ValueType>
bool operator!=(const math_vector<ValueType>& lArg, const math_vector<ValueType>& rArg)
{
    return !(lArg == rArg);
}

template <typename ValueType>
auto operator*(const math_vector<ValueType>& lArg, const math_vector<ValueType>& rArg)
{
    if (lArg.size() != rArg.size())
        throw std::invalid_argument{"Cannot compute a dot product on 2 vectors with different sizes"};

    ValueType output = 0;
    for (std::size_t i = 0; i < lArg.size(); ++i)
        output += lArg[i] * rArg[i];

    return output;
}

template <typename ValueType>
auto operator*(math_vector<ValueType> vec, ValueType scalar)
{
    return vec *= scalar;
}

template <typename ValueType>
auto operator*(ValueType scalar, math_vector<ValueType> vec)
{
    return vec *= scalar;
}

template <typename ValueType>
auto operator/(math_vector<ValueType> vec, ValueType scalar)
{
    return vec /= scalar;
}

template <typename ValueType>
auto operator+(math_vector<ValueType> lArg, const math_vector<ValueType>& rArg)
{
    return lArg += rArg;
}

template <typename ValueType>
auto operator-(math_vector<ValueType> lArg, const math_vector<ValueType>& rArg)
{
    return lArg -= rArg;
}