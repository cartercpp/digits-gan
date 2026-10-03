//
// Created by cartercpp on 10/2/26.
//

#pragma once

#include <initializer_list>
#include <vector>
#include <memory_resource>
#include <cstddef>
#include "math_vector.hpp"
#include "matrix.hpp"

class neural_network
{
public:

    // CONSTRUCTORS

    explicit neural_network(
        const std::initializer_list<matrix<double>> weightMatrices,
        const std::initializer_list<math_vector<double>> biasVectors,
        const std::initializer_list<std::size_t> neuronsPerLayer,
        double learningRate,
        std::pmr::memory_resource* resource = std::pmr::get_default_resource()
    )
        pre(weightMatrices.size() == biasVectors.size())
        pre(weightMatrices.size() + 1 == neuronsPerLayer.size())
        pre(neuronsPerLayer.size() >= 3);

    explicit neural_network(
        const std::initializer_list<std::size_t> neuronsPerLayer,
        double learningRate,
        std::pmr::memory_resource* resource = std::pmr::get_default_resource()
    )
        pre(neuronsPerLayer.size() >= 3);

    // METHODS

    [[nodiscard]] std::pmr::vector<math_vector<double>> forward(const math_vector<double>& input) const;
    [[nodiscard]] math_vector<double> predict(const math_vector<double>& input) const;

    math_vector<double> backward(
        const std::pmr::vector<math_vector<double>>& activations,
        math_vector<double> delta,
        bool deltaIsPreActivation
    );
    void update_weights_and_biases();
    void zero_deltas();

private:

    [[nodiscard]] static math_vector<double> Sigmoid(math_vector<double>);
    [[nodiscard]] static math_vector<double> SigmoidDerivative(math_vector<double>);
    [[nodiscard]] static math_vector<double> Relu(math_vector<double>);
    [[nodiscard]] static math_vector<double> ReluDerivative(math_vector<double>);

    std::pmr::vector<matrix<double>> m_weightMatrices,
                                     m_weightDeltas;
    std::pmr::vector<math_vector<double>> m_biasVectors,
                                          m_biasDeltas;
    std::pmr::vector<std::size_t> m_neuronsPerLayer;
    double m_learningRate;
    mutable std::pmr::memory_resource* m_resource;
};