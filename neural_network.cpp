//
// Created by cartercpp on 10/2/26.
//

#include "neural_network.hpp"
#include <initializer_list>
#include <vector>
#include <memory_resource>
#include <random>
#include <utility>
#include <cstddef>
#include <cmath>
#include "math_vector.hpp"
#include "matrix.hpp"

math_vector<double> neural_network::Relu(math_vector<double> vec)
{
    for (std::size_t i = 0; i < vec.size(); ++i)
        vec[i] = (vec[i] > 0) * vec[i];

    return vec;
}

math_vector<double> neural_network::ReluDerivative(math_vector<double> vec)
{
    for (std::size_t i = 0; i < vec.size(); ++i)
        vec[i] = vec[i] > 0;

    return vec;
}

math_vector<double> neural_network::Sigmoid(math_vector<double> vec)
{
    for (std::size_t i = 0; i < vec.size(); ++i)
        vec[i] = 1 / (1 + std::exp(-vec[i]));

    return vec;
}

math_vector<double> neural_network::SigmoidDerivative(math_vector<double> vec)
{
    for (std::size_t i = 0; i < vec.size(); ++i)
        vec[i] = vec[i] * (1 - vec[i]);

    return vec;
}

std::pmr::vector<math_vector<double>> neural_network::forward(const math_vector<double>& input) const
{
    if (input.size() != m_neuronsPerLayer[0])
        throw std::invalid_argument{"Incorrect # of inputs given"};

    std::pmr::vector<math_vector<double>> activations{m_resource};
    activations.reserve(m_neuronsPerLayer.size());
    activations.push_back(input);

    for (std::size_t i = 0; i < m_weightMatrices.size(); ++i)
    {
        if (i + 1 < m_weightMatrices.size())
            activations.emplace_back(Relu(m_weightMatrices[i] * activations[i] + m_biasVectors[i]));
        else
            activations.emplace_back(Sigmoid(m_weightMatrices[i] * activations[i] + m_biasVectors[i]));
    }

    return activations;
}

math_vector<double> neural_network::predict(const math_vector<double>& input) const
{
    return forward(input).back();
}

math_vector<double> neural_network::backward(
    const std::pmr::vector<math_vector<double>>& activations,
    math_vector<double> delta,
    bool deltaIsPreActivation
)
{
    if (activations.size() != m_neuronsPerLayer.size())
        throw std::invalid_argument{"`activations` doesn't match NN configuration"};

    for (std::size_t i = 0; i < activations.size(); ++i)
        if (activations[i].size() != m_neuronsPerLayer[i])
            throw std::invalid_argument{"`activations` doesn't match NN configuration"};

    for (std::size_t i = 0; i + 1 < activations.size(); ++i)
    {
        if (i == 0)
        {
            if (!deltaIsPreActivation)
                delta = delta.multiply(SigmoidDerivative(activations[activations.size() - 1]));
        }
        else
            delta = delta.multiply(ReluDerivative(activations[activations.size() - i - 1]));

        m_weightDeltas[m_weightDeltas.size() - i - 1]
            += outer_product(delta, activations[activations.size() - i - 2]);
        m_biasDeltas[m_biasDeltas.size() - i - 1] += delta;

        delta = m_weightMatrices[m_weightMatrices.size() - i - 1].transpose() * delta;
    }

    return delta;
}

void neural_network::update_weights_and_biases()
{
    for (std::size_t i = 0; i < m_weightMatrices.size(); ++i)
    {
        m_weightMatrices[i] -= m_learningRate * m_weightDeltas[i];
        m_biasVectors[i] -= m_learningRate * m_biasDeltas[i];
    }

    zero_deltas();
}

void neural_network::zero_deltas()
{
    for (std::size_t i = 0; i < m_weightDeltas.size(); ++i)
    {
        matrix<double>& weightDeltaRef{m_weightDeltas[i]};

        for (std::size_t row = 0; row < weightDeltaRef.rows(); ++row)
            for (std::size_t column = 0; column < weightDeltaRef.columns(); ++column)
                weightDeltaRef[row][column] = 0;
    }

    for (std::size_t i = 0; i < m_biasDeltas.size(); ++i)
    {
        math_vector<double>& biasDeltaRef{m_biasDeltas[i]};

        for (std::size_t index = 0; index < biasDeltaRef.size(); ++index)
            biasDeltaRef[index] = 0;
    }
}

neural_network::neural_network(
    const std::initializer_list<matrix<double>> weightMatrices,
    const std::initializer_list<math_vector<double>> biasVectors,
    const std::initializer_list<std::size_t> neuronsPerLayer,
    double learningRate,
    std::pmr::memory_resource* resource
    )
        : m_weightMatrices{resource},
        m_weightDeltas{resource},
        m_biasVectors{resource},
        m_biasDeltas{resource},
        m_neuronsPerLayer{resource},
        m_learningRate{learningRate},
        m_resource{resource}
{
    m_neuronsPerLayer.reserve(neuronsPerLayer.size());
    for (const std::size_t neurons : neuronsPerLayer)
        m_neuronsPerLayer.push_back(neurons);

    m_weightMatrices.reserve(weightMatrices.size());
    m_weightDeltas.reserve(weightMatrices.size());
    for (const matrix<double>& weightMatrix : weightMatrices)
    {
        matrix<double> mat(weightMatrix.rows(), weightMatrix.columns(), 0, m_resource);
        for (std::size_t row = 0; row < weightMatrix.rows(); ++row)
            for (std::size_t column = 0; column < weightMatrix.columns(); ++column)
                mat[row][column] = weightMatrix[row][column];

        m_weightMatrices.emplace_back(std::move(mat));
        m_weightDeltas.emplace_back(weightMatrix.rows(), weightMatrix.columns(), 0, m_resource);
    }

    m_biasVectors.reserve(biasVectors.size());
    m_biasDeltas.reserve(biasVectors.size());
    for (const math_vector<double>& biasVector : biasVectors)
    {
        math_vector<double> vec(biasVector.size(), 0, m_resource);
        for (std::size_t i = 0; i < biasVector.size(); ++i)
            vec[i] = biasVector[i];

        m_biasVectors.emplace_back(std::move(vec));
        m_biasDeltas.emplace_back(biasVector.size(), 0, m_resource);
    }
}

neural_network::neural_network(
    const std::initializer_list<std::size_t> neuronsPerLayer,
    double learningRate,
    std::pmr::memory_resource* resource
    )
        : m_neuronsPerLayer{resource},
        m_learningRate{learningRate},
        m_resource{resource},
        m_weightMatrices{resource},
        m_weightDeltas{resource},
        m_biasVectors{resource},
        m_biasDeltas{resource}
{
    m_neuronsPerLayer.reserve(neuronsPerLayer.size());
    for (const std::size_t neurons : neuronsPerLayer)
        m_neuronsPerLayer.push_back(neurons);
    
    m_weightMatrices.reserve(m_neuronsPerLayer.size() - 1);
    m_weightDeltas.reserve(m_neuronsPerLayer.size() - 1);
    m_biasVectors.reserve(m_neuronsPerLayer.size() - 1);
    m_biasDeltas.reserve(m_neuronsPerLayer.size() - 1);

    std::mt19937 rng{std::random_device{}()};
    for (std::size_t layer = 1; layer < m_neuronsPerLayer.size(); ++layer)
    {
        const std::size_t layerSize = m_neuronsPerLayer[layer],
                          prevLayerSize = m_neuronsPerLayer[layer - 1];

        matrix<double> weights(layerSize, prevLayerSize, 0, m_resource),
                       weightDeltas(layerSize, prevLayerSize, 0, m_resource);
        math_vector<double> biases(layerSize, 0, m_resource),
                            biasDeltas(layerSize, 0, m_resource);

        std::normal_distribution<double> dist(0, 1 / std::sqrt(static_cast<double>(2 * prevLayerSize)));
        for (std::size_t i = 0; i < layerSize; ++i)
            for (std::size_t i2 = 0; i2 < prevLayerSize; ++i2)
                weights[i][i2] = dist(rng);

        m_weightMatrices.emplace_back(std::move(weights));
        m_weightDeltas.emplace_back(std::move(weightDeltas));
        m_biasVectors.emplace_back(std::move(biases));
        m_biasDeltas.emplace_back(std::move(biasDeltas));
    }
}
