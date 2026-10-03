//
// Created by cartercpp on 10/2/26.
//

#pragma once

#include <initializer_list>
#include <memory_resource>
#include <cstddef>
#include "math_vector.hpp"
#include "neural_network.hpp"

class conditional_gan
{
public:

    // CONSTRUCTORS

    explicit conditional_gan(
        const std::initializer_list<std::size_t> generatorNeuronsPerLayer,
        const std::size_t labels,
        double generatorLearningRate,
        const std::initializer_list<std::size_t> discriminatorNeuronsPerLayer,
        double discriminatorLearningRate,
        std::pmr::memory_resource* resource = std::pmr::get_default_resource()
    )
        pre(generatorNeuronsPerLayer.size() >= 3)
        pre(discriminatorNeuronsPerLayer.size() >= 3)
        pre(*generatorNeuronsPerLayer.begin() > labels)
        pre(*(generatorNeuronsPerLayer.begin() + generatorNeuronsPerLayer.size() - 1)
            == *discriminatorNeuronsPerLayer.begin())
        pre(*(discriminatorNeuronsPerLayer.begin() + discriminatorNeuronsPerLayer.size() - 1) == labels);

    // METHODS

    void train_generator(std::size_t label);
    void train_discriminator(const math_vector<double>& realImage, std::size_t label);

    [[nodiscard]] math_vector<double> generate(std::size_t label) const;
    [[nodiscard]] math_vector<double> discriminate(const math_vector<double>& pixels) const;

    [[nodiscard]] std::pmr::memory_resource* resource() const;

private:

    neural_network m_generatorNN,
                   m_discriminatorNN;
    std::size_t m_noiseSize,
                m_labels;
    mutable std::pmr::memory_resource* m_resource;
};