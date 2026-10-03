//
// Created by cartercpp on 10/2/26.
//

#include "conditional_gan.hpp"
#include <stdexcept>
#include <initializer_list>
#include <memory_resource>
#include <random>
#include <cstddef>
#include "math_vector.hpp"
#include "neural_network.hpp"

conditional_gan::conditional_gan(
    const std::initializer_list<std::size_t> generatorNeuronsPerLayer,
    const std::size_t labels,
    double generatorLearningRate,
    const std::initializer_list<std::size_t> discriminatorNeuronsPerLayer,
    double discriminatorLearningRate,
    std::pmr::memory_resource* resource
) : m_generatorNN(generatorNeuronsPerLayer, generatorLearningRate, resource),
    m_discriminatorNN(discriminatorNeuronsPerLayer, discriminatorLearningRate, resource),
    m_noiseSize{*generatorNeuronsPerLayer.begin() - labels},
    m_labels{labels},
    m_resource{resource}
{
}

void conditional_gan::train_generator(std::size_t label)
{
    // add label to input:
    math_vector<double> input(m_noiseSize + m_labels, 0, m_resource);
    input[m_noiseSize + label] = 1;

    // add noise to input:
    std::mt19937 rng{std::random_device{}()};
    std::uniform_real_distribution<double> dist(-0.5, 0.5);
    for (std::size_t i = 0; i < m_noiseSize; ++i)
        input[i] = dist(rng);

    // generate image & get discriminator activations:
    const std::pmr::vector<math_vector<double>> generatorActivations{m_generatorNN.forward(input)};
    const math_vector<double>& generatedImage{generatorActivations.back()};
    const std::pmr::vector<math_vector<double>> discriminatorActivations{m_discriminatorNN.forward(generatedImage)};

    // trick discriminator:
    math_vector<double> fakeTarget(m_labels, 0, m_resource);
    fakeTarget[label] = true;
    const math_vector<double> fakeDelta{discriminatorActivations.back() - fakeTarget};

    const math_vector<double> discriminatorDelta
        = m_discriminatorNN.backward(discriminatorActivations, fakeDelta, true);
    m_discriminatorNN.zero_deltas();

    // use that to train generator:
    m_generatorNN.backward(generatorActivations, discriminatorDelta, false);
    m_generatorNN.update_weights_and_biases();
}

void conditional_gan::train_discriminator(const math_vector<double>& realImage, std::size_t label)
{
    {
        const std::pmr::vector<math_vector<double>> discriminatorActivations{m_discriminatorNN.forward(realImage)};
        const math_vector<double>& discriminatorPredictions{discriminatorActivations.back()};

        math_vector<double> target(m_labels, 0, m_resource);
        target[label] = 1;
        const math_vector<double> delta{discriminatorPredictions - target};

        m_discriminatorNN.backward(discriminatorActivations, delta, true);
        m_discriminatorNN.update_weights_and_biases();
    }

    {
        math_vector<double> input(m_noiseSize + m_labels, 0, m_resource);
        input[m_noiseSize + label] = 1;

        std::mt19937 rng{std::random_device{}()};
        std::uniform_real_distribution<double> dist(-0.5, 0.5);
        for (std::size_t i = 0; i < m_noiseSize; ++i)
            input[i] = dist(rng);

        const math_vector<double> fakeImage{m_generatorNN.predict(input)};
        const std::pmr::vector<math_vector<double>> discriminatorActivations{m_discriminatorNN.forward(fakeImage)};
        const math_vector<double>& discriminatorPredictions{discriminatorActivations.back()};

        m_discriminatorNN.backward(discriminatorActivations, discriminatorPredictions, true);
        m_discriminatorNN.update_weights_and_biases();
    }
}

math_vector<double> conditional_gan::generate(std::size_t label) const
{
    math_vector<double> input(m_noiseSize + m_labels, 0, m_resource);
    input[m_noiseSize + label] = 1;

    std::mt19937 rng{std::random_device{}()};
    std::uniform_real_distribution<double> dist(-0.5, 0.5);
    for (std::size_t i = 0; i < m_noiseSize; ++i)
        input[i] = dist(rng);

    return m_generatorNN.predict(input);
}

math_vector<double> conditional_gan::discriminate(const math_vector<double>& pixels) const
{
    return m_discriminatorNN.predict(pixels);
}

std::pmr::memory_resource* conditional_gan::resource() const
{
    return m_resource;
}