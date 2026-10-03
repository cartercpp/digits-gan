/*
    MNIST training + live terminal preview for Carter's conditional_gan.

    Put this file beside conditional_gan.hpp/.cpp, neural_network.hpp/.cpp,
    matrix.hpp, and math_vector.hpp. Use your project's compiler settings;
    its pre(...) declarations require C++26 contract support.

    Download the two MNIST TRAINING files once (Linux/macOS):
      mkdir -p mnist
      curl -fL https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz -o mnist/train-images-idx3-ubyte.gz
      curl -fL https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz -o mnist/train-labels-idx1-ubyte.gz
      gzip -dk mnist/train-images-idx3-ubyte.gz mnist/train-labels-idx1-ubyte.gz

    The mirror/file names are also used by torchvision's MNIST loader:
      https://github.com/pytorch/vision/blob/main/torchvision/datasets/mnist.py

    Run: ./main ./mnist
         ./main ./mnist --epochs 30 --label 7
         ./main ./mnist --steps 100 --no-ui

    Keys: 0-9 = select digit, A = cycle digits, P = pause, Q = quit.
    A UTF-8 terminal supports the full-resolution half-block display:
    28 columns x 14 character rows represent ALL 28 x 28 pixels.
    A tall terminal (42+ rows) also shows a real MNIST reference below it.
    --ascii uses a plain character ramp (pairs of rows are averaged).

    Every run trains from scratch. The current GAN API has no weight-saving
    or fixed-noise sampling method: each preview uses new random noise.
    The displayed D scores are the requested discriminator output, not
    externally measured digit accuracy. Recognizable digits take training.
*/

#include "conditional_gan.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory_resource>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>
#endif

namespace
{
using Clock = std::chrono::steady_clock;
constexpr std::size_t imageWidth = 28;
constexpr std::size_t pixelCount = imageWidth * imageWidth;
constexpr std::size_t classCount = 10;
constexpr std::size_t noiseSize = 64;
volatile std::sig_atomic_t stopRequested = 0;

void handle_signal(int) { stopRequested = 1; }

struct Options
{
    std::filesystem::path directory = "mnist";
    std::uint64_t epochs = 20;
    std::uint64_t maxSteps = 0; // Zero means use the epoch limit.
    double generatorLearningRate = 0.01;
    double discriminatorLearningRate = 0.002;
    std::size_t label = 0;
    bool automatic = true;
    bool ascii = false;
    bool noUI = false;
    bool help = false;
};

std::uint64_t parse_unsigned(std::string_view value)
{
    std::uint64_t result = 0;
    const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);
    if (error != std::errc{} || end != value.data() + value.size())
        throw std::invalid_argument("Expected a nonnegative integer: " + std::string(value));
    return result;
}

double parse_rate(const std::string& value)
{
    std::size_t consumed = 0;
    const double rate = std::stod(value, &consumed);
    if (consumed != value.size() || !std::isfinite(rate) || rate <= 0)
        throw std::invalid_argument("Learning rates must be finite and positive.");
    return rate;
}

Options parse_options(int argc, char** argv)
{
    Options options;
    bool directoryGiven = false;
    for (int i = 1; i < argc; ++i)
    {
        const std::string argument = argv[i];
        const auto next = [&]() -> std::string {
            if (i + 1 >= argc)
                throw std::invalid_argument("Missing value after " + argument);
            return argv[++i];
        };
        if (argument == "--help" || argument == "-h") options.help = true;
        else if (argument == "--epochs") options.epochs = parse_unsigned(next());
        else if (argument == "--steps") options.maxSteps = parse_unsigned(next());
        else if (argument == "--g-lr") options.generatorLearningRate = parse_rate(next());
        else if (argument == "--d-lr") options.discriminatorLearningRate = parse_rate(next());
        else if (argument == "--label")
        {
            const auto label = parse_unsigned(next());
            if (label >= classCount) throw std::invalid_argument("--label must be 0 through 9.");
            options.label = static_cast<std::size_t>(label);
            options.automatic = false;
        }
        else if (argument == "--ascii") options.ascii = true;
        else if (argument == "--no-ui") options.noUI = true;
        else if (!argument.empty() && argument.front() != '-' && !directoryGiven)
        {
            options.directory = argument;
            directoryGiven = true;
        }
        else throw std::invalid_argument("Unknown argument: " + argument);
    }
    if (options.epochs == 0) throw std::invalid_argument("--epochs must be at least 1.");
    return options;
}

void print_help()
{
    std::cout << "Usage: ./main [mnist-directory] [options]\n"
                 "  --epochs N    Training passes through MNIST (default: 20)\n"
                 "  --steps N     Stop training after N steps (0: epoch limit)\n"
                 "  --label N     Start on digit N without automatic cycling\n"
                 "  --g-lr X      Generator learning rate (default: 0.01)\n"
                 "  --d-lr X      Discriminator learning rate (default: 0.002)\n"
                 "  --ascii       Plain character images instead of color blocks\n"
                 "  --no-ui       Print periodic progress and a final ASCII sample\n\n"
                 "Keys: 0-9 select a digit, A auto-cycle, P pause, Q quit.\n"
                 "After training, an interactive preview stays open until Q.\n"
                 "Expected files (uncompressed):\n"
                 "  train-images-idx3-ubyte\n"
                 "  train-labels-idx1-ubyte\n"
                 "Download/extraction commands are at the top of main.cpp.\n";
}

std::uint32_t read_big_endian(std::istream& stream)
{
    std::array<unsigned char, 4> bytes{};
    if (!stream.read(reinterpret_cast<char*>(bytes.data()), bytes.size()))
        throw std::runtime_error("Truncated MNIST header.");
    return (std::uint32_t{bytes[0]} << 24) | (std::uint32_t{bytes[1]} << 16)
         | (std::uint32_t{bytes[2]} << 8) | std::uint32_t{bytes[3]};
}

std::filesystem::path find_idx_file(const std::filesystem::path& directory,
                                   const std::string& name, const std::string& alternate)
{
    for (const auto& candidate : {name, alternate})
        if (std::filesystem::is_regular_file(directory / candidate))
            return directory / candidate;
    throw std::runtime_error("Cannot find " + (directory / name).string()
        + ". Download both training files and extract the .gz archives first; see main.cpp.");
}

struct MNIST
{
    // Keep the dataset as bytes: about 47 MB, instead of 376 MB of doubles.
    std::vector<std::uint8_t> images;
    std::vector<std::uint8_t> labels;
    std::array<std::vector<std::size_t>, classCount> byLabel;

    void copy_image(std::size_t index, math_vector<double>& destination) const
    {
        const std::size_t offset = index * pixelCount;
        for (std::size_t pixel = 0; pixel < pixelCount; ++pixel)
            destination[pixel] = static_cast<double>(images[offset + pixel]) / 255.0;
    }
};

MNIST load_mnist(const std::filesystem::path& directory)
{
    const auto imagesPath = find_idx_file(directory, "train-images-idx3-ubyte", "train-images.idx3-ubyte");
    const auto labelsPath = find_idx_file(directory, "train-labels-idx1-ubyte", "train-labels.idx1-ubyte");
    std::ifstream imagesFile(imagesPath, std::ios::binary);
    std::ifstream labelsFile(labelsPath, std::ios::binary);
    if (!imagesFile || !labelsFile) throw std::runtime_error("Could not open the MNIST files.");
    if (read_big_endian(imagesFile) != 2051 || read_big_endian(labelsFile) != 2049)
        throw std::runtime_error("Invalid MNIST IDX magic numbers. Use uncompressed IDX files, not .gz or CSV.");
    const std::uint32_t count = read_big_endian(imagesFile);
    const std::uint32_t labelCount = read_big_endian(labelsFile);
    const std::uint32_t rows = read_big_endian(imagesFile);
    const std::uint32_t columns = read_big_endian(imagesFile);
    if (count == 0 || count != labelCount || rows != imageWidth || columns != imageWidth)
        throw std::runtime_error("MNIST must contain matching, nonempty labels and 28x28 images.");
    const std::uint64_t imageBytes = std::uint64_t{count} * pixelCount;
    if (imageBytes > std::numeric_limits<std::size_t>::max()
        || imageBytes > static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max())
        || std::filesystem::file_size(imagesPath) != 16 + imageBytes
        || std::filesystem::file_size(labelsPath) != 8 + std::uint64_t{count})
        throw std::runtime_error("MNIST file sizes do not match their headers (possibly truncated files).");
    MNIST data;
    data.images.resize(static_cast<std::size_t>(imageBytes));
    data.labels.resize(count);
    if (!imagesFile.read(reinterpret_cast<char*>(data.images.data()), static_cast<std::streamsize>(imageBytes))
        || !labelsFile.read(reinterpret_cast<char*>(data.labels.data()), static_cast<std::streamsize>(count)))
        throw std::runtime_error("Could not read the complete MNIST dataset.");
    for (std::size_t i = 0; i < data.labels.size(); ++i)
    {
        if (data.labels[i] >= classCount) throw std::runtime_error("MNIST contains a label outside 0-9.");
        data.byLabel[data.labels[i]].push_back(i);
    }
    for (const auto& indices : data.byLabel)
        if (indices.empty()) throw std::runtime_error("The dataset needs at least one example of every digit 0-9.");
    return data;
}

class DefaultResourceScope
{
    std::pmr::memory_resource* previous;
public:
    explicit DefaultResourceScope(std::pmr::memory_resource* resource)
        : previous(std::pmr::set_default_resource(resource)) {}
    ~DefaultResourceScope() { std::pmr::set_default_resource(previous); }
    DefaultResourceScope(const DefaultResourceScope&) = delete;
    DefaultResourceScope& operator=(const DefaultResourceScope&) = delete;
};

class Terminal
{
    bool active = false;
#if defined(__unix__) || defined(__APPLE__)
    termios original{};
#endif
public:
    explicit Terminal(bool enabled)
    {
#if defined(__unix__) || defined(__APPLE__)
        if (enabled && ::isatty(STDIN_FILENO) && ::isatty(STDOUT_FILENO)
            && ::tcgetattr(STDIN_FILENO, &original) == 0)
        {
            termios mode = original;
            mode.c_lflag &= static_cast<tcflag_t>(~(ICANON | ECHO));
            mode.c_cc[VMIN] = 0;
            mode.c_cc[VTIME] = 0;
            active = ::tcsetattr(STDIN_FILENO, TCSANOW, &mode) == 0;
            if (active) std::cout << "\x1b[?1049h\x1b[?25l\x1b[2J\x1b[H" << std::flush;
        }
#else
        (void)enabled;
#endif
    }
    ~Terminal()
    {
        if (active)
        {
#if defined(__unix__) || defined(__APPLE__)
            ::tcsetattr(STDIN_FILENO, TCSANOW, &original);
#endif
            std::cout << "\x1b[0m\x1b[?25h\x1b[?1049l" << std::flush;
        }
    }
    Terminal(const Terminal&) = delete;
    Terminal& operator=(const Terminal&) = delete;
    bool interactive() const { return active; }
    int key() const
    {
#if defined(__unix__) || defined(__APPLE__)
        char value = 0;
        if (active && ::read(STDIN_FILENO, &value, 1) == 1)
            return static_cast<unsigned char>(value);
#endif
        return -1;
    }
    unsigned rows() const
    {
#if defined(__unix__) || defined(__APPLE__)
        winsize size{};
        if (active && ::ioctl(STDOUT_FILENO, TIOCGWINSZ, &size) == 0 && size.ws_row)
            return size.ws_row;
#endif
        return 24;
    }
};

std::string image_panel(const math_vector<double>& pixels, const std::string& title, bool ascii)
{
    constexpr std::string_view ramp = " .:-=+*#%@";
    std::ostringstream out;
    out << "  + " << title << ' ' << std::string(26 - title.size(), '-') << "+\n";
    for (std::size_t row = 0; row < imageWidth; row += 2)
    {
        out << "  |";
        for (std::size_t column = 0; column < imageWidth; ++column)
        {
            const double upper = pixels[row * imageWidth + column];
            const double lower = pixels[(row + 1) * imageWidth + column];
            if (!std::isfinite(upper) || !std::isfinite(lower))
                throw std::runtime_error("Generated pixels are non-finite. Try smaller --g-lr and --d-lr values.");
            if (ascii)
            {
                const double intensity = std::clamp((upper + lower) * 0.5, 0.0, 1.0);
                out << ramp[static_cast<std::size_t>(intensity * (ramp.size() - 1))];
            }
            else
            {
                const int top = static_cast<int>(255 * std::clamp(upper, 0.0, 1.0));
                const int bottom = static_cast<int>(255 * std::clamp(lower, 0.0, 1.0));
                // U+2580: foreground is the upper pixel, background the lower.
                out << "\x1b[38;2;" << top << ';' << top << ';' << top << 'm'
                    << "\x1b[48;2;" << bottom << ';' << bottom << ';' << bottom << "m\xE2\x96\x80";
            }
        }
        if (!ascii) out << "\x1b[0m";
        out << "|\n";
    }
    out << "  +" << std::string(imageWidth, '-') << "+\n";
    return out.str();
}

void draw_preview(const Terminal& terminal, const Options& options, conditional_gan& gan,
                  const MNIST& data, const std::array<std::size_t, classCount>& references,
                  std::uint64_t steps, std::uint64_t epoch, double stepsPerSecond,
                  bool paused, bool finished, std::pmr::memory_resource* resource)
{
    const auto generated = gan.generate(options.label);
    math_vector<double> real(pixelCount, 0, resource);
    data.copy_image(references[options.label], real);
    const double fakeScore = gan.discriminate(generated)[options.label];
    const double realScore = gan.discriminate(real)[options.label];
    if (!std::isfinite(fakeScore) || !std::isfinite(realScore))
        throw std::runtime_error("Discriminator outputs became non-finite. Lower the learning rates.");
    std::ostringstream out;
    const bool color = !options.ascii;
    if (color) out << "\x1b[1;36m";
    out << "  MNIST / CONDITIONAL GAN\n";
    if (color) out << "\x1b[0m";
    out << "  " << (finished ? "FINISHED" : paused ? "PAUSED" : "TRAINING")
        << "   Epoch " << std::min(epoch + 1, options.epochs) << '/' << options.epochs << '\n'
        << "  Step " << steps << "   " << std::fixed << std::setprecision(1)
        << stepsPerSecond << " steps/s\n"
        << image_panel(generated, "GENERATED " + std::to_string(options.label), options.ascii);
    if (terminal.rows() >= 42)
        out << '\n' << image_panel(real, "REAL MNIST " + std::to_string(options.label), options.ascii);
    out << std::setprecision(2)
        << "  D[" << options.label << "]  fake " << fakeScore << " | real " << realScore << '\n'
        << "  Digit " << options.label << " / " << (options.automatic ? "AUTO" : "SELECTED") << '\n'
        << "  [0-9] digit   [A] auto\n"
        << "  [P] pause     [Q] quit\n"
        << "  Fresh noise at every preview";

    // Clear each line, not the entire screen: avoids flashing. No trailing
    // newline, so a 24-row terminal does not scroll on the last line.
    std::istringstream lines(out.str());
    std::string line;
    std::string frame = "\x1b[H";
    bool first = true;
    while (std::getline(lines, line))
    {
        if (!first) frame += '\n';
        frame += line;
        frame += "\x1b[K";
        first = false;
    }
    frame += "\x1b[0m\x1b[J";
    std::cout << frame << std::flush;
}

void run(Options options, const MNIST& data)
{
    // Persistent weights and temporary activations share one resource in
    // the current GAN API. Use a pool backed by a monotonic arena: freed
    // buffers return to the pool and can be reused on the next step.
    // A bare monotonic resource would retain every temporary allocation.
    // NEVER release either resource while the GAN or its vectors are alive.
    std::pmr::monotonic_buffer_resource trainingArena{8 * 1024 * 1024};
    std::pmr::pool_options poolOptions;
    poolOptions.max_blocks_per_chunk = 16;
    poolOptions.largest_required_pool_block = 64 * 1024;
    std::pmr::unsynchronized_pool_resource trainingPool{poolOptions, &trainingArena};

    // Implicit copies of the math classes' PMR vectors select the default
    // resource. Redirect it too, for this single-threaded training session.
    // Declaration order ensures all such objects die before the pool/arena.
    DefaultResourceScope defaultResource{&trainingPool};

    conditional_gan gan(
        {noiseSize + classCount, 128, 256, pixelCount}, classCount,
        options.generatorLearningRate,
        {pixelCount, 128, 64, classCount}, options.discriminatorLearningRate,
        &trainingPool
    );
    math_vector<double> realImage(pixelCount, 0, &trainingPool);
    std::mt19937 rng{std::random_device{}()};
    std::vector<std::size_t> order(data.labels.size());
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), rng);
    std::array<std::size_t, classCount> references{};
    for (std::size_t label = 0; label < classCount; ++label)
    {
        std::uniform_int_distribution<std::size_t> pick(0, data.byLabel[label].size() - 1);
        references[label] = data.byLabel[label][pick(rng)];
    }

    std::uint64_t steps = 0, epoch = 0, rateSteps = 0;
    std::size_t cursor = 0;
    bool paused = false, finished = false, redraw = true;
    double stepsPerSecond = 0;
    auto lastFrame = Clock::now(), lastCycle = lastFrame, lastRate = lastFrame;
    {
        Terminal terminal{!options.noUI};
        if (!terminal.interactive())
            std::cout << "Training from scratch; use a Linux/macOS terminal for the live display.\n";
        while (!stopRequested)
        {
            for (int key = terminal.key(); key != -1; key = terminal.key())
            {
                if (key == 'q' || key == 'Q') stopRequested = 1;
                else if (key == 'p' || key == 'P') paused = !paused;
                else if (key == 'a' || key == 'A')
                {
                    options.automatic = !options.automatic;
                    lastCycle = Clock::now();
                }
                else if (key >= '0' && key <= '9')
                {
                    options.label = static_cast<std::size_t>(key - '0');
                    options.automatic = false;
                }
                redraw = true;
            }
            if (stopRequested) break;
            if (!paused && !finished)
            {
                const std::size_t index = order[cursor];
                const std::size_t label = data.labels[index];
                data.copy_image(index, realImage); // Normalize bytes to [0, 1].
                // This method already trains D on BOTH a real and a fake image.
                gan.train_discriminator(realImage, label);
                gan.train_generator(label);
                ++steps;
                if (++cursor == order.size())
                {
                    cursor = 0;
                    ++epoch;
                    std::shuffle(order.begin(), order.end(), rng);
                }
                finished = epoch >= options.epochs || (options.maxSteps != 0 && steps >= options.maxSteps);
                if (finished) redraw = true;
            }

            const auto now = Clock::now();
            const double rateInterval = std::chrono::duration<double>(now - lastRate).count();
            if (rateInterval >= 1.0)
            {
                stepsPerSecond = static_cast<double>(steps - rateSteps) / rateInterval;
                rateSteps = steps;
                lastRate = now;
            }
            if (options.automatic && now - lastCycle >= std::chrono::seconds{4})
            {
                options.label = (options.label + 1) % classCount;
                lastCycle = now;
                redraw = true;
            }
            const auto interval = terminal.interactive() ? std::chrono::milliseconds{250}
                                                        : std::chrono::milliseconds{2000};
            if (redraw || now - lastFrame >= interval)
            {
                if (terminal.interactive())
                    draw_preview(terminal, options, gan, data, references, steps, epoch,
                                 stepsPerSecond, paused, finished, &trainingPool);
                else
                    std::cout << "Step " << steps << " | epochs completed " << epoch
                              << " | " << std::fixed << std::setprecision(1) << stepsPerSecond
                              << " steps/s\n" << std::flush;
                lastFrame = now;
                redraw = false;
            }
            if (finished && !terminal.interactive()) break;
            if (paused || finished) std::this_thread::sleep_for(std::chrono::milliseconds{20});
        }
    } // Restore the terminal before printing the final summary.
    std::cout << "\n" << (finished ? "Training limit reached" : "Stopped") << " after " << steps
              << " training steps.\n"
              << image_panel(gan.generate(options.label), "GENERATED " + std::to_string(options.label), true);
}
} // namespace

int main(int argc, char** argv)
{
    try
    {
        const Options options = parse_options(argc, argv);
        if (options.help) { print_help(); return 0; }
        std::signal(SIGINT, handle_signal);
        std::signal(SIGTERM, handle_signal);
        std::cout << "Loading MNIST from " << options.directory << "...\n";
        const MNIST data = load_mnist(options.directory);
        std::cout << "Loaded " << data.labels.size() << " images (28x28).\n";
        run(options, data);
    }
    catch (const std::exception& error)
    {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}