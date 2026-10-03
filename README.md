# Digits GAN

A conditional generative adversarial network implemented from scratch in C++26 that learns to generate **28×28 MNIST handwritten digits (0–9)**. Watch the generator's output change during training, choose a digit, and compare it with a real MNIST example directly in the terminal.

The project uses custom `math_vector`, `matrix`, `neural_network`, and `conditional_gan` classes, with manual backpropagation and CPU training. No external ML libraries or GPU are required.

## Quick start

Use a Linux terminal with Git and GCC 15 or newer. The interactive display also supports macOS with a suitable compiler. A UTF-8 terminal with true-color support gives the best display; use `--ascii` for plain characters.

```bash
git clone https://github.com/cartercpp/digits-gan.git
cd digits-gan

g++ -std=c++26 -O3 -march=native '-Dpre(...)=' \
    main.cpp conditional_gan.cpp neural_network.cpp -o main &&
./main ./mnist
```

The repository includes the two uncompressed MNIST training files in `mnist/`, so a normal clone is ready to run. Run these commands from the repository root. The compiler command builds an executable for your machine.

**GCC 15 compatibility:** `'-Dpre(...)='` removes the constructor precondition annotations for this build. C++26 mode stays enabled, but those contract checks are disabled. Keep the quotes around the macro definition. Selecting `-std=c++26` alone does not enable every C++26 feature.

### Build with C++26 contracts

[GCC lists C++26 contracts as supported starting with GCC 16](https://gcc.gnu.org/projects/cxx-status.html#cxx26). If your installed compiler supports them, use [`-fcontracts`](https://gcc.gnu.org/onlinedocs/gcc/C_002b_002b-Dialect-Options.html) instead of the compatibility macro:

```bash
g++-16 -std=c++26 -fcontracts -O3 -march=native \
    main.cpp conditional_gan.cpp neural_network.cpp -o main &&
./main ./mnist
```

Use `g++` in place of `g++-16` if that is how your GCC 16+ compiler is installed. Check the compiler you are invoking with `g++ --version` or `g++-16 --version`.

### Optional CMake build

The supplied `CMakeLists.txt` requires **CMake 4.3 or newer** and selects C++26. With GCC 16+ installed as `g++-16`:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=g++-16 \
    -DCMAKE_CXX_FLAGS=-fcontracts
cmake --build build --parallel
./build/DigitsGAN ./mnist
```

The direct GCC commands above do not require CMake.

## MNIST data

The program reads these **uncompressed IDX files**:

- `mnist/train-images-idx3-ubyte`
- `mnist/train-labels-idx1-ubyte`

They contain 60,000 training images and their digit labels. Pixels are normalized from bytes to values in `[0, 1]` during training.

If the files are missing, download them with `curl` and extract them with `gzip`:

```bash
mkdir -p mnist
for name in train-images-idx3-ubyte train-labels-idx1-ubyte; do
    if [ ! -f "mnist/$name" ]; then
        curl -fL "https://ossci-datasets.s3.amazonaws.com/mnist/$name.gz" \
            -o "mnist/$name.gz" && gzip -dk "mnist/$name.gz"
    fi
done
```

This is the MNIST mirror used by [torchvision's dataset loader](https://github.com/pytorch/vision/blob/main/torchvision/datasets/mnist.py); torchvision itself is not needed. To use another dataset directory, pass its path as the first argument.

## Run and interact

```bash
# Live training with the default settings
./main ./mnist

# Train for up to 30 epochs and preview digit 7
./main ./mnist --epochs 30 --label 7

# Short run without the interactive display
./main ./mnist --steps 100 --no-ui

# Plain-character display
./main ./mnist --ascii

# Show all options
./main --help
```

| Key | Action |
| --- | --- |
| `0`–`9` | Select the digit to generate and turn off automatic cycling |
| `A` | Toggle automatic cycling through digits |
| `P` | Pause or resume training |
| `Q` or `Ctrl+C` | Quit |

The **GENERATED** panel displays pixels returned by `conditional_gan::generate()`. At **42 terminal rows or more**, a **REAL MNIST** panel appears below it for comparison. The color display represents all 28×28 pixels using 28 columns and 14 character rows. ASCII mode averages pairs of pixel rows.

The displayed `D[label]` values are the discriminator's scores for the selected digit, not a measured accuracy. Each preview uses fresh random noise, so samples can still change while training is paused or finished.

After the training limit is reached, the interactive preview stays open until you quit. With `--no-ui`, the program prints progress, outputs a final ASCII sample, and exits. Non-interactive input/output also uses this mode.

### Options

| Option | Default | Meaning |
| --- | --- | --- |
| `[mnist-directory]` | `mnist` | Directory containing the uncompressed training files |
| `--epochs N` | `20` | Maximum passes through the training dataset; must be at least 1 |
| `--steps N` | `0` | Additional training-step limit; 0 disables this limit |
| `--label N` | Starts at `0`, cycles automatically | Preview digit 0–9 and disable automatic cycling |
| `--g-lr X` | `0.01` | Generator learning rate; must be positive |
| `--d-lr X` | `0.002` | Discriminator learning rate; must be positive |
| `--ascii` | Off | Use plain-character images |
| `--no-ui` | Off | Use progress output instead of the live display |
| `--help`, `-h` | Off | Print usage information |

Training stops when either the epoch limit or a nonzero step limit is reached. `--label` and the digit keys change the preview; training continues to use all digit classes.

## How it works

| Network | Layers | Role |
| --- | --- | --- |
| Generator | `74 → 128 → 256 → 784` | Turns 64 noise values plus a 10-value one-hot digit label into 784 pixels |
| Discriminator | `784 → 128 → 64 → 10` | Produces one sigmoid score per digit class |

Both networks use ReLU hidden layers and sigmoid outputs. For each shuffled training example, `main.cpp` calls:

1. `train_discriminator(realImage, label)`: update the discriminator on a real image with a one-hot target, then on a generated image with an all-zero target.
2. `train_generator(label)`: backpropagate through the discriminator to update the generator toward the requested digit's one-hot target. The discriminator's weights stay fixed during this update.

The implementation uses sigmoid binary-cross-entropy gradients and stochastic gradient descent. Every run initializes a new model; saving/loading weights is not implemented. Early samples are noisy, and the quality depends on training time and learning rates.

### Training memory

The math and neural-network classes use `std::pmr` containers. `main.cpp` passes an `std::pmr::unsynchronized_pool_resource` backed by an `std::pmr::monotonic_buffer_resource` to the GAN. The pool reuses temporary allocations across training steps, while the arena stays alive for the entire session. Its initial allocation size is 8 MiB and it can grow.

The default PMR resource is also temporarily redirected to the pool so implicit copies use the same resource. The resources are destroyed after the model and its vectors.

## Source files

| File | Purpose |
| --- | --- |
| `main.cpp` | MNIST loading, training loop, command-line options, and terminal visualization |
| `conditional_gan.hpp` / `.cpp` | Conditional generation and adversarial training |
| `neural_network.hpp` / `.cpp` | Feedforward networks, backpropagation, and weight updates |
| `matrix.hpp` | Matrix operations and outer products |
| `math_vector.hpp` | Vector operations and dot products |

## Troubleshooting

- **Errors at `pre(...)`:** use the quoted compatibility macro in the quick start, or GCC 16+ with `-fcontracts`.
- **`./main: No such file or directory`:** fix any compilation errors first and run from the directory containing the executable.
- **Missing or invalid MNIST files:** extract both `.gz` archives and pass the directory containing the resulting IDX files.
- **No live display:** run in a regular Linux/macOS terminal with input and output attached to it. Use `--ascii` if your terminal does not render the color blocks correctly.
- **Non-finite outputs:** restart with smaller learning rates, for example `./main ./mnist --g-lr 0.001 --d-lr 0.0002`.

More projects: [ML from scratch](https://github.com/cartercpp/ml-from-scratch).
