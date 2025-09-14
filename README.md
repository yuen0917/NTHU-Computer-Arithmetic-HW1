# EE5410 Computer Arithmetic - Homework 1

## Assignment Description

Implement perceptron models for binary classification on the Iris dataset with various optimization strategies:

### Question (a): IEEE 754 Precision Formats

Compare performance across different floating-point precisions:

- **Binary16 (Half Precision)**: 16-bit half-precision floating-point
- **Binary32 (Single Precision)**: 32-bit single-precision floating-point
- **Binary64 (Double Precision)**: 64-bit double-precision floating-point

### Question (b): Individual Optimization Methods

Implement three independent optimization methods:

- **Quantized Perceptron**: Input quantization optimization (8-bit quantization)
- **Lookup Table Perceptron**: Lookup table optimization (pre-computed activation function)
- **Bit-shift Perceptron**: Bit-shift multiplication optimization
- **Combined Perceptron**: Integration of all three optimization methods

### Question (c): Ultra-optimized Inference Model

Maximize inference efficiency within 1% accuracy tolerance:

- **Feature Selection**: Multi-criteria based on variance, correlation, and mutual information
- **Decision Rules**: Single-feature thresholds, bitwise operations, lookup tables
- **Adaptive Optimization**: Automatically select the fastest method meeting accuracy requirements

## Program Architecture

- **BasePerceptron**: Base class for IEEE 754 precision handling
- **Mixin Classes**: Modular optimization strategies (QuantizationMixin, LookupTableMixin, BitShiftMixin)
- **OptimizedPerceptronBase**: Common functionality for optimized perceptrons
- **UltraOptimizedPerceptron**: Ultra-optimized inference model

## Usage

### Option 1: Using uv (Recommended)

You can use uv to manage dependencies and run the program:

```bash
# Install uv with Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Install uv with Linux and macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Run the program
uv run python HW1.py
```

### Option 2: Using Python directly

Alternatively, you can run the program directly with Python without uv:

```bash
# Install dependencies using pip
pip install numpy pandas scikit-learn

# Run the program
python HW1.py
```

## Output Results

The program displays:

- Accuracy and inference time for each model (nanosecond precision)
- Performance comparison table
- Best accuracy and fastest speed models
- "All the same" when all models have identical accuracy

## Dependencies

- numpy
- pandas
- scikit-learn
