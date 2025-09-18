# NTHU EE5410 Computer Arithmetic - Homework 1

## Assignment Description

Implement perceptron models for binary classification on the Iris dataset with various optimization strategies:

### Question (a): IEEE 754 Precision Formats

Compare performance across different floating-point precisions:

- **Binary16 (Half Precision)**: 16-bit half-precision floating-point
- **Binary32 (Single Precision)**: 32-bit single-precision floating-point
- **Binary64 (Double Precision)**: 64-bit double-precision floating-point

### Question (b): Training Optimization Methods

Vectorized and numerically optimized training variants (inference follows each class's implementation):

- **Vec-Update (VecUpdatePerceptron)**: Vectorized weight updates only (vectorized training; elementwise forward).
- **Vec-Quant (VecQuantPerceptron)**: Same training as above plus input quantization preprocessing (8-bit quantization).
- **Vec-Combo (VecComboPerceptron)**: Combines quantization, vectorized forward, and vectorized updates.
- **Vec-Full (VecFullPerceptron)**: Both training and inference use vectorized forward and updates.

### Question (c): Ultra-optimized Inference Model

Maximize inference efficiency within 1% accuracy tolerance (binary labels only, `y ∈ {0,1}`):

- **Feature Selection**: Combine three criteria, normalize, and take Top-k (k=2 for LUT):
  - Variance (higher is better)
  - Pearson correlation with `y` (absolute value; higher is better)
  - Conditional entropy H(Y|Z) (compute then negate so "lower is better" becomes "higher is better")
- **Decision Rules (auto-select the fastest that meets accuracy requirement):**
  - Single-bit rule (single threshold on one feature)
  - Bitwise rule (single feature with threshold)
  - 2D lookup table (8×8 quantized grid)
  - Fallback: multi-feature threshold voting if above fail to meet tolerance
- **Inference**: Only the vectorized inference path is kept.

## Program Architecture

- **BasePerceptron**: Base perceptron with IEEE 754 precision handling
- **EfficientTrainingMixin**: Vectorized forward and update utilities for training
- **OptimizedPerceptronBase**: Common training/inference with precision selection
- **VecUpdate/VecQuant/VecFull/VecCombo**: Optimized family for question (b)
- **UltraOptimizedPerceptron**: Ultra-optimized model for (c) with vectorized inference only

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

The program prints:

- Accuracy and inference time for each model (in microseconds, us); training time is shown in milliseconds (ms)
- A summary table (model, accuracy, training/inference times, category)
- The highest-accuracy model and the fastest training/inference models
- "All the same" when all models have identical accuracy

## Dependencies

- numpy
- pandas
- scikit-learn
