import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import struct
import time

class BasePerceptron:
    """Base perceptron class with common fit and predict methods"""

    def __init__(self, learning_rate=0.01, max_epochs=1000):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.weights = None
        self.bias = None

    def convert_to_precision(self, x):
        """Convert input to specific precision format - to be overridden"""
        raise NotImplementedError("Subclasses must implement convert_to_precision")

    def convert_to_float(self, x):
        """Convert from precision format to float for operations - to be overridden"""
        raise NotImplementedError("Subclasses must implement convert_to_float")

    def fit(self, X, y):
        """Train the perceptron with specific precision"""
        n_samples, n_features = X.shape

        # Initialize weights and bias in specific precision
        self.weights = self.convert_to_precision(np.random.randn(n_features))
        self.bias = self.convert_to_precision(0.0)

        for _ in range(self.max_epochs):
            for i in range(n_samples):
                # Convert inputs to specific precision
                x_i = self.convert_to_precision(X[i])
                y_i = y[i]

                # Forward pass with specific precision
                z = self.convert_to_float(self.bias)
                for j in range(n_features):
                    z += self.convert_to_float(self.weights[j] * x_i[j])

                # Apply threshold function
                prediction = 1 if z >= 0 else 0

                # Update weights if prediction is wrong
                if prediction != y_i:
                    error = y_i - prediction

                    # Update bias
                    self.bias = self.convert_to_precision(
                        self.convert_to_float(self.bias) +
                        self.learning_rate * error
                    )

                    # Update weights
                    for j in range(n_features):
                        self.weights[j] = self.convert_to_precision(
                            self.convert_to_float(self.weights[j]) +
                            self.learning_rate * error * self.convert_to_float(x_i[j])
                        )

    def predict(self, X):
        """Make predictions using specific precision"""
        predictions = []
        for i in range(X.shape[0]):
            x_i = self.convert_to_precision(X[i])
            z = self.convert_to_float(self.bias)
            for j in range(len(self.weights)):
                z += self.convert_to_float(self.weights[j] * x_i[j])
            predictions.append(1 if z >= 0 else 0)
        return np.array(predictions)

class Binary16Perceptron(BasePerceptron):
    """IEEE 754 binary16 format perceptron implementation"""

    def convert_to_precision(self, x):
        """Convert input to binary16 format"""
        return np.float16(x)

    def convert_to_float(self, x):
        """Convert binary16 back to float32 for operations"""
        return np.float32(x)

class Binary32Perceptron(BasePerceptron):
    """IEEE 754 binary32 format perceptron implementation"""

    def convert_to_precision(self, x):
        """Convert input to binary32 format"""
        return np.float32(x)

    def convert_to_float(self, x):
        """Convert binary32 back to float64 for operations"""
        return np.float64(x)

class Binary64Perceptron(BasePerceptron):
    """IEEE 754 binary64 format perceptron implementation"""

    def convert_to_precision(self, x):
        """Convert input to binary64 format"""
        return np.float64(x)

    def convert_to_float(self, x):
        """Binary64 is already the highest precision, no conversion needed"""
        return x  # Direct return since binary64 is already the highest precision

# Mixin classes for different optimization strategies
class QuantizationMixin:
    """Mixin for input quantization optimization"""

    def _quantize_input(self, x, bits=8):
        """Quantize input to reduce precision"""
        # Scale to [0, 2^bits - 1]
        x_scaled = (x - x.min()) / (x.max() - x.min()) * (2**bits - 1)
        # Round and scale back
        x_quantized = np.round(x_scaled) / (2**bits - 1) * (x.max() - x.min()) + x.min()
        return x_quantized

class LookupTableMixin:
    """Mixin for lookup table optimization"""

    def _create_activation_lut(self):
        """Create lookup table for activation function"""
        # Pre-compute common threshold values
        lut = {}
        for i in range(-100, 101):
            lut[i/10.0] = 1 if i >= 0 else 0
        return lut

    def _lookup_activation(self, z):
        """Use lookup table for activation function"""
        z_rounded = round(z, 1)
        return self.activation_lut.get(z_rounded, 1 if z >= 0 else 0)

class BitShiftMixin:
    """Mixin for bit-shift multiplication optimization"""

    def _bit_shift_multiply(self, a, b, shift_bits=8):
        """Approximate multiplication using bit shifts"""
        # Convert to fixed point representation
        a_fixed = int(a * (2**shift_bits))
        b_fixed = int(b * (2**shift_bits))
        result = (a_fixed * b_fixed) >> shift_bits
        return result / (2**shift_bits)

class OptimizedPerceptronBase:
    """Base class for optimized perceptrons with common functionality"""

    def __init__(self, learning_rate=0.01, max_epochs=1000):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.weights = None
        self.bias = None

    def _initialize_weights(self, n_features):
        """Initialize weights and bias"""
        self.weights = np.random.randn(n_features) * 0.1
        self.bias = 0.0

    def _forward_pass(self, x_i, n_features):
        """Standard forward pass - can be overridden by mixins"""
        z = self.bias
        for j in range(n_features):
            z += self.weights[j] * x_i[j]
        return z

    def _activation_function(self, z):
        """Standard activation function - can be overridden by mixins"""
        return 1 if z >= 0 else 0

    def _update_weights(self, x_i, y_i, prediction, n_features):
        """Update weights if prediction is wrong"""
        if prediction != y_i:
            error = y_i - prediction
            self.bias += self.learning_rate * error
            for j in range(n_features):
                self.weights[j] += self.learning_rate * error * x_i[j]

    def fit(self, X, y):
        """Training with optimizations"""
        n_samples, n_features = X.shape

        # Preprocess inputs (can be overridden by mixins)
        X_processed = self._preprocess_inputs(X)

        # Initialize weights
        self._initialize_weights(n_features)

        for _ in range(self.max_epochs):
            for i in range(n_samples):
                x_i = X_processed[i]
                y_i = y[i]

                # Forward pass
                z = self._forward_pass(x_i, n_features)

                # Activation function
                prediction = self._activation_function(z)

                # Update weights
                self._update_weights(x_i, y_i, prediction, n_features)

    def predict(self, X):
        """Prediction with optimizations"""
        X_processed = self._preprocess_inputs(X)
        predictions = []

        for i in range(X.shape[0]):
            x_i = X_processed[i]
            z = self._forward_pass(x_i, len(self.weights))
            prediction = self._activation_function(z)
            predictions.append(prediction)

        return np.array(predictions)

    def _preprocess_inputs(self, X):
        """Preprocess inputs - can be overridden by mixins"""
        return X

class QuantizedPerceptron(QuantizationMixin, OptimizedPerceptronBase):
    """Perceptron with only input quantization optimization"""

    def _preprocess_inputs(self, X):
        """Quantize inputs"""
        return self._quantize_input(X)

class LookupTablePerceptron(LookupTableMixin, OptimizedPerceptronBase):
    """Perceptron with only lookup table optimization"""

    def __init__(self, learning_rate=0.01, max_epochs=1000):
        super().__init__(learning_rate, max_epochs)
        self.activation_lut = self._create_activation_lut()

    def _activation_function(self, z):
        """Use lookup table for activation"""
        return self._lookup_activation(z)

class BitShiftPerceptron(BitShiftMixin, OptimizedPerceptronBase):
    """Perceptron with only bit-shift multiplication optimization"""

    def _forward_pass(self, x_i, n_features):
        """Use bit-shift multiplication for forward pass"""
        z = self.bias
        for j in range(n_features):
            z += self._bit_shift_multiply(self.weights[j], x_i[j])
        return z

class OptimizedPerceptron(QuantizationMixin, LookupTableMixin, BitShiftMixin, OptimizedPerceptronBase):
    """Perceptron with all three optimization methods combined"""

    def __init__(self, learning_rate=0.01, max_epochs=1000):
        super().__init__(learning_rate, max_epochs)
        self.activation_lut = self._create_activation_lut()

    def _preprocess_inputs(self, X):
        """Quantize inputs"""
        return self._quantize_input(X)

    def _forward_pass(self, x_i, n_features):
        """Use bit-shift multiplication for forward pass"""
        z = self.bias
        for j in range(n_features):
            z += self._bit_shift_multiply(self.weights[j], x_i[j])
        return z

    def _activation_function(self, z):
        """Use lookup table for activation"""
        return self._lookup_activation(z)

class UltraOptimizedPerceptron:
    """Ultra-optimized inference model with 1% accuracy tolerance - Best method for c)"""

    def __init__(self, accuracy_tolerance=0.01):
        self.weights = None
        self.bias = None
        self.feature_masks = None
        self.accuracy_tolerance = accuracy_tolerance
        self.baseline_accuracy = None  # Store baseline accuracy for comparison
        self.optimization_method = "lookup_table"  # Default to fastest method

    def _select_optimal_features(self, X, y, max_features=3):
        """Select optimal features using multiple criteria with proper normalization"""
        n_features = X.shape[1]

        # Method 1: Variance-based selection
        variances = np.var(X, axis=0)

        # Method 2: Correlation with target
        correlations = []
        for i in range(n_features):
            corr = np.corrcoef(X[:, i], y)[0, 1]
            correlations.append(abs(corr) if not np.isnan(corr) else 0)

        # Method 3: Mutual information with discretization (more robust)
        mutual_info = []
        for i in range(n_features):
            # Discretize continuous feature into 10 bins
            try:
                bins = np.percentile(X[:, i], np.linspace(0, 100, 11))
                z = np.clip(np.digitize(X[:, i], bins[1:-1], right=True), 0, 9)

                # Calculate mutual information on discretized data
                mi = 0
                for bin_val in range(10):
                    mask = z == bin_val
                    if np.sum(mask) > 0:
                        p_val = np.sum(mask) / len(mask)
                        p_class_given_val = np.sum(y[mask]) / np.sum(mask)
                        if 0 < p_class_given_val < 1:
                            mi += p_val * (-p_class_given_val * np.log2(p_class_given_val) -
                                         (1-p_class_given_val) * np.log2(1-p_class_given_val))
                mutual_info.append(mi)
            except:
                # Fallback to simple correlation if discretization fails
                mutual_info.append(abs(correlations[i]))

        # Normalize scores to avoid scale mismatch
        def _normalize_scores(scores):
            scores = np.asarray(scores, dtype=float)
            std = np.std(scores)
            return (scores - np.mean(scores)) / (std + 1e-9) if std > 0 else scores

        # Combine normalized scores
        norm_var = _normalize_scores(variances)
        norm_corr = _normalize_scores(correlations)
        norm_mi = _normalize_scores(mutual_info)

        scores = (norm_var + norm_corr + norm_mi) / 3
        self.feature_masks = np.argsort(scores)[-max_features:]

        return X[:, self.feature_masks]

    def _create_optimized_rules(self, X, y):
        """Create optimized decision rules with multiple thresholds (weights only used in multi-feature combination)"""
        thresholds = []
        weights = []

        for i in range(X.shape[1]):
            # Find best threshold for this feature (no weight testing for single feature)
            unique_vals = np.unique(X[:, i])
            best_thresh = unique_vals[0]
            best_acc = 0

            # Test different thresholds (sample for efficiency)
            for thresh in unique_vals[::max(1, len(unique_vals)//10)]:  # Sample thresholds
                pred = (X[:, i] >= thresh).astype(int)
                acc = np.mean(pred == y)

                if acc > best_acc:
                    best_acc = acc
                    best_thresh = thresh

            thresholds.append(best_thresh)
            weights.append(1.0)  # Default weight, will be used in multi-feature combination

        return thresholds, weights

    def _create_ultra_fast_lookup_table(self, X, y):
        """Create ultra-fast lookup table with maximum optimization"""
        # Use smaller bins for maximum speed (8x8 = 64 entries)
        n_bins = 8
        lut = np.zeros((n_bins, n_bins), dtype=np.uint8)  # Use uint8 for memory efficiency
        vote_counts = np.zeros((n_bins, n_bins, 2), dtype=np.uint32)  # Use uint32 to avoid overflow

        # Pre-compute quantization parameters for speed with zero-division protection
        self.quantization_params = []
        for i in range(X.shape[1]):
            min_val, max_val = X[:, i].min(), X[:, i].max()
            rng = max_val - min_val
            scale = rng / (n_bins - 1) if rng > 0 else 1.0  # Protection against zero division
            self.quantization_params.append((min_val, max_val, scale))

        # Quantize features with optimized calculation
        X_quantized = np.zeros_like(X, dtype=np.uint8)
        for i in range(X.shape[1]):
            min_val, max_val, scale = self.quantization_params[i]
            X_quantized[:, i] = np.clip(np.floor((X[:, i] - min_val) / scale), 0, n_bins - 1).astype(np.uint8)

        # Count votes for each bin (vectorized)
        for i in range(len(X)):
            idx1 = X_quantized[i, 0]
            idx2 = X_quantized[i, 1] if X.shape[1] > 1 else 0
            vote_counts[idx1, idx2, y[i]] += 1

        # Fill lookup table with majority vote (vectorized)
        lut = (vote_counts[:, :, 1] > vote_counts[:, :, 0]).astype(np.uint8)

        return lut

    def _create_bitwise_rules(self, X, y):
        """Create bitwise decision rules for maximum speed"""
        # Find the most discriminative single feature
        best_feature = 0
        best_accuracy = 0
        best_threshold = 0

        for i in range(X.shape[1]):
            unique_vals = np.unique(X[:, i])
            for thresh in unique_vals[::max(1, len(unique_vals)//5)]:  # Sample fewer thresholds
                pred = (X[:, i] >= thresh).astype(int)
                acc = np.mean(pred == y)
                if acc > best_accuracy:
                    best_accuracy = acc
                    best_feature = i
                    best_threshold = thresh

        return best_feature, best_threshold, best_accuracy

    def _create_single_bit_rule(self, X, y):
        """Create single-bit decision rule for maximum speed"""
        # Find the single most discriminative threshold
        best_accuracy = 0
        best_rule = None

        for i in range(X.shape[1]):
            unique_vals = np.unique(X[:, i])
            for thresh in unique_vals:
                pred = (X[:, i] >= thresh).astype(int)
                acc = np.mean(pred == y)
                if acc > best_accuracy:
                    best_accuracy = acc
                    best_rule = (i, thresh)

        return best_rule, best_accuracy

    def fit(self, X, y):
        """Train ultra-optimized model with maximum speed optimization"""
        # Step 1: Calculate real baseline accuracy
        maj = int(np.mean(y) >= 0.5)  # Majority class
        self.baseline_accuracy = max(np.mean(y==0), np.mean(y==1))
        min_acceptable_accuracy = self.baseline_accuracy - self.accuracy_tolerance

        # Step 2: Try single-bit rule first (fastest possible)
        single_rule, single_acc = self._create_single_bit_rule(X, y)

        # Step 3: Try bitwise rules (second fastest)
        bitwise_feature, bitwise_thresh, bitwise_acc = self._create_bitwise_rules(X, y)

        # Step 4: Try ultra-fast lookup table
        X_2d = self._select_optimal_features(X, y, 2)  # Use only 2 features for lookup table
        self.lookup_table = self._create_ultra_fast_lookup_table(X_2d, y)
        self.feature_masks = self.feature_masks  # Set by _select_optimal_features

        # Test lookup table accuracy
        lut_pred = self.predict(X)
        lut_acc = np.mean(lut_pred == y)

        # Step 5: Choose the fastest method that meets accuracy requirements
        if single_acc >= min_acceptable_accuracy:
            self.optimization_method = "single_bit"
            self.single_rule = single_rule
            self.feature_masks = [single_rule[0]]
            print(f"Ultra-Optimized: Single-bit rule (feature {single_rule[0]}, threshold {single_rule[1]:.4f})")
        elif bitwise_acc >= min_acceptable_accuracy:
            self.optimization_method = "bitwise"
            self.bitwise_feature = bitwise_feature
            self.bitwise_threshold = bitwise_thresh
            self.feature_masks = [bitwise_feature]
            print(f"Ultra-Optimized: Bitwise rule (feature {bitwise_feature}, threshold {bitwise_thresh:.4f})")
        elif lut_acc >= min_acceptable_accuracy:
            self.optimization_method = "lookup_table"
            print(f"Ultra-Optimized: Lookup table (2 features, 64 entries)")
        else:
            # Fallback to more complex method
            self.optimization_method = "fallback"
            self.thresholds, self.feature_weights = self._create_optimized_rules(X, y)
            self.feature_masks = np.arange(X.shape[1])
            print(f"Ultra-Optimized: Fallback method ({X.shape[1]} features)")

    def predict(self, X):
        """Ultra-fast prediction using the selected optimization method"""
        if self.optimization_method == "single_bit":
            # Single-bit rule: fastest possible (1 comparison)
            feature_idx, threshold = self.single_rule
            return (X[:, feature_idx] >= threshold).astype(int)

        elif self.optimization_method == "bitwise":
            # Bitwise rule: very fast (1 comparison)
            return (X[:, self.bitwise_feature] >= self.bitwise_threshold).astype(int)

        elif self.optimization_method == "lookup_table":
            # Ultra-fast lookup table: O(1) lookup
            X_selected = X[:, self.feature_masks]
            predictions = []

            for i in range(X.shape[0]):
                x_i = X_selected[i]
                # Optimized quantization
                idx1 = int((x_i[0] - self.quantization_params[0][0]) / self.quantization_params[0][2])
                idx1 = np.clip(idx1, 0, 7)

                if X_selected.shape[1] > 1:
                    idx2 = int((x_i[1] - self.quantization_params[1][0]) / self.quantization_params[1][2])
                    idx2 = np.clip(idx2, 0, 7)
                    prediction = self.lookup_table[idx1, idx2]
                else:
                    prediction = self.lookup_table[idx1, 0]

                predictions.append(prediction)

            return np.array(predictions)

        else:  # fallback
            # Fallback method
            X_selected = X[:, self.feature_masks]
            predictions = []

            for i in range(X.shape[0]):
                x_i = X_selected[i]
                votes = []
                for j, (thresh, weight) in enumerate(zip(self.thresholds, self.feature_weights)):
                    vote = 1 if x_i[j] >= thresh else 0
                    votes.append(vote * weight)

                prediction = 1 if np.sum(votes) >= np.sum(self.feature_weights) / 2 else 0
                predictions.append(prediction)

            return np.array(predictions)

    def _estimate_memory_usage(self):
        """Estimate memory usage of the optimized model"""
        if self.optimization_method == "single_bit":
            return "1 threshold (4 bytes)"
        elif self.optimization_method == "bitwise":
            return "1 threshold (4 bytes)"
        elif self.optimization_method == "lookup_table":
            lut_bytes = self.lookup_table.size  # uint8 entries
            param_bytes = len(self.quantization_params) * 3 * 8  # 3 floats per feature
            return f"LUT: {lut_bytes} bytes + Params: {param_bytes} bytes = {lut_bytes + param_bytes} bytes total"
        else:
            return f"{len(self.thresholds)} thresholds + {len(self.feature_weights)} weights"

    def get_optimization_info(self):
        """Get information about optimization level"""
        info = {
            'features_used': len(self.feature_masks) if self.feature_masks is not None else 0,
            'has_lookup_table': hasattr(self, 'lookup_table'),
            'lookup_table_size': self.lookup_table.size if hasattr(self, 'lookup_table') else 0,
            'thresholds_count': len(self.thresholds) if hasattr(self, 'thresholds') else 0,
            'accuracy_tolerance': self.accuracy_tolerance
        }
        return info

def load_and_preprocess_data():
    """Load and preprocess Iris dataset"""
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Filter only Setosa (0) and Versicolor (1)
    mask = y < 2
    X = X[mask]
    y = y[mask]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model performance"""
    # Use time.perf_counter() for higher precision
    start_time = time.perf_counter()
    predictions = model.predict(X_test)
    inference_time = time.perf_counter() - start_time

    accuracy = accuracy_score(y_test, predictions)

    # Convert to nanoseconds for better precision display
    time_ns = inference_time * 1_000_000_000
    print(f"{model_name}: Accuracy={accuracy:.4f}, Time={time_ns:.0f}ns")

    return accuracy, inference_time

def main():
    """Main function to run all experiments"""
    print("=== EE5410 Computer Arithmetic - HW1 ===")
    print("Perceptron Classification on Iris Dataset\n")

    # Load data
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data()
    print(f"Dataset: {X_train.shape[0]} training, {X_test.shape[0]} test samples\n")

    # Question (a): IEEE 754 different precision formats comparison
    print("Question (a): IEEE 754 Precision Formats")
    print("-" * 50)

    # Binary16 (Half Precision)
    binary16_model = Binary16Perceptron(learning_rate=0.01, max_epochs=1000)
    binary16_model.fit(X_train, y_train)
    binary16_acc, binary16_time = evaluate_model(binary16_model, X_test, y_test, "Binary16")

    # Binary32 (Single Precision)
    binary32_model = Binary32Perceptron(learning_rate=0.01, max_epochs=1000)
    binary32_model.fit(X_train, y_train)
    binary32_acc, binary32_time = evaluate_model(binary32_model, X_test, y_test, "Binary32")

    # Binary64 (Double Precision)
    binary64_model = Binary64Perceptron(learning_rate=0.01, max_epochs=1000)
    binary64_model.fit(X_train, y_train)
    binary64_acc, binary64_time = evaluate_model(binary64_model, X_test, y_test, "Binary64")

    # Question (b): Individual optimization methods
    print("\nQuestion (b): Individual Optimization Methods")
    print("-" * 50)

    # Quantized Perceptron (only quantization)
    quantized_model = QuantizedPerceptron(learning_rate=0.01, max_epochs=1000)
    quantized_model.fit(X_train, y_train)
    quantized_acc, quantized_time = evaluate_model(quantized_model, X_test, y_test, "Quantized")

    # Lookup Table Perceptron (only lookup table)
    lookup_model = LookupTablePerceptron(learning_rate=0.01, max_epochs=1000)
    lookup_model.fit(X_train, y_train)
    lookup_acc, lookup_time = evaluate_model(lookup_model, X_test, y_test, "Lookup Table")

    # Bit-shift Perceptron (only bit-shift multiplication)
    bitshift_model = BitShiftPerceptron(learning_rate=0.01, max_epochs=1000)
    bitshift_model.fit(X_train, y_train)
    bitshift_acc, bitshift_time = evaluate_model(bitshift_model, X_test, y_test, "Bit-shift")

    # Combined Optimized Perceptron (all three methods)
    optimized_model = OptimizedPerceptron(learning_rate=0.01, max_epochs=1000)
    optimized_model.fit(X_train, y_train)
    optimized_acc, optimized_time = evaluate_model(optimized_model, X_test, y_test, "Combined")

    # Question (c): Ultra-optimized inference model
    print("\nQuestion (c): Ultra-optimized inference model")
    print("-" * 50)

    ultra_model = UltraOptimizedPerceptron(accuracy_tolerance=0.01)
    ultra_model.fit(X_train, y_train)
    ultra_acc, ultra_time = evaluate_model(ultra_model, X_test, y_test, "Ultra-Optimized")

    # Summary comparison
    print("\n" + "="*60)
    print("Performance Summary")
    print("="*60)

    all_models = ["Binary16", "Binary32", "Binary64", "Quantized", "Lookup Table", "Bit-shift", "Combined", "Ultra-Optimized"]
    all_accuracies = [binary16_acc, binary32_acc, binary64_acc, quantized_acc, lookup_acc, bitshift_acc, optimized_acc, ultra_acc]
    all_times = [binary16_time, binary32_time, binary64_time, quantized_time, lookup_time, bitshift_time, optimized_time, ultra_time]

    print(f"{'Model':<15} {'Accuracy':<10} {'Time (ns)':<12} {'Category':<12}")
    print("-" * 60)

    categories = ["Precision", "Precision", "Precision", "Optimization", "Optimization", "Optimization", "Optimization", "Ultra-Opt"]

    for i, (model, acc, time_taken, category) in enumerate(zip(all_models, all_accuracies, all_times, categories)):
        time_ns = time_taken * 1_000_000_000
        print(f"{model:<15} {acc:<10.4f} {time_ns:<12.0f} {category:<12}")

    # Best performance summary
    print(f"\nBest Performance:")
    best_accuracy_idx = np.argmax(all_accuracies)
    fastest_idx = np.argmin(all_times)

    # Check if all accuracies are the same
    if len(set(all_accuracies)) == 1:
        print(f"Highest Accuracy: All the same ({all_accuracies[best_accuracy_idx]:.4f})")
    else:
        print(f"Highest Accuracy: {all_models[best_accuracy_idx]} ({all_accuracies[best_accuracy_idx]:.4f})")

    fastest_time_ns = all_times[fastest_idx] * 1_000_000_000
    print(f"Fastest: {all_models[fastest_idx]} ({fastest_time_ns:.0f}ns)")

if __name__ == "__main__":
    main()
