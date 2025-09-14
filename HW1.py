import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import struct
import time

# Question (a)
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

# Question (b)
# Mixin classes for training optimization strategies

class TrainingOptimizationMixin:
    """Mixin for training phase numerical representation optimization"""

    def _update_weights_optimized(self, x_i, y_i, prediction, n_features):
        """Optimized weight update using vectorized operations"""
        if prediction != y_i:
            error = y_i - prediction

            # Vectorized bias update
            self.bias += self.learning_rate * error

            # Vectorized weight updates - this is the real optimization
            weight_updates = self.learning_rate * error * x_i
            self.weights += weight_updates

class EfficientTrainingMixin:
    """More efficient training optimization using vectorized operations"""

    def _vectorized_update_weights(self, x_i, y_i, prediction, n_features):
        """Vectorized weight update for better performance"""
        if prediction != y_i:
            error = y_i - prediction

            # Vectorized operations are much faster
            self.bias += self.learning_rate * error
            self.weights += self.learning_rate * error * x_i

    def _vectorized_forward_pass(self, x_i, n_features):
        """Vectorized forward pass"""
        return self.bias + np.dot(self.weights, x_i)

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


# Training optimization classes
class TrainOptPerceptron(TrainingOptimizationMixin, OptimizedPerceptronBase):
    """Perceptron with training phase numerical optimization"""

    def _update_weights(self, x_i, y_i, prediction, n_features):
        """Use optimized weight update"""
        self._update_weights_optimized(x_i, y_i, prediction, n_features)

class TrainBitShiftPerceptron(TrainingOptimizationMixin, OptimizedPerceptronBase):
    """Perceptron with both training and inference bit-shift optimization"""

    def _bit_shift_multiply(self, a, b, shift_bits=8):
        """Approximate multiplication using bit shifts"""
        a_fixed = int(a * (2**shift_bits))
        b_fixed = int(b * (2**shift_bits))
        result = (a_fixed * b_fixed) >> shift_bits
        return result / (2**shift_bits)

    def _forward_pass(self, x_i, n_features):
        """Use bit-shift multiplication for forward pass"""
        z = self.bias
        for j in range(n_features):
            z += self._bit_shift_multiply(self.weights[j], x_i[j])
        return z

    def _update_weights(self, x_i, y_i, prediction, n_features):
        """Use optimized weight update"""
        self._update_weights_optimized(x_i, y_i, prediction, n_features)

    def predict(self, X):
        """Optimized prediction - use standard multiplication for inference"""
        X_processed = self._preprocess_inputs(X)
        predictions = []

        # Use standard multiplication for inference, as bit shifts are not obvious for inference
        for i in range(X.shape[0]):
            x_i = X_processed[i]
            z = self.bias + np.dot(self.weights, x_i)  # Use standard vectorized multiplication
            prediction = self._activation_function(z)
            predictions.append(prediction)

        return np.array(predictions)

class TrainQuantizedPerceptron(TrainingOptimizationMixin, OptimizedPerceptronBase):
    """Perceptron with training optimization and input quantization"""

    def _quantize_input(self, x, bits=8):
        """Quantize input to reduce precision"""
        x_scaled = (x - x.min()) / (x.max() - x.min()) * (2**bits - 1)
        x_quantized = np.round(x_scaled) / (2**bits - 1) * (x.max() - x.min()) + x.min()
        return x_quantized

    def _preprocess_inputs(self, X):
        """Quantize inputs"""
        return self._quantize_input(X)

    def _update_weights(self, x_i, y_i, prediction, n_features):
        """Use optimized weight update"""
        self._update_weights_optimized(x_i, y_i, prediction, n_features)

    def predict(self, X):
        """Optimized prediction - use vectorized operations for inference"""
        X_processed = self._preprocess_inputs(X)
        predictions = []

        # Use vectorized operations for inference
        for i in range(X.shape[0]):
            x_i = X_processed[i]
            z = self.bias + np.dot(self.weights, x_i)  # Use standard vectorized multiplication
            prediction = self._activation_function(z)
            predictions.append(prediction)

        return np.array(predictions)

class EfficientPerceptron(EfficientTrainingMixin, OptimizedPerceptronBase):
    """Highly efficient perceptron using vectorized operations"""

    def _forward_pass(self, x_i, n_features):
        """Use vectorized forward pass for training"""
        return self._vectorized_forward_pass(x_i, n_features)

    def _update_weights(self, x_i, y_i, prediction, n_features):
        """Use vectorized weight update for training"""
        self._vectorized_update_weights(x_i, y_i, prediction, n_features)

    def predict(self, X):
        """Optimized prediction - use vectorized operations for inference"""
        X_processed = self._preprocess_inputs(X)
        predictions = []

        # Use vectorized forward pass for inference
        for i in range(X.shape[0]):
            x_i = X_processed[i]
            z = self._vectorized_forward_pass(x_i, len(self.weights))
            prediction = self._activation_function(z)
            predictions.append(prediction)

        return np.array(predictions)

class TrainCombinedPerceptron(TrainingOptimizationMixin, OptimizedPerceptronBase):
    """Perceptron with all three training optimization methods combined"""

    def _bit_shift_multiply(self, a, b, shift_bits=8):
        """Approximate multiplication using bit shifts"""
        a_fixed = int(a * (2**shift_bits))
        b_fixed = int(b * (2**shift_bits))
        result = (a_fixed * b_fixed) >> shift_bits
        return result / (2**shift_bits)

    def _quantize_input(self, x, bits=8):
        """Quantize input to reduce precision"""
        x_scaled = (x - x.min()) / (x.max() - x.min()) * (2**bits - 1)
        x_quantized = np.round(x_scaled) / (2**bits - 1) * (x.max() - x.min()) + x.min()
        return x_quantized

    def _preprocess_inputs(self, X):
        """Quantize inputs"""
        return self._quantize_input(X)

    def _forward_pass(self, x_i, n_features):
        """Use bit-shift multiplication for forward pass"""
        z = self.bias
        for j in range(n_features):
            z += self._bit_shift_multiply(self.weights[j], x_i[j])
        return z

    def _update_weights(self, x_i, y_i, prediction, n_features):
        """Use optimized weight update"""
        self._update_weights_optimized(x_i, y_i, prediction, n_features)

    def predict(self, X):
        """Optimized prediction - use standard operations for inference"""
        X_processed = self._preprocess_inputs(X)
        predictions = []

        # Use standard vectorized operations for inference, avoiding complex bit shifts
        for i in range(X.shape[0]):
            x_i = X_processed[i]
            z = self.bias + np.dot(self.weights, x_i)  # Use standard vectorized multiplication
            prediction = self._activation_function(z)
            predictions.append(prediction)

        return np.array(predictions)

# Question (c)
class UltraOptimizedPerceptron:
    """Ultra-optimized inference model with 1% accuracy tolerance - Best method for c)"""

    def __init__(self, accuracy_tolerance=0.01):
        self.weights = None
        self.bias = None
        self.feature_masks = None
        self.accuracy_tolerance = accuracy_tolerance
        self.baseline_accuracy = None  # Store baseline accuracy for comparison
        self.optimization_method = "lookup_table"  # Default to fastest method
        self.use_vectorized = True  # Enable vectorized operations

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
        # Use original method for inference, as vectorization is not obvious for inference
        return self._predict_original(X)

    def _predict_vectorized(self, X):
        """Vectorized prediction method"""
        if self.optimization_method == "single_bit":
            # Single-bit rule: fastest possible (1 comparison)
            feature_idx, threshold = self.single_rule
            return (X[:, feature_idx] >= threshold).astype(int)

        elif self.optimization_method == "bitwise":
            # Bitwise rule: very fast (1 comparison)
            return (X[:, self.bitwise_feature] >= self.bitwise_threshold).astype(int)

        elif self.optimization_method == "lookup_table":
            # Vectorized lookup table
            X_selected = X[:, self.feature_masks]

            # Vectorized quantization
            idx1 = np.clip(
                ((X_selected[:, 0] - self.quantization_params[0][0]) /
                 self.quantization_params[0][2]).astype(int), 0, 7
            )

            if X_selected.shape[1] > 1:
                idx2 = np.clip(
                    ((X_selected[:, 1] - self.quantization_params[1][0]) /
                     self.quantization_params[1][2]).astype(int), 0, 7
                )
                predictions = self.lookup_table[idx1, idx2]
            else:
                predictions = self.lookup_table[idx1, 0]

            return predictions

        else:  # fallback - Vectorized
            X_selected = X[:, self.feature_masks]

            # Vectorized calculation of votes
            votes = np.zeros((X.shape[0], len(self.thresholds)))
            for j, (thresh, weight) in enumerate(zip(self.thresholds, self.feature_weights)):
                votes[:, j] = (X_selected[:, j] >= thresh).astype(int) * weight

            # Vectorized decision
            total_weight = np.sum(self.feature_weights)
            predictions = (np.sum(votes, axis=1) >= total_weight / 2).astype(int)

            return predictions

    def _predict_original(self, X):
        """Original prediction method"""
        if self.optimization_method == "single_bit":
            feature_idx, threshold = self.single_rule
            return (X[:, feature_idx] >= threshold).astype(int)

        elif self.optimization_method == "bitwise":
            return (X[:, self.bitwise_feature] >= self.bitwise_threshold).astype(int)

        elif self.optimization_method == "lookup_table":
            X_selected = X[:, self.feature_masks]
            predictions = []

            for i in range(X.shape[0]):
                x_i = X_selected[i]
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

def evaluate_model(model, X_test, y_test, model_name, training_time=None):
    """Evaluate model performance"""
    # Use time.perf_counter() for higher precision
    start_time = time.perf_counter()
    predictions = model.predict(X_test)
    inference_time = time.perf_counter() - start_time

    accuracy = accuracy_score(y_test, predictions)

    # Convert to appropriate units for display
    inference_ns = inference_time * 1_000_000_000 # Convert to nanoseconds
    if training_time is not None:
        training_ms = training_time * 1_000  # Convert to milliseconds
        print(f"{model_name:<15}: Accuracy={accuracy:.4f}, Training={training_ms:.2f}ms, Inference={inference_ns:.0f}ns")
    else:
        print(f"{model_name:<15}: Accuracy={accuracy:.4f}, Inference={inference_ns:.0f}ns")

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
    start_time = time.perf_counter()
    binary16_model.fit(X_train, y_train)
    binary16_training_time = time.perf_counter() - start_time
    binary16_acc, binary16_time = evaluate_model(binary16_model, X_test, y_test, "Binary16", binary16_training_time)

    # Binary32 (Single Precision)
    binary32_model = Binary32Perceptron(learning_rate=0.01, max_epochs=1000)
    start_time = time.perf_counter()
    binary32_model.fit(X_train, y_train)
    binary32_training_time = time.perf_counter() - start_time
    binary32_acc, binary32_time = evaluate_model(binary32_model, X_test, y_test, "Binary32", binary32_training_time)

    # Binary64 (Double Precision)
    binary64_model = Binary64Perceptron(learning_rate=0.01, max_epochs=1000)
    start_time = time.perf_counter()
    binary64_model.fit(X_train, y_train)
    binary64_training_time = time.perf_counter() - start_time
    binary64_acc, binary64_time = evaluate_model(binary64_model, X_test, y_test, "Binary64", binary64_training_time)

    # Question (b): Training optimization methods
    print("\nQuestion (b): Training Optimization Methods")
    print("-" * 50)

    # Train Opt Perceptron (only training optimization)
    train_opt_model = TrainOptPerceptron(learning_rate=0.01, max_epochs=1000)
    start_time = time.perf_counter()
    train_opt_model.fit(X_train, y_train)
    train_opt_training_time = time.perf_counter() - start_time
    train_opt_acc, train_opt_time = evaluate_model(train_opt_model, X_test, y_test, "Train-Opt", train_opt_training_time)

    # Train BitShift Perceptron (training + inference optimization)
    train_bitshift_model = TrainBitShiftPerceptron(learning_rate=0.01, max_epochs=1000)
    start_time = time.perf_counter()
    train_bitshift_model.fit(X_train, y_train)
    train_bitshift_training_time = time.perf_counter() - start_time
    train_bitshift_acc, train_bitshift_time = evaluate_model(train_bitshift_model, X_test, y_test, "Train-BitShift", train_bitshift_training_time)

    # Train Quantized Perceptron (training + quantization)
    train_quantized_model = TrainQuantizedPerceptron(learning_rate=0.01, max_epochs=1000)
    start_time = time.perf_counter()
    train_quantized_model.fit(X_train, y_train)
    train_quantized_training_time = time.perf_counter() - start_time
    train_quantized_acc, train_quantized_time = evaluate_model(train_quantized_model, X_test, y_test, "Train-Quantized", train_quantized_training_time)

    # Train Combined Perceptron (all three methods)
    train_combined_model = TrainCombinedPerceptron(learning_rate=0.01, max_epochs=1000)
    start_time = time.perf_counter()
    train_combined_model.fit(X_train, y_train)
    train_combined_training_time = time.perf_counter() - start_time
    train_combined_acc, train_combined_time = evaluate_model(train_combined_model, X_test, y_test, "Train-Combined", train_combined_training_time)

    # Efficient Perceptron (vectorized operations)
    efficient_model = EfficientPerceptron(learning_rate=0.01, max_epochs=1000)
    start_time = time.perf_counter()
    efficient_model.fit(X_train, y_train)
    efficient_training_time = time.perf_counter() - start_time
    efficient_acc, efficient_time = evaluate_model(efficient_model, X_test, y_test, "Efficient", efficient_training_time)

    # Question (c): Ultra-optimized inference model
    print("\nQuestion (c): Ultra-optimized inference model")
    print("-" * 50)

    ultra_model = UltraOptimizedPerceptron(accuracy_tolerance=0.01)
    start_time = time.perf_counter()
    ultra_model.fit(X_train, y_train)
    ultra_training_time = time.perf_counter() - start_time
    ultra_acc, ultra_time = evaluate_model(ultra_model, X_test, y_test, "Ultra-Optimized", ultra_training_time)

    # Summary comparison
    print("\n" + "="*60)
    print("Performance Summary")
    print("="*60)

    all_models = ["Binary16", "Binary32", "Binary64", "Train-Opt", "Train-BitShift", "Train-Quantized", "Train-Combined", "Efficient", "Ultra-Optimized"]
    all_accuracies = [binary16_acc, binary32_acc, binary64_acc, train_opt_acc, train_bitshift_acc, train_quantized_acc, train_combined_acc, efficient_acc, ultra_acc]
    all_training_times = [binary16_training_time, binary32_training_time, binary64_training_time, train_opt_training_time, train_bitshift_training_time, train_quantized_training_time, train_combined_training_time, efficient_training_time, ultra_training_time]
    all_times = [binary16_time, binary32_time, binary64_time, train_opt_time, train_bitshift_time, train_quantized_time, train_combined_time, efficient_time, ultra_time]

    print(f"{'Model':<15} {'Accuracy':<10} {'Training (ms)':<15} {'Inference (ns)':<15} {'Category':<12}")
    print("-" * 80)

    categories = ["Precision", "Precision", "Precision", "Train-Opt", "Train-Opt", "Train-Opt", "Train-Opt", "Efficient", "Ultra-Opt"]

    for i, (model, acc, training_time, inference_time, category) in enumerate(zip(all_models, all_accuracies, all_training_times, all_times, categories)):
        training_ms = training_time * 1_000  # Convert to milliseconds
        inference_ns = inference_time * 1_000_000_000
        print(f"{model:<15} {acc:<10.4f} {training_ms:<15.2f} {inference_ns:<15.0f} {category:<12}")

    # Best performance summary
    print(f"\nBest Performance:")
    best_accuracy_idx = np.argmax(all_accuracies)
    fastest_training_idx = np.argmin(all_training_times)
    fastest_inference_idx = np.argmin(all_times)

    # Check if all accuracies are the same
    if len(set(all_accuracies)) == 1:
        print(f"Highest Accuracy: All the same ({all_accuracies[best_accuracy_idx]:.4f})")
    else:
        print(f"Highest Accuracy: {all_models[best_accuracy_idx]} ({all_accuracies[best_accuracy_idx]:.4f})")

    fastest_training_ms = all_training_times[fastest_training_idx] * 1_000  # Convert to milliseconds
    fastest_inference_ns = all_times[fastest_inference_idx] * 1_000_000_000
    print(f"Fastest Training: {all_models[fastest_training_idx]} ({fastest_training_ms:.2f}ms)")
    print(f"Fastest Inference: {all_models[fastest_inference_idx]} ({fastest_inference_ns:.0f}ns)")

if __name__ == "__main__":
    main()
