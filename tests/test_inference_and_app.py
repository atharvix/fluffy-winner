"""Tests for inference engine, input validation, and predictor module."""

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.datasets import load_breast_cancer

from src.data.dataset import prepare_breast_cancer_data
from src.inference.predictor import HybridPredictor, PredictionResult


@pytest.fixture(scope="module")
def predictor():
    """Load cached HybridPredictor instance."""
    return HybridPredictor(checkpoint_path="results/hybrid_quantum/best_model.pt", random_seed=42)


def test_predictor_initialization_and_metadata(predictor):
    """Verify that predictor loads model architecture and exact 30 feature names."""
    assert predictor.model is not None
    assert len(predictor.EXPECTED_FEATURES) == 30
    assert len(predictor.feature_names) == 30
    assert predictor.scaler is not None
    assert predictor.scaler.mean_.shape == (30,)


def test_single_sample_inference(predictor):
    """Verify single-sample dictionary inference and output bounds."""
    splits = prepare_breast_cancer_data(random_seed=42)
    sample_dict = {name: float(splits.X_test_raw[0, i]) for i, name in enumerate(predictor.EXPECTED_FEATURES)}

    result: PredictionResult = predictor.predict(sample_dict)

    assert result.sample_count == 1
    assert len(result.probabilities_benign) == 1
    assert len(result.probabilities_malignant) == 1
    assert 0.0 <= result.probabilities_benign[0] <= 1.0
    assert 0.0 <= result.probabilities_malignant[0] <= 1.0
    assert np.isclose(result.probabilities_benign[0] + result.probabilities_malignant[0], 1.0)
    assert result.predicted_class_names[0] in ["malignant", "benign"]

    # Verify intermediate quantum values
    assert result.encoder_angles.shape == (1, 4)
    assert -np.pi - 1e-5 <= result.encoder_angles.min() and result.encoder_angles.max() <= np.pi + 1e-5
    assert result.quantum_expectation_values.shape == (1, 4)
    assert -1.0 - 1e-5 <= result.quantum_expectation_values.min() and result.quantum_expectation_values.max() <= 1.0 + 1e-5


def test_batch_dataframe_inference(predictor):
    """Verify batch DataFrame inference matches individual sample predictions."""
    splits = prepare_breast_cancer_data(random_seed=42)
    df = pd.DataFrame(splits.X_test_raw[:5], columns=predictor.EXPECTED_FEATURES)

    results: PredictionResult = predictor.predict(df)

    assert results.sample_count == 5
    assert len(results.predicted_labels) == 5
    assert len(results.confidences) == 5
    assert results.encoder_angles.shape == (5, 4)
    assert results.quantum_expectation_values.shape == (5, 4)


def test_validation_rejects_missing_columns(predictor):
    """Verify validation failure when required feature columns are missing."""
    df_missing = pd.DataFrame({"mean radius": [14.0, 15.0], "mean texture": [20.0, 21.0]})
    is_valid, errors = predictor.validate_dataframe(df_missing)

    assert not is_valid
    assert any("Missing" in err for err in errors)

    with pytest.raises(ValueError, match="DataFrame validation failed"):
        predictor.predict(df_missing)


def test_validation_rejects_non_numeric_and_nulls(predictor):
    """Verify validation failure on non-numeric or NaN values."""
    splits = prepare_breast_cancer_data(random_seed=42)
    df_invalid = pd.DataFrame(splits.X_test_raw[:3], columns=predictor.EXPECTED_FEATURES)

    # 1. Null values test
    df_null = df_invalid.copy()
    df_null.iloc[0, 0] = np.nan
    is_valid_null, errors_null = predictor.validate_dataframe(df_null)
    assert not is_valid_null
    assert any("missing/null" in err for err in errors_null)

    # 2. String/Object dtype test
    df_str = df_invalid.copy()
    df_str["mean radius"] = ["invalid_str", "12.0", "13.0"]
    is_valid_str, errors_str = predictor.validate_dataframe(df_str)
    assert not is_valid_str
    assert any("must be numeric" in err for err in errors_str)


def test_preprocessor_is_not_refitted_on_inference_data(predictor):
    """Verify that the preprocessor scaler mean/scale remain unchanged before and after inference."""
    original_mean = predictor.scaler.mean_.copy()
    original_var = predictor.scaler.var_.copy()

    # Pass extreme input data
    extreme_data = np.ones((10, 30), dtype=np.float32) * 10000.0
    predictor.predict(extreme_data)

    np.testing.assert_array_equal(predictor.scaler.mean_, original_mean)
    np.testing.assert_array_equal(predictor.scaler.var_, original_var)
