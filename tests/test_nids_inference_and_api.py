"""Integration and validation tests for NIDS Predictor and FastAPI Backend."""

import asyncio
import io
import json
import numpy as np
import pandas as pd
import pytest
import torch
from fastapi import HTTPException, UploadFile

from api.main import health, model_info, predict_json, predict_csv, PredictRequest, SingleFlowRecord
from src.data.nids_dataset import RAW_INPUT_FEATURES, prepare_nids_data
from src.inference.nids_predictor import NIDSPredictor, NIDSPredictionResult
from src.models.hybrid import HybridQuantumClassifier


@pytest.fixture(scope="module")
def sample_feature_df():
    """Extract a small valid sample DataFrame from dataset splits."""
    splits = prepare_nids_data(sample_size=100, random_seed=42)
    return splits.X_test_raw.iloc[:10].copy()


@pytest.fixture(scope="module")
def predictor():
    """Instantiate NIDSPredictor using completed run checkpoint."""
    return NIDSPredictor(
        checkpoint_path="results/network_intrusion/controlled_study/Classical_MLP_seed_42/best_model.pt",
        preprocessor_path="results/network_intrusion/preprocessor.joblib",
    )


def test_predictor_initialization(predictor):
    """Verify predictor attributes and configuration."""
    assert predictor.model is not None
    assert predictor.preprocessor is not None
    assert len(predictor.EXPECTED_FEATURES) == 42
    assert predictor.device == torch.device("cpu")


def test_single_sample_and_batch_inference(predictor, sample_feature_df):
    """Verify inference on DataFrame and single dictionary record."""
    # Batch DataFrame
    res = predictor.predict(sample_feature_df)
    assert isinstance(res, NIDSPredictionResult)
    assert res.sample_count == 10
    assert len(res.predicted_labels) == 10
    assert len(res.predicted_class_names) == 10

    # Probabilities sum to 1.0
    prob_sums = res.probabilities_attack + res.probabilities_benign
    np.testing.assert_allclose(prob_sums, np.ones(10), atol=1e-5)

    # Single dictionary
    single_dict = sample_feature_df.iloc[0].to_dict()
    res_single = predictor.predict(single_dict)
    assert res_single.sample_count == 1
    assert res_single.predicted_labels[0] in [0, 1]
    assert res_single.predicted_class_names[0] in ["BENIGN", "ATTACK"]


def test_consistent_predictions_between_csv_and_json(predictor, sample_feature_df):
    """Verify bitwise identical predictions between JSON record dict and CSV DataFrame."""
    dict_records = sample_feature_df.to_dict(orient="records")
    res_json = predictor.predict(dict_records)

    csv_bytes = sample_feature_df.to_csv(index=False).encode("utf-8")
    df_from_csv = pd.read_csv(io.BytesIO(csv_bytes))
    res_csv = predictor.predict(df_from_csv)

    np.testing.assert_allclose(res_json.probabilities_attack, res_csv.probabilities_attack, atol=1e-5)
    np.testing.assert_array_equal(res_json.predicted_labels, res_csv.predicted_labels)


def test_validation_rejects_missing_columns(predictor, sample_feature_df):
    """Verify DataFrame missing mandatory features is rejected."""
    bad_df = sample_feature_df.drop(columns=["proto", "dur"])
    is_valid, errors = predictor.validate_dataframe(bad_df)
    assert not is_valid
    assert any("Missing" in err for err in errors)

    with pytest.raises(ValueError, match="Input validation failed"):
        predictor.predict(bad_df)


def test_validation_rejects_non_numeric_and_nulls(predictor, sample_feature_df):
    """Verify invalid non-numeric values in numeric columns or nulls trigger rejection."""
    # Null values
    null_df = sample_feature_df.copy()
    null_df.loc[0, "dur"] = np.nan
    is_valid, errors = predictor.validate_dataframe(null_df)
    assert not is_valid
    assert any("null values" in err for err in errors)

    # Non-numeric in numeric column
    bad_type_df = sample_feature_df.copy()
    bad_type_df["spkts"] = bad_type_df["spkts"].astype(object)
    bad_type_df.loc[0, "spkts"] = "not_a_number_packet"
    is_valid, errors = predictor.validate_dataframe(bad_type_df)
    assert not is_valid


def test_quantum_measurements_bounds_and_gradients():
    """Verify quantum Pauli-Z measurements [-1, 1], angles [-pi, pi], and gradient flow."""
    model = HybridQuantumClassifier(input_dim=63, n_qubits=4, n_layers=2)
    x = torch.randn(8, 63, requires_grad=True)

    angles = model.encode_features(x)
    assert angles.shape == (8, 4)
    assert torch.all(angles >= -torch.pi - 1e-4)
    assert torch.all(angles <= torch.pi + 1e-4)

    expvals = model.quantum_features(x)
    assert expvals.shape == (8, 4)
    assert torch.all(expvals >= -1.0 - 1e-4)
    assert torch.all(expvals <= 1.0 + 1e-4)

    # Gradient flow end-to-end
    logits = model(x)
    loss = logits.sum()
    loss.backward()

    # Check gradients in encoder, quantum layer, and output head
    for name, p in model.named_parameters():
        assert p.grad is not None, f"Parameter {name} has None grad!"
        assert torch.isfinite(p.grad).all(), f"Parameter {name} has non-finite grad!"


def test_api_health_endpoint(predictor):
    """Test GET /api/health."""
    import api.main
    api.main._PREDICTOR = predictor
    data = health()
    assert data["status"] in ["healthy", "degraded"]
    assert "service" in data
    assert "timestamp" in data
    assert "uptime_seconds" in data


def test_api_model_info_endpoint(predictor):
    """Test GET /api/model-info."""
    import api.main
    api.main._PREDICTOR = predictor
    data = model_info()
    assert "dataset" in data
    assert data["input_feature_count"] == 42
    assert "class_label_mapping" in data
    assert "parameter_breakdown" in data


def test_api_predict_json_endpoint(predictor, sample_feature_df):
    """Test POST /api/predict with valid records."""
    import api.main
    api.main._PREDICTOR = predictor
    records = sample_feature_df.iloc[:2].to_dict(orient="records")
    payload = PredictRequest(records=[SingleFlowRecord(**r) for r in records], threshold=0.5)

    data = predict_json(payload)
    assert data["status"] == "success"
    assert data["sample_count"] == 2
    assert len(data["predictions"]) == 2

    p0 = data["predictions"][0]
    assert p0["predicted_label"] in ["BENIGN", "ATTACK"]
    assert "confidence" in p0
    assert "probabilities" in p0


def test_api_predict_csv_endpoint(predictor, sample_feature_df):
    """Test POST /api/predict/csv with file upload."""
    import api.main
    api.main._PREDICTOR = predictor
    csv_bytes = sample_feature_df.iloc[:3].to_csv(index=False).encode("utf-8")

    upload_json = UploadFile(filename="test_flows.csv", file=io.BytesIO(csv_bytes))
    response_json = predict_csv(file=upload_json, threshold=0.5, format="json")
    data = json.loads(response_json.body.decode("utf-8")) if hasattr(response_json, "body") else response_json
    assert data["status"] == "success"
    assert data["sample_count"] == 3

    upload_csv = UploadFile(filename="test_flows.csv", file=io.BytesIO(csv_bytes))
    response_csv = predict_csv(file=upload_csv, threshold=0.5, format="csv")
    assert "text/csv" in response_csv.headers["content-type"]
    assert "predicted_label" in response_csv.body.decode("utf-8")


def test_api_error_handling_malformed_requests(predictor):
    """Test API error handling on malformed JSON and empty CSV."""
    import api.main
    api.main._PREDICTOR = predictor

    # Empty records
    with pytest.raises(HTTPException) as exc_info:
        predict_json(PredictRequest(records=[]))
    assert exc_info.value.status_code == 400

    # Empty CSV
    upload_empty = UploadFile(filename="empty.csv", file=io.BytesIO(b""))
    with pytest.raises(HTTPException) as exc_info_csv:
        predict_csv(file=upload_empty)
    assert exc_info_csv.value.status_code == 400
