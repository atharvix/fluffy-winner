"""FastAPI Backend for Hybrid Classical–Quantum Network Intrusion Detection System.

Exposes production inference endpoints for Google Stitch or any custom web frontend:
- GET /api/health: Service readiness and model status.
- GET /api/model-info: Model architecture, feature schema, class mappings, circuit metadata.
- POST /api/predict: Validated JSON batch inference with quantum state inspection.
- POST /api/predict/csv: Validated CSV file upload inference.
"""

import io
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(1)

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from src.data.ciciot2023_dataset import TARGET_LABEL_MAPPING
from src.inference.nids_predictor import NIDSPredictor, NIDSPredictionResult


# Instantiate FastAPI Application
app = FastAPI(
    title="Hybrid Classical–Quantum Network Intrusion Detection API",
    description=(
        "Production inference backend for variational quantum and data re-uploading "
        "hybrid classifiers trained on the CIC-IoT-2023 (45M Flows) network intrusion dataset."
    ),
    version="1.0.0",
)

# CORS configuration for local frontend development
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:8000",
    "http://localhost:8080",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:8000",
    "http://127.0.0.1:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Global Predictor Instance with lazy load / graceful error handling
_PREDICTOR: Optional[NIDSPredictor] = None
_START_TIME = time.time()


def get_predictor() -> NIDSPredictor:
    """Retrieve or initialize the singleton NIDSPredictor."""
    global _PREDICTOR
    if _PREDICTOR is None:
        try:
            _PREDICTOR = NIDSPredictor()
        except Exception as e:
            raise RuntimeError(f"Model failed to initialize: {str(e)}")
    return _PREDICTOR


# Pydantic Schemas for JSON inference
class SingleFlowRecord(BaseModel):
    model_config = {"extra": "allow"}
    id: Optional[Union[str, int]] = Field(None, description="Optional record identifier")


class PredictRequest(BaseModel):
    records: List[SingleFlowRecord] = Field(..., description="List of network flow records to classify")
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Decision threshold for ATTACK label (defaults to model's tuned threshold)")



@app.get("/api/config", summary="Runtime Frontend Configuration")
def runtime_config(request: Request) -> Dict[str, Any]:
    pred = get_predictor()
    base_url = str(request.base_url).rstrip("/")
    return {
        "api_base_url": base_url,
        "model_name": pred.model_name,
        "model_family": pred.model_family,
        "dataset": getattr(pred, "dataset_name", "CIC-IoT-2023 (45,019,243 Flows)"),
        "input_feature_count": len(pred.EXPECTED_FEATURES),
        "decision_threshold": round(float(pred.decision_threshold), 4),
    }

@app.get("/api/health", summary="Service Health and Model Status")
def health() -> Dict[str, Any]:
    """
    Check API health and model readiness.
    Distinguishes between a running service process and a successfully loaded model.
    """
    uptime_seconds = time.time() - _START_TIME
    model_ready = False
    model_info = "Not loaded"

    try:
        predictor = get_predictor()
        model_ready = True
        model_info = predictor.model_name
    except Exception as e:
        model_ready = False
        model_info = f"Failed to load: {str(e)}"

    return {
        "status": "healthy" if model_ready else "degraded",
        "service": "Hybrid Classical-Quantum NIDS Inference Service",
        "model_loaded": model_ready,
        "selected_model": model_info,
        "device": "cpu",
        "uptime_seconds": round(uptime_seconds, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0",
    }


@app.get("/api/model-info", summary="Model Metadata and Input Schema")
def model_info() -> Dict[str, Any]:
    """
    Return active model architecture, dataset provenance, feature schema, and parameter counts.
    """
    try:
        predictor = get_predictor()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP(503),
            detail=f"Inference model is currently unavailable: {str(e)}",
        )

    config = predictor.config
    circuit_metadata = config.get("circuit_metadata", {})
    param_breakdown = config.get("parameter_breakdown", {})

    return {
        "model_name": predictor.model_name,
        "model_family": predictor.model_family,
        "dataset": "CIC-IoT-2023 (45M Flows)",
        "dataset_provenance": (
            "Australian Centre for Cyber Security (ACCS), UNSW Canberra. "
            "Authors: Nour Moustafa and Jill Slay (2015)."
        ),
        "input_feature_count": len(predictor.EXPECTED_FEATURES),
        "raw_features": predictor.EXPECTED_FEATURES,
        "numerical_features": predictor.NUMERICAL_FEATURES,
        "categorical_features": predictor.CATEGORICAL_FEATURES,
        "class_label_mapping": TARGET_LABEL_MAPPING,
        "positive_class": "ATTACK (1)",
        "negative_class": "BENIGN (0)",
        "decision_threshold": round(float(predictor.decision_threshold), 4),
        "parameter_breakdown": param_breakdown,
        "circuit_metadata": circuit_metadata,
        "environment": config.get("environment", {}),
    }


@app.post("/api/predict", summary="Predict on JSON Batch")
def predict_json(request: PredictRequest) -> Dict[str, Any]:
    """
    Perform inference on a batch of network flow feature records.
    Returns predicted label, class probabilities, confidences, and quantum circuit state inspection.
    """
    try:
        predictor = get_predictor()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model unavailable: {str(e)}",
        )

    if not request.records:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Request records list cannot be empty.",
        )

    # Convert records to dictionary list
    records_dict = [r.model_dump() for r in request.records]
    df = pd.DataFrame(records_dict)

    active_threshold = (
        float(request.threshold)
        if request.threshold is not None
        else float(predictor.decision_threshold)
    )

    try:
        res: NIDSPredictionResult = predictor.predict(df, threshold=active_threshold)
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Input validation error: {str(ve)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failure: {str(e)}",
        )

    predictions_out: List[Dict[str, Any]] = []
    for i in range(res.sample_count):
        rec_id = records_dict[i].get("id", i)
        pred_item: Dict[str, Any] = {
            "record_id": rec_id,
            "index": i,
            "predicted_label": res.predicted_class_names[i],
            "predicted_class_id": int(res.predicted_labels[i]),
            "confidence": round(float(res.confidences[i]), 4),
            "probabilities": {
                "ATTACK": round(float(res.probabilities_attack[i]), 4),
                "BENIGN": round(float(res.probabilities_benign[i]), 4),
            },
            "raw_logit": round(float(res.raw_logits[i]), 4),
        }

        # Include quantum inspection if available in model architecture
        if res.encoder_angles is not None or res.quantum_expectation_values is not None:
            pred_item["quantum_inspection"] = {
                "encoder_angles": (
                    [round(float(a), 4) for a in res.encoder_angles[i]]
                    if res.encoder_angles is not None
                    else None
                ),
                "pauli_z_expectations": (
                    [round(float(e), 4) for e in res.quantum_expectation_values[i]]
                    if res.quantum_expectation_values is not None
                    else None
                ),
            }

        predictions_out.append(pred_item)

    attack_count = sum(1 for p in predictions_out if p["predicted_label"] == "ATTACK")
    benign_count = res.sample_count - attack_count

    return {
        "status": "success",
        "sample_count": res.sample_count,
        "summary": {
            "attack_count": attack_count,
            "benign_count": benign_count,
            "attack_ratio": round(attack_count / res.sample_count, 4),
        },
        "decision_threshold": round(active_threshold, 4),
        "predictions": predictions_out,
    }


@app.post("/api/predict/csv", summary="Predict on Uploaded CSV File")
def predict_csv(
    file: UploadFile = File(...),
    threshold: Optional[float] = Query(None, ge=0.0, le=1.0),
    format: str = Query("json", pattern="^(json|csv)$"),
) -> Response:
    """
    Accepts CSV file upload containing CIC-IoT-2023 (45M Flows) flow features, validates schema,
    executes inference, and returns structured JSON or downloadable CSV with predictions.
    """
    try:
        predictor = get_predictor()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model unavailable: {str(e)}",
        )

    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a .csv extension.",
        )

    contents = file.file.read()
    if len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded CSV file is empty.",
        )

    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to parse CSV file: {str(e)}",
        )

    # Validate against expected schema
    is_valid, validation_errors = predictor.validate_dataframe(df)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"CSV schema validation failed: {'; '.join(validation_errors)}",
        )

    active_threshold = (
        float(threshold) if threshold is not None else float(predictor.decision_threshold)
    )

    try:
        res: NIDSPredictionResult = predictor.predict(df, threshold=active_threshold)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference failed: {str(e)}",
        )

    if format == "csv":
        out_df = df.copy()
        out_df["predicted_label"] = res.predicted_class_names
        out_df["prob_attack"] = np.round(res.probabilities_attack, 4)
        out_df["prob_benign"] = np.round(res.probabilities_benign, 4)
        out_df["confidence"] = np.round(res.confidences, 4)

        csv_buffer = io.StringIO()
        out_df.to_csv(csv_buffer, index=False)
        return Response(
            content=csv_buffer.getvalue(),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=nids_predictions_{file.filename}"
            },
        )

    # Return structured JSON
    predictions_out = []
    for i in range(res.sample_count):
        predictions_out.append(
            {
                "index": i,
                "predicted_label": res.predicted_class_names[i],
                "confidence": round(float(res.confidences[i]), 4),
                "probabilities": {
                    "ATTACK": round(float(res.probabilities_attack[i]), 4),
                    "BENIGN": round(float(res.probabilities_benign[i]), 4),
                },
            }
        )

    attack_count = int(np.sum(res.predicted_labels == 1))
    benign_count = res.sample_count - attack_count

    return JSONResponse(
        content={
            "status": "success",
            "filename": file.filename,
            "sample_count": res.sample_count,
            "summary": {
                "attack_count": attack_count,
                "benign_count": benign_count,
                "attack_ratio": round(attack_count / res.sample_count, 4),
            },
            "decision_threshold": round(active_threshold, 4),
            "predictions": predictions_out,
        }
    )

# Eagerly warm up predictor on main thread to avoid Windows OpenMP/AnyIO thread crash
get_predictor()

