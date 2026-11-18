"""FastAPI Backend for Hybrid Classical–Quantum Network Intrusion Detection System.

Exposes production inference endpoints for Google Stitch or any custom web frontend:
- GET /api/health: Service readiness and model status.
- GET /api/model-info: Model architecture, feature schema, class mappings, circuit metadata.
- POST /api/predict: Validated JSON batch inference with quantum state inspection.
- POST /api/predict/csv: Validated CSV file upload inference.
"""

import io
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from src.data.nids_dataset import (
    RAW_CATEGORICAL_FEATURES,
    RAW_INPUT_FEATURES,
    RAW_NUMERICAL_FEATURES,
    TARGET_LABEL_MAPPING,
)
from src.inference.nids_predictor import NIDSPredictor, NIDSPredictionResult


# Instantiate FastAPI Application
app = FastAPI(
    title="Hybrid Classical–Quantum Network Intrusion Detection API",
    description=(
        "Production inference backend for variational quantum and data re-uploading "
        "hybrid classifiers trained on the UNSW-NB15 network intrusion dataset."
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
    id: Optional[Union[str, int]] = Field(None, description="Optional record identifier")
    dur: float = Field(..., description="Record total duration")
    proto: str = Field(..., description="Transaction protocol (e.g. tcp, udp)")
    service: str = Field(..., description="Destination service (e.g. http, ftp, -)")
    state: str = Field(..., description="Connection state (e.g. FIN, INT, CON)")
    spkts: int = Field(..., description="Source to destination packet count")
    dpkts: int = Field(..., description="Destination to source packet count")
    sbytes: int = Field(..., description="Source to destination transaction bytes")
    dbytes: int = Field(..., description="Destination to source transaction bytes")
    rate: float = Field(..., description="Packet rate")
    sttl: int = Field(..., description="Source to destination time to live")
    dttl: int = Field(..., description="Destination to source time to live")
    sload: float = Field(..., description="Source bits per second")
    dload: float = Field(..., description="Destination bits per second")
    sloss: int = Field(..., description="Source packets retransmitted or dropped")
    dloss: int = Field(..., description="Destination packets dropped")
    sinpkt: float = Field(..., description="Source interpacket arrival time (msec)")
    dinpkt: float = Field(..., description="Destination interpacket arrival time (msec)")
    sjit: float = Field(..., description="Source jitter (msec)")
    djit: float = Field(..., description="Destination jitter (msec)")
    swin: int = Field(..., description="Source TCP window advertisement")
    stcpb: int = Field(..., description="Source TCP base sequence number")
    dtcpb: int = Field(..., description="Destination TCP base sequence number")
    dwin: int = Field(..., description="Destination TCP window advertisement")
    tcprtt: float = Field(..., description="TCP connection setup round-trip time")
    synack: float = Field(..., description="TCP SYN to SYN-ACK setup time")
    ackdat: float = Field(..., description="TCP SYN-ACK to ACK setup time")
    smean: int = Field(..., description="Mean packet size transmitted by source")
    dmean: int = Field(..., description="Mean packet size transmitted by destination")
    trans_depth: int = Field(..., description="Pipelined HTTP request depth")
    response_body_len: int = Field(..., description="HTTP response content length")
    ct_srv_src: int = Field(..., description="Connections to same service from source IP")
    ct_state_ttl: int = Field(..., description="Connections with same state and TTL")
    ct_dst_ltm: int = Field(..., description="Connections to same destination address")
    ct_src_dport_ltm: int = Field(..., description="Connections from same source to same destination port")
    ct_dst_sport_ltm: int = Field(..., description="Connections from same destination to same source port")
    ct_dst_src_ltm: int = Field(..., description="Connections between same source and destination")
    is_ftp_login: int = Field(..., description="1 if FTP session accessed by user/password")
    ct_ftp_cmd: int = Field(..., description="Count of FTP commands in session")
    ct_flw_http_mthd: int = Field(..., description="Count of HTTP request methods")
    ct_src_ltm: int = Field(..., description="Connections from same source IP in last 100")
    ct_srv_dst: int = Field(..., description="Connections to same service at destination IP")
    is_sm_ips_ports: int = Field(..., description="1 if source and destination IP and ports equal")


class PredictRequest(BaseModel):
    records: List[SingleFlowRecord] = Field(..., description="List of network flow records to classify")
    threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="Decision threshold for ATTACK label")


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
        "dataset": "UNSW-NB15",
        "dataset_provenance": (
            "Australian Centre for Cyber Security (ACCS), UNSW Canberra. "
            "Authors: Nour Moustafa and Jill Slay (2015)."
        ),
        "input_feature_count": len(RAW_INPUT_FEATURES),
        "raw_features": RAW_INPUT_FEATURES,
        "numerical_features": RAW_NUMERICAL_FEATURES,
        "categorical_features": RAW_CATEGORICAL_FEATURES,
        "class_label_mapping": TARGET_LABEL_MAPPING,
        "positive_class": "ATTACK (1)",
        "negative_class": "BENIGN (0)",
        "decision_threshold": 0.5,
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

    try:
        res: NIDSPredictionResult = predictor.predict(df, threshold=request.threshold)
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
        "decision_threshold": request.threshold,
        "predictions": predictions_out,
    }


@app.post("/api/predict/csv", summary="Predict on Uploaded CSV File")
async def predict_csv(
    file: UploadFile = File(...),
    threshold: float = Query(0.5, ge=0.0, le=1.0),
    format: str = Query("json", pattern="^(json|csv)$"),
) -> Response:
    """
    Accepts CSV file upload containing UNSW-NB15 flow features, validates schema,
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

    contents = await file.read()
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

    try:
        res: NIDSPredictionResult = predictor.predict(df, threshold=threshold)
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
            "decision_threshold": threshold,
            "predictions": predictions_out,
        }
    )
