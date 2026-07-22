from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter

from ..registry import get_registry
from .failure_predictor import FailurePredictor

logger = logging.getLogger(__name__)

predictor_router = APIRouter(prefix="/api/v1/predictor", tags=["predictor"])


def _get_predictor() -> FailurePredictor:
    reg = get_registry()
    return reg.get_or_create("predictor", FailurePredictor)


@predictor_router.post("/predict")
def predict_failure(telemetry: dict) -> dict:
    return _get_predictor().predict_failure(telemetry)


@predictor_router.post("/train")
def train_predictor(telemetry_data: List[dict], labels: List[int]) -> dict:
    return _get_predictor().train(telemetry_data, labels)


@predictor_router.post("/analyze-cluster")
def analyze_cluster(cluster_id: str, node_count: int = 4) -> dict:
    return _get_predictor().analyze_cluster(cluster_id, node_count)
