# fastserver.py

import argparse
import pickle
import numpy as np
import torch
import gpytorch
from sklearn.preprocessing import StandardScaler
from fastapi import FastAPI, Body, HTTPException, Response
from inference import model_pb2

import sys
import os

from training.methods.svgp.gp import SVGP

# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI()

knn_model    = None

gp_models = []

likelihood   = None
gp_scaler    = None
_model_choice= "knn"
_dof_choice  = "6"

weight = np.array([1]*6 +[1]*6 + [1]*6, dtype=np.float32)

# -------------------------------------
def load_all(model_choice, dof_choice):
    global knn_model, gp_models, likelihood, gp_scaler, _model_choice, _dof_choice, weight
    dir = "training/results/2025-06-05_15-16-27"
    _model_choice = model_choice
    _dof_choice = dof_choice

    if dof_choice == "6":
        knn_path = dir + "/knn/knn_6dof.pkl"
        gp_path = [dir + f"/svgp/svgp_6dof_output_{i}.pth" for i in range(6)]

    else:
        knn_path = dir + "/knn/knn_3dof.pkl"

        gp_path = [dir + f"/svgp/svgp_3dof_output_{i}.pth" for i in range(6)]
    
    if model_choice == "knn":
        print(f"[startup] Loading KNN from {knn_path}")
        with open(knn_path, "rb") as f:
            knn_model = pickle.load(f)

    else:
        print(f"[startup] Loading all 6 SVGPs")

        for i, path in enumerate(gp_path):
            gp, scaler = SVGP.load(path)
            gp_models.append(gp)

            if i == 0:
                gp_scaler = scaler

        for gp in gp_models:
            gp.scaler = gp_scaler

def model_predict(features: list[float]) -> list[float]:
    print(_model_choice + _dof_choice)
    X = np.array(features, dtype=np.float32).reshape(1, -1)

    if _model_choice == "knn":
        return knn_model.predict(X)[0].tolist()
    else:
        X_scaled = gp_scaler.transform(X)
        predictions = []
        for model in gp_models:
            mu, _ = model.sample(X_scaled)
            predictions.append(mu[0])
        return np.array(predictions)


@app.post("/predict", response_class=Response)
def com(body: bytes = Body(..., media_type="application/octet-stream")):
    # Parse into our protobuf msg
    msg = model_pb2.InputFeatures()
    try:
        msg.ParseFromString(body)
    except Exception:
        # invalid bytes
        raise HTTPException(status_code=400, detail="Invalid protobuf")

    # do da inference
    residuals = model_predict(msg.features)
    out = model_pb2.Prediction(residuals=residuals)
    #serialize
    data = out.SerializeToString()

    return Response(content=data, media_type="application/octet-stream")

#-------------------------MAIN---------------------------#

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Sync FastAPI server for KNN/GP"
    )
    parser.add_argument(
        "--model", choices=["knn","gp"], default="knn",
        help="Which model to serve (default: knn)"
    )
    parser.add_argument(
        "--dof", choices=["3","6"], default="6",
        help="How many DOF for the model (default: 6)"
    )
    parser.add_argument(
        "--host", default="0.0.0.0",
        help="Host to bind"
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="Port to listen on"
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of Uvicorn workers"
    )
    args = parser.parse_args()

    # Load once at startup
    load_all(args.model, args.dof)

    # Start Uvicorn
    import uvicorn
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
    )