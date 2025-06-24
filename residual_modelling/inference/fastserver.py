# fastserver.py

import argparse
import pickle
import numpy as np
import torch
import gpytorch
from sklearn.preprocessing import StandardScaler
from fastapi import FastAPI, Body, HTTPException, Response
from contextlib import asynccontextmanager

from inference import model_pb2

import sys
import os

from training.methods.svgp.gp import SVGP
from training.methods.mtgp.mtgp import MTGP

import time
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup: nothing special to do here beyond what you already do at import
    yield
    # shutdown: write one line of averages
    if total_requests:
        avg = total_time / total_requests
        ts  = datetime.now().isoformat()

        # define and create your output directory
        out_dir = os.path.join("data_and_plots", "inftime")
        os.makedirs(out_dir, exist_ok=True)

        # write into data/time/
        fname = os.path.join(out_dir, "inference_averages_tet.csv")

        with open(fname, "a") as f:
            f.write(f"{ts}_{_model_choice}_{_dof_choice}_{total_requests}_{avg:.6f}s\n")

app = FastAPI(lifespan=lifespan)

total_time     = 0.0
total_requests = 0

knn_model    = None
mtgp_model   = None
svgp_models  = []

likelihood   = None
svgp_scaler  = None
mtgp_scaler  = None
_model_choice= "knn"
_dof_choice  = "6"

weight = np.array([1]*6 + [1]*6, dtype=np.float32)

# -------------------------------------
def load_all(model_choice, dof_choice):
    global knn_model, mtgp_model ,svgp_models, likelihood, mtgp_scaler, svgp_scaler, _model_choice, _dof_choice, weight
    dir = "training/results/tet/2025-06-17_16-19-02" #OBS: change with new model folder
    _model_choice = model_choice
    _dof_choice = dof_choice

    if dof_choice == "6":
        knn_path = dir + "/knn/knn_6dof.pkl"
        mtgp_path = dir + "/mtgp/mtgp_6dof.pth"
        svgp_path = [dir + f"/svgp/svgp_6dof_output_{i}.pth" for i in range(6)]

    else:
        knn_path = dir + "/knn/knn_3dof.pkl"
        mtgp_path = dir + "/mtgp/mtgp_3dof.pth"
        svgp_path = [dir + f"/svgp/svgp_3dof_output_{i}.pth" for i in range(6)]
    
    if model_choice == "knn":
        print(f"[startup] Loading KNN from {knn_path}")
        with open(knn_path, "rb") as f:
            knn_model = pickle.load(f)

    elif model_choice == "mtgp":
        print(f"[startup] Loading MTGP from {mtgp_path}")
        mtgp_model, mtgp_scaler = MTGP.load(mtgp_path)
    
    else:
        print(f"[startup] Loading all 6 SVGPs")

        for i, path in enumerate(svgp_path):
            svgp, scaler = SVGP.load(path)
            svgp_models.append(svgp)

            if i == 0:
                svgp_scaler = scaler

        for svgp in svgp_models:
            svgp.scaler = svgp_scaler

def model_predict(features: list[float]) -> list[float]:
    print(_model_choice + _dof_choice)
    X = np.array(features, dtype=np.float32).reshape(1, -1)

    if _model_choice == "knn":
        return knn_model.predict(X)[0].tolist()
    
    elif _model_choice == "mtgp":
        pred = mtgp_model.predict(X)
        print(pred[0])
        return pred[0]

    else:
        X_scaled = svgp_scaler.transform(X)
        predictions = []
        for model in svgp_models:
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
    start = time.perf_counter()
    residuals = model_predict(msg.features)
    elapsed = time.perf_counter() - start

    global total_time, total_requests
    total_time     += elapsed
    total_requests += 1

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
        "--model", choices=["knn","mtgp", "svgp"], default="knn",
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