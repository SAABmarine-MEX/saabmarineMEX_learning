# fastserver.py

import argparse
import pickle
import numpy as np
import torch
import gpytorch
from sklearn.preprocessing import StandardScaler
from fastapi import FastAPI, Body, HTTPException, Response
import model_pb2

import sys
import os

#TODO make cleaner & check if works
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'training', 'methods', 'svgp')))
from gp import SVGP

# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI()

knn_model    = None

gp_x = None
gp_y = None
gp_z = None
gp_wx = None
gp_wy = None
gp_wz = None

likelihood   = None
gp_scalar    = None
_model_choice= "knn"
_dof_choice  = "6"

weight = np.array([1]*6 +[1]*6 + [1]*6, dtype=np.float32)

# -------------------------------------
def load_all(model_choice, dof_choice):
    global knn_model, gp_x, gp_y, gp_z, gp_wx, gp_wy, gp_wz, likelihood, gp_scalar, _model_choice, _dof_choice, weight

    _model_choice = model_choice
    _dof_choice = dof_choice

    if dof_choice == "6":
        knn_path = "knn_6dof.pkl"

        gp_path_x = "svgp_6dof_output_0.pth"
        gp_path_y = "svgp_6dof_output_1.pth"
        gp_path_z = "svgp_6dof_output_2.pth"
        gp_path_wx = "svgp_6dof_output_3.pth"
        gp_path_wy = "svgp_6dof_output_4.pth"
        gp_path_wz = "svgp_6dof_output_5.pth"
    else:
        knn_path = "knn_3dof.pkl"

        gp_path_x = "svgp_3dof_output_0.pth"
        gp_path_y = "svgp_3dof_output_1.pth"
        gp_path_z = "svgp_3dof_output_2.pth"
        gp_path_wx = "svgp_3dof_output_3.pth"
        gp_path_wy = "svgp_3dof_output_4.pth"
        gp_path_wz = "svgp_3dof_output_5.pth"
    
    if model_choice == "knn":
        print(f"[startup] Loading KNN from {knn_path}")
        with open(knn_path, "rb") as f:
            knn_model = pickle.load(f)

    else:
        print(f"[startup] Loading all 6 SVGPs")

        gp_x = model_load(gp_path_x)
        gp_y = model_load(gp_path_y)
        gp_z = model_load(gp_path_z)
        gp_wx = model_load(gp_path_wx)
        gp_wy = model_load(gp_path_wy)
        gp_wz = model_load(gp_path_wz)


def model_load(path):
    c = torch.load(path, map_location="cpu")
    gp_scalar = StandardScaler()
    gp_scalar.mean_ = c["scaler_mean"]
    gp_scalar.scale_ = c["scaler_scale"]
    gp_model = SVGP.load(nind=c["n_inducing"], fname=path)
    gp_model.eval()
    gp_model.likelihood.eval()

    #TODO change how gp is saved in gp.py

    return (gp_model)


def model_predict(features: list[float]) -> list[float]:
    print(_model_choice + _dof_choice)
    X = np.array(features, dtype=np.float32).reshape(1, -1)
    if _model_choice == "knn":
        return knn_model.predict(X)[0].tolist()
    else:
        predict_x = single_gp_predict(gp_x,X)
        predict_y = single_gp_predict(gp_y,X)
        predict_z = single_gp_predict(gp_z,X)
        predict_wx = single_gp_predict(gp_wx,X)
        predict_wy = single_gp_predict(gp_wy,X)
        predict_wz = single_gp_predict(gp_wz,X)
        return [predict_x, predict_y, predict_z, predict_wx, predict_wy, predict_wz]

        #TODO check what X and mu is and make this work correctly 

def single_gp_predict(gp,x):
    mu, sigma = gp.sample(x)
    return mu


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