# fastserver.py

import argparse
import pickle
import numpy as np
import torch
import gpytorch
from sklearn.preprocessing import StandardScaler
from fastapi import FastAPI, Body, HTTPException, Response
import model_pb2

# ─────────────────────────────────────────────────────────────────────────────
class MTGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        num_tasks = train_y.shape[1]
        
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_tasks)
        
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=num_tasks, rank=1)

        # super().__init__(train_x, train_y, likelihood)
        # D = train_x.shape[1]

        # self.mean_module = gpytorch.means.MultitaskMean(
        # gpytorch.means.ConstantMean(), num_tasks=num_tasks)  
        
        # const_kern = gpytorch.kernels.ConstantKernel()
        # lin_kern   = gpytorch.kernels.LinearKernel(ard_num_dims=D)
        # matern_kern = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=D))

        # base_covar = const_kern + lin_kern + matern_kern

        # self.covar_module = gpytorch.kernels.MultitaskKernel(
        #     base_covar, num_tasks=num_tasks, rank=1)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI()

knn_model    = None
gp_model     = None
likelihood   = None
gp_scalar    = None
_model_choice= "knn"
_dof_choice  = "6"

weight = np.array([1]*6 +[1]*6 + [1]*6, dtype=np.float32)

# -------------------------------------
def load_models(model_choice, dof_choice):
    global knn_model, gp_model, likelihood, gp_scalar, _model_choice, _dof_choice, weight

    _model_choice = model_choice
    _dof_choice = dof_choice

    if dof_choice == "6":
        knn_path = "knn_6dof.pkl"
        gp_path = "gp_6dof.pth"
    else:
        knn_path = "knn_3dof.pkl"
        gp_path = "gp_3dof.pth"
    
    if model_choice == "knn":
        print(f"[startup] Loading KNN from {knn_path}")
        with open(knn_path, "rb") as f:
            knn_model = pickle.load(f)

    else:
        print(f"[startup] Loading GP from {gp_path}")
        chk = torch.load(gp_path, map_location="cpu", weights_only=False)

        # rebuild GP
        n_out=int(chk["num_tasks"])
        #print(f"[startup] GP has {n_out} dimensions")
        gp_scalar = StandardScaler()
        gp_scalar.mean_, gp_scalar.scale_ = (chk["scaler_mean"], chk["scaler_scale"])    

        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=n_out)

        #to get the right shapes
        input_dim = weight.shape[0]
        dummy_x = torch.zeros(1, input_dim)
        dummy_y = torch.zeros(1, n_out)
        gp_model = MTGP(dummy_x, dummy_y, likelihood)

        gp_model.load_state_dict(chk["model_state"])
        likelihood.load_state_dict(chk["likelihood_state"])

        gp_model.eval()
        likelihood.eval()

def model_predict(features: list[float]) -> list[float]:
    print(_model_choice + _dof_choice)
    X = np.array(features, dtype=np.float32).reshape(1, -1)
    if _model_choice == "knn":
        return knn_model.predict(X)[0].tolist()
    else:
        Xs = gp_scalar.transform(X)
        Xs = Xs * weight
        xt = torch.tensor(Xs)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            out = likelihood(gp_model(xt)).mean.numpy()[0]
        return out.tolist()


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
    load_models(args.model, args.dof)

    # Start Uvicorn
    import uvicorn
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
    )