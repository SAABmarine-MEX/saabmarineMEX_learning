#!/usr/bin/env python3
import socket, struct, threading, argparse
import pickle, numpy as np
import torch, gpytorch
from model_pb2 import InputFeatures, Prediction

#MTGP class (same as residualdynamic.py)
class MTGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_tasks
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=num_tasks, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

def load_knn(path):
    data = pickle.load(open(path,'rb'))
    return data['knn'], data['scaler']

def load_gp(path):
    chk = torch.load(path)
    # rebuild model + likelihood
    scaler = None  # will read scaler from chk
    gp_model = MTGP(None, None, None)  # dummy args
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=chk['likelihood_state']['noise_covar.raw_noise'].shape[0])
    gp_model.load_state_dict(chk['model_state'])
    likelihood.load_state_dict(chk['likelihood_state'])
    gp_model.eval(); likelihood.eval()
    scaler = (chk['scaler_mean'], chk['scaler_scale'])
    return gp_model, likelihood, scaler

def handle_client(conn, args):
    if args.model=='knn':
        knn, scaler = load_knn(args.knn)
    else:
        gp_model, likelihood, (mean, scale) = load_gp(args.gp)
        scaler = StandardScaler()
        scaler.mean_, scaler.scale_ = mean, scale

    while True:
        raw = conn.recv(4)
        if not raw: break
        L = struct.unpack(">I", raw)[0]
        data = b''
        while len(data)<L:
            chunk = conn.recv(L-len(data))
            if not chunk: return
            data += chunk
        inp = InputFeatures()
        inp.ParseFromString(data)
        feat = np.array(inp.features, dtype=np.float32).reshape(1,-1)
        Xs = scaler.transform(feat)

        if args.model=='knn':
            res = knn.predict(Xs)[0]
        else:
            xt = torch.tensor(Xs, dtype=torch.float32)
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                out = gp_model(xt)
                res = likelihood(out).mean.numpy()[0]
        out_msg = Prediction()
        out_msg.residuals.extend(res.tolist())
        out_b = out_msg.SerializeToString()
        conn.sendall(struct.pack(">I", len(out_b)) + out_b)
    conn.close()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['knn','gp'], required=True)
    p.add_argument('--knn', default='kl_server/knn_model.pkl')
    p.add_argument('--gp', default='kl_server/gp_model.pth')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=5005)
    args = p.parse_args()

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind((args.host, args.port))
    srv.listen()
    print(f"[Python] Serving {args.model} on {args.host}:{args.port}")
    while True:
        conn, addr = srv.accept()
        threading.Thread(target=handle_client, args=(conn,args), daemon=True).start()

if __name__=="__main__":
    main()
