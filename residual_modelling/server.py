#!/usr/bin/env python3
import socket, struct, threading, argparse
import pickle, numpy as np
import torch, gpytorch
from model_pb2 import InputFeatures, Prediction
import time

# timing stuff
#tmings = {}
timings =[
        [], # time to receive data
        [], # time to send data
        [], # time to run knn
        ]


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
    resknn = pickle.load(open(path,'rb'))
    return resknn

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
        resknn = load_knn(args.knn)
    else:
        gp_model, likelihood, (mean, scale) = load_gp(args.gp)
        scaler = StandardScaler()
        scaler.mean_, scaler.scale_ = mean, scale

    try:
        while True:
            # Measure time of receiving data
            
            # Recieve data, feutures from the sim
            raw = conn.recv(4)
            if not raw: break
            L = struct.unpack(">I", raw)[0]
            data = b''
            while len(data)<L:
                chunk = conn.recv(L-len(data))
                if not chunk: return
                data += chunk
            
            start_recv = time.perf_counter()
            inp = InputFeatures()
            inp.ParseFromString(data)
            feat = np.array(inp.features, dtype=np.float32).reshape(1,-1)
            
            end_recv = time.perf_counter()
            diff_recv = end_recv - start_recv
            timings[0].append(diff_recv)
            print("[Python] Received features time:", diff_recv)

            #print("[Python] Got features:", inp.features[:6], "...")       # show first 6 for brevity

            if args.model=='knn':
                # time of knn inference
                start_knn_inference = time.perf_counter()
                res = resknn.predict(feat)[0]
                end_knn_inference = time.perf_counter()
                diff_knn_inference = end_knn_inference - start_knn_inference
                timings[1].append(diff_knn_inference)
                print("[Python] KNN inference time:", diff_knn_inference)
            else:
                # time of gp inference
                start_gp_inference = time.perf_counter()
                xt = torch.tensor(Xs, dtype=torch.float32)
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    out = gp_model(xt)
                    res = likelihood(out).mean.numpy()[0]
                end_gp_inference = time.perf_counter()
                diff_gp_inference = end_gp_inference - start_gp_inference
                print("[Python] GP inference time:", diff_gp_inference)


            
            #print("[Python] Sending residuals:", res.tolist())

            # Measure time of sending data
            start_send = time.perf_counter()
            out_msg = Prediction()
            out_msg.residuals.extend(res.tolist())
            out_b = out_msg.SerializeToString()
            conn.sendall(struct.pack(">I", len(out_b)) + out_b)
            end_send = time.perf_counter()
            diff_send = end_send - start_send
            timings[2].append(diff_send)
            print("[Python] Sent residuals time:", diff_send)
    finally:
        means = [sum(block) / len(block) if block else 0 for block in timings]
        print("Mean timings (in seconds):")
        print(f"Receive: {means[0]:.6f}")
        print(f"KNN:     {means[1]:.6f}")
        print(f"Send:    {means[2]:.6f}")

    conn.close()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['knn','gp'], required=True)
    p.add_argument('--knn', default='knn_data.pkl')
    p.add_argument('--gp', default='gp_model.pth')
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
