import numpy as np
import torch
import gpytorch
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer


class MTGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        num_tasks = train_y.shape[1]

        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_tasks)
        
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=num_tasks, rank=1)

        # num_tasks = train_y.shape[1]
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


def train_mtgp(train_x, train_y, lr, iters, w):
    #scalar
    scaler = StandardScaler().fit(train_x)
    train_x_scaled = scaler.transform(train_x)
    train_x_scaled = train_x_scaled * w

    tx = torch.tensor(train_x_scaled, dtype=torch.float32)
    ty = torch.tensor(train_y, dtype=torch.float32)

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=train_y.shape[1])


    #reduced noise
    #y_var = torch.var(ty, dim=0)
    #likelihood.noise = y_var.mean().item() * 0.01 


    model = MTGP(tx, ty, likelihood)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(iters):
        optimizer.zero_grad()
        output = model(tx)
        loss = -mll(output, ty)
        loss.backward()
        optimizer.step()
        #print(f'Iter {i+1}/{iters} - Loss: {loss.item():.3f}') 

    model.eval(); likelihood.eval()
    return model, likelihood, scaler



def tune_gp(X_train, y_train, X_val, y_val, lr_list=(1e-2, 1e-1, 5e-1), iters_list=(200, 500, 800)):
    # tune learning rate and optimizer steps
    best_rmse   = np.inf
    best_params = None

    for lr in lr_list:
        for n_iters in iters_list:
            # train on training split
            model, likelihood, scaler = train_mtgp(
                X_train, y_train, lr=lr, iters = n_iters
            )
            # predict on val split
            Xs = scaler.transform(X_val)
            tx = torch.tensor(Xs, dtype=torch.float32)
            model.eval(); likelihood.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                yhat = likelihood(model(tx)).mean.cpu().numpy()

            rmse = np.sqrt(np.mean((yhat - y_val)**2))
            print(f"  lr={lr:.2e}, iters={n_iters} → RMSE={rmse:.4f}")
            if rmse < best_rmse:
                best_rmse   = rmse
                best_params = (lr, n_iters)

    lr_opt, iters_opt = best_params
    print(f"Best GP params: lr={lr_opt:.2e}, iters={iters_opt}, val_RMSE={best_rmse:.4f}")

    #Re-train final on the entire training set
    final_model, final_likelihood, final_scaler = train_mtgp(
        X_train, y_train, lr=lr_opt, iters=iters_opt)
    
    return final_model, final_likelihood, final_scaler


def main():
    weight = np.array([1]*6 +[1]*6 + [1]*6, dtype=np.float32)

    print("Loading data...")
    data_dir = "../../data/data1/"
    data_file = "data.npz"
    data = np.load(data_dir + data_file)
    data_x = data["x"]
    data_y = data["y"]
    print("Data X shape:", data_x.shape)
    print("Data Y shape:", data_y.shape)

    print("Training MTGP...")
        #gp_model, gp_likelihood, gp_scalar = tune_gp( x_train, y_train, x_val, y_val, num_tasks= data_y.shape[1],lr_list=(1e-2,1e-1,5e-1), iters_list=(200,500,800))
    gp_model, gp_likelihood, gp_scalar = train_mtgp(data_x, data_y, lr=0.01, iters=300, w = weight)

    gp_checkpoint = {
    "model_state":      gp_model.state_dict(),
    "likelihood_state": gp_likelihood.state_dict(),
    "scaler_mean":      gp_scalar.mean_,
    "scaler_scale":     gp_scalar.scale_,
    "num_tasks":        data_y.shape[1],
    }


    print(f"Saving GP 3d0f")
    result_dir = "../../results/mtgp/"
    results_path = result_dir + "mtgp_3dof.pth"
    torch.save(gp_checkpoint, results_path)

if __name__ == "__main__":
    main()

