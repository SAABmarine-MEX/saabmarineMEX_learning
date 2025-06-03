import numpy as np
import torch
import gpytorch
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer


class MTGP(gpytorch.models.ExactGP):
    def __init__(self, num_inputs=18, num_tasks=6): # NOTE: num_tasks is determined by train_y.shape[1]
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)

        # dumby data to initialize the parent class, but will be overridden in fit()
        train_x = torch.ones((10, num_inputs), dtype=torch.float32)
        train_y = torch.ones((10, num_tasks), dtype=torch.float32)

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


    def predict(self, data_x):
        # Predict with the model
        X_scaled = self.sclr.transform(data_x)
        tx = torch.tensor(X_scaled, dtype=torch.float32)
        self.eval(); self.lh.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            yhat = self.likelihood(self(tx)).mean.cpu().numpy()
        return yhat


    def save(self, path):
        if not path.endswith('.pth'):
            path += '.pth'
        # Save the model state
        checkpoint = {
            "model_state": self.state_dict(), 
            # "likelihood_state": self.likelihood.state_dict(),
            "likelihood_state": self.lh.state_dict(),
            # "scaler_mean": self.scaler.mean_,
            # "scaler_scale": self.scaler.scale_,
            # "num_tasks": self.likelihood.num_tasks,
            "scaler_mean": self.sclr.mean_,
            "scaler_scale": self.sclr.scale_,
            "num_tasks": self.lh.num_tasks,
        }
        torch.save(checkpoint, path)


    def fit(self, data_x, data_y, lr=0.01, iters=300, w=np.array([1]*6 +[1]*6 + [1]*6, dtype=np.float32)):
        # Override the dummy data
        #tx = torch.tensor(data_x, dtype=torch.float32)
        #ty = torch.tensor(data_y, dtype=torch.float32)

        #scalar
        scaler = StandardScaler().fit(data_x)
        data_x_scaled = scaler.transform(data_x)
        data_x_scaled = data_x_scaled * w

        tx = torch.tensor(data_x_scaled, dtype=torch.float32)
        ty = torch.tensor(data_y, dtype=torch.float32)

        # Override the dummy data
        self.set_train_data(inputs=tx, targets=ty, strict=False)


        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=data_y.shape[1])

        #reduced noise
        #y_var = torch.var(ty, dim=0)
        #likelihood.noise = y_var.mean().item() * 0.01 

        self.train()
        likelihood.train()

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, self)

        for i in range(iters):
            optimizer.zero_grad()
            output = self(tx)
            loss = -mll(output, ty)
            loss.backward()
            optimizer.step()
            #print(f'Iter {i+1}/{iters} - Loss: {loss.item():.3f}') 

        self.eval(); likelihood.eval()
        self.sclr = scaler  # Store the scaler in the model for later use
        self.lh = likelihood  # Store the likelihood in the model

        return self, likelihood, scaler # TODO: remove ideally

    def tune_gp(self, X_train, y_train, X_val, y_val, lr_list=(1e-2, 1e-1, 5e-1), iters_list=(200, 500, 800)):
        # tune learning rate and optimizer steps
        best_rmse   = np.inf
        best_params = None

        for lr in lr_list:
            for n_iters in iters_list:
                # train on training split
                model, likelihood, scaler = self.fit(
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
        final_model, final_likelihood, final_scaler = self.fit(
            X_train, y_train, lr=lr_opt, iters=iters_opt)

        #return final_model, final_likelihood, final_scaler


    """
    def fit2(self, train_x, train_y, likelihood, lr=0.01, iters=300):
        # scalar
        scaler = StandardScaler().fit(train_x)
        train_x_scaled = scaler.transform(train_x)

        tx = torch.tensor(train_x_scaled, dtype=torch.float32)
        ty = torch.tensor(train_y, dtype=torch.float32)

        self.train()
        likelihood.train()

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, self)

        for i in range(iters):
            optimizer.zero_grad()
            output = self(tx)
            loss = -mll(output, ty)
            loss.backward()
            optimizer.step()
            # print(f'Iter {i+1}/{iters} - Loss: {loss.item():.3f}')

        self.eval(); likelihood.eval()
        return self, likelihood, scaler
    """


