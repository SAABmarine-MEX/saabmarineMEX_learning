import numpy as np
import matplotlib.pyplot as plt
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
        
        self.scaler = StandardScaler()
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)

        self.device = torch.device('cpu')
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
        x_scaled = self.scaler.transform(data_x)
        x_t = torch.from_numpy(x_scaled).to(self.device).float()
        self.eval(); self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            y_dist = self.likelihood(self(x_t))
            y = y_dist.mean.cpu().detach().numpy()
        return y


    def save(self, path):
        if not path.endswith('.pth'):
            path += '.pth'
        # Save the model state
        checkpoint = {
            "model_state": self.state_dict(), 
            "likelihood_state": self.likelihood.state_dict(),
            "scaler_mean": self.scaler.mean_,
            "scaler_scale": self.scaler.scale_,
            #"num_tasks": self.likelihood.num_tasks,
        }
        torch.save(checkpoint, path)


    def fit(self, data_x, data_y, lr=0.1, iters=500, w=np.array([1]*6 +[1]*6+[1]*6, dtype=np.float32)):
        # Override the dummy data

        #scalar
        scaler = StandardScaler().fit(data_x)
        data_x_scaled = scaler.transform(data_x)
        data_x_scaled = data_x_scaled * w

        tx = torch.tensor(data_x_scaled, dtype=torch.float32)
        ty = torch.tensor(data_y, dtype=torch.float32)

        self.set_train_data(inputs=tx, targets=ty, strict=False)


        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=data_y.shape[1])

        self.train()
        likelihood.train()
        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, self)

        self.loss = list()
        for i in range(iters):

            optimizer.zero_grad()
            output = self(tx)
            loss = -mll(output, ty)
            loss.backward()
            optimizer.step()

            self.loss.append(loss.item())

            print(f'Iter {i+1}/{iters} - Loss: {loss.item():.3f}') 

        self.eval(); likelihood.eval()
        self.scaler = scaler  # Store the scaler in the model for later use
        self.likelihood = likelihood  # Store the likelihood in the model

    def plot_loss(self, fname):

        # plot
        fig, ax = plt.subplots(1)
        ax.plot(self.loss, 'k-')

        # format
        ax.set_xlabel('Iteration')
        ax.set_ylabel('ELBO')
        ax.set_yscale('log')
        plt.tight_layout()

        # save
        fig.savefig(fname, bbox_inches='tight', dpi=1000)

    @classmethod
    def load(cls, fname):
        chk = torch.load(fname, map_location="cpu")
        
        gp = cls()
        gp.load_state_dict(chk["model_state"])
        gp.eval()
        gp.likelihood.eval()

        scaler = StandardScaler()
        scaler.mean_ = chk["scaler_mean"]
        scaler.scale_ = chk["scaler_scale"]
        gp.scaler = scaler

        return gp, scaler

