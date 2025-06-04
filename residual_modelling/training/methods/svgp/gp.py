#!/usr/bin/env python3

import torch, numpy as np, tqdm, matplotlib.pyplot as plt
from gpytorch.models import VariationalGP, ExactGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.means import ConstantMean
from gpytorch.kernels import MaternKernel, ScaleKernel, GaussianSymmetrizedKLKernel, InducingPointKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import VariationalELBO, PredictiveLogLikelihood, ExactMarginalLogLikelihood
import gpytorch.settings
from .convergence import ExpMAStoppingCriterion
#from gp_mapping.convergence import ExpMAStoppingCriterion
from sklearn.preprocessing import StandardScaler


class SVGP(VariationalGP):

    def __init__(self, n_inducing=400, n_inputs=18): # TODO: make this more dynamic. "dummy and override"

        # number of inducing points and optimisation samples
        assert isinstance(n_inducing, int)
        self.m = n_inducing

        self.n_inputs = n_inputs

        # variational distribution and strategy
        # NOTE: we put random normal dumby inducing points
        # here, which we'll change in self.fit
        vardist = CholeskyVariationalDistribution(self.m)
        varstra = VariationalStrategy(
            self,
            torch.randn((self.m, n_inputs)), #
            vardist,
            learn_inducing_locations=True
        )
        VariationalGP.__init__(self, varstra)

        # kernel — implemented in self.forward
        self.mean = ConstantMean()
        self.cov = MaternKernel(ard_num_dims=n_inputs)
        # self.cov = GaussianSymmetrizedKLKernel()
        self.cov = ScaleKernel(self.cov, ard_num_dims=n_inputs)

        self.scaler = None
        
        # likelihood
        self.likelihood = GaussianLikelihood()

        # hardware allocation
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.likelihood.to(self.device).float()
        self.to(self.device).float()

    def forward(self, input):
        m = self.mean(input)
        v = self.cov(input)
        return MultivariateNormal(m, v)

    #def fit(self, inputs, targets, covariances=None, n_samples=5000, max_iter=10000,
    #        learning_rate=1e-3, rtol=1e-4, n_window=100, auto=True, verbose=True):
    def fit(self, inputs, targets, covariances=None, n_samples=1000, max_iter=1000,
            learning_rate=1e-1, rtol=1e-12, n_window=200, auto=False, verbose=True):
        '''
        Optimises the hyperparameters of the GP kernel and likelihood.
        inputs: (nx2) numpy array
        targets: (n,) numpy array
        n_samples: number of samples to take from the inputs/targets at every optimisation epoch
        max_iter: maximum number of optimisation epochs
        learning_rate: optimiser step size
        rtol: change between -MLL values over ntol epoch that determine termination if auto==True
        ntol: number of epochs required to maintain rtol in order to terminate if auto==True
        auto: if True terminate based on rtol and ntol, else terminate at max_iter
        verbose: if True show progress bar, else nothing
        '''
        self.scaler = StandardScaler()
        inputs_scaled = self.scaler.fit_transform(inputs)

        # inducing points randomly distributed over data
        indpts = np.random.choice(inputs_scaled.shape[0], self.m, replace=True)
        self.variational_strategy.inducing_points.data = torch.from_numpy(inputs_scaled[indpts]).to(self.device).float()

        # number of random samples
        n = inputs_scaled.shape[0]
        n = n_samples if n >= n_samples else n

        # objective
        mll = VariationalELBO(self.likelihood, self, n, combine_terms=True)

        # stochastic optimiser
        opt = torch.optim.Adam(self.parameters(),lr=learning_rate)
        # opt = torch.optim.SGD(self.parameters(),lr=learning_rate)

        # convergence criterion
        # print("N window ", n_window)
        if auto: criterion = ExpMAStoppingCriterion(rel_tol=rtol, minimize=True, n_window=n_window)

        # episode iteratior
        epochs = range(max_iter)
        epochs = tqdm.tqdm(epochs) if verbose else epochs

        # train
        self.train()
        self.likelihood.train()
        self.loss = list()
        for _ in epochs:

            # randomly sample from the dataset
            idx = np.random.choice(inputs_scaled.shape[0], n, replace=False)


            ## UI covariance
            if covariances is None or covariances.shape[1] == 18:
                input = torch.from_numpy(inputs_scaled[idx]).to(self.device).float()
                target = torch.from_numpy(targets[idx]).to(self.device).float()

            # if the inputs are distributional, sample them
            if covariances is not None:
                covariance = torch.from_numpy(covariances[idx]).to(self.device).float()
                input = MultivariateNormal(input, covariance).rsample()

            # compute loss, compute gradient, and update
            loss = -mll(self(input), target)
            opt.zero_grad()
            loss.backward()
            opt.step()

            # verbosity and convergence check
            if verbose:
                epochs.set_description('Loss {:.4f}'.format(loss.item()))
                self.loss.append(loss.detach().cpu().numpy())
            if auto and criterion.evaluate(loss.detach()):
                break

    def sample(self, x):

        '''
        Samples the posterior at x
        x: (n,2) numpy array
        returns:
            mu: (n,) numpy array of predictive mean at x
            sigma: (n,) numpy array of predictive variance at x
        '''

        ## On your source code, call:
        # self.likelihood.eval()
        # self.eval()
        ## before using this function to toggle evaluation mode
        if self.scaler is None:
            raise ValueError("Scaler not initialized")
        # sanity
        assert len(x.shape) == 2 and x.shape[1] == 18, \
            f"Expected input shape (n, {self.n_inputs}), but got {x.shape}"

        x_scaled = self.scaler.transform(x)
        # sample posterior
        # TODO: fast_pred_var activates LOVE. Test performance on PF
        # https://towardsdatascience.com/gaussian-process-regression-using-gpytorch-2c174286f9cc
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            x_tensor = torch.from_numpy(x_scaled).to(self.device).float()
            dist = self.likelihood(self(x_tensor))
            return dist.mean.cpu().numpy(), dist.variance.cpu().numpy()

    def save_posterior(self, n, xlb, xub, ylb, yub, fname, verbose=True):

        '''
        Samples the GP posterior on a inform grid over the
        rectangular region defined by (xlb, xub) and (ylb, yub)
        and saves it as a pointcloud array.

        n: determines n² number of sampling locations
        xlb, xub: lower and upper bounds of x sampling locations
        ylb, yub: lower and upper bounds of y sampling locations
        fname: path to save array at (use .npy extension)
        '''

        # sanity
        assert('.npy' in fname)

        # toggle evaluation mode
        self.likelihood.eval()
        self.eval()
        torch.cuda.empty_cache()

        # posterior sampling locations
        inputs = [
            np.linspace(xlb, xub, n),
            np.linspace(ylb, yub, n)
        ]
        inputs = np.meshgrid(*inputs)
        inputs = [_.flatten() for _ in inputs]
        inputs = np.vstack(inputs).transpose()

        # split the array into smaller ones for memory
        inputs = np.split(inputs, 4000, axis=0)

        # compute the posterior for each batch
        means, variances = list(), list()
        with torch.no_grad():
            for i, input in enumerate(inputs):
                if verbose: print('Batch {}'.format(i))
                mean, variance = self.sample(input)
                means.append(mean)
                variances.append(variance)

        # assemble probabalistic pointcloud
        cloud = np.hstack((
            np.vstack(inputs),
            np.hstack(means).reshape(-1, 1),
            np.hstack(variances).reshape(-1, 1)
        ))

        # save it
        np.save(fname, cloud)

    def plot(self, inputs, targets, fname, n=80, n_contours=50, track=None):

        '''
        Plots:
            ax[0]: raw inputs and targets,
            ax[1]: posterior predictive mean,
            ax[2]: posterior predictive variance
        inputs: (n, n_inputs) numpy array of inputs
        output: (n,) numpy array of targets
        fname: path to save plot at (extension determines file type, e.g. .png or .pdf)
        n: determines n² number of sampling locations to plot GP posterior
        n_contours: number of contours to show output magnitude with
        '''

        # sanity
        assert inputs.shape[0] == targets.shape[0]
        assert inputs.shape[1] == self.n_inputs

        # toggle evaluation mode
        self.likelihood.eval()
        self.eval()
        torch.cuda.empty_cache()

        # posterior sampling locations
        inputsg = [
            np.linspace(min(inputs[:,0]), max(inputs[:,0]), n),
            np.linspace(min(inputs[:,1]), max(inputs[:,1]), n)
        ]
        inputst = np.meshgrid(*inputsg)
        s = inputst[0].shape
        inputst = [_.flatten() for _ in inputst]
        inputst = np.vstack(inputst).transpose()
        inputst = torch.from_numpy(inputst).to(self.device).float()

        # sample
        with torch.no_grad():
            outputs = self(inputst)
            outputs = self.likelihood(outputs)
            mean = outputs.mean.cpu().numpy().reshape(s)
            variance = outputs.variance.cpu().numpy().reshape(s)

        # plot raw, mean, and variance
        levels = np.linspace(min(targets), max(targets), n_contours)
        fig, ax = plt.subplots(3, sharex=True, sharey=True)
        cr = ax[0].scatter(inputs[:,0], inputs[:,1], c=targets, cmap='jet', s=0.4, edgecolors='none')
        cm = ax[1].contourf(*inputsg, mean, cmap='jet', levels=levels)
        cv = ax[2].contourf(*inputsg, variance, levels=n_contours)
        indpts = self.variational_strategy.inducing_points.data.cpu().numpy()
        ax[2].plot(indpts[:,0], indpts[:,1], 'ko', markersize=1, alpha=0.2)

        # colorbars
        fig.colorbar(cr, ax=ax[0])
        fig.colorbar(cm, ax=ax[1])
        fig.colorbar(cv, ax=ax[2])

        # formatting
        ax[0].set_aspect('equal')
        ax[0].set_title('Raw data')
        ax[0].set_ylabel('$y~[m]$')
        ax[1].set_aspect('equal')
        ax[1].set_title('Mean')
        ax[1].set_ylabel('$y~[m]$')
        ax[2].set_aspect('equal')
        ax[2].set_title('Variance')
        ax[2].set_xlabel('$x~[m]$')
        ax[2].set_ylabel('$y~[m]$')
        plt.tight_layout()

        if track != None:
            ax[0].plot(track[:,0], track[:,1], "-b", linewidth=0.2)

        # save
        fig.savefig(fname, bbox_inches='tight', dpi=1000)

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
        
    def save(self, fname):
        assert self.scaler is not None, "Scaler not initialized"
        if '.pth' not in fname:
            fname += '.pth'
        torch.save({
            "model_state": self.state_dict(),
            "scaler_mean": self.scaler.mean_,
            "scaler_scale": self.scaler.scale_,
            "n_inducing": self.m,
            "input_dim": self.n_inputs,
        },fname)

    @classmethod
    def load(cls, fname):
        chk = torch.load(fname, map_location="cpu")
        
        gp = cls(n_inducing=chk["n_inducing"], n_inputs=chk["input_dim"])
        gp.load_state_dict(chk["model_state"])
        gp.eval()
        gp.likelihood.eval()

        scaler = StandardScaler()
        scaler.mean_ = chk["scaler_mean"]
        scaler.scale_ = chk["scaler_scale"]
        gp.scaler = scaler

        return gp, scaler

    def predict(self, tx):
        '''
        Predicts the output for the given input tx.
        tx: (n, n_inputs) numpy array of inputs
        returns:
            yhat: (n,) numpy array of predicted outputs
        '''

        # sanity
        assert tx.shape[1] == self.n_inputs

        # toggle evaluation mode
        self.likelihood.eval()
        self.eval()
        torch.cuda.empty_cache()

        # predict
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            tx = torch.from_numpy(tx).to(self.device).float()
            yhat = self.likelihood(self(tx)).mean.cpu().numpy()

        return yhat

    # def save finns redan
