import torch
import gpytorch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple

# --- CONFIGURATION ---
env_prior_path = "envs/real_dynamic/prior_env/prior.x86_64"
env_real_path = "envs/real_dynamic/real_env/real.x86_64"

n_steps = 300  # steps per episode
change_action = 20
n = 0
num_tasks = 6  # 6D force rescaling

# Data storage
data_x = []  # sim pos, sim_vel, sim_acc, action
data_y = []  # Force rescaling (6D)

# --- ENVIRONMENT SETUP & DATA COLLECTION ---
while n < 1:
    env_sim = UnityEnvironment(file_name=env_prior_path, seed=1, worker_id=0, side_channels=[])
    print("Loaded prior env!")
    env_real = UnityEnvironment(file_name=env_real_path, seed=1, worker_id=1, side_channels=[])
    print("Loaded real env!")
    
    env_sim.reset()
    env_real.reset()

    # Retrieve behavior
    behavior_name_sim = list(env_sim.behavior_specs.keys())[0]
    behavior_spec_sim = env_sim.behavior_specs[behavior_name_sim]
    behavior_name_real = list(env_real.behavior_specs.keys())[0]
    behavior_spec_real = env_real.behavior_specs[behavior_name_real]
    num_agents = len(env_sim.get_steps(behavior_name_sim)[0])
    action_size = num_tasks

    actions = np.random.uniform(-1, 1, (num_agents, action_size)).astype(np.float32)

    prev_sim_vel = np.zeros(6)
    prev_real_vel = np.zeros(6)

    print(f"\nStarting simulation {n+1} for data collection...")
    for step in range(n_steps):
        env_sim.step()
        env_real.step()
        
        sim_steps, _ = env_sim.get_steps(behavior_name_sim)
        real_steps, _ = env_real.get_steps(behavior_name_real)
        
        for agent_id in sim_steps.agent_id:

            sim_pos = sim_steps[agent_id].obs[0][:6]
            real_pos = real_steps[agent_id].obs[0][:6]
            sim_vel = sim_steps[agent_id].obs[0][6:12]
            real_vel = real_steps[agent_id].obs[0][6:12]

            sim_acc = sim_vel - prev_sim_vel
            real_acc = real_vel - prev_real_vel

            force_rescale = real_acc / (sim_acc + 1e-10)
            
            data_x.append(np.concatenate([sim_pos, sim_vel, sim_acc, actions[agent_id]]))
            data_y.append(force_rescale)

            #if step % change_action == 0:
                #actions = np.random.uniform(-1, 1, (num_agents, action_size)).astype(np.float32)
                #print(f"Step {step}: Updated Actions:\n{actions}")
            action_tuple = ActionTuple(continuous=actions)
            env_sim.set_actions(behavior_name_sim, action_tuple)
            env_real.set_actions(behavior_name_real, action_tuple)

            print(f"\nStep {step+1}")
            print(f"Action: {actions}")   

            prev_real_vel = real_vel
            prev_sim_vel = sim_vel

    env_sim.close()
    env_real.close()
    n += 1

# Convert to tensors
train_x = torch.tensor(np.array(data_x), dtype=torch.float32)
train_y = torch.tensor(np.array(data_y), dtype=torch.float32)

# --- MULTITASK GP MODEL ---
class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(gpytorch.means.ConstantMean(), num_tasks=num_tasks)
        self.covar_module = gpytorch.kernels.MultitaskKernel(gpytorch.kernels.RBFKernel(), num_tasks=num_tasks, rank=1)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)
model = MultitaskGPModel(train_x, train_y, likelihood)

# TRAIN GP MODEL
model.train()
likelihood.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(50):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    optimizer.step()
    print(f'Iter {i+1}/50 - Loss: {loss.item():.3f}')

print("\nTraining completed!")

# TEST GP MODEL
print("\nTESTING GP.")
env_sim = UnityEnvironment(file_name=env_prior_path, seed=1, worker_id=0, side_channels=[])
env_real = UnityEnvironment(file_name=env_real_path, seed=1, worker_id=1, side_channels=[])
env_sim.reset()
env_real.reset()

prev_sim_vel = np.zeros(6)
prev_real_vel = np.zeros(6)

actions = np.random.uniform(-1, 1, (num_agents, action_size)).astype(np.float32)
model.eval()
likelihood.eval()

for step in range(n_steps):
    env_sim.step()
    env_real.step()
    
    sim_steps, _ = env_sim.get_steps(behavior_name_sim)
    real_steps, _ = env_real.get_steps(behavior_name_real)
    
    for agent_id in sim_steps.agent_id:
        sim_pos = sim_steps[agent_id].obs[0][:6]
        real_pos = real_steps[agent_id].obs[0][:6]
        sim_vel = sim_steps[agent_id].obs[0][6:12]
        real_vel = real_steps[agent_id].obs[0][6:12]

        sim_acc = (sim_vel - prev_sim_vel)
        real_acc = (real_vel - prev_real_vel)
        
        test_x = torch.tensor(np.concatenate([sim_pos, sim_vel, sim_acc, actions[agent_id]]), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            predicted_scaling = likelihood(model(test_x)).mean.numpy()
        corrected_action = actions * predicted_scaling
        
        print(f"\nStep {step+1}")
        print(f"  Original Action: {actions}")
        #print(f"  Knn Data: {knn_test}")
        print(f"  Predicted Scaling: {predicted_scaling}")
        print(f"  Corrected Action: {corrected_action}")

        action_tuple_knn = ActionTuple(continuous=corrected_action)
        action_tuple = ActionTuple(continuous=actions)
        env_sim.set_actions(behavior_name_sim, action_tuple_knn)
        env_real.set_actions(behavior_name_real, action_tuple)
        
        prev_sim_vel = sim_vel
        prev_real_vel = real_vel

env_sim.close()
env_real.close()
print("Testing GP complete!")
