import numpy as np
import pickle
from sklearn.neighbors import KNeighborsRegressor
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple

# --- CONFIGURATION ---
env_prior_path = "envs/res_test_prior/res_prior.x86_64"
env_real_path = "envs/res_test_real/res_real.x86_64"

n_steps = 10  # Number of steps per episode
dt = 1 / 50  # Time step (assuming 50Hz simulation)
n_neigh = 5  # Number of neighbors in KNN

# --- INITIALIZE ENVIRONMENTS ---
env_sim = UnityEnvironment(file_name=env_prior_path, seed=1, worker_id=0, side_channels=[])
print("Loaded prior env!")
env_real = UnityEnvironment(file_name=env_real_path, seed=1, worker_id=1, side_channels=[])
print("Loaded real env!")

env_sim.reset()
env_real.reset()

# Retrieve behavior name
behavior_name_sim = list(env_sim.behavior_specs.keys())[0]
behavior_spec_sim = env_sim.behavior_specs[behavior_name_sim]
num_agents = len(env_sim.get_steps(behavior_name_sim)[0])
action_size = behavior_spec_sim.action_spec.continuous_size  # Expecting 2D force (2 actions: fx, fy)

behavior_name_real = list(env_real.behavior_specs.keys())[0]
behavior_spec_real = env_real.behavior_specs[behavior_name_real]


# --- DATA COLLECTION ---
data_x = []  # Features: [state_diff (4D) + action (2D)]
data_y = []  # Target: Force rescaling (2D)


print("ENV SIM")
print("Observation Shapes:", [obs.shape for obs in behavior_spec_sim.observation_specs])
print("Continuous Action Space:", behavior_spec_sim.action_spec.continuous_size)
print("Discrete Action Space:", behavior_spec_sim.action_spec.discrete_size)
print("\nENV REAL")
print("Observation Shapes:", [obs.shape for obs in behavior_spec_real.observation_specs])
print("Continuous Action Space:", behavior_spec_real.action_spec.continuous_size)
print("Discrete Action Space:", behavior_spec_real.action_spec.discrete_size)



print("Starting simulation for data collection...")

for step in range(n_steps):
    # Generate random 6DOF actions (scaled between -1 and 1)
    # actions = np.random.uniform(-1, 1, (num_agents, action_size)).astype(np.float32)
    # Initialize actions with zeros
    actions = np.zeros((num_agents, action_size), dtype=np.float32)
    # Set only the X-direction (first component) to +1
    actions[:, 1] = 1  # Assuming the first component represents force in X direction

    # Apply actions to both simulations
    action_tuple = ActionTuple(continuous=actions)
    print("Actions: ", actions)
    env_sim.set_actions(behavior_name_sim, action_tuple)
    env_real.set_actions(behavior_name_real, action_tuple)

    # Step both environments
    env_sim.step()
    env_real.step()

    # Retrieve updated states
    sim_steps, _ = env_sim.get_steps(behavior_name_sim)
    real_steps, _ = env_real.get_steps(behavior_name_real)

    for agent_id in sim_steps.agent_id:
        # Extract state (Position: [px, py], Velocity: [vx, vy])

        sim_pos = sim_steps[agent_id].obs[0][:2]
        sim_vel = sim_steps[agent_id].obs[0][2:4]
        real_pos = real_steps[agent_id].obs[0][:2]
        real_vel = real_steps[agent_id].obs[0][2:4]



        # Compute state difference
        state_diff = np.concatenate([sim_pos - real_pos, sim_vel - real_vel])

        #Compute missing acceleration (residual acceleration correction)
        force_rescale = (real_vel - sim_vel) / dt

        # Store in dataset
        data_x.append(np.concatenate([state_diff, actions[agent_id]]))
        data_y.append(force_rescale)

data_x = [[0.1,0,0.1,0,1,0],[0.3,0,0.2,0,1,0],[0.6,0,0.3,0,1,0],[1.0,0,0.4,0,1,0],[1.5,0,0.5,0,1,0],[2.1,0,0.6,0,1,0],[2.8,0,0.7,0,1,0],[3.6,0,0.8,0,1,0],[4.5,0,0.9,0,1,0],[5.5,0,1.0,0,1,0]]
data_y = [[-5,0],[-10,0],[-15,0],[-20,0],[-25,0],[-30,0],[-35,0],[-40,0],[-45,0],[-50,0]]
# print
for i, (features, target) in enumerate(zip(data_x, data_y)):
    print(f"Sample {i + 1}:")
    px, py = features[0:2]
    vx, vy = features[2:4]
    fx, fy = target 
    print(f"  Position:   px={px:.3f}, py={py:.3f}")
    print(f"  Velocity:   vx={vx:.3f}, vy={vy:.3f}")
    print(f"  Correction: fx={fx:.3f}, fy={fy:.3f}")
    print("-" * 80)  # Separator for readability


# Save dataset
with open("residual_data_6dof.pkl", "wb") as f:
    pickle.dump((data_x, data_y), f)

print("Data collection complete!")

# --- TRAIN KNN MODEL ---
data_x = np.array(data_x)  # Shape: (N, 6) -> 4 state diff + 2 action inputs
data_y = np.array(data_y)  # Shape: (N, 2) -> Force rescaling (fx, fy)

knn_model = KNeighborsRegressor(n_neighbors=n_neigh, weights="distance")
knn_model.fit(data_x, data_y)

# Save trained model
with open("knn_model_6dof.pkl", "wb") as f:
    pickle.dump(knn_model, f)

print("KNN model trained and saved!")
# --- PRINT FORCE RESCALING RECOMMENDATIONS ---

print("\nForce Rescaling Predictions from KNN:")
for i, features in enumerate(data_x):
    px, py = features[:2]  # Position differences
    vx, vy = features[2:4]  # Velocity differences
    fx, fy = features[4:6]  # Original applied forces

    # 🔥 Use the trained KNN model to predict force rescaling
    predicted_rescale = knn_model.predict(features.reshape(1, -1))[0]  # Predict new force scaling
    rescale_fx, rescale_fy = predicted_rescale  # Extract force components

    print(f"Sample {i + 1}:")
    print(f"  Position Diff:   px={px:.3f}, py={py:.3f}")
    print(f"  Velocity Diff:   vx={vx:.3f}, vy={vy:.3f}")
    print(f"  Original Forces: fx={fx:.3f}, fy={fy:.3f}")
    print(f"  KNN Predicted Scaling: fx*{rescale_fx:.3f}, fy*{rescale_fy:.3f}")
    print("-" * 80)

# --- CLOSE ENVIRONMENTS ---
env_sim.close()
env_real.close()
print("Simulation complete!")
