import numpy as np
import pickle
import time
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple

# --- CONFIGURATION ---
env_prior_path = "envs/res_test_prior_v3/res_prior_v3.x86_64"
env_real_path = "envs/res_test_real_v3/res_real_v3.x86_64"

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
action_size = behavior_spec_sim.action_spec.continuous_size  # Expecting 6DOF forces

behavior_name_real = list(env_real.behavior_specs.keys())[0]
behavior_spec_real = env_real.behavior_specs[behavior_name_real]
print("ENV SIM")
print("Observation Shapes:", [obs.shape for obs in behavior_spec_sim.observation_specs])
print("Continuous Action Space:", behavior_spec_sim.action_spec.continuous_size)
print("Discrete Action Space:", behavior_spec_sim.action_spec.discrete_size)
print("\nENV REAL")
print("Observation Shapes:", [obs.shape for obs in behavior_spec_real.observation_specs])
print("Continuous Action Space:", behavior_spec_real.action_spec.continuous_size)
print("Discrete Action Space:", behavior_spec_real.action_spec.discrete_size)
# --- DATA COLLECTION ---
data_x = []  # Features: [state_diff (12D) + action (6D)]
data_y = []  # Target: Force rescaling (6D)

print("Starting simulation for data collection...")

for step in range(n_steps):
    actions = np.zeros((num_agents, action_size), dtype=np.float32)
    actions[:, 1] = 1
    # Apply actions to both simulations
    action_tuple = ActionTuple(continuous=actions)
    env_sim.set_actions(behavior_name_sim, action_tuple)
    env_real.set_actions(behavior_name_real, action_tuple)

    # Step both environments
    env_sim.step()
    env_real.step()

    # Retrieve updated states
    sim_steps, _ = env_sim.get_steps(behavior_name_sim)
    real_steps, _ = env_real.get_steps(behavior_name_real)

    prev_sim_vel= [0,0,0,0,0,0]
    prev_real_vel = [0,0,0,0,0,0]

    for agent_id in sim_steps.agent_id:
        # Extract 6DOF states: [px, py, pz, roll, pitch, yaw], velocities: [vx, vy, vz, ωx, ωy, ωz]
        sim_pos_u = sim_steps[agent_id].obs[0][:6]
        sim_vel_u = sim_steps[agent_id].obs[0][6:12]
        real_pos_u = real_steps[agent_id].obs[0][:6]
        real_vel_u = real_steps[agent_id].obs[0][6:12]
        
        # --- Convert Position & Rotation (Unity → NED) ---
        sim_pos = np.array([
        sim_pos_u[2], sim_pos_u[0], -sim_pos_u[1],   # Position [Z → X, X → Y, -Y → Z]
        -sim_pos_u[4], -sim_pos_u[3], sim_pos_u[5]    # z y x   -y -x z  
        ])
        real_pos = np.array([
        real_pos_u[2], real_pos_u[0], -real_pos_u[1],
        -real_pos_u[4], -real_pos_u[3], real_pos_u[5]
        ])

    # --- Convert Velocity & Angular Velocity (Unity → NED) ---
        sim_vel = np.array([
        sim_vel_u[2], sim_vel_u[0], -sim_vel_u[1],   # Velocity [Vz → X, Vx → Y, -Vy → Z]
        -sim_vel_u[4], -sim_vel_u[3], sim_vel_u[5]    # z y x   -y -x z  
        ])
        real_vel = np.array([
        real_vel_u[2], real_vel_u[0], -real_vel_u[1],
        -real_vel_u[4], -real_vel_u[3], real_vel_u[5]
        ])


        # Compute state difference
        state_diff = np.concatenate([sim_pos - real_pos, sim_vel - real_vel])

        # Compute real and simulated accelerations
        real_acc = (real_vel - prev_real_vel) / dt  # Avoid division by zero
        sim_acc = (sim_vel - prev_sim_vel) / dt  # Avoid division by zero

        # Compute force rescale factor
        force_rescale = real_acc / (sim_acc + 1e-10)  # Avoid division by zero

        # Store in dataset
        data_x.append(np.concatenate([state_diff, actions[agent_id]]))
        data_y.append(force_rescale)

        # Update previous velocities for next step
        prev_real_vel = real_vel
        prev_sim_vel = sim_vel

# Convert to NumPy arrays
data_x = np.array(data_x)
data_y = np.array(data_y)

# Save dataset
with open("residual_data_6dof.pkl", "wb") as f:
    pickle.dump((data_x, data_y), f)

print("Data collection complete!")

# --- NUMPY-BASED KNN IMPLEMENTATION ---
def knn_predict(queries, data_x, data_y, k=n_neigh):
    """
    Predicts force scaling factors for multiple queries using KNN.
    """
    queries = np.atleast_2d(queries)  # Ensure queries is a 2D array

    # Compute Euclidean distances between all queries and training samples
    distances = np.linalg.norm(data_x[None, :, :] - queries[:, None, :], axis=2)  # Shape: (num_queries, num_samples)

    # Get indices of the k-nearest neighbors
    knn_indices = np.argsort(distances, axis=1)[:, :k]  # Shape: (num_queries, k)

    # Gather distances and corresponding target values
    knn_distances = np.take_along_axis(distances, knn_indices, axis=1)  # Shape: (num_queries, k)
    knn_targets = np.take(data_y, knn_indices, axis=0)  # Shape: (num_queries, k, 6DOF)

    # Compute inverse distance weights
    weights = 1 / (knn_distances + 1e-10)  # Avoid division by zero
    weights /= np.sum(weights, axis=1, keepdims=True)  # Normalize weights

    # Compute weighted average of force scaling factors
    weighted_predictions = np.sum(knn_targets * weights[:, :, None], axis=1)  # Shape: (num_queries, 6DOF)

    return weighted_predictions
n=0
# --- CONTINUOUS PREDICTION LOOP ---
print("\n Running Continuous KNN Predictions...")
# Simulated batch of test inputs (3 samples per loop iteration)
test_pos_mismatch = np.array([
    [1, 0, 0, 0, 0, 0],  
    [10, 0, 0, 0, 0, 0],  
    [0, 0, 0, 0, 0, 0]
])  # Position mismatch (6DOF)

test_vel_mismatch = np.array([
    [0.2, 0, 0, 0, 0, 0],  
    [1, 0, 0, 0, 0, 0],  
    [0, 0, 0, 0, 0, 0]
])  # Velocity mismatch (6DOF)

test_force_input = np.array([
    [0.5, 0, 0, 0, 0, 0],  
    [1, 0, 0, 0, 0, 0],  
    [1, 0, 0, 0, 0, 0]
])  # Force input (6DOF)
for i in range(len(test_pos_mismatch)):
    # Combine test data
    test_inputs = np.hstack([test_pos_mismatch[i], test_vel_mismatch[i], test_force_input[i]])  # Shape: (num_queries, 18)

    #Predict force corrections for all test samples
    predicted_scaling = knn_predict(test_inputs, data_x, data_y)

    # Print results
    print(f"\n Prediction {i+1}:")
    print(f"  Position Mismatch: {test_pos_mismatch[i]}")
    print(f"  Velocity Mismatch: {test_vel_mismatch[i]}")
    print(f"  Predicted Force Scaling: {predicted_scaling}")

    # Simulate real-time loop (adjust as needed)
    time.sleep(0.05)

# --- CLOSE ENVIRONMENTS ---
env_sim.close()
env_real.close()
print("Simulation complete!")
