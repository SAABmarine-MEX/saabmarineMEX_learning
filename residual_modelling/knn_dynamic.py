import numpy as np
import pickle
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
env_prior_path = "envs/real_dynamic/prior_env/prior.x86_64"
env_real_path = "envs/real_dynamic/real_env/real.x86_64"

n_steps = 400  # Number of steps per episode
#dt = 1 / 20 
k = 5  # Number of neighbors in KNN
change_action = 50
data_x = []  # Features: [state_diff (12D) + action (6D)]
data_y = []  # Target: Force rescaling (6D)
n = 0
training_action_sequence = [[1, 0, 0.5, 0, 0, 0],
                            [0, 1, 0.5, 0, 0, 0],
                            [-1, 0, 0.5, 0, 0, 0],
                            [0, -1, 0.5, 0, 0, 0]]
test_action_sequence = [[0.8, 0, 0.5, 0, 0, 0],
                        [0, 0.8, 0.5, 0, 0, 0],
                        [-0.8, 0, 0.5, 0, 0, 0],
                        [0, -0.8, 0.5, 0, 0, 0]]

while n<1:
    prev_sim_vel= [0,0,0,0,0,0]
    prev_real_vel = [0,0,0,0,0,0]

    # --- INITIALIZE ENVIRONMENTS ---
    env_sim = UnityEnvironment(file_name=env_prior_path, seed=1, worker_id=0, side_channels=[])
    print("Loaded prior env!")
    env_real = UnityEnvironment(file_name=env_real_path, seed=1, worker_id=1, side_channels=[])
    print("Loaded real env!")

    env_sim.reset()
    env_real.reset()

    # Retrieve behavior
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

    print(f"\nStarting simulation {n} for data collection...")

    current_action_index = 0
    actions = np.tile(action_sequence[current_action_index], (num_agents, 1)).astype(np.float32)

    print(f"Current actions:\n{actions}")
    for step in range(n_steps):
        # Step
        env_sim.step()
        env_real.step()

        # States
        sim_steps, _ = env_sim.get_steps(behavior_name_sim)
        real_steps, _ = env_real.get_steps(behavior_name_real)

        for agent_id in sim_steps.agent_id:
            #pos: [px, py, pz, roll, pitch, yaw], vel: [vx, vy, vz, ωx, ωy, ωz]
            sim_pos_u = sim_steps[agent_id].obs[0][:6]
            sim_vel_u = sim_steps[agent_id].obs[0][6:12]
            real_pos_u = real_steps[agent_id].obs[0][:6]
            real_vel_u = real_steps[agent_id].obs[0][6:12]
            
            # --- Convert Position & Rotation (Unity → NED) ---
            sim_pos = np.array([
            sim_pos_u[2], sim_pos_u[0], -sim_pos_u[1],   
            sim_pos_u[5], sim_pos_u[3], -sim_pos_u[4]    # x y z = z x -y  
            ])
            real_pos = np.array([
            real_pos_u[2], real_pos_u[0], -real_pos_u[1],
            real_pos_u[5], real_pos_u[3], -real_pos_u[4]
            ])

        # --- Convert Velocity & Angular Velocity (Unity → NED) ---
            sim_vel = np.array([
            sim_vel_u[2], sim_vel_u[0], -sim_vel_u[1],   
            sim_vel_u[5], sim_vel_u[3], -sim_vel_u[4]    # x y z = z x -y     
            ])
            real_vel = np.array([
            real_vel_u[2], real_vel_u[0], -real_vel_u[1],
            real_vel_u[5], real_vel_u[3], -real_vel_u[4]
            ])

            # Skip first step to avoid large initial acceleration
            #if step > 0:

            # Compute state difference
            #state_diff = np.concatenate([sim_pos - real_pos, sim_vel - real_vel])
            #state = np.concatenate([sim_vel])
            # Compute real and simulated accelerations
            real_acc = (real_vel - prev_real_vel) #/ dt
            sim_acc = (sim_vel - prev_sim_vel)# / dt

            # Compute force rescale factor
            force_rescale = real_acc / (sim_acc + 1e-10)  # Avoid division by zero
            
            if step % change_action == 0:
                current_action_index = (current_action_index + 1) % len(training_action_sequence)
                actions[:] = np.tile(action_sequence[current_action_index], (num_agents, 1)).astype(np.float32)
                print(f"Current actions:\n{actions}")
                input()
                #Apply actions
            action_tuple = ActionTuple(continuous=actions)

            env_sim.set_actions(behavior_name_sim, action_tuple)
            env_real.set_actions(behavior_name_real, action_tuple)
            # Store in dataset
            if step > 0:
                data_x.append(np.concatenate([sim_vel, sim_acc, actions[agent_id]]))
                data_y.append(force_rescale)

            # Update previous velocities for next step
            prev_real_vel = real_vel
            prev_sim_vel = sim_vel

    # np arrays

    # split train test
    #test_size = 10
    #train_x, test_x = data_x[:-test_size], data_x[-test_size:]
    #train_y, test_y = data_y[:-test_size], data_y[-test_size:]

    #with open("residual_data_6dof_train.pkl", "wb") as f:
        #pickle.dump((knn_x, knn_y), f)

    #with open("residual_data_6dof_test.pkl", "wb") as f:
    #    pickle.dump((test_x, test_y), f)


    # Simulated batch of test inputs
    #test_inputs = test_x

    #predicted_scaling = knn_predict(test_inputs)

    #for i, pred in enumerate(predicted_scaling):
    # Print results
    #    print(f"\n Prediction {i+1}:")
    #    np.set_printoptions(precision=2, suppress=True)  # Limit to 2 decimals
    #   print(f"  Data_x: {test_inputs[i]}")
    #    print(f"  Data_y: {test_y[i]}")
    #    print(f"  Predicted Force Scaling: {pred}")


    # --- CLOSE ENVIRONMENTS ---
    env_sim.close()
    env_real.close()
    n=n+1
# Save full dataset
knn_x = np.array(data_x)
knn_y = np.array(data_y)
with open("residual_data_6dof_train.pkl", "wb") as f:
    pickle.dump((knn_x, knn_y), f)
knn = KNeighborsRegressor(n_neighbors=k, weights='distance', algorithm='auto', leaf_size=30)

# Fit KNN model to training data
knn.fit(knn_x, knn_y)
knn_predictions = knn.predict(knn_x)

num_actions = 6 
plt.figure(figsize=(12, 6))
for i in range(num_actions):
    plt.scatter(knn_x[:, -num_actions + i], knn_y[:, i], alpha=0.5, label=f"Action {i+1}")

plt.xlabel("Action Input")
plt.ylabel("Force Scaling Factor")
plt.title("Force Scaling Factor vs. Action Input")
plt.legend()
plt.grid()
plt.show()


def knn_predict(queries):
    queries = np.atleast_2d(queries)  # Ensure queries is a 2D array
    return knn.predict(queries)

print("\nTraining data collection complete! Full dataset saved.")


env_sim = UnityEnvironment(file_name=env_prior_path, seed=1, worker_id=0, side_channels=[])
print("Loaded sim knn env!")
env_real = UnityEnvironment(file_name=env_real_path, seed=1, worker_id=1, side_channels=[])
print("Loaded real env!")

env_sim.reset()
env_real.reset()

behavior_name_sim = list(env_sim.behavior_specs.keys())[0]
behavior_spec_sim = env_sim.behavior_specs[behavior_name_sim]
num_agents = len(env_sim.get_steps(behavior_name_sim)[0])
action_size = behavior_spec_sim.action_spec.continuous_size  # Expecting 6DOF forces

behavior_name_real = list(env_real.behavior_specs.keys())[0]
behavior_spec_real = env_real.behavior_specs[behavior_name_real]

prev_sim_vel= [0,0,0,0,0,0]
prev_real_vel = [0,0,0,0,0,0]
data_test = []
actions = np.random.uniform(-1, 1, (num_agents, action_size)).astype(np.float32)
for step in range(n_steps):
    # Step
    env_sim.step()
    env_real.step()

    # States
    sim_steps, _ = env_sim.get_steps(behavior_name_sim)
    real_steps, _ = env_real.get_steps(behavior_name_real)
    for agent_id in sim_steps.agent_id:
        #pos: [px, py, pz, roll, pitch, yaw], vel: [vx, vy, vz, ωx, ωy, ωz]
        sim_pos_u = sim_steps[agent_id].obs[0][:6]
        sim_vel_u = sim_steps[agent_id].obs[0][6:12]
        real_pos_u = real_steps[agent_id].obs[0][:6]
        real_vel_u = real_steps[agent_id].obs[0][6:12]
        
        # --- Convert Position & Rotation (Unity → NED) ---
        sim_pos = np.array([
        sim_pos_u[2], sim_pos_u[0], -sim_pos_u[1],   
        sim_pos_u[5], sim_pos_u[3], -sim_pos_u[4]    # x y z = z x -y  
        ])
        real_pos = np.array([
        real_pos_u[2], real_pos_u[0], -real_pos_u[1],
        real_pos_u[5], real_pos_u[3], -real_pos_u[4]
        ])

    # --- Convert Velocity & Angular Velocity (Unity → NED) ---
        sim_vel = np.array([
        sim_vel_u[2], sim_vel_u[0], -sim_vel_u[1],   
        sim_vel_u[5], sim_vel_u[3], -sim_vel_u[4]    # x y z = z x -y     
        ])
        real_vel = np.array([
        real_vel_u[2], real_vel_u[0], -real_vel_u[1],
        real_vel_u[5], real_vel_u[3], -real_vel_u[4]
        ])

        print("UNITY OBSERVED IN TEST")
        #if step > 0:

        # Compute state difference
        #state_diff = np.concatenate([sim_pos - real_pos, sim_vel - real_vel])
        #state = np.concatenate([sim_vel])
        # Compute real and simulated accelerations
        real_acc = (real_vel - prev_real_vel) #/ dt
        sim_acc = (sim_vel - prev_sim_vel)# / dt
        print("UNITY SAVED ACC")

        # Compute force rescale factor
        #force_rescale = real_acc / (sim_acc + 1e-10)  # Avoid division by zero

        # Compute force rescale factor
        force_rescale = real_acc / (sim_acc + 1e-10)  # Avoid division by zero
        if step % change_action == 0:
            current_action_index = (current_action_index + 1) % len(test_action_sequence)
            actions[:] = np.tile(test_action_sequence[current_action_index], (num_agents, 1)).astype(np.float32)
            
            print(f"Current actions:\n{actions}")

        #Apply actions
        knn_test = np.array(data_test)
        predicted_scaling = knn_predict(knn_test)

        # Apply rescaled actions to simulation
        corrected_action = actions[agent_id] * predicted_scaling
        action_tuple = ActionTuple(continuous=actions)
        env_sim.set_actions(behavior_name_sim, action_tuple)
        env_real.set_actions(behavior_name_real, action_tuple)
        # Store in dataset
        if step > 0:
            data_test.append(np.concatenate([sim_vel, sim_acc, corrected_action[agent_id]]))
            data_y.append(force_rescale)

        # Update previous velocities for next step
        prev_real_vel = real_vel
        prev_sim_vel = sim_vel


    print("TEST PREDICTION")

    knn_test = np.array(data_test)
    predicted_scaling = knn_predict(knn_test)

    # Apply rescaled actions to simulation
    corrected_action = actions[agent_id] * predicted_scaling

    # Print action comparison
    print(f"\nStep {step+1}")

    print(f"  Original Action: {actions[agent_id]}")
    print(f"  Predicted Scaling: {predicted_scaling}")
    print(f"  Corrected Action: {corrected_action}")

    action_tuple_knn = ActionTuple(continuous=corrected_action)
    action_tuple = ActionTuple(continuous=actions)
    env_sim.set_actions(behavior_name_sim, action_tuple_knn)
    env_real.set_actions(behavior_name_real, action_tuple)
    env_sim.step()
    env_real.step()
    # States
    sim_steps, _ = env_sim.get_steps(behavior_name_sim)
    real_steps, _ = env_real.get_steps(behavior_name_real)

env_sim.close()
env_real.close()


from sklearn.metrics import mean_absolute_error

# Define range of k values to test
k_values = range(1, 21)  # Testing k from 1 to 20
mae_scores = []

# Iterate through different k values
for k in k_values:
    knn = KNeighborsRegressor(n_neighbors=k, weights='distance')
    knn.fit(knn_x, knn_y)
    knn_predictions = knn.predict(knn_x)
    
    # Compute Mean Absolute Error (MAE)
    mae = mean_absolute_error(knn_y, knn_predictions)
    mae_scores.append(mae)


print("Testing knn complete!")

# --- Plot MAE vs. k ---
plt.figure(figsize=(10, 5))
plt.plot(k_values, mae_scores, marker='o', linestyle='-', color='b')
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Mean Absolute Error (MAE)")
plt.title("KNN Performance: MAE vs. k")
plt.grid()
plt.show()

print("Testing knn complete!")