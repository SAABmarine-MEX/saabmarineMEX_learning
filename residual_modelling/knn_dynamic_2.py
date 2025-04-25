import numpy as np
import pickle
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
env_prior_path = "envs/real_dynamic/prior_env3/prior.x86_64"
env_real_path = "envs/real_dynamic/real_env3/real.x86_64"

n_steps = 360  # steps per episode
k = 5  # k neigbours
change_action = 30
data_x = []  # sim pos(ROT ONLY), sim_vel, sim_acc, action
data_y = []  # Force rescaling (6D)
n = 0


# z, x, -y = x, y, z
train_action_sequence = [[0, 0, 0, 0, 0, 0],
                        [0.8, 0, 0, 0, 0, 0],
                        [0, 0.8, 0, 0, 0, 0],
                        [-0.8, 0, 0, 0, 0, 0],
                        [0, -0.8, 0, 0, 0, 0],
                        [0.5, 0, 0, 0, 0, 0.2],
                        [0.5, 0, 0, 0, 0, 0.2],
                        [0, -1, 0.1, 0, 0, 0],
                        [0, 1, -0.1, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 1]]
test_action_sequence = [[0, 0, 0, 0, 0, 0],
                        [0.5, 0, 0, 0, 0, 0],
                        [0, 0.5, 0, 0, 0, 0],
                        [-0.5, 0, 0, 0, 0, 0],
                        [0, -0.5, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0.5, 0, 0],
                        [0, 0, 0, 0, 0.5, 0],
                        [0, 0, 0, 0, 0, 0.5],
                        [0, 0, 0, 0, 0, 0],
                        [0.3, 0, 0, 0, 0, 0.3],
                        [0.3, 0, 0, 0, 0, 0.3]]

while n < 1:
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
    #action_size = behavior_spec_sim.action_spec.continuous_size  # Expecting 6DOF forces

    behavior_name_real = list(env_real.behavior_specs.keys())[0]
    #behavior_spec_real = env_real.behavior_specs[behavior_name_real]

    print(f"\nStarting simulation {n+1} for data collection...")

    # Initialize actions 

    current_action_index = 0
    actions = np.tile(train_action_sequence[current_action_index], (num_agents, 1)).astype(np.float32)
    print(f"Current actions:\n{actions}")

    prev_sim_vel = np.zeros(6)
    prev_real_vel = np.zeros(6)

    for step in range(n_steps):
        # --- OBSERVE STATES FIRST ---
        env_sim.step()
        env_real.step()

        sim_steps, _ = env_sim.get_steps(behavior_name_sim)
        real_steps, _ = env_real.get_steps(behavior_name_real)

        for agent_id in sim_steps.agent_id:
            # Extract velocity data            
            sim_pos = sim_steps[agent_id].obs[0][3:6]
            #real_pos = real_steps[agent_id].obs[0][3:6]
            sim_vel = sim_steps[agent_id].obs[0][6:12]
            real_vel = real_steps[agent_id].obs[0][6:12]

            # Compute real and simulated accelerations
            real_acc = (real_vel - prev_real_vel)
            sim_acc = (sim_vel - prev_sim_vel)

            # Compute force rescale factor
            force_rescale = real_acc / (sim_acc + 1e-10)  # Avoid division by zero

            # Store data
            data_x.append(np.concatenate([sim_vel, sim_acc, actions[agent_id]]))#
            data_y.append(force_rescale)

            #Change actions
            '''if step % change_action == 0:
                actions = np.random.uniform(-1, 1, (num_agents, action_size)).astype(np.float32)
                print(f"Step {step}: Updated Actions:\n{actions}")'''
            if step % change_action == 0:
                current_action_index = (current_action_index + 1) % len(train_action_sequence)
                actions[:] = np.tile(train_action_sequence[current_action_index], (num_agents, 1)).astype(np.float32)
                print(f"\nStep {step+1}")
                print(f"Current actions:\n{actions}")
            # Send actions to the environments
            action_tuple = ActionTuple(continuous=actions)
            env_sim.set_actions(behavior_name_sim, action_tuple)
            env_real.set_actions(behavior_name_real, action_tuple)

            # Update previous velocities for next step
            prev_real_vel = real_vel
            prev_sim_vel = sim_vel

    env_sim.close()
    env_real.close()
    n += 1

# Save full dataset
data_x = np.array(data_x)
data_y = np.array(data_y)
#with open("knn_data.pkl", "wb") as f:
#    pickle.dump((knn_x, knn_y), f)

#print("\nTraining data collection complete! Full dataset saved.")

#Plot
num_actions = 6
plt.figure(figsize=(12, 6))
for i in range(num_actions):
    plt.scatter(data_x[:, -num_actions + i], data_y[:, i], alpha=0.5, label=f"Action {i+1}")

plt.xlabel("Action Input")
plt.ylabel("Force Scaling Factor")
plt.title("Force Scaling Factor vs. Action Input")
plt.legend()
plt.grid()
plt.show()

# --- TRAINING KNN ---
#scaler = StandardScaler()
#data_x_scaled = scaler.fit_transform(data_x)  
knn = KNeighborsRegressor(n_neighbors=k, weights='distance', algorithm='auto', leaf_size=30)
knn.fit(data_x, data_y)

# --- TESTING KNN ---
print("\nTESTING KNN.")
env_sim = UnityEnvironment(file_name=env_prior_path, seed=1, worker_id=0, side_channels=[])
env_real = UnityEnvironment(file_name=env_real_path, seed=1, worker_id=1, side_channels=[])
env_sim.reset()
env_real.reset()

prev_sim_vel = np.zeros(6)
prev_real_vel = np.zeros(6)
#data_test = []
#data_test = np.array(data_test)
current_action_index = 0
actions = np.tile(test_action_sequence[current_action_index], (num_agents, 1)).astype(np.float32)
print(f"Current actions:\n{actions}")
#actions = np.random.uniform(-1, 1, (num_agents, action_size)).astype(np.float32)
for step in range(n_steps):
    # Step environments
    env_sim.step()
    env_real.step()

    sim_steps, _ = env_sim.get_steps(behavior_name_sim)
    real_steps, _ = env_real.get_steps(behavior_name_real)

    for agent_id in sim_steps.agent_id:
        sim_pos = sim_steps[agent_id].obs[0][3:6]
        real_pos = real_steps[agent_id].obs[0][3:6]
        sim_vel = sim_steps[agent_id].obs[0][6:12]
        real_vel = real_steps[agent_id].obs[0][6:12]


        real_acc = (real_vel - prev_real_vel)
        sim_acc = (sim_vel - prev_sim_vel)

        data_test = np.array(np.concatenate([sim_vel, sim_acc, actions[agent_id]]))#
        prev_real_vel = real_vel
        prev_sim_vel = sim_vel

        '''if step % change_action == 0:
            actions = np.random.uniform(-1, 1, (num_agents, action_size)).astype(np.float32)
            print(f"Current actions:\n{actions}")'''
        if step % change_action == 0:
            current_action_index = (current_action_index + 1) % len(test_action_sequence)
            actions[:] = np.tile(test_action_sequence[current_action_index], (num_agents, 1)).astype(np.float32)
            
            

        #scaler = StandardScaler()
        data_test = np.array(data_test).reshape(1, -1)  # Ensure 2D shape (1, features)
        #data_test_scaled = scaler.fit_transform(data_test)
        predicted_scaling = knn.predict(data_test)
        corrected_action = actions * predicted_scaling
        #corrected_action = actions + predicted_scaling

        print(f"\nStep {step+1}")
        print(f"  Original Action: {actions}")
        #print(f"  Knn Data: {knn_test}")
        print(f"  Predicted Scaling: {predicted_scaling}")
        print(f"  Corrected Action: {corrected_action}")

        action_tuple_knn = ActionTuple(continuous=corrected_action)
        action_tuple = ActionTuple(continuous=actions)
        env_sim.set_actions(behavior_name_sim, action_tuple_knn)
        env_real.set_actions(behavior_name_real, action_tuple)

env_sim.close()
env_real.close()

print("Testing KNN complete!")
