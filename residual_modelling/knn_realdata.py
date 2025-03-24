import numpy as np
import pickle
import pandas as pd

import rosbag2_py
import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt

# --- CONFIGURATION ---

n_steps = 300  # steps per episode
k = 5  # k neighbors
data_x = []  # sim pos, sim_vel, sim_acc, action
data_y = []  # Force rescaling (6D)

import os

rosbag_folder = "rosbag2_2024_12_08-18_10_08"
rosbag_file = "rosbag2_2024_12_08-18_10_08_0.db3"
rosbag_path = os.path.join(os.getcwd(), rosbag_folder, rosbag_file)  # Full path

print(f"Checking if rosbag exists at: {rosbag_path}")
if not os.path.exists(rosbag_path):
    raise FileNotFoundError(f"ERROR: ROS Bag file not found at {rosbag_path}")

# --- LOAD ROS BAG 2 ---
rclpy.init()
reader = rosbag2_py.SequentialReader()
storage_options = rosbag2_py.StorageOptions(uri=rosbag_path, storage_id="sqlite3")
converter_options = rosbag2_py.ConverterOptions(input_serialization_format="", output_serialization_format="")
reader.open(storage_options, converter_options)

# Find topic types
topic_types = reader.get_all_topics_and_types()
topic_type_map = {t.name: t.type for t in topic_types}

# Define expected topics
pose_topic = "/brov2heavy/pose"
control_topic = "/mavros/rc/override"

pose_msg_type = get_message(topic_type_map[pose_topic])
control_msg_type = get_message(topic_type_map[control_topic])

# --- EXTRACT DATA FROM ROSBAG ---
pose_data = []
control_data = []

while reader.has_next():
    (topic, data, t) = reader.read_next()
    msg_type = topic_type_map[topic]

    if topic == pose_topic:
        msg = deserialize_message(data, pose_msg_type)
        pose_data.append([t, msg.pose.position.x, msg.pose.position.y, msg.pose.position.z,
                          msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z])
    
    elif topic == control_topic:
        msg = deserialize_message(data, control_msg_type)
        control_data.append([t, msg.channels])

# Convert to Pandas DataFrame
pose_df = pd.DataFrame(pose_data, columns=["timestamp", "x", "y", "z", "qx", "qy", "qz"])
control_df = pd.DataFrame(control_data, columns=["timestamp", "channels"])

# --- PROCESS DATA ---
pose_df["timestamp"] = pd.to_numeric(pose_df["timestamp"])
control_df["timestamp"] = pd.to_numeric(control_df["timestamp"])

positions = pose_df[["x", "y", "z", "qx", "qy", "qz"]].values
control_inputs = np.vstack(control_df["channels"].values)  # Convert list of arrays to 2D array
control_inputs = control_inputs[:, :6]

#for i, pwm in enumerate(control_inputs):
   #print(f"Row {i}: {pwm}")

# Compute time differences
dt_pose = np.gradient(pose_df["timestamp"]) / 1e9  # Convert nanoseconds to seconds
#dt_control = np.gradient(control_df["timestamp"]) / 1e9

# Extract timestamps as NumPy arrays
control_timestamps = control_df["timestamp"].values
pose_timestamps = pose_df["timestamp"].values

aligned_positions = []

aligned_positions = []
valid_control_timestamps = []

for control_time in control_timestamps:
    idx = np.searchsorted(pose_timestamps, control_time, side="left")
    
    # Skip if index is out of bounds (control_time is beyond available pose timestamps)
    if idx >= len(positions):
        #print(f"Skipping control_time {control_time} - No valid pose timestamp available.")
        continue  # Skip this control input
    
    aligned_positions.append(positions[idx])
    valid_control_timestamps.append(control_time)  # Keep only valid timestamps
# Convert to NumPy arrays
aligned_positions = np.array(aligned_positions)
valid_control_timestamps = np.array(valid_control_timestamps)

dt_control = np.gradient(valid_control_timestamps) / 1e9

# Compute velocities and accelerations
velocities = np.gradient(aligned_positions, axis=0) / dt_control[:, None]
accelerations = np.gradient(velocities, axis=0) / dt_control[:, None]

scaled_controls = (control_inputs - 1500.0) / 400.0

#for i, pose in enumerate(aligned_positions):
#    print(f"Row {i}: {pose}")

for i, pwm in enumerate(scaled_controls):
    print(f"Row {i}: {pwm}")
    if i % 300 == 0:  # Pause at step 2
        print("Press Enter to continue...")
        input()


# --- INITIALIZE SIM ENVIRONMENT ---
env_prior_path = "envs/real_dynamic/prior_env/prior.x86_64"
env_sim = UnityEnvironment(file_name=env_prior_path, seed=1, worker_id=0, side_channels=[])
print("Loaded prior env!")
env_sim.reset()

# Retrieve behavior
behavior_name_sim = list(env_sim.behavior_specs.keys())[0]
behavior_spec_sim = env_sim.behavior_specs[behavior_name_sim]
num_agents = len(env_sim.get_steps(behavior_name_sim)[0])
action_size = behavior_spec_sim.action_spec.continuous_size 

# Initialize previous velocities
prev_sim_vel = np.zeros(6)

for step in range(min(n_steps, len(dt_control))):  # Ensure we don't exceed ROS data size
    # --- OBSERVE STATES ---
    env_sim.step()
    sim_steps, _ = env_sim.get_steps(behavior_name_sim)

    for agent_id in sim_steps.agent_id:
        # Extract velocity data from the simulation            
        sim_pos = sim_steps[agent_id].obs[0][:6]
        sim_vel = sim_steps[agent_id].obs[0][6:12]

        # Compute simulated acceleration
        sim_acc = (sim_vel - prev_sim_vel) / dt_control[step]

        # Get corresponding real-world acceleration from ROS dataset
        real_acc = accelerations[step]

        # Compute force rescale factor
        force_rescale = real_acc / (sim_acc + 1e-10)  # Avoid division by zero

        # Store data
        data_x.append(np.concatenate([sim_pos, sim_vel, sim_acc, scaled_controls[step]]))
        data_y.append(force_rescale)

        # Select action from aligned ROS data
        action_tuple = ActionTuple(continuous=np.array([scaled_controls[step]]))
        env_sim.set_actions(behavior_name_sim, action_tuple)

        print(f"\nStep {step+1}")
        print(f"Action: {scaled_controls[step]}")

        # Update previous velocity
        prev_sim_vel = sim_vel

env_sim.close()

# Save dataset
knn_x = np.array(data_x)
knn_y = np.array(data_y)
with open("knn_data.pkl", "wb") as f:
    pickle.dump((knn_x, knn_y), f)

print("\nTraining data prepared and saved!")

# --- TRAINING KNN ---
knn = KNeighborsRegressor(n_neighbors=k, weights='distance', algorithm='auto')
knn.fit(knn_x, knn_y)

print("KNN model trained!")

# --- TESTING KNN ---
print("\nTesting KNN with sample inputs...")
test_sample = knn_x[:1]  # Take the first sample
predicted_scaling = knn.predict(test_sample)

print("Original Input:", test_sample)
print("Predicted Force Scaling:", predicted_scaling)

# Plot results
plt.figure(figsize=(12, 6))
num_actions = aligned_positions.shape[1]
for i in range(num_actions):
    plt.scatter(knn_x[:, -num_actions + i], knn_y[:, i], alpha=0.5, label=f"Action {i+1}")

plt.xlabel("Control Input (PWM)")
plt.ylabel("Force Scaling Factor")
plt.title("Force Scaling Factor vs. Control Input")
plt.legend()
plt.grid()
plt.show()

print("Testing complete!")
