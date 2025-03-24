import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

import rosbag2_py
import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from sklearn.neighbors import KNeighborsRegressor


def load_rosbag(rosbag_path, pose_topic="/brov2heavy/pose", control_topic="/mavros/rc/override"):
    rclpy.init()
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=rosbag_path, storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions(input_serialization_format="", output_serialization_format="")
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    topic_type_map = {t.name: t.type for t in topic_types}

    pose_msg_type = get_message(topic_type_map[pose_topic])
    control_msg_type = get_message(topic_type_map[control_topic])

    pose_data, control_data = [], []

    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic == pose_topic:
            msg = deserialize_message(data, pose_msg_type)
            pose_data.append([t, msg.pose.position.x, msg.pose.position.y, msg.pose.position.z,
                              msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z])
        elif topic == control_topic:
            msg = deserialize_message(data, control_msg_type)
            control_data.append([t, msg.channels])

    pose_df = pd.DataFrame(pose_data, columns=["timestamp", "x", "y", "z", "qx", "qy", "qz"])
    control_df = pd.DataFrame(control_data, columns=["timestamp", "channels"])

    pose_df["timestamp"] = pd.to_numeric(pose_df["timestamp"])
    control_df["timestamp"] = pd.to_numeric(control_df["timestamp"])

    return pose_df, control_df


def align_data(pose_df, control_df):
    positions = pose_df[["x", "y", "z", "qx", "qy", "qz"]].values
    control_inputs = np.vstack(control_df["channels"].values)[:, :6]
    scaled_controls = (control_inputs - 1500.0) / 400.0

    control_ts = control_df["timestamp"].values
    pose_ts = pose_df["timestamp"].values

    aligned_positions = []
    valid_ts = []

    for t in control_ts:
        idx = np.searchsorted(pose_ts, t, side="left")
        if idx >= len(positions): continue
        aligned_positions.append(positions[idx])
        valid_ts.append(t)

    aligned_positions = np.array(aligned_positions)
    valid_ts = np.array(valid_ts)
    dt_control = np.gradient(valid_ts) / 1e9

    velocities = np.gradient(aligned_positions, axis=0) / dt_control[:, None]
    accelerations = np.gradient(velocities, axis=0) / dt_control[:, None]

    return aligned_positions, scaled_controls, dt_control, accelerations


def run_simulation(aligned_positions, scaled_controls, dt_control, accelerations, n_steps, env_path):
    data_x, data_y = [], []

    env_sim = UnityEnvironment(file_name=env_path, seed=1, worker_id=0, side_channels=[])
    env_sim.reset()

    behavior_name = list(env_sim.behavior_specs.keys())[0]
    spec = env_sim.behavior_specs[behavior_name]
    prev_sim_vel = np.zeros(6)

    for step in range(min(n_steps, len(dt_control))):
        env_sim.step()
        sim_steps, _ = env_sim.get_steps(behavior_name)

        for agent_id in sim_steps.agent_id:
            sim_obs = sim_steps[agent_id].obs[0]
            sim_pos = sim_obs[:6]
            sim_vel = sim_obs[6:12]
            sim_acc = (sim_vel - prev_sim_vel) / dt_control[step]
            real_acc = accelerations[step]
            force_rescale = real_acc / (sim_acc + 1e-10)

            data_x.append(np.concatenate([sim_pos, sim_vel, sim_acc, scaled_controls[step]]))
            data_y.append(force_rescale)

            action = ActionTuple(continuous=np.array([scaled_controls[step]]))
            env_sim.set_actions(behavior_name, action)

            prev_sim_vel = sim_vel

    env_sim.close()
    return np.array(data_x), np.array(data_y)


def train_knn(data_x, data_y, k):
    knn = KNeighborsRegressor(n_neighbors=k, weights='distance', algorithm='auto')
    knn.fit(data_x, data_y)
    return knn


def main():
    n_steps = 300
    k = 5

    rosbag_folder = "rosbag2_2024_12_08-18_10_08"
    rosbag_file = "rosbag2_2024_12_08-18_10_08_0.db3"
    rosbag_path = os.path.join(os.getcwd(), rosbag_folder, rosbag_file)

    if not os.path.exists(rosbag_path):
        raise FileNotFoundError(f"ROS Bag file not found at {rosbag_path}")

    print("Loading rosbag...")
    pose_df, control_df = load_rosbag(rosbag_path)

    print("Aligning data...")
    aligned_pos, scaled_ctrls, dt_ctrl, accels = align_data(pose_df, control_df)

    print("Running Unity simulation...")
    env_path = "envs/real_dynamic/prior_env2/prior.x86_64"
    data_x, data_y = run_simulation(aligned_pos, scaled_ctrls, dt_ctrl, accels, n_steps, env_path)

    print("Saving dataset...")
    with open("knn_data.pkl", "wb") as f:
        pickle.dump((data_x, data_y), f)

    print("Training KNN...")
    knn = train_knn(data_x, data_y, k)

    print("Testing KNN...")
    test_sample = data_x[:1]
    pred = knn.predict(test_sample)

    print("Sample input:", test_sample)
    print("Predicted scaling:", pred)

    plt.figure(figsize=(12, 6))
    for i in range(aligned_pos.shape[1]):
        plt.scatter(data_x[:, -aligned_pos.shape[1] + i], data_y[:, i], alpha=0.5, label=f"Action {i+1}")
    plt.xlabel("Control Input (PWM)")
    plt.ylabel("Force Scaling Factor")
    plt.title("Force Scaling Factor vs. Control Input")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
