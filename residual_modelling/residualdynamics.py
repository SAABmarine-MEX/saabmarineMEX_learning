#---imports---

import os
import glob
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

import torch
import gpytorch

import rosbag2_py

from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message

from brov_msgs.msg import SyncedPoseControl
from brov_msgs.msg import StampedRcOverride
from mavros_msgs.msg import OverrideRCIn

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple

from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from scipy.spatial.transform import Rotation as R

#----------------------------------------------------

def get_rosbag_paths(parent_folder):
    rosbag_paths = []

    # Loop over each subfolder inside the parent folder
    for bag_dir in os.listdir(parent_folder):
        full_path = os.path.join(parent_folder, bag_dir)
        if os.path.isdir(full_path):
            # Look for .db3 files in the subfolder
            db3_files = glob.glob(os.path.join(full_path, "*.db3"))
            if db3_files:
                rosbag_paths.append(db3_files[0])  # Assume first .db3 file is the main one

    rosbag_paths.sort()
    return rosbag_paths

#--------------------------------------------------------

def load_rosbag(rosbag_path, synced_topic="/synced_pose_control"):
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=rosbag_path, storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr")
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    topic_type_map = {t.name: t.type for t in topic_types}

    if synced_topic not in topic_type_map:
        raise ValueError(f"Topic {synced_topic} not found in the bag file.")

    data = []
    found_first_action = False  #flag

    while reader.has_next():
        topic, raw_data, t = reader.read_next()
        if topic == synced_topic:
            msg = deserialize_message(raw_data, SyncedPoseControl)

            channels = list(msg.rc_stamped.rc.channels)[:6]
            if not found_first_action:
                # Check if all 6 channels are zero
                if all(ch == 1500 for ch in channels):
                    continue  # still in the skip phase
                else:
                    found_first_action = True  

            # Extract pose
            position = msg.pose.pose.position
            orientation = msg.pose.pose.orientation
            linvel = msg.twist.twist.linear
            angvel = msg.twist.twist.angular

            #channels = [channels[i] for i in [4, 5, 2, 1, 0, 3]]  # pitch, roll, z, yaw, x, y

            data.append([
                t,
                position.x, position.y, position.z,
                orientation.x, orientation.y, orientation.z, orientation.w,
                linvel.x, linvel.y, linvel.z,
                angvel.x,angvel.y,angvel.z,
                channels
            ])

    df = pd.DataFrame(data, columns=[
        "timestamp", 
        "x", "y", "z", "qx", "qy", "qz", "qw",
        "vx","vy","vz","wx","wy","wz",
        "channels"
    ])
    print(df)

    df["timestamp"] = pd.to_numeric(df["timestamp"])
    return df
#------------------------------

def process_data(df):
    pos = df[["x", "y", "z"]].values
    pos_zero = pos - pos[0:1, :]  
    linvel = df[["vx","vy","vz"]].values
    angvel = df[["wx","wy","wz"]].values    

    quaternions = df[["qx", "qy", "qz", "qw"]].values
    timestamps = df["timestamp"].values.astype(np.float64)

    control_inputs = np.vstack(df["channels"].values)
    scaled_controls = (control_inputs - 1500.0) / 400.0
    scaled_controls[2] *=-1
    # Time delta in seconds
    dt = np.gradient(timestamps) / 1e9
    
    # Steps between each data point (assuming nominal step = 0.02s)
    dt_steps = np.round(dt / 0.02).astype(int)

    # Print timing info
    #print("Time between control steps (in seconds):")
    #print(dt)
    print(f"\nMean dt: {np.mean(dt):.6f} s")
    print(f"Min dt: {np.min(dt):.6f} s")
    print(f"Max dt: {np.max(dt):.6f} s")
    input()

    # Linear kinematics
    #linear_vel_world = np.gradient(pos, axis=0) / dt[:, None]

    # Angular kinematics from quaternion → euler
    rotations = R.from_quat(quaternions)
    euler_angles = rotations.as_euler('xyz', degrees=False)
    euler_zero   = euler_angles - euler_angles[0:1, :]


    linear_vel_local = rotations.inv().apply(linvel)
    linear_acc = np.gradient(linear_vel_local, axis=0) / dt[:, None]


    #angular_vel = np.gradient(euler_angles, axis=0) / dt[:, None]
    angular_acc = np.gradient(angvel, axis=0) / dt[:, None]
        # Combine into single arrays

    #acc = np.gradient(vel, axis=0) / dt[:, None]

    positions = np.hstack([pos_zero, euler_zero])   # (N, 6)
    vel = np.hstack([linear_vel_local, angvel])   # (N, 6)
    acc = np.hstack([linear_acc, angular_acc])   # (N, 6)
    
    for i in range(len(df)):
        print(f"\nControls: {scaled_controls[i]}")
        print(f"\nPositions: {positions[i]}")
        print(f"\nAccelerations: {acc[i]}")
        #if i % 100 == 0:
            #input()
    #input()
    return {
        "positions": positions,
        "scaled_controls": scaled_controls,
        "dt": dt,
        "steps": dt_steps,
        "velocities": vel,
        #"euler_angles": euler_angles,
        "accelerations": acc,
        "timestamps": timestamps
    }

#--------------------------------

def train_multitask_gp(train_x, train_y, num_tasks, lr=0.1, iters=500):
    class MultitaskGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super().__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.MultitaskMean(
                gpytorch.means.ConstantMean(), num_tasks=num_tasks
            )
            self.covar_module = gpytorch.kernels.MultitaskKernel(
                gpytorch.kernels.RBFKernel(), num_tasks=num_tasks, rank=1
            )

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)
    model = MultitaskGPModel(train_x, train_y, likelihood)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(iters):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
        print(f'Iter {i+1}/{iters} - Loss: {loss.item():.3f}') 

    return model, likelihood

#--------------------------------------

def train_knn(data_x, data_y, k):
    pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("knn",    KNeighborsRegressor(n_neighbors=5, weights="distance"))
    ])
    pipeline.fit(data_x, data_y)
    # scale, TODO check if better scaling is needed

    return pipeline

#--------------------------------------
def run_simulation(scaled_controls, dt, dt_steps, pos, vel, accelerations, n_steps, env_path, sim_timestep=0.02):
    data_x, data_y = [], []
    sim_pos = []
    sim_vel = []
    sim_acc = []
    real_pos = []
    real_vel = []
    real_acc = []
    rc=[]


    env_sim = UnityEnvironment(file_name=env_path, seed=1, worker_id=0, side_channels=[])
    env_sim.reset()

    behavior_name = list(env_sim.behavior_specs.keys())[0]
    prev_sim_vel = np.zeros(6)

    for step in range(n_steps):
        current_control = scaled_controls[step]
        real_pos_s = pos[step]
        real_vel_s = vel[step]
        real_acc_s = accelerations[step]
        num_sim_steps = max(1, dt_steps[step])
        print(current_control)
        # Apply control at the start of this interval
        action = ActionTuple(continuous=np.array([current_control]))
        #env_sim.set_actions(behavior_name, action)

        # Step the simulation for the full duration
        for _ in range(num_sim_steps):
            env_sim.set_actions(behavior_name, action)

            env_sim.step()

        # Get observation *after* simulation steps are completed
        sim_steps, _ = env_sim.get_steps(behavior_name)
        
        # go back one in current control to measure velocity after control input(next_vel-vel)

        for agent_id in sim_steps.agent_id:
            sim_obs = sim_steps[agent_id].obs[0]
            sim_pos_s = sim_obs[0:3]  # ned [x, y, z]
            sim_rot_q = sim_obs[3:7]  #quaternions
            sim_vel_s = sim_obs[7:13]  # ned [vx, vy, vz, wx, wy, wz]
            


            qs = np.array(sim_rot_q)  # OBS CHECK order = [x, y, z, w]
            '''for i in range(len(qs)-1):
                if np.dot(qs[i], qs[i+1]) < 0:
                    qs[i+1] *= -1'''
            rotations = R.from_quat(qs)
            euler = rotations.as_euler('xyz', degrees=False)
            sim_rot = np.unwrap(euler)
            #OBS CHECK AXES



            sim_dt = sim_timestep * num_sim_steps

            sim_acc_s = (sim_vel_s - prev_sim_vel) / sim_dt

            # Initialize rescale
            #force_rescale = np.ones_like(real_acc)
            print("Sim vel:", sim_vel_s)
            print("Sim acc:", sim_acc_s)
            print("Real acc:", real_acc)
            print("Applied control:", current_control)

            # Only rescale axes where control input is nonzero
            '''for i in range(len(real_acc)):
                if abs(real_acc[i]) < 0.01 and abs(sim_acc[i]) < 0.01:
                    real_acc[i] = 0.01 
                    sim_acc[i] = 0.01
                elif abs(real_acc[i]) >= 0.01 and abs(sim_acc[i]) < 0.01:    
                    sign = np.sign(sim_acc[i]) #
                    if sign == 0:
                        sign = np.sign(real_acc[i]) #
                    sim_acc[i] += sign * 0.01
                force_rescale[i] = real_acc[i] / (sim_acc[i])
                print(f"\nAction {i+1}: Rescale = {force_rescale[i]:.6f}")'''

                # else: leave rescale[i] = 0
            #force_rescale = real_acc / (sim_acc + 1e-10)
            force_rescale = real_acc_s - sim_acc_s
            for i in range(len(force_rescale)):
                print(f"\nAction {i+1}: Rescale = {force_rescale[i]:.6f}")
                
            # Input: sim vel + acc + control | Output: force rescale
            

            sim_pos.append(np.concatenate([sim_pos_s, sim_rot]))
            sim_vel.append(sim_vel_s)
            sim_acc.append(sim_acc_s)
            real_pos.append(real_pos_s)
            real_vel.append(real_vel_s)
            real_acc.append(real_acc_s)
            rc.append(current_control)
            data_x.append(np.concatenate([sim_vel_s, sim_acc_s, current_control]))
            data_y.append(force_rescale)

            prev_sim_vel = sim_vel_s.copy()
    env_sim.close()
    #input()
    return {
        "data_x": np.array(data_x),
        "data_y": np.array(data_y),
        "sim_pos": np.array(sim_pos),
        "real_pos": np.array(real_pos),
        "sim_vel": np.array(sim_vel),
        "real_vel": np.array(real_vel),
        "sim_acc": np.array(sim_acc),
        "real_acc": np.array(real_acc),
        "controls": np.array(rc),
    }


#--------------------------------------
def plot_all(actions, sim_pos, real_pos, sim_vel, real_vel, residuals):
    '''
    print("Input shapes:")
    print(f"  actions:     {actions.shape}")
    print(f"  sim_pos:     {sim_pos.shape}")
    print(f"  real_pos:    {real_pos.shape}")
    print(f"  sim_vel:     {sim_vel.shape}")
    print(f"  real_vel:    {real_vel.shape}")
    print(f"  residuals:   {residuals.shape}")

    print("\nFirst row of each:")
    print(f"  actions[0]:   {actions[0]}")
    print(f"  sim_pos[0]:   {sim_pos[0]}")
    print(f"  real_pos[0]:  {real_pos[0]}")
    print(f"  sim_vel[0]:   {sim_vel[0]}")
    print(f"  real_vel[0]:  {real_vel[0]}")
    print(f"  residuals[0]: {residuals[0]}")
    input()
    '''
    # --- the rest of your function ---
    dof_labels = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw']
    num_dofs = 6
    timesteps = actions.shape[0]

    x_axis = np.linspace(0, 100, timesteps)  # Race progress (%)

    fig, axs = plt.subplots(4, num_dofs, figsize=(20, 10), sharex=True)
    fig.subplots_adjust(hspace=0.3)

    #change the actions
    perm = [4, 5, 2, 1, 0, 3]
    actions = actions[:, perm]



    for i in range(num_dofs):
        # actions
        axs[0, i].plot(x_axis, actions[:, i], color='tab:blue')
        axs[0, i].set_title(f'{dof_labels[i]}')
        if i == 0:
            axs[0, i].set_ylabel("Action [PWM]")

        #Velocities (Sim vs Real)
        axs[1, i].plot(x_axis, sim_vel[:, i], label="Sim", color='tab:orange')
        axs[1, i].plot(x_axis, real_vel[:, i], label="Real", color='tab:green')
        if i == 0:
            axs[1, i].set_ylabel("Velocity [m/s]")
        elif i == 3:
            axs[1, i].set_ylabel("Velocity [rad/s]") 

        #Residuals
        axs[2, i].plot(x_axis, residuals[:, i], color='tab:red')
        if i == 0:
            axs[2, i].set_ylabel("Residual")
        
        #POS
        axs[3, i].plot(x_axis, sim_pos[:, i], label="Sim", color='tab:orange')
        axs[3, i].plot(x_axis, real_pos[:, i], label="Real", color='tab:green')
        if i == 0:
            axs[3, i].set_ylabel("Position [m]")
        elif i == 3:
            axs[3, i].set_ylabel("Position [rad]") 

        # Formatting
        for j in range(4):
            axs[j, i].grid(True)
            if j == 3:
                axs[j, i].set_xlabel("Race Progress [%]")

    # Add legend just once
    axs[1, 0].legend(loc='upper right', fontsize='small')
    axs[2, 0].legend(loc='upper right', fontsize='small')
    
    plt.tight_layout()
    plt.show()

        # 3D trajectory plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(sim_pos[:,0], sim_pos[:,1], sim_pos[:,2], label='Sim', color='orange')
    ax.plot(real_pos[:,0], real_pos[:,1], real_pos[:,2], label='Real', color='green')
    ax.set_title('3D Position')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.tight_layout()
    plt.show()

#----------------------------Main------------------------------

def main():
    k = 5
    n_steps = 1000

    data_x, data_y = [], []
    bag_dir = "../../ros2_ws2/fixed_sync"  # path bags
    #rosbag_paths = get_rosbag_paths(rosbag_folder)

    # wanted bags
    bags = {
        "fixed_sync",
        "fixed_sync_2",
        "fixed_sync_3",
        "fixed_sync_4"
    }

    rosbag_paths = []
    for name in bags:
        bag_folder = os.path.join(bag_dir, name)
        db3 = os.path.join(bag_folder, f"{name}_0.db3")
        if os.path.exists(db3):
            rosbag_paths.append(db3)
        else:
            print(f"[!] No .db3 in {bag_folder} (expected {db3})")
    print(f"Found {len(rosbag_paths)} rosbag files.")
    input()




    for rosbag_path in rosbag_paths:
        if not os.path.exists(rosbag_path):
            print(f"[!] Skipping missing ROS Bag: {rosbag_path}")
            continue
        #rosbag_path = os.path.join(os.getcwd(), "../../ros2_ws2/src/saabmarineMEX_ros2/bags/rosbag2_2025_04_09-12_37_45/rosbag2_2025_04_09-12_37_45_0.db3")
        print(f"Processing: {rosbag_path}")
        df = load_rosbag(rosbag_path)
        data = process_data(df)
        
        pos_s = data["positions"]
        scaled_ctrls = data["scaled_controls"]
        dt = data["dt"] #time between data
        dt_steps = data["steps"] #steps per dt
        vel_s = data["velocities"]
        acc_s = data["accelerations"]

        print("Running Unity simulation...")
        #nv_path = "envs/real_dynamic/prior_env3/prior.x86_64"
        env_path = "envs/sitl_envs/v4/prior/prior.x86_64"

        result = run_simulation(scaled_ctrls, dt, dt_steps, pos_s, vel_s, acc_s, n_steps, env_path)

        print(f"\nPlotting results for {rosbag_path}...\n")
        plot_all(
            result["controls"],
            result["sim_pos"],
            result["real_pos"],
            result["sim_vel"],
            result["real_vel"],
            result["data_y"]
        )

        data_x.append(result["data_x"])
        data_y.append(result["data_y"])

    data_x = np.vstack(data_x)
    data_y = np.vstack(data_y)

    print("Training KNN...")
    knn = train_knn(data_x, data_y, k)
    
    print("Saving dataset...")
    with open("knn_data.pkl", "wb") as f:
        pickle.dump((knn), f)


    print("Testing KNN...")
    test_sample = data_x[:10]
    pred = knn.predict(test_sample)

    # Train GP model
    print("Training GP model...")
    train_x_tensor = torch.tensor(data_x, dtype=torch.float32)
    train_y_tensor = torch.tensor(data_y, dtype=torch.float32)

    gp_model, gp_likelihood = train_multitask_gp(train_x_tensor, train_y_tensor, num_tasks=data_y.shape[1])

    print("Testing GP...")
    gp_model.eval()
    gp_likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_x_tensor = torch.tensor(test_sample, dtype=torch.float32)
        gp_prediction = gp_model(test_x_tensor)
        gp_mean = gp_prediction.mean.numpy()

    print("GP predicted scaling:", gp_mean)

    # Optional: Compare side-by-side
    print("\n--- Comparison ---")
    print("KNN:", pred[0])
    print("GP :", gp_mean[0])

    plt.figure(figsize=(8, 4))
    bar_width = 0.3
    indices = np.arange(data_y.shape[1])

    plt.bar(indices, pred[0], bar_width, label="KNN")
    plt.bar(indices + bar_width, gp_mean[0], bar_width, label="GP")
    plt.xticks(indices + bar_width / 2, [f"Action {i+1}" for i in range(data_y.shape[1])])
    plt.ylabel("Predicted Scaling")
    plt.title("KNN vs GP Prediction")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
