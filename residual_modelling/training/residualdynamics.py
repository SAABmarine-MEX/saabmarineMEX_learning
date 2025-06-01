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
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit
from sklearn.preprocessing import FunctionTransformer

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
                rosbag_paths.append(db3_files[0]) 
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
    df["timestamp"] = pd.to_numeric(df["timestamp"])

    return df
#------------------------------

def process_data(df, bin_size=0.2, sim_timestep=0.02):

    # Extract raw data
    pos = df[['x', 'y', 'z']].values
    # Zero positions at t=0
    pos_zero = pos - pos[0:1, :]

    # Raw velocities
    linvel = df[['vx', 'vy', 'vz']].values
    angvel = df[['wx', 'wy', 'wz']].values

    # local velocities in body frame
    quats = df[['qx', 'qy', 'qz', 'qw']].values
    rotations = R.from_quat(quats)
    linvel_local = rotations.inv().apply(linvel)
    angvel_local  = rotations.inv().apply(angvel)
    vel = np.hstack([linvel_local, angvel_local])

    # Euler angles
    euler = rotations.as_euler('xyz', degrees=False)
    euler_zero = euler - euler[0:1, :] #zeroed at t=0

    positions = np.hstack([pos_zero, euler_zero])

    # Scale control inputs
    control_inputs = np.vstack(df['channels'].values)
    scaled_controls = (control_inputs - 1500.0) / 400.0
    # Invert Z channel
    scaled_controls[:, 2] *= -1

    # Raw timestamps in ns
    timestamps = df['timestamp'].values.astype(np.float64)

    # Prepare buffers for time-based binning
    agg_pos = []
    agg_ctrl = []
    agg_vel = []
    agg_ts = []
    # Buffers
    pos_buf = []
    ctrl_buf = []
    vel_buf = []

    bin_start = timestamps[0]
    for t, p, u, v in zip(timestamps, positions, scaled_controls, vel):
        pos_buf.append(p)
        ctrl_buf.append(u)
        vel_buf.append(v)

        # Check if bin duration reached
        if (t - bin_start) >= bin_size * 1e9:
            # Aggregate by averaging
            agg_pos.append(np.mean(pos_buf, axis=0))
            agg_ctrl.append(np.mean(ctrl_buf, axis=0))
            agg_vel.append(np.mean(vel_buf, axis=0))
            # Use bin start time for timestamp
            agg_ts.append(bin_start)

            # Reset buffers
            bin_start = t
            pos_buf.clear()
            ctrl_buf.clear()
            vel_buf.clear()

    # Convert lists to arrays
    agg_positions  = np.vstack(agg_pos)
    agg_controls   = np.vstack(agg_ctrl)
    agg_velocities = np.vstack(agg_vel)
    agg_timestamps = np.array(agg_ts)
    #agg_velocities[3], agg_velocities[4] = agg_velocities[4], agg_velocities[3]

    # Compute true dt between bins (in seconds)
    dt = np.diff(agg_timestamps, prepend=agg_timestamps[0]) / 1e9
    dt[0] = dt[1]

    # Every bin spans ~bin_size seconds --> fixed sim steps
    steps = np.full(len(dt), int(round(bin_size / sim_timestep)), dtype=int)
    #print(steps)
    # Compute accelerations from velocities
    # accel[i] = (vel[i] - vel[i-1]) / dt[i]
    acc = np.diff(agg_velocities, axis=0) / dt[1:, None]
    # Prepend first accel to match length
    acc = np.vstack([acc[0], acc])
    # print(len(steps))
    # input()
    return {
        'positions':    agg_positions,
        'scaled_controls': agg_controls,
        'dt':           dt,
        'steps':        steps,
        'velocities':   agg_velocities,
        'accelerations': acc,
        'timestamps':   agg_timestamps
    }
#--------------------------------

def tune_gp(X_train, y_train, X_val, y_val, lr_list=(1e-2, 1e-1, 5e-1), iters_list=(200, 500, 800)):
    # tune learning rate and optimizer steps
    best_rmse   = np.inf
    best_params = None

    for lr in lr_list:
        for n_iters in iters_list:
            # train on training split
            model, likelihood, scaler = train_multitask_gp(
                X_train, y_train, lr=lr, iters = n_iters
            )
            # predict on val split
            Xs = scaler.transform(X_val)
            tx = torch.tensor(Xs, dtype=torch.float32)
            model.eval(); likelihood.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                yhat = likelihood(model(tx)).mean.cpu().numpy()

            rmse = np.sqrt(np.mean((yhat - y_val)**2))
            print(f"  lr={lr:.2e}, iters={n_iters} → RMSE={rmse:.4f}")
            if rmse < best_rmse:
                best_rmse   = rmse
                best_params = (lr, n_iters)

    lr_opt, iters_opt = best_params
    print(f"Best GP params: lr={lr_opt:.2e}, iters={iters_opt}, val_RMSE={best_rmse:.4f}")

    #Re-train final on the entire training set
    final_model, final_likelihood, final_scaler = train_multitask_gp(
        X_train, y_train, lr=lr_opt, iters=iters_opt)
    
    return final_model, final_likelihood, final_scaler
#--------------------------------

def train_multitask_gp(train_x, train_y, lr, iters, w):
    class MTGP(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            num_tasks = train_y.shape[1]

            super().__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.MultitaskMean(
                gpytorch.means.ConstantMean(), num_tasks=num_tasks)
            
            self.covar_module = gpytorch.kernels.MultitaskKernel(
                gpytorch.kernels.RBFKernel(), num_tasks=num_tasks, rank=1)

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

    #scalar
    scaler = StandardScaler().fit(train_x)
    train_x_scaled = scaler.transform(train_x)
    train_x_scaled = train_x_scaled * w

    tx = torch.tensor(train_x_scaled, dtype=torch.float32)
    ty = torch.tensor(train_y, dtype=torch.float32)

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=train_y.shape[1])


    #reduced noise
    #y_var = torch.var(ty, dim=0)
    #likelihood.noise = y_var.mean().item() * 0.01 


    model = MTGP(tx, ty, likelihood)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(iters):
        optimizer.zero_grad()
        output = model(tx)
        loss = -mll(output, ty)
        loss.backward()
        optimizer.step()
        #print(f'Iter {i+1}/{iters} - Loss: {loss.item():.3f}') 

    model.eval(); likelihood.eval()
    return model, likelihood, scaler
#--------------------------------------

def train_knn(data_x, data_y, k, w):
    pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("knn",    KNeighborsRegressor(n_neighbors=k, weights="uniform"))
    ])
    pipeline.fit(data_x, data_y)
    # scale, TODO check if better scaling is needed

    return pipeline
#-------------------------------------

def tune_knn(X_train, y_train, X_val, y_val, w):
    best_rmse = np.inf
    best_cfg  = None

    for k in range(3,11):
        for p in (1,2):
            knn = Pipeline([
                ("scaler", StandardScaler()),
                ("weight", w),
                ("knn",    KNeighborsRegressor(n_neighbors=k, p=p, weights="uniform"))
            ])
            knn.fit(X_train, y_train)
            yhat = knn.predict(X_val)
            rmse = np.sqrt(np.mean((yhat - y_val)**2))
            if rmse < best_rmse:
                best_rmse = rmse
                best_cfg = (k,p)

    print(f"Best (k,p)={best_cfg} with RMSE={best_rmse:.4f}")

    k_opt, p_opt = best_cfg
    final_knn = Pipeline([
        ("scaler", StandardScaler()),
        ("knn",    KNeighborsRegressor(
                    n_neighbors=k_opt,
                    p=p_opt,
                    weights="uniform"))
                        ])
    final_knn.fit(X_train, y_train)

    return final_knn
#--------------------------------------

def run_simulation(scaled_controls, dt_steps, pos, vel, accelerations, n_steps, env_path, sim_timestep=0.02):
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
    prev_sim_vel = vel[0]

    for step in range(n_steps):
        current_control = scaled_controls[step]        
        #if step > 1:
        real_pos_s = pos[step + 1]
        real_vel_s = vel[step + 1]
        real_acc_s = accelerations[step + 1]
        num_sim_steps = dt_steps[step]

        #apply control
        action = ActionTuple(continuous=np.array([current_control]))

        # Step the simulation for the full duration
        for _ in range(2):
            env_sim.set_actions(behavior_name, action)
            env_sim.step()

        # Get observation *after* simulation steps are completed
        sim_steps, _ = env_sim.get_steps(behavior_name)
        for agent_id in sim_steps.agent_id:
            sim_obs = sim_steps[agent_id].obs[0]
            sim_pos_s = sim_obs[0:3]  # ned [x, y, z]
            sim_rot_q = sim_obs[3:7]  #quaternions
            sim_vel_s = sim_obs[7:13]  # ned [vx, vy, vz, wx, wy, wz]
            sim_vel_s[3], sim_vel_s[4] = sim_vel_s[4], sim_vel_s[3], 

            qs = np.array(sim_rot_q)  # OBS CHECK order = [x, y, z, w]
            rotations = R.from_quat(qs)
            euler = rotations.as_euler('xyz', degrees=False)
            sim_rot = np.unwrap(euler)
            #OBS CHECK AXES

            sim_dt = sim_timestep * num_sim_steps
            sim_acc_s = (sim_vel_s - prev_sim_vel) / sim_dt

            # Initialize rescale
            
            if step > 0:
                force_rescale = real_acc_s - sim_acc_s
                # for i in range(len(force_rescale)):
                #     print(f"\nAction {i+1}: Rescale = {force_rescale[i]:.6f}")
                
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

def plot_all(actions, sim_pos, real_pos, sim_vel, real_vel, residuals, name):

    dof_labels = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw']
    num_dofs = 6
    timesteps = actions.shape[0]

    x_axis = np.linspace(0, 100, timesteps)  #Race progress (%)

    fig, axs = plt.subplots(3, num_dofs, figsize=(20, 10), sharex=True)
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
        # axs[3, i].plot(x_axis, sim_pos[:, i], label="Sim", color='tab:orange')
        # axs[3, i].plot(x_axis, real_pos[:, i], label="Real", color='tab:green')
        # if i == 0:
        #     axs[3, i].set_ylabel("Position [m]")
        # elif i == 3:
        #     axs[3, i].set_ylabel("Position [rad]") 

        #Formatting
        for j in range(3):
            axs[j, i].grid(True)
            if j == 2:
                axs[j, i].set_xlabel("Race Progress [%]")

    # Add legend
    axs[1, 0].legend(loc='upper right', fontsize='small')
    axs[2, 0].legend(loc='upper right', fontsize='small')
    
    plt.tight_layout()
    plt.savefig( name + ".png", dpi=300, bbox_inches="tight")

        # 3D trajectory plot
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot(sim_pos[:,0], sim_pos[:,1], sim_pos[:,2], label='Sim', color='orange')
    # ax.plot(real_pos[:,0], real_pos[:,1], real_pos[:,2], label='Real', color='green')
    # ax.set_title('3D Position')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.legend()
    # plt.tight_layout()
    # plt.show()

#----------------------------Main------------------------------

def main():
    k = 5
    n_steps = 200
    bag_dir = "../../brov_tank_bags2"  # path bags

    weight = np.array([1]*6 +[1]*6 + [1]*6, dtype=np.float32)
    weight_tf = FunctionTransformer(lambda X: X * weight, validate=True)

    data_x = []
    data_y = []
    # wanted bags
    #3dof
    #600 points 3 plots for training

    #Only keyboard test
    bags =  [ 
        "rosbag2_2025_05_14-16_33_33", #
        "rosbag2_2025_05_14-17_12_31" #
    ]   

    # bags =  [ 
    #     "rosbag2_2025_05_14-17_37_39", #train
    #     "rosbag2_2025_05_14-17_12_31",  #train    
    #     "rosbag2_2025_05_14-16_33_33", #
    #     "rosbag2_2025_05_14-17_34_28" #val
    # ]   

    rosbag_paths = []
    for name in bags:
        bag_folder = os.path.join(bag_dir, name)
        db3 = os.path.join(bag_folder, f"{name}_0.db3")
        if os.path.exists(db3):
            rosbag_paths.append(db3)
        else:
            print(f"[!] No .db3 in {bag_folder} (expected {db3})")
    # print(f"Found {len(rosbag_paths)} rosbag files.")
    # input()
    bagnr = 0
    for rosbag_path in rosbag_paths:
        if not os.path.exists(rosbag_path):
            #print(f"[!] Skipping missing ROS Bag: {rosbag_path}")
            continue
        print(f"Processing: {rosbag_path}")
        bagnr += 1

        df = load_rosbag(rosbag_path)
        data = process_data(df)
        
        pos_s = data["positions"]
        scaled_ctrls = data["scaled_controls"]
        #dt = data["dt"] #time between data
        dt_steps = data["steps"] #steps per dt
        vel_s = data["velocities"]
        acc_s = data["accelerations"]

        total_bins = scaled_ctrls.shape[0]
        n_chunks   = (total_bins - 10) // n_steps
        
        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_steps
            end   = start + n_steps
            pos_seg   = pos_s[start : end+1]
            vel_seg   = vel_s[start : end+1]
            acc_seg   = acc_s[start : end+1]
            ctrl_seg  = scaled_ctrls[start : end]
            step_seg  = dt_steps[start : end]

            print("Running Unity simulation...")
            env_path = "envs/sitl_envs/v5/prior/prior.x86_64"
            result = run_simulation(ctrl_seg, step_seg, pos_seg, vel_seg, acc_seg, n_steps, env_path)

            print(f"\nPlotting results for {rosbag_path}...\n")
            plot_all(
                result["controls"],
                result["sim_pos"],
                result["real_pos"],
                result["sim_vel"],
                result["real_vel"],
                result["data_y"],
                "resultplot_3dof_" + str(bagnr) + "_seg" + str(chunk_idx)
            )

            data_x.append(result["data_x"])
            data_y.append(result["data_y"])

    data_x = np.vstack(data_x)
    data_y = np.vstack(data_y)


    # data_x: shape (N,18); data_y: shape (N,6)
    # noise_stds = np.std(data_y, axis=0)     # shape (6,)
    # k = 2.0                                 # threshold multiplier
    # thresholds = k * noise_stds            # shape (6,)

    # # zero out any residual below its threshold
    # y_filtered = np.where(
    #     np.abs(data_y) > thresholds,      # condition: “big enough to keep”
    #     data_y,                           # keep original if True
    #     0.0                               # else set to zero
    # )
    # N = data_x.shape[0]
    # mid = N // 2                  # first half = train, second half = val

    # x_train, x_val = data_x[:mid],     data_x[mid:]
    # y_train, y_val = data_y[:mid],     data_y[mid:]
    #y_train, y_val = y_filtered[:mid],     y_filtered[mid:]
    
    #print("Tuning KNN 3dof hyperparameters…")
    #knn = tune_knn(x_train, y_train, x_val, y_val, weight_tf)

    print("Training KNN 3dof...")
    knn = train_knn(data_x, data_y, k, weight_tf)

    # Train GP model
    print("Training GP 3dof...")
    #gp_model, gp_likelihood, gp_scalar = tune_gp( x_train, y_train, x_val, y_val, num_tasks= data_y.shape[1],lr_list=(1e-2,1e-1,5e-1), iters_list=(200,500,800))
    gp_model, gp_likelihood, gp_scalar = train_multitask_gp(data_x, data_y, lr=0.01, iters=300, w = weight)

    gp_checkpoint = {
    "model_state":      gp_model.state_dict(),
    "likelihood_state": gp_likelihood.state_dict(),
    "scaler_mean":      gp_scalar.mean_,
    "scaler_scale":     gp_scalar.scale_,
    "num_tasks":        data_y.shape[1],
    }

    print("Saving KNN 3dof...")
    with open("knn_3dof.pkl", "wb") as f:
        pickle.dump(knn, f)

    print(f"Saving GP 3d0f")
    torch.save(gp_checkpoint, "gp_3dof.pth")
#-----------------------------6DOF-------------------------------------------
    data_x = []
    data_y = []
    # wanted bags
    #6dof
    #check length total

    bags =  [ 
        "rosbag2_2025_05_14-17_20_40" #
    ]   

    # bags = [
    #     "rosbag2_2025_05_15-13_30_38", #
    #     "rosbag2_2025_05_14-17_41_18",
    #     "rosbag2_2025_05_14-17_20_40",  #
    #     "rosbag2_2025_05_14-17_26_50"
    #     ]
    
    rosbag_paths = []
    for name in bags:
        bag_folder = os.path.join(bag_dir, name)
        db3 = os.path.join(bag_folder, f"{name}_0.db3")
        if os.path.exists(db3):
            rosbag_paths.append(db3)
        else:
            print(f"[!] No .db3 in {bag_folder} (expected {db3})")
    print(f"Found {len(rosbag_paths)} rosbag files.")
    #input()
    bagnr = 0
    for rosbag_path in rosbag_paths:
        if not os.path.exists(rosbag_path):
            print(f"[!] Skipping missing ROS Bag: {rosbag_path}")
            continue
        print(f"Processing: {rosbag_path}")
        bagnr += 1
        df = load_rosbag(rosbag_path)
        data = process_data(df)
        
        pos_s = data["positions"]
        scaled_ctrls = data["scaled_controls"]
        dt_steps = data["steps"] #steps per dt
        vel_s = data["velocities"]
        acc_s = data["accelerations"]

        total_bins = scaled_ctrls.shape[0]
        n_chunks   = (total_bins - 10) // n_steps
        
        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_steps
            end   = start + n_steps
            pos_seg   = pos_s[start : end+1]
            vel_seg   = vel_s[start : end+1]
            acc_seg   = acc_s[start : end+1]
            ctrl_seg  = scaled_ctrls[start : end]
            step_seg  = dt_steps[start : end]

            print("Running Unity simulation...")
            env_path = "envs/sitl_envs/v5/prior/prior.x86_64"
            result = run_simulation(ctrl_seg, step_seg, pos_seg, vel_seg, acc_seg, n_steps, env_path)

            print(f"\nPlotting results for {rosbag_path}...\n")
            plot_all(
                result["controls"],
                result["sim_pos"],
                result["real_pos"],
                result["sim_vel"],
                result["real_vel"],
                result["data_y"],
                "resultplot_6dof_" + str(bagnr) + "_seg" + str(chunk_idx)
            )

            data_x.append(result["data_x"])
            data_y.append(result["data_y"])

    data_x = np.vstack(data_x)
    data_y = np.vstack(data_y)


    # data_x: shape (N,18); data_y: shape (N,6)
    # noise_stds = np.std(data_y, axis=0)     # shape (6,)
    # k = 2.0                                 # threshold multiplier
    # thresholds = k * noise_stds            # shape (6,)

    # # zero out any residual below its threshold
    # y_filtered = np.where(
    #     np.abs(data_y) > thresholds,      # condition: “big enough to keep”
    #     data_y,                           # keep original if True
    #     0.0                               # else set to zero
    # )
    # N = data_x.shape[0]
    # mid = N // 2                  # first half = train, second half = val

    # x_train, x_val = data_x[:mid],     data_x[mid:]
    # y_train, y_val = data_y[:mid],     data_y[mid:]
    #y_train, y_val = y_filtered[:mid],     y_filtered[mid:]

    # print("Tuning KNN 6dof params…")
    # knn = tune_knn(x_train, y_train, x_val, y_val, weight_tf)

    print("Training KNN 6dof...")
    knn = train_knn(data_x, data_y, k, weight_tf)
    
    print("Saving knn 6dof")
    with open("knn_6dof.pkl", "wb") as f:
        pickle.dump(knn, f)

    # Train GP model
    print("Training GP 6dof...")
    gp_model, gp_likelihood, gp_scalar = train_multitask_gp(data_x, data_y, lr=0.01, iters=300, w = weight)
    #gp_model, gp_likelihood, gp_scalar = tune_gp(x_train, y_train, x_val, y_val, num_tasks=data_y.shape[1],lr_list=(1e-2,1e-1,5e-1), iters_list=(200,500,800))

    gp_checkpoint = {
    "model_state":      gp_model.state_dict(),
    "likelihood_state": gp_likelihood.state_dict(),
    "scaler_mean":      gp_scalar.mean_,
    "scaler_scale":     gp_scalar.scale_,
    "num_tasks":        data_y.shape[1],
    }

    print(f"Saving GP 6dof")
    torch.save(gp_checkpoint, "gp_6dof.pth")


if __name__ == "__main__":
    main()
