#---imports---
import os
import glob
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

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

import signal
import subprocess
import sys
import time
#----------------------------------------------------

def get_rosbag_paths(parent_folder):
    rosbag_paths = []

    # Loop each subfolder
    for bag_dir in os.listdir(parent_folder):
        full_path = os.path.join(parent_folder, bag_dir)
        if os.path.isdir(full_path):
            # .db3 files
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
def process_data(df, bin_size=0.1, sim_timestep=0.1):

    # Extract raw data
    pos = df[['x', 'y', 'z']].values
    # Zero positions at t=0
    pos_zero = pos - pos[0:1, :]

    # Raw velocities and angular velocities
    linvel = df[['vx', 'vy', 'vz']].values
    angvel = df[['wx', 'wy', 'wz']].values

    # local velocities in body frame
    quats = df[['qx', 'qy', 'qz', 'qw']].values
    rotations = R.from_quat(quats)
    linvel_local = rotations.inv().apply(linvel)
    angvel_local  = rotations.inv().apply(angvel)
    
    # Stack velocity vectors (6 DOF)
    vel = np.hstack([linvel_local, angvel_local])

    # Euler angles zeroed at t=0
    euler = rotations.as_euler('xyz', degrees=False)
    euler_zero = euler - euler[0:1, :]

    # Combined position+orientation state (6D)
    positions = np.hstack([pos_zero, euler_zero])

    # Scale control inputs
    control_inputs = np.vstack(df['channels'].values)
    scaled_controls = (control_inputs - 1500.0) / 400.0
    # Invert Z channel if needed
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
    agg_velocities[3], agg_velocities[4] = agg_velocities[4], agg_velocities[3]

    # Compute dt in seconds
    dt = np.diff(agg_timestamps, prepend=agg_timestamps[0]) / 1e9
    dt[0] = dt[1]

    steps = np.full(len(dt), int(round(bin_size / sim_timestep)), dtype=int)
    #print(steps)
    acc = np.diff(agg_velocities, axis=0) / dt[1:, None]
    # Prepend first accel to match length
    acc = np.vstack([acc[0], acc])
    #print(len(steps))
    #input()
    return {
        'positions':    agg_positions,
        'scaled_controls': agg_controls,
        'dt':           dt,
        'steps':        steps,
        'velocities':   agg_velocities,
        'accelerations': acc,
        'timestamps':   agg_timestamps
    }

#--------------------------------------

def run_simulation(scaled_controls, dt_steps, pos, vel, accelerations, n_steps, env_path, sim_timestep=0.1):
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
        #timestart = time.perf_counter()
        for _ in range(1):
            env_sim.set_actions(behavior_name, action)
            env_sim.step()
        #timeend = time.perf_counter()
        #print(f"Sim step = {timeend - timestart:.6f} seconds")

        # Get observation *after* simulation steps are completed
        sim_steps, _ = env_sim.get_steps(behavior_name)
        for agent_id in sim_steps.agent_id:
            sim_obs = sim_steps[agent_id].obs[0]
            sim_pos_s = sim_obs[0:3]  # ned [x, y, z]
            sim_rot_q = sim_obs[3:7]  #quaternions
            sim_vel_s = sim_obs[7:13]  # ned [vx, vy, vz, wx, wy, wz]
            sim_vel_s[3], sim_vel_s[4] = sim_vel_s[4], sim_vel_s[3], 


            qs = np.array(sim_rot_q)  # [x, y, z, w]
            rotations = R.from_quat(qs)
            euler = rotations.as_euler('xyz', degrees=False)
            sim_rot = np.unwrap(euler)

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

                data_x.append(np.concatenate([sim_vel_s, current_control]))
                data_y.append(force_rescale)

            prev_sim_vel = sim_vel_s.copy()
    env_sim.close()
    print(len(rc))
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
def plot_all(actions, sim_pos, real_pos, sim_vel, real_vel, residuals, name, dir):
    # --- the rest of your function ---
    dof_labels = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw']
    num_dofs = 6
    timesteps = actions.shape[0]

    #x_axis = np.linspace(0, 100, timesteps)  #Race progress (%)
    x_axis = np.arange(timesteps) * 0.1

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
                axs[j, i].set_xlabel("Time [s]")

    # Add legend just once
    axs[1, 0].legend(loc='upper right', fontsize='small')
    axs[2, 0].legend(loc='upper right', fontsize='small')
    
    plt.tight_layout()
    out_path = os.path.join(dir, f"{name}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    #3D trajectory plot
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
    out_path_3d = os.path.join(dir, f"3D_{name}.png")
    plt.savefig(out_path_3d, dpi=300, bbox_inches="tight")
    plt.close()
#--------------------------------------------------

def analyze_results(result):

    sim_vel = result["sim_vel"]
    real_vel = result["real_vel"]
    sim_acc = result["sim_acc"]
    real_acc = result["real_acc"]

    #vel MAE (mean absolute error)
    vel_abs_err = np.abs(sim_vel - real_vel)
    vel_mae = np.mean(vel_abs_err, axis=0)

    #acc RMSE
    acc_mse  = np.mean((sim_acc - real_acc)**2, axis=0)
    acc_rmse = np.sqrt(acc_mse)

    return vel_mae, acc_rmse
#--------------------------------------------------
def process_and_save(rosbag_path, dof, data_dir, plot_dir, sim_timestep=0.1 ):
    """
    1. Load + bin the bag once
    2. For each model (zero-shot, knn, gp), run_simulation & compute RMSE
    3. Save one NPZ containing:
        - binned “physics” arrays: timestamps, positions, velocities,
            accelerations, scaled_controls
        - for each model: sim_pos_<m>, real_pos_<m>, sim_vel_<m>, real_vel_<m>,
                        data_x_<m>, data_y_<m>, controls_<m>
    4. Return dict: { model: (vel_rmse, acc_rmse) }
    """
    bag_name = os.path.splitext(os.path.basename(rosbag_path))[0]
    os.makedirs(data_dir, exist_ok=True)

    # load
    df = load_rosbag(rosbag_path)
    processed = process_data(df)

    total_bins = processed['scaled_controls'].shape[0]
    bins_for_sim = total_bins - 50
    if bins_for_sim <= 10:
        raise RuntimeError(f"Not enough bins in {rosbag_path} (got {total_bins=}) to drop last 50.")

    # array align
    timestamps = processed['timestamps'][: bins_for_sim + 1]        # (bins_for_sim+1,)
    positions = processed['positions'][: bins_for_sim + 1]          # (bins_for_sim+1, 6)
    velocities = processed['velocities'][: bins_for_sim + 1]        # (bins_for_sim+1, 6)
    accelerations = processed['accelerations'][: bins_for_sim + 1]  # (bins_for_sim+1, 6)
    scaled_controls = processed['scaled_controls'][: bins_for_sim]  # (bins_for_sim, 6)
    steps = processed['steps'][: bins_for_sim]                      # (bins_for_sim,)

    positions_zeroed = positions - positions[0:1, :]  # zero offset

    models = ["zero-shot", "knn", "mtgp", "svgp"]
    metrics_per_model = {}

    save_dict = {
        "timestamps":      timestamps,
        "positions":       positions,
        "velocities":      velocities,
        "accelerations":   accelerations,
        "scaled_controls": scaled_controls
    }
    real_saved = False
    for model in models:
        # If KNN or GP, launch server first
        if model in ("knn", "mtgp", "svgp"):
            env_path = "envs/res_inference/empty/tether/3m_1-5Cd_novattenyta/env.x86_64"
            srv = launch_server(model, dof)
            time.sleep(5)
        else:
            env_path = "envs/tether/3m_1-5Cd_novattenyta/env.x86_64"
    
        sim_out = run_simulation(
            scaled_controls=scaled_controls,
            dt_steps=steps,
            pos=positions_zeroed,
            vel=velocities,
            accelerations=accelerations,
            n_steps=bins_for_sim,
            env_path=env_path,
            sim_timestep=sim_timestep
        )
        #input()
        vel_mae, acc_rmse = analyze_results(sim_out)
        metrics_per_model[model] = (vel_mae, acc_rmse)

        plot_name = f"{bag_name}_{dof}dof_{model}"
        plot_all(
            sim_out["controls"],    # (bins_for_sim-1, 6)
            sim_out["sim_pos"],     # (bins_for_sim-1, 6)
            sim_out["real_pos"],    # (bins_for_sim-1, 6)
            sim_out["sim_vel"],     # (bins_for_sim-1, 6)
            sim_out["real_vel"],    # (bins_for_sim-1, 6)
            sim_out["data_y"],      # (bins_for_sim-1, 6) residuals
            plot_name,
            plot_dir
        )

        if not real_saved:
            save_dict["real_pos"] = sim_out["real_pos"]    # (bins_for_sim-1, 6)
            save_dict["real_vel"] = sim_out["real_vel"]    # (bins_for_sim-1, 6)
            save_dict["real_acc"] = sim_out["real_acc"]    # (bins_for_sim-1, 6)
            save_dict["controls"] = sim_out["controls"]
            real_saved = True

        suffix = {
            "zero-shot": "_zero_shot",
            "knn":       "_knn",
            "mtgp":        "_mtgp",
            "svgp":        "_svgp"
        }[model]

        save_dict[f"sim_pos{suffix}"]  = sim_out["sim_pos"]
        save_dict[f"sim_vel{suffix}"]  = sim_out["sim_vel"]
        save_dict[f"sim_acc{suffix}"]  = sim_out["sim_acc"] 
        save_dict[f"data_x{suffix}"]   = sim_out["data_x"]
        save_dict[f"data_y{suffix}"]   = sim_out["data_y"]

        if model in ("knn", "mtgp", "svgp"):
            stop_server(srv)
            srv.wait()

    dof_str = f"{dof}dof"
    out_filename = f"{bag_name}_{dof_str}_data_all.npz"
    out_path = os.path.join(data_dir, out_filename)

    np.savez_compressed(out_path, **save_dict)
    print(f"[saved NPZ] {rosbag_path} → {out_path}")

    return metrics_per_model

#--------------------------------------------------
def plot_comparisons(metrics_across_bags, dof, dir):

    labels = ["X", "Y", "Z", "Roll", "Pitch", "Yaw"]
    n_axes = 6
    x = np.arange(n_axes)
    width = 0.2
    models = ["zero-shot", "knn", "mtgp", "svgp"]

    # Compute per-model average across bags (stack over axis=0)
    avg_vel = {}
    avg_acc = {}
    for model in models:
        vel_list = [m[0] for m in metrics_across_bags[model]]
        acc_list = [m[1] for m in metrics_across_bags[model]]
        vel_stack = np.vstack(vel_list)  # (n_bags, 6)
        acc_stack = np.vstack(acc_list)  # (n_bags, 6)
        avg_vel[model] = np.mean(vel_stack, axis=0)  # (6,)
        avg_acc[model] = np.mean(acc_stack, axis=0)  # (6,)

    # --- Plot velocity MAE (6 axes) ---
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, model in enumerate(models):
        ax.bar(x + i*width, avg_vel[model], width, label=model)
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Velocity MAE")
    ax.grid(True, which="major", axis="y", linestyle="--", linewidth=0.5, color="grey", alpha=0.7)
    ax.minorticks_on()
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    fig.tight_layout(rect=[0, 0, 0.85, 1])

    vel_fname = f"velocity_mae_{dof}dof.png"
    path_vel = os.path.join(dir, vel_fname)
    plt.savefig(path_vel, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved plot] {vel_fname}")

    # --- Plot acceleration RMSE (6 axes) ---
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    for i, model in enumerate(models):
        ax2.bar(x + i*width, avg_acc[model], width, label=model)
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Acceleration RMSE")
    ax2.grid(True, which="major", axis="y", linestyle="--", linewidth=0.5, color="grey", alpha=0.7)
    ax2.minorticks_on()
    ax2.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    fig2.tight_layout(rect=[0, 0, 0.85, 1])

    acc_fname = f"acceleration_rmse_{dof}dof.png"
    path_acc = os.path.join(dir, acc_fname)
    plt.savefig(path_acc, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"[saved plot] {acc_fname}")
#--------------------------------------------------

def launch_server(model, dof, port="8000"):
    cmd = [sys.executable, "-m", "inference.fastserver", "--model", model, "--dof", str(dof), "--port", port]
    return subprocess.Popen([
        "gnome-terminal",
        "--",
        "bash", "-lc",
        " ".join(cmd)# + "; exec bash"
    ],preexec_fn=os.setsid)

def stop_server(proc, port=8000):
    # kill
    pgid = os.getpgid(proc.pid)
    os.killpg(pgid, signal.SIGTERM)
    subprocess.run(["fuser", "-k", "-TERM", f"{port}/tcp"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

#----------------------------Main------------------------------

def main():
    data_dir = "data_and_plots/dictfiles_plot_tet"
    plot_dir = "data_and_plots/plots_inf_tet"
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    bag_dir = "training/ros2_bags/27-5"  # path bags
    
    bags_3dof = ["rosbag2_2025_05_27-11_57_50",
                "rosbag2_2025_05_27-11_10_52_sync"]

    bags_6dof = ["rosbag2_2025_05_27-12_04_31",
                "rosbag2_2025_05_27-11_40_38",
                "rosbag2_2025_05_27-11_46_03",
                "rosbag2_2025_05_27-12_09_45"]

    metrics = {
        3: {"zero-shot": [], "knn": [], "mtgp": [], "svgp": []},
        6: {"zero-shot": [], "knn": [], "mtgp": [], "svgp": []}
    }

    for dof, bag_list in ((3, bags_3dof), (6, bags_6dof)):
        for bag_base in bag_list:
            rosbag_path = os.path.join(bag_dir, bag_base, f"{bag_base}_0.db3")
            model_metrics = process_and_save(rosbag_path, dof, data_dir, plot_dir)
            # model_metrics is { model: (vel_rmse_array, acc_rmse_array) }, both shape (6,)
            for model, (v_rmse, a_rmse) in model_metrics.items():
                metrics[dof][model].append((v_rmse, a_rmse))

        # After collecting all bags of this DOF, plot RMSE
        plot_comparisons(metrics[dof], dof, plot_dir)

if __name__ == "__main__":
    main()
