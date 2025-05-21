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

    # Compute true dt in seconds
    dt = np.diff(agg_timestamps, prepend=agg_timestamps[0]) / 1e9
    dt[0] = dt[1]

    steps = np.full(len(dt), int(round(bin_size / sim_timestep)), dtype=int)
    #print(steps)
    # Compute accelerations from velocities
    # accel[i] = (vel[i] - vel[i-1]) / dt[i]
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

    # --- the rest of your function ---
    dof_labels = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw']
    num_dofs = 6
    timesteps = actions.shape[0]

    #x_axis = np.linspace(0, 100, timesteps)  #Race progress (%)
    x_axis = np.arange(timesteps) * 0.2

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
    plt.savefig(name + ".png", dpi=300, bbox_inches="tight")
    plt.close()

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
#--------------------------------------------------

def analyze_results(result):

    sim_vel = result["sim_vel"]
    real_vel = result["real_vel"]
    sim_acc = result["sim_acc"]
    real_acc = result["real_acc"]

    #vel RMSE
    vel_mse  = np.mean((sim_vel - real_vel)**2, axis=0)
    vel_rmse = np.sqrt(vel_mse)

    # 2) acc RMSE
    acc_mse  = np.mean((sim_acc - real_acc)**2, axis=0)
    acc_rmse = np.sqrt(acc_mse)

    return vel_rmse, acc_rmse
#--------------------------------------------------
def plot_comparisons(metrics_dict):

    labels   = ['X','Y','Z','Roll','Pitch','Yaw']
    models   = ['zero-shot','knn','gp']
    
    width  = 0.2  #

    for dof in (3, 6):
        x = np.arange(len(labels))

        # --- RMSE for this DOF ---
        fig, ax = plt.subplots(figsize=(8,4))
        for i, model in enumerate(models):
            rmse, _ = metrics_dict[(model, dof)]
            ax.bar(x + i*width, rmse, width, label=model)

        ax.set_xticks(x + width)
        ax.set_xticklabels(labels)
        ax.set_ylabel("RMSE velocity")
        #ax.set_title(f"{dof}-DOF RMSE Comparison")

        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6, prune='both'))
        ax.minorticks_on()
        ax.grid(True,  which='major', axis='y',
                linestyle='--', linewidth=0.5, color='grey', alpha=0.7)
        ax.grid(True,  which='minor', axis='y',
                linestyle=':',  linewidth=0.25, color='grey', alpha=0.5)

        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
        fig.tight_layout(rect=[0,0,0.85,1])
        out_rmse = f"comparison_rmse_{dof}dof.png"
        fig.savefig(out_rmse, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"saved {out_rmse}")

        # --- rmse acc for this DOF ---
        fig2, ax2 = plt.subplots(figsize=(8,4))
        for i, model in enumerate(models):
            _, avg_res = metrics_dict[(model, dof)]
            ax2.bar(x + i*width, np.abs(avg_res), width, label=model)

        ax2.set_xticks(x + width)
        ax2.set_xticklabels(labels)
        ax2.set_ylabel("RMSE acceleration")
        #ax2.set_title(f"{dof}-DOF Absolute Avg Residual Comparison")

        ax2.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6, prune='both'))
        ax2.minorticks_on()
        ax2.grid(True,  which='major', axis='y',
                linestyle='--', linewidth=0.5, color='grey', alpha=0.7)
        ax2.grid(True,  which='minor', axis='y',
                linestyle=':',  linewidth=0.25, color='grey', alpha=0.5)

        ax2.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
        fig2.tight_layout(rect=[0,0,0.85,1])
        out_avgres = f"comparison_avgres_{dof}dof.png"
        fig2.savefig(out_avgres, dpi=300, bbox_inches='tight')
        plt.close(fig2)
        print(f"saved {out_avgres}")
#--------------------------------------------------

def runandanalyze(bag_paths, n_steps, model, dof):
    if model == "zero-shot":
        env_path = "envs/sitl_envs/v5/prior/prior.x86_64"
    else:
        env_path = "envs/res_inference/empty/brov_empty.x86_64"
    segments = []  # list of (bag, start, end)

    for bag in bag_paths:

        print(f"Processing: {bag}")    
        df = load_rosbag(bag)
        data = process_data(df)
        total_bins = data["scaled_controls"].shape[0]
        n_chunks   = (total_bins - 1) // n_steps

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_steps
            end   = start + n_steps
            segments.append((bag, start, end, data))

    total_windows = len(segments)
    half = total_windows // 2
    
    all_rmse   = []
    all_avgres = []

    for global_idx, (bag, start, end, data) in enumerate(segments):

        # if global_idx > half:
        #     continue

        pos_seg  = data["positions"][ start:end+1 ]
        vel_seg  = data["velocities"][start:end+1 ]
        acc_seg  = data["accelerations"][start:end+1 ]
        ctrl_seg = data["scaled_controls"][start:end]
        step_seg = data["steps"][start:end]

        result = run_simulation(ctrl_seg, step_seg, pos_seg, vel_seg, acc_seg, n_steps, env_path)

        name = "results_" + model + "_" + str(dof) + "-dof_" + str(global_idx)

        plot_all(
            result["controls"],
            result["sim_pos"],
            result["real_pos"],
            result["sim_vel"],
            result["real_vel"],
            result["data_y"],
            name
        )
        rmse, avgres = analyze_results(result)
        all_rmse.append(rmse)
        all_avgres.append(avgres)

    labels = ['X','Y','Z','Roll','Pitch','Yaw']

    rmse_mean   = np.mean(np.vstack(all_rmse),   axis=0)
    avgres_mean = np.mean(np.vstack(all_avgres), axis=0)

    metrics_fname = f"{model}{dof}dof_metrics.txt"
    with open(metrics_fname, 'w') as f:
        f.write("Axis," + ",".join(labels) + "\n")
        f.write("RMSE," + ",".join(f"{v:.6f}" for v in rmse_mean) + "\n")
        f.write("AvgResidual," + ",".join(f"{v:.6f}" for v in avgres_mean) + "\n")
    print(f"saved aggregated metrics to {metrics_fname}")

    return rmse_mean, avgres_mean
#--------------------------------------

def launch_server(model, dof, port="8000"):
    cmd = [sys.executable, "fastserver.py", "--model", model, "--dof", str(dof), "--port", port]
    return subprocess.Popen([
        "gnome-terminal",
        "--",
        "bash", "-lc",
        " ".join(cmd)   
    ],preexec_fn=os.setsid)

def stop_server(proc, port=8000):
    # kill
    pgid = os.getpgid(proc.pid)
    os.killpg(pgid, signal.SIGTERM)
    subprocess.run(["fuser", "-k", f"{port}/tcp"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

#----------------------------Main------------------------------

def main():
    n_steps = 200
    bag_dir = "../../brov_tank_bags2"  # path bags
    metrics = {}

    # # wanted bags
    # bags_3dof = [ 
    #     "rosbag2_2025_05_14-17_37_39", #train
    #     "rosbag2_2025_05_14-17_12_31",  #train    
    #     "rosbag2_2025_05_14-16_33_33", #
    #     "rosbag2_2025_05_14-17_34_28" #val
    # ]   
    # bags_6dof = [
    #     "rosbag2_2025_05_15-13_30_38", #
    #     "rosbag2_2025_05_14-17_20_40",  #
    #     "rosbag2_2025_05_14-17_26_50",
    #     "rosbag2_2025_05_14-17_41_18"
    #     ]
    
    # bags_3dof = ["rosbag2_2025_05_14-17_08_32"]
    # bags_6dof = ["rosbag2_2025_05_14-17_18_19"]

    bags_3dof = ["rosbag2_2025_05_14-16_33_33",
                "rosbag2_2025_05_14-17_12_31"]
    
    
    bags_6dof = ["rosbag2_2025_05_14-17_20_40"]


    bag_paths3 = [os.path.join(bag_dir, b, f"{b}_0.db3") for b in bags_3dof]
    bag_paths6 = [os.path.join(bag_dir, b, f"{b}_0.db3") for b in bags_6dof]

    for bag_paths, dof in [(bag_paths3,3),(bag_paths6,6)]:
    # zero-shot first half
        rmse0, avg0 = runandanalyze(bag_paths, n_steps, "zero-shot", dof)
        metrics[("zero-shot",dof)] = (rmse0, avg0)

        for model in ("knn","gp"):
            #TODO Check that new model actually load.

            srv = launch_server(model, dof); time.sleep(1)
            r, a = runandanalyze(bag_paths, n_steps, model, dof)
            stop_server(srv)
            srv.wait()
            
            #srv.terminate(); srv.wait(5)
            metrics[(model,dof)] = (r, a)

    plot_comparisons(metrics)

if __name__ == "__main__":
    main()
