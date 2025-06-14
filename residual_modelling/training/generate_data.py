import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


from scipy.spatial.transform import Rotation as R

import rosbag2_py
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message

from brov_msgs.msg import SyncedPoseControl
from brov_msgs.msg import StampedRcOverride
from mavros_msgs.msg import OverrideRCIn

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple


def main():
    # Prep input
    #n_steps = 200
    data_x = []
    data_y = []
    env_path = "envs/tether/1_2Cd/env_tether_1_2Cd.x86_64"
    
    #bag_dir = "ros2_bags/brov_tank_bags2"
    bag_dir = "training/ros2_bags/27-5"

    plotdir = "data_and_plots/training_plots_tet"
    os.makedirs(plotdir, exist_ok=True)
    
    # 3DOF bags
    bags3 = ["rosbag2_2025_05_27-12_58_29",
            "rosbag2_2025_05_27-11_34_13"]

    # Get the bag paths TODO: could make as function in py
    rosbag_paths = []
    for name in bags3:
        bag_folder = os.path.join(bag_dir, name)
        db3 = os.path.join(bag_folder, f"{name}_0.db3")
        if os.path.exists(db3):
            rosbag_paths.append(db3)
        else:
            print(f"[!] No .db3 in {bag_folder} (expected {db3})")

    # Process the bags
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
        #n_chunks   = (total_bins - 10) // n_steps
        
        """
        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_steps
            end   = start + n_steps
            pos_seg   = pos_s[start : end+1]
            vel_seg   = vel_s[start : end+1]
            acc_seg   = acc_s[start : end+1]
            ctrl_seg  = scaled_ctrls[start : end]
            step_seg  = dt_steps[start : end]

            print("Running Unity simulation...")
            env_path = "../envs/sitl_envs/v5/prior/prior.x86_64"
            result = run_simulation(ctrl_seg, step_seg, pos_seg, vel_seg, acc_seg, n_steps, env_path)
            """
        start = 0
        end   = total_bins - 50
        print(end)
        pos_seg   = pos_s[start : end+1]
        vel_seg   = vel_s[start : end+1]
        acc_seg   = acc_s[start : end+1]
        ctrl_seg  = scaled_ctrls[start : end]
        step_seg  = dt_steps[start : end]

        print("Running Unity simulation...")
        result = run_simulation(ctrl_seg, step_seg, pos_seg, vel_seg, acc_seg, end, env_path)

        # print(f"\nPlotting results for {rosbag_path}...\n")
        plot_all(
            result["controls"],
            result["sim_pos"],
            result["real_pos"],
            result["sim_vel"],
            result["real_vel"],
            result["data_y"],
            "trainingplot_3dof_" + str(bagnr),
            plotdir
            )
        
        data_x.append(result["data_x"])
        data_y.append(result["data_y"])

    data3dof_x = np.vstack(data_x)
    data3dof_y = np.vstack(data_y)

    #-----------------------------6DOF-------------------------------------------
    data_x = []
    data_y = []
    # wanted bags
    # 6dof bags

    bags6 = [
        "rosbag2_2025_05_27-12_54_51",
        "rosbag2_2025_05_27-11_51_32",
        "rosbag2_2025_05_27-13_04_58"
    ]

    rosbag_paths2 = []
    for name in bags6:
        bag_folder = os.path.join(bag_dir, name)
        db3 = os.path.join(bag_folder, f"{name}_0.db3")
        if os.path.exists(db3):
            rosbag_paths2.append(db3)
        else:
            print(f"[!] No .db3 in {bag_folder} (expected {db3})")
    print(f"Found {len(rosbag_paths2)} rosbag files.")
    #input()
    bagnr = 0
    for rosbag_path in rosbag_paths2:
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


        start = 0
        end   = total_bins - 50
        print(end)
        pos_seg   = pos_s[start : end+1]
        vel_seg   = vel_s[start : end+1]
        acc_seg   = acc_s[start : end+1]
        ctrl_seg  = scaled_ctrls[start : end]
        step_seg  = dt_steps[start : end]

        print("Running Unity simulation...")
        result6 = run_simulation(ctrl_seg, step_seg, pos_seg, vel_seg, acc_seg, end, env_path)

        # print(f"\nPlotting results for {rosbag_path}...\n")
        plot_all(
            result6["controls"],
            result6["sim_pos"],
            result6["real_pos"],
            result6["sim_vel"],
            result6["real_vel"],
            result6["data_y"],
            "trainingplot_6dof_" + str(bagnr),
            plotdir
        )
        
        data_x.append(result["data_x"])
        data_y.append(result["data_y"])

    data6dof_x = np.vstack(data_x)
    data6dof_y = np.vstack(data_y)

    print(len(data3dof_x))
    print(len(data6dof_x))

    # Split and save the data
    # Trim both to the minimum length
    min_len = min(len(data3dof_x), len(data6dof_x))

    data3dof_x = data3dof_x[:min_len]
    data3dof_y = data3dof_y[:min_len]

    data6dof_x = data6dof_x[:min_len]
    data6dof_y = data6dof_y[:min_len]

    # split index
    split_idx = int(0.95 * min_len)

    # split
    train3dof_x, eval3dof_x = data3dof_x[:split_idx], data3dof_x[split_idx:]
    train3dof_y, eval3dof_y = data3dof_y[:split_idx], data3dof_y[split_idx:]

    train6dof_x, eval6dof_x = data6dof_x[:split_idx], data6dof_x[split_idx:]
    train6dof_y, eval6dof_y = data6dof_y[:split_idx], data6dof_y[split_idx:]

    # Setup directories for saving data
    folder_path = "training/data/data9_tet/"
    train_dir = os.path.join(folder_path, "train/")
    eval_dir = os.path.join(folder_path, "eval/")

    os.makedirs(folder_path, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    # Save training data
    np.savez(train_dir + "data3dof.npz", x=train3dof_x, y=train3dof_y)
    np.savez(train_dir + "data6dof.npz", x=train6dof_x, y=train6dof_y)

    # Save evaluation data
    np.savez(eval_dir + "data3dof.npz", x=eval3dof_x, y=eval3dof_y)
    np.savez(eval_dir + "data6dof.npz", x=eval6dof_x, y=eval6dof_y)


    # Generate the readme content
    readme_lines = [
        "This dataset contains:",
        "- 3dof",
        f"  - Number of bags: {len(rosbag_paths)}",
        "   - List of used bags:"
    ]
    readme_lines += [f"     - {bag}" for bag in rosbag_paths]

    readme_lines += [
            "- 6dof",
            f"  - Number of bags: {len(rosbag_paths2)}",
            "   - List of used bags:"
    ]
    readme_lines += [f"     - {bag}" for bag in rosbag_paths2]
    
    readme_text = "\n".join(readme_lines)


    # Save the README
    with open(os.path.join(folder_path, 'README.txt'), 'w') as f:
        f.write(readme_text)


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

def process_data(df, bin_size=0.1, sim_timestep=0.1):

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

    # Compute true dt between bins (in seconds)
    dt = np.diff(agg_timestamps, prepend=agg_timestamps[0]) / 1e9
    dt[0] = dt[1]

    # Every bin spans ~bin_size seconds --> fixed sim steps
    steps = np.full(len(dt), int(round(bin_size / sim_timestep)), dtype=int)
    #print(steps)
    # Compute accelerations from velocities
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


def plot_all(actions, sim_pos, real_pos, sim_vel, real_vel, residuals, name, dir):

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

    # Add legend
    axs[1, 0].legend(loc='upper right', fontsize='small')
    axs[2, 0].legend(loc='upper right', fontsize='small')
    
    plt.tight_layout()
    out_path = os.path.join(dir, f"{name}.png")
    plt.savefig( out_path, dpi=300, bbox_inches="tight")
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

            qs = np.array(sim_rot_q)
            rotations = R.from_quat(qs)
            euler = rotations.as_euler('xyz', degrees=False)
            sim_rot = np.unwrap(euler)


            sim_dt = sim_timestep * num_sim_steps
            sim_acc_s = (sim_vel_s - prev_sim_vel) / sim_dt

            # Initialize rescale
            
            if step > 0:
                force_rescale = real_acc_s - sim_acc_s # Acc diff

                # Input: sim vel + acc + control | Output: force rescale (acc diff, real-sim)
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



if __name__ == "__main__":
    main()

