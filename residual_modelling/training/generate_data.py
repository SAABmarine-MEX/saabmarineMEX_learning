import os
import numpy as np
import utils

def main():
    # Prep input
    n_steps = 200
    data_x = []
    data_y = []

    bag_dir = "ros2_bags/brov_tank_bags2"
        #Only keyboard test
    bags =  [ 
        "rosbag2_2025_05_14-16_33_33", #
        "rosbag2_2025_05_14-17_12_31" #
    ]

    # Get the bag paths TODO: could make as function in utils.py
    rosbag_paths = []
    for name in bags:
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

        df = utils.load_rosbag(rosbag_path)
        data = utils.process_data(df)
        
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
            #env_path = "envs/sitl_envs/v5/prior/prior.x86_64"
            env_path = "../envs/sitl_envs/v5/prior/prior.x86_64"
            result = utils.run_simulation(ctrl_seg, step_seg, pos_seg, vel_seg, acc_seg, n_steps, env_path)

            print(f"\nPlotting results for {rosbag_path}...\n")
            utils.plot_all(
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

    # Folder to save in
    folder_path = "data/data1/"
    os.makedirs(folder_path, exist_ok=True)

    # Generate the readme content
    readme_lines = [
        "This dataset contains:",
        f"- Number of bags: {len(rosbag_paths)}",
        "- List of used bags:"
    ]
    readme_lines += [f"  - {bag}" for bag in rosbag_paths]

    readme_text = "\n".join(readme_lines)

    # Save the README
    with open(os.path.join(folder_path, 'README.txt'), 'w') as f:
        f.write(readme_text)

    # np.save("data/data_x.npy", data_x)
    # np.save("data/data_y.npy", data_y)
    np.savez(folder_path + "data.npz", x=data_x, y=data_y)


if __name__ == "__main__":
    main()

