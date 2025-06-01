#!/usr/bin/env python3

# from process import process
import os
from gp import SVGP
import numpy as np
from optparse import OptionParser
import numpy as np
#import open3d as o3d


def train_svgp(gp_inputs_type, survey_name, data_x, data_y, dof):
    inputs = data_x
    print("Inputs:", inputs.shape)
    targets = data_y
    print("Targets:", targets.shape)

    print(gp_inputs_type)
    if gp_inputs_type == 'di':
        name = "svgp_di"
        covariances = None
    else:
        ## UI
        covariances = cloud['covs']
        print("Covariances ", covariances.shape)
        name = "svgp_ui"    

    name += "_" + dof


    # 2. Train GP
    n_inputs = inputs.shape[1]
    # Initialise GP with 1000 inducing points
    gp = SVGP(400, n_inputs)
    gp.fit(inputs, targets, covariances=covariances, n_samples=1000, 
            max_iter=1000, learning_rate=1e-1, rtol=1e-12, n_window=2000, 
            auto=False, verbose=True)


    # 3. Post processing
    # Save GP
    print("Saving trained GP")
    gp.save(survey_name + name + '.pth')

    # TODO: try replicate below outcommented for this case
    # Save figures
    print("Plotting results")
    #gp.plot(inputs, targets, name + '.png',
    #         n=100, n_contours=100) # NOTE: this one only works for 2D inputs. we have 18 inputs
    gp.plot_loss(name + '_loss.png')
    
    # Save loss for tunning of stopping criterion
    np.save(name + '_loss.npy', np.asarray(gp.loss))

    # Save posterior
    # print("Saving posterior")
    # x = inputs[:,0]
    # y = inputs[:,1]
    # gp.save_posterior(1000, min(x), max(x), min(y), max(y), 
    #                   name + '_post.npy', verbose=False)


def trace_kernel(gp_path):

    gp = SVGP.load(1000, gp_path)
    gp.likelihood.eval()
    gp.eval()

    print("Kernel")
    ip = gp.variational_strategy.inducing_points.data
    print(np.trace(gp.cov(ip).cpu().numpy()))


def load_plot(gp_path, survey_name, trajectory_name):
    gp = SVGP.load(400, gp_path)
    gp.likelihood.eval()
    gp.eval()
    
    """
    pcd = o3d.io.read_point_cloud(survey_name)
    pcd = pcd.uniform_down_sample(every_k_points=3)
    points = np.asarray(pcd.points)

    inputs = points[:, [0,1]]
    print("Inputs ", inputs.shape)
    targets = points[:,2]
    print("Targets ", targets.shape)

    track_file = np.load(trajectory_name)
    track = track_file["track_position"]

    name = "svgp_di"
    gp.plot(inputs, targets, name + '.png',
             n=100, n_contours=100, track=track)
    """


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--gp_inputs", dest="gp_inputs",
                  default="di", help="di or ui inputs for training.")
    parser.add_option("--survey_name", dest="survey_name",
                  default="../../../results/svgp/", help="Name for folder to store results.")
    parser.add_option("--dof", dest="gp_dof",
                  default="x", help="Which dof to train; 'x', 'y', 'z', 'roll', 'pitch', 'yaw' or 'all'.")

    (options, args) = parser.parse_args()
    gp_inputs_type = options.gp_inputs
    survey_name = options.survey_name
    gp_dof = options.gp_dof


    # Load data
    data_dir = "../../data/data1/"
    data_file = "data.npz"
    data = np.load(data_dir + data_file)
    data_x = data["x"]
    data_y = data["y"]
    print("Data X shape:", data_x.shape)
    print("Data Y shape:", data_y.shape)

    print("Train dof:", gp_dof)
    dof_to_indices = {
        'x': 0, 'y': 1, 'z': 2,
        'roll': 3, 'pitch': 4, 'yaw': 5,
    }
    if gp_dof == "all":
        for dof, index in dof_to_indices.items():
            print(f"Training for dof: {dof}")
            train_svgp(gp_inputs_type, survey_name, 
                       data_x, data_y[:, index], dof)
    else:
        train_svgp(gp_inputs_type, survey_name,
                   data_x, data_y[:, dof_to_indices[gp_dof]], gp_dof)

    # load_plot(gp_path, survey_name, track_path)
    # trace_kernel(survey_name)
