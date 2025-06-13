from optparse import OptionParser
import numpy as np
import os
from datetime import datetime

from training.methods.knn.knn import KNN
from training.methods.mtgp.mtgp import MTGP
from training.methods.svgp.gp import SVGP


def main():
    # parser
    parser = OptionParser()
    parser.add_option("-m", "--model", dest="model",
                default="all", help="Model type: 'knn', 'mtgp', 'svgp', or 'all'")
    parser.add_option("-c", "--complexity", dest="complexity",
                default="all", help="'3dof', '6dof' or 'all'")

    (options, args) = parser.parse_args()
    model_type = options.model
    complexity = options.complexity
    print("Selected model type:", model_type)

    # directories
    data_dir = "training/data/"
    data_folder = "data9_tet/" # TODO: add data dir as parser option
    train_data_dir = data_dir + data_folder + "train/"
    eval_data_dir  = data_dir + data_folder + "eval/"
    results_dir = create_results_dir()  # timestamped directory TODO make parser option

    elbo_dir = "data_and_plots/lossplots_tet"
    os.makedirs(elbo_dir, exist_ok=True)

    # train process
    for comp in ["3dof", "6dof"]:
        if complexity == "all" or complexity == comp:
            print(f"Training for complexity: {comp}")
            # load training and evaluation data
            train_data_x, train_data_y = load_data(train_data_dir, comp) 
            print("Train data X shape:", train_data_x.shape)
            print("Train data Y shape:", train_data_y.shape)

            eval_data_x, eval_data_y = load_data(eval_data_dir, comp)
            print("Eval data X shape:", eval_data_x.shape)
            print("Eval data Y shape:", eval_data_y.shape)

            # train model(s)
            for m_type in ["knn", "mtgp", "svgp"]:
                if model_type == "all" or model_type == m_type:
                    results_model_dir = results_dir + f"{m_type}/"
                    os.makedirs(results_model_dir, exist_ok=True)

                    if m_type == "svgp":
                        # this model only gives one output, so we need to loop through the outputs
                        print("Training SVGP models...")
                        for i in range(train_data_y.shape[1]):
                            # NOTE: index are represent (x, y, z, roll, pitch, yaw) = (0, 1, 2, 3, 4, 5)
                            print(f"Training SVGP for output index {i}...")
                            model = model_factory(m_type)#, output_index=i)
                            model.fit(inputs=train_data_x, targets=train_data_y[:, i])

                            elbo = os.path.join(elbo_dir, f"elbo_axis{i}_{comp}")
                            model.plot_loss(elbo)

                            print("Saving model...")
                            file_name = f"{m_type}_{comp}_output_{i}"
                            model.save(results_model_dir + file_name)

                            print("Evaluating model...")
                            yhat = model.predict(eval_data_x)
                            rmse = np.sqrt(np.mean((yhat - eval_data_y[:, i])**2))
                            print(f"RMSE on evaluation data for output index {i}: {rmse:.4f}")
                    else:
                        print(f"Training {m_type} model...")
                        model = model_factory(m_type)
                        model.fit(data_x=train_data_x, data_y=train_data_y)

                        print("Saving model...")
                        file_name = f"{m_type}_{comp}" 
                        model.save(results_model_dir + file_name)

                        print("Evaluating model...")
                        yhat = model.predict(data_x=eval_data_x)
                        rmse = np.sqrt(np.mean((yhat - eval_data_y)**2))
                        print(f"RMSE on evaluation data: {rmse:.4f}")
                        if m_type =="mtgp":
                            mt_elbo = os.path.join(elbo_dir, f"elbo_axis_{comp}_{m_type}")
                            model.plot_loss(mt_elbo)


def create_results_dir():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = f"training/results/tet/{timestamp}/"

    return results_dir


def model_factory(name, **kwargs):
    if name == "knn":
        return KNN(**kwargs)
    elif name == "mtgp":
        return MTGP(**kwargs)
    elif name == "svgp":
        return SVGP(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {name}")


def load_data(data_dir, complexity):
    data_file = "data" + complexity + ".npz"
    data = np.load(data_dir + data_file)
    data_x = data["x"]
    data_y = data["y"]
    return data_x, data_y


if __name__ == "__main__":
    main()
    #print("klar")
