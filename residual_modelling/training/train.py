from optparse import OptionParser
import numpy as np

from methods.knn.knn import KNN
from methods.mtgp.mtgp import MTGP
from methods.svgp.gp import SVGP


def main():
    # 1. Option parser
    parser = OptionParser()
    parser.add_option("-m", "--model", dest="model",
                  default="all", help="Model type: 'knn', 'mtgp', 'svgp', or 'all'")
    parser.add_option("-c", "--complexity", dest="complexity",
                      default="3dof", help="'3dof' or '6dof'") # TODO: implement 'all'

    (options, args) = parser.parse_args()
    model_type = options.model
    complexity = options.complexity
    print("Selected model type:", model_type)

    # 2. Set up directories
    data_dir = "data/"
    data_folder = "data2/"
    train_data_dir = data_dir + "train/" + data_folder
    eval_data_dir = data_dir + "eval/" + data_folder


    # 3. Train process
    for comp in ["3dof", "6dof"]:
        if complexity == "all" or complexity == comp:
            print(f"Training for complexity: {comp}")
            # load training and evaluation data
            train_data_x, train_data_y = load_data(train_data_dir, comp) # TODO: make sure the data files are in the right name structure for this to work. train/data_complexity.npz and eval/data_complexity.npz
            eval_data_x, eval_data_y = load_data(eval_data_dir, comp)

            # train model(s)
            for m_type in ["knn", "mtgp", "svgp"]:
                if model_type == "all" or model_type == m_type:
                    results_dir = "results/" + m_type + "/"
                    if m_type == "svgp":
                        # this model only gives one output, so we need to loop through the outputs
                        print("Training SVGP models...")
                        for i in range(train_data_y.shape[1]):
                            print(f"Training SVGP for output index {i}...")
                            model = model_factory(m_type)#, output_index=i)
                            model.fit(inputs=train_data_x, targets=train_data_y[:, i])

                            print("Saving model...")
                            file_name = f"{m_type}_{comp}_output_{i}"
                            model.save(results_dir + file_name)

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
                        model.save(results_dir + file_name)

                        print("Evaluating model...")
                        yhat = model.predict(data_x=eval_data_x)
                        rmse = np.sqrt(np.mean((yhat - eval_data_y)**2))
                        print(f"RMSE on evaluation data: {rmse:.4f}")

                        # TODO: add plots






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
