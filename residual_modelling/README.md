# Residual Dynamic Modelling for BlueROV2 sim

This repository implements a residual learning pipeline to improve underwater vehicle (BlueROV2) simulations by learning from real-world data (underwater MoCap).
The system combines ROS2 bag processing, Unity simulation and data-driven learning methods (KNN & GP) to estimate residual dynamics between simulation and reality.

## Dependencies
To install dependencies use:

```bash
pip install -r requirements.txt
```

To install ml-agents go to: https://github.com/SAABmarine-MEX/saabmarineMEX_learning/rl_training

To get appropriate ros2 for the real-world data go to: https://github.com/SAABmarine-MEX/saabmarineMEX_ros2

and

```bash
source install/setup.bash
```

## Folder Structure

```
.
├── data_and_plots/        (Data and plots from running the scripts)
├── envs/                  (Unity environments)
├── inference/
│   ├── fastserver.py      (FastAPI server runs the models via a Protobuf interface) 
│   ├── model_pb2.py       (Auto-generated Protobuf file)
│   └── res_inf.py         (3. Run the model inference in the sim)  
└── training/
    ├── generate_data.py   (1. Generate the residual difference data, sim vs. real)
    ├── methods/           (Model scripts (KNN, MTGP & SVGP)
    ├── results/
    ├── ros2_bags/
    └── train.py           (2. Train the models: KNN, MTGP & SVGP)
```

## Usage

### 1. Data Generation
Run `generate_data.py` to:

- Load and bin synchronized data from ROS2 bags.

- Run Unity simulation to generate residual targets.

- Save data in `npz` format for training and evaluation.
                           
```bash 
python -m training.generate_data
```

### 2. Model Training
Train one or all supported models (KNN, MTGP, SVGP):

```bash 
python -m training.train --model [knn|mtgp|svgp|all] --complexity [3dof|6dof|all]
```

### 3. Evaluation
Use `res_inf.py` to:

- Launch simulations with & without residual models.

- Launch fastserver.py for real-time residual inference 

- Compute residual metrics and visualize performance.
                     
```bash
python -m inference.res_inf
```

## Outputs

- Training/eval datasets in `training/data/`
  
- Trained models in `training/results/`

- Inference plots in `data_and_plots/`

- Evaluation metrics saved as `.npz` and `.png`
