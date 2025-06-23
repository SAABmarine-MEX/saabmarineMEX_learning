# RL Training
The rl training utilizes [Unity Machine Learning Agents Toolkit (ML-Agents)](https://github.com/Unity-Technologies/ml-agents/blob/develop/docs/ML-Agents-Overview.md). This is used in the Unity simulation to create environment builds (more about the simulation can be found in [SAABmarine-MEX](https://github.com/SAABmarine-MEX)). These builds are then used here on the training side of things.

## Folder Structure
```
.
├── builds/ (containts the environment builds from the simulation)
├── config/ (contains the training configuration yaml files)
└── results/ (contains the results from the training)
```

## Installation

### Local installation
```
conda create -n mlagents python=3.10.12 && conda activate mlagents
conda install numpy=1.23.5
```
Depending if you are using a CUDA GPU or CPU for training the bellow command will differ.

* For CPU:
```
pip3 install torch~=2.2.1 --index-url https://download.pytorch.org/whl/cpu
```
* For CUDA GPU:
```
pip3 install torch~=2.2.1 --index-url https://download.pytorch.org/whl/cu121
```

```
# Clone this somewhere approriate outside this repo
git clone --branch release_22 https://github.com/Unity-Technologies/ml-agents.git

cd ml-agents/
python -m pip install ./ml-agents-envs
python -m pip install ./ml-agents

# As a last check, if you get this to run without errors then you have had a successfull installation!
mlagents-learn --help
```

### Google Cloud Platform (GCP) VM installation
As most of the work of this project was done with a laptop without CUDA GPU, a GCP VM was create in order to give better training performance for the RL.

Optional: Local machine if you want to use a pre-emptible instance 
```
gcloud compute instances add-metadata mlagents-preemptible \
  --zone=europe-west1-b \
  --metadata-from-file shutdown-script=./shutdownscript.sh
```

Create a VM with some reasonable instance: 
```
Deep Learning VM with CUDA 12.1 M126
Debian 11, Python 3.10. With CUDA 12.1 preinstalled.
```
For a T4 or L4 GPU instance works reasonably well with 12-16 vCPUs. Use spot/preemptible instances for a massive price cut!!!


#### Main installation process in the VM
```
conda create -n mlagents python=3.10.12 && conda activate mlagents

pip3 install torch~=2.2.1 --index-url https://download.pytorch.org/whl/cu121

mkdir workspace
cd workspace
git clone https://github.com/martkartasev/ml-agents.git -b vf_save_better

pip install -e ./ml-agents/ml-agents-envs
pip install -e ./ml-agents/ml-agents
```

Rights in folder to copy to in VM
```
mkdir config
mkdir builds
sudo chmod +777 ./builds/ # On VM
sudo chmod +777 ./config/
```

#### Bind tensorboard

Got to open ports in GCP first! Use --bind_all to bind to network and make it available.

```
tensorboard --logdir results --bind_all
```

## How to train

`mlagents-learn <trainer-config-file> --env=<env_name> --run-id=<run-identifier>`

Pass `--no-graphics` to run training without graphics. This will increase the efficiency of the training since scene rendering is not needed. It can also be directly added into the behavior file. See `config/Brov.yaml` for example.

Your `<trainer-config-file>` is found in `config/` and `<env_name>` is found in `envs/`.

## Training Using Concurrent Unity Instances
"In order to run concurrent Unity instances during training, set the number of environment instances using the command line option `--num-envs=<n>` when you invoke `mlagents-learn`. Optionally, you can also set the --base-port, which is the starting port used for the concurrent Unity instances.", https://github.com/Unity-Technologies/ml-agents/blob/develop/docs/Training-ML-Agents.md. 

# GCP stuff

## Connect

gcloud compute ssh mlagents-preemptible

## Copy files from local machine

 gcloud compute scp --recurse ~/mex/DRL-Python/rl_training/builds/* mlagents-preemptible:/home/albin/workspace/builds --zone=europe-west1-b


## To run on GCP 

This commands are ran in the terminal of the GCP instance.

`sudo apt install -y libgl1 libglib2.0-0 libnss3 libxrandr2 libxcursor1 libxinerama1 libxi6 libasound2 libpulse0`

`sudo apt install -y xvfb`

 xvfb-run -a mlagents-learn config/Brov.yaml --env=builds/env_simple.x86_64 --run-id=simple_test1 --no-graphics
