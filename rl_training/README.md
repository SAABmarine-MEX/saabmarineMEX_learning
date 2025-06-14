# How to train

`mlagents-learn <trainer-config-file> --env=<env_name> --run-id=<run-identifier>`

Pass `--no-graphics` to run training without graphics. This will increase the efficiency of the training since scene rendering is not needed. It can also be directly added into the behavior file. See `config/Brov.yaml` for example.

Your `<trainer-config-file>` is found in `config/` and `<env_name>` is found in `envs/`

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
