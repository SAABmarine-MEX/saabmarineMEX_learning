# How to train

`mlagents-learn <trainer-config-file> --env=<env_name> --run-id=<run-identifier>`

Pass `--no-graphics` to run training without graphics. This will increase the efficiency of the training since scene rendering is not needed. It can also be directly added into the behavior file. See `config/Brov.yaml` for example.

Your `<trainer-config-file>` is found in `config/` and `<env_name>` is found in `envs/`

## Training Using Concurrent Unity Instances
"In order to run concurrent Unity instances during training, set the number of environment instances using the command line option `--num-envs=<n>` when you invoke `mlagents-learn`. Optionally, you can also set the --base-port, which is the starting port used for the concurrent Unity instances.", https://github.com/Unity-Technologies/ml-agents/blob/develop/docs/Training-ML-Agents.md. 
