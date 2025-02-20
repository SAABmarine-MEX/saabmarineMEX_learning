"""
Structure:
1. Start the two simulations DONE
- sim1: raw simulation 
- sim2: with different physical params)

2. Measure difference between the two DONE
a) Same action/input for both
b) Measure position difference. with state from center of mass. the same will be given by the mocap system for the brov irl

3. residual dynamic modelling
- based on state diff and actuation input (direction 6dof) -> the missing acceleration to achive correctness 
(strategy from the drone paper)
"""

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
import numpy as np

# Init the two envs
env_prior_path = "envs/res_test_prior/res_prior.x86_64"
env_real_path = "envs/res_test_real/res_real.x86_64"
env_prior = UnityEnvironment(file_name=env_prior_path, seed=1, worker_id=0, side_channels=[])
print("Loaded prior env!")
env_real = UnityEnvironment(file_name=env_real_path, seed=1, worker_id=1, side_channels=[])
print("Loaded real env!")

env_real.reset()
env_prior.reset()

# Run the two envs with the same input
# Get the behavior name (assuming there's only one behavior in the environment)
behavior_name = list(env_prior.behavior_specs.keys())[0]
# Get the BehaviorSpec to check action spaces. Assuming they have the same behavior
behavior_spec_sim = env_prior.behavior_specs[behavior_name]
behavior_spec_real = env_real.behavior_specs[behavior_name]
# Print observation and action space details
print("ENV SIM")
print("Observation Shapes:", [obs.shape for obs in behavior_spec_sim.observation_specs])
print("Continuous Action Space:", behavior_spec_sim.action_spec.continuous_size)
cont_action_size = behavior_spec_sim.action_spec.continuous_size
print("Discrete Action Space:", behavior_spec_sim.action_spec.discrete_size)
print("\nENV REAL")
print("Observation Shapes:", [obs.shape for obs in behavior_spec_real.observation_specs])
print("Continuous Action Space:", behavior_spec_real.action_spec.continuous_size)
print("Discrete Action Space:", behavior_spec_real.action_spec.discrete_size)


# Residual modelling
# List to store residual errors for analysis
residual_errors = []
# Run for a fixed number of steps to measure differences
num_steps = 10
prior_decision_steps, terminal_steps_sim = env_prior.get_steps(behavior_name)
real_decision_steps, terminal_steps_real = env_real.get_steps(behavior_name)
for step in range(num_steps):
    # Get the current decision steps (agents that need actions) from both environments

    # Determine the number of agents currently requesting decisions
    # TODO: add something later to make sure it is adjusted for both env's number of agents. Now we just assume they are the same number
    num_agents = len(prior_decision_steps)

    # Max in 1 direction, the rest is 0 
    actions = np.zeros((num_agents, cont_action_size), dtype=np.float32)
    actions[:, 1] = 1
    print(1)
    # Wrap the actions in an ActionTuple (here we assume continuous actions only)
    action_tuple = ActionTuple(continuous=actions, discrete=None)
    print(2)
    # Apply the same actions to both environments
    env_prior.set_actions(behavior_name, action_tuple)
    env_real.set_actions(behavior_name, action_tuple)
    print(3)
    # Step both environments forward
    env_prior.step()
    env_real.step()
    print(4)
    # Retrieve the updated decision steps after the environment step
    prior_decision_steps, _ = env_prior.get_steps(behavior_name)
    real_decision_steps, _  = env_real.get_steps(behavior_name)
    print(5)
    # Compute the residual error between corresponding observations from both environments
    for agent_id in prior_decision_steps.agent_id:
        # Assume the first observation vector is the one of interest
        print("prior:")
        # transform.localPosition
        prior_obs = prior_decision_steps[agent_id].obs[1]
        real_obs  = real_decision_steps[agent_id].obs[1]
        print(f"0  {prior_obs[0]}  1  {prior_obs[1]}  2  {prior_obs[2]}")
        print("Real:")
        print(f"0  {real_obs[0]}  1  {real_obs[1]}  2  {real_obs[2]}")
        #print(real_obs)
        #print(real_decision_steps[agent_id].obs)
        # Here we use the Euclidean norm as the error metric; you could use other metrics as needed
        error = np.linalg.norm(prior_obs - real_obs)
        residual_errors.append(error)
        print(f"Step {step}, Agent {agent_id}: Residual error = {error:.4f}")

# Close the environments when finished
env_prior.close()
env_real.close()