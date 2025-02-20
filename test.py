import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple

# This is a non-blocking call that only loads the environment.
env_path = "envs/3DBall_example/3DBall.x86_64"
env = UnityEnvironment(file_name=env_path, seed=1, side_channels=[])
# Start interacting with the environment.
env.reset()

# Get the behavior name (assuming there's only one behavior in the environment)
behavior_name = list(env.behavior_specs.keys())[0]

# Get the BehaviorSpec to check action spaces
behavior_spec = env.behavior_specs[behavior_name]

# Print observation and action space details
print("Observation Shapes:", [obs.shape for obs in behavior_spec.observation_specs])
print("Continuous Action Space:", behavior_spec.action_spec.continuous_size)
print("Discrete Action Space:", behavior_spec.action_spec.discrete_size)

# Main interaction loop
for episode in range(5):  # Run for 5 episodes
    env.reset()
    
    while True:
        # Get DecisionSteps and TerminalSteps for the current step
        decision_steps, terminal_steps = env.get_steps(behavior_name)

        # Check if all agents are done
        if len(terminal_steps) > 0:
            break  # Episode ends when at least one agent has reached a terminal state
        
        # Create an action array for all agents
        num_agents = len(decision_steps)
        continuous_size = behavior_spec.action_spec.continuous_size
        
        if continuous_size > 0:
            # Generate random continuous actions (shape: num_agents x continuous_size)
            actions = np.random.uniform(-1, 1, size=(num_agents, continuous_size)).astype(np.float32)
            action_tuple = ActionTuple(continuous=actions, discrete=None)
        else:
            # If using discrete actions (not in 3DBall, but for reference)
            discrete_size = behavior_spec.action_spec.discrete_size
            actions = np.random.randint(0, 2, size=(num_agents, discrete_size)).astype(np.int32)
            action_tuple = ActionTuple(continuous=None, discrete=actions)
        
        # Set actions for all agents
        env.set_actions(behavior_name, action_tuple)

        # Perform a simulation step
        env.step()      

# Close the environment when done
env.close()