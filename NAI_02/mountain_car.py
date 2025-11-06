"""
==============================================================
Project: Fuzzy Logic Controller for MountainCar-v0
==============================================================

 Problem Description
----------------------
The MountainCar-v0 environment (from the Gymnasium library) is a classic
control problem. A car is stuck in a valley between two hills and its
engine is not powerful enough to drive directly up the right hill.

The goal is to reach the top of the right hill (position ≥ 0.5).
To do this, the car must first reverse left to gain momentum,
then accelerate right to climb the hill.

Instead of using reinforcement learning, this project demonstrates
a **fuzzy logic control system** — an approach based on intuitive
human reasoning using "if–then" rules such as:

    "If the car is on the left and moving backward, push right."

 Fuzzy system inputs:
    • `position` — car's horizontal location (range: [-1.2, 0.6])
    • `velocity` — car's movement speed (range: [-0.07, 0.07])
    • `slope` — steepness of the terrain (derived from position)

 Output:
    • `force` — driving direction and intensity (left / neutral / right)

The fuzzy controller continuously reads environment values, evaluates
the fuzzy rules, and decides which discrete action to apply in real time.

 Authors
------------
• Marek Jenczyk
• Oskar Skomra

 Environment Setup
---------------------
1. Install the required Python packages:
   ```bash
   pip install gymnasium[classic-control]
   pip install scikit-fuzzy numpy matplotlib
"""

import gymnasium as gym
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# ==============================================================
# Define fuzzy input/output variables
# ==============================================================

# Antecedents (inputs): position, velocity, and slope
# Consequent (output): force to apply
position = ctrl.Antecedent(np.arange(-1.2, 0.6, 0.01), 'position')
velocity = ctrl.Antecedent(np.arange(-0.07, 0.07, 0.001), 'velocity')
slope = ctrl.Antecedent(np.arange(-1, 1, 0.05), 'slope')
force = ctrl.Consequent(np.arange(-1, 1, 0.1), 'force')

# ==============================================================
# Define membership functions
# ==============================================================

# Position: left, center, right
position['left'] = fuzz.trimf(position.universe, [-1.2, -1.2, -0.5])
position['center'] = fuzz.trimf(position.universe, [-1.0, -0.3, 0.2])
position['right'] = fuzz.trimf(position.universe, [-0.2, 0.6, 0.6])

# Velocity: backward, zero, forward
velocity['backward'] = fuzz.trimf(velocity.universe, [-0.07, -0.07, 0])
velocity['zero'] = fuzz.trimf(velocity.universe, [-0.02, 0, 0.02])
velocity['forward'] = fuzz.trimf(velocity.universe, [0, 0.07, 0.07])

# Slope: downhill, flat, uphill
slope['downhill'] = fuzz.trimf(slope.universe, [-1, -1, 0])
slope['flat'] = fuzz.trimf(slope.universe, [-0.2, 0, 0.2])
slope['uphill'] = fuzz.trimf(slope.universe, [0, 1, 1])

# Force (output): push left, neutral, push right
force['left'] = fuzz.trimf(force.universe, [-1, -1, 0])
force['neutral'] = fuzz.trimf(force.universe, [-0.3, 0, 0.3])
force['right'] = fuzz.trimf(force.universe, [0, 1, 1])


# ==============================================================
# Define fuzzy control rules
# ==============================================================

rules = [
    ctrl.Rule(position['left'] & velocity['backward'], force['right']),
    ctrl.Rule(position['left'] & velocity['forward'], force['right']),
    ctrl.Rule(position['center'] & velocity['forward'] & slope['uphill'], force['right']),
    ctrl.Rule(position['center'] & velocity['backward'] & slope['downhill'], force['left']),
    ctrl.Rule(position['right'] & velocity['forward'], force['right']),
    ctrl.Rule(position['right'] & velocity['backward'], force['left']),
    ctrl.Rule(velocity['zero'] & slope['flat'], force['neutral']),
]

# Build the fuzzy control system
force_ctrl = ctrl.ControlSystem(rules)
force_sim = ctrl.ControlSystemSimulation(force_ctrl)


# ==============================================================
# Helper functions
# ==============================================================

def get_slope(pos: float) -> float:
    """
    Compute the slope (steepness) of the MountainCar terrain
    at a given position.

    The terrain shape is y = sin(3x), so the slope (dy/dx)
    is the derivative: 3 * cos(3x).

    Args:
        pos (float): current position of the car

    Returns:
        float: slope value in the range [-3, 3]
    """
    return np.cos(3 * pos) * 3


def decide_action(pos: float, vel: float, slp: float) -> int:
    """
    Compute the fuzzy control output based on position, velocity,
    and slope, then map it to a discrete action for the Gymnasium
    environment.

    Args:
        pos (float): current car position (-1.2 to 0.6)
        vel (float): current car velocity (-0.07 to 0.07)
        slp (float): current terrain slope (-1 to 1, normalized)

    Returns:
        int: discrete action:
             0 = push left
             1 = do nothing
             2 = push right
    """
    # Clamp inputs to valid ranges
    force_sim.input['position'] = np.clip(pos, -1.2, 0.6)
    force_sim.input['velocity'] = np.clip(vel, -0.07, 0.07)
    force_sim.input['slope'] = np.clip(slp, -1, 1)

    # Compute fuzzy inference
    force_sim.compute()

    # Retrieve crisp output force
    force_value = force_sim.output.get('force', 0)

    # Convert continuous force value into discrete Gym action
    if force_value < -0.2:
        return 0  # push left
    elif force_value > 0.2:
        return 2  # push right
    else:
        return 1  # no push


# ==============================================================
# Environment simulation loop
# ==============================================================

env = gym.make("MountainCar-v0", render_mode="human")

def run_simulation(seed: int = 44):
    """
    Run one episode of the MountainCar-v0 environment using
    the fuzzy logic controller.

    The environment is reset with a fixed seed for reproducibility.
    At each step, position, velocity, and slope are read, passed to
    the fuzzy controller, and the resulting action is applied until
    the car reaches the goal or the episode ends.
    """
    obs, info = env.reset(seed=seed)
    done = False

    while not done:
        pos, vel = obs
        slp = get_slope(pos)
        action = decide_action(pos, vel, slp)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    print("Simulation finished successfully.")

if __name__ == "__main__":
    run_simulation()
    env.close()
