import numpy as np
import matplotlib.pyplot as plt

# Constants
rho_mat = 800       # kg/m³ (cable material density)
rho_water = 1025    # kg/m³ (water density)
diameter = 0.02     # m (cable diameter)
Cd = 1.2            # drag coefficient
g = 9.82            # gravity

# Derived constants
A = np.pi * (diameter / 2)**2  # cross-sectional area
b_per_meter = (rho_water - rho_mat) * A * g  # N/m (buoyancy per depth)


# Simulated ranges
depths = np.linspace(0, 3, 100)          # 0 to 3 meters
speeds = np.linspace(0, 1, 50)           # 0 to 1 m/s


# Create a grid of depths and speeds
D, V = np.meshgrid(depths, speeds)       # D: depth, V: horizontal speed


# Force calculations
F_buoy = b_per_meter * D                 # Upward force from buoyant tether
#F_drag = 0.5 * rho_water * Cd * A * V**2  # Drag force

# Drag formula: F = -0.5 * rho * Cd * D * L * v * |v|
F_drag = 0.5 * rho_water * Cd * diameter * D * V**2  # Horizontal drag

# Total force
F_total = np.sqrt(F_buoy**2 + F_drag**2)


# Plotting
fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

c1 = axs[0].contourf(D, V, F_buoy, 20, cmap='viridis')
axs[0].set_title("Buoyant Force (N)")
axs[0].set_xlabel("Depth (m)")
axs[0].set_ylabel("Horizontal Speed (m/s)")
fig.colorbar(c1, ax=axs[0])

c2 = axs[1].contourf(D, V, F_drag, 20, cmap='plasma')
axs[1].set_title("Horizontal Drag (N)")
axs[1].set_xlabel("Depth (m)")
fig.colorbar(c2, ax=axs[1])

c3 = axs[2].contourf(D, V, F_total, 20, cmap='magma')
axs[2].set_title("Total Tether Force Magnitude (N)")
axs[2].set_xlabel("Depth (m)")
fig.colorbar(c3, ax=axs[2])

plt.tight_layout()
plt.show()



