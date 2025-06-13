import numpy as np
import matplotlib.pyplot as plt

# === Constants (from Unity setup) ===
rho_mat = 800         # kg/m³
rho_water = 1025      # kg/m³
diameter = 0.02       # m
Cd = 1.2              # drag coefficient
g = 9.82              # m/s²

# === Derived constants ===
A = np.pi * (diameter / 2)**2
b_per_meter = (rho_water - rho_mat) * A * g  # N/m

# === Simulation parameters ===
depths = np.linspace(0, 3, 50)      # 0–3 m depth
v_range = np.linspace(0, 1, 50)     # 0–1 m/s

# Create meshgrid for velocities (vx and vy)
VX, VY = np.meshgrid(v_range, v_range)
speed_squared = VX**2 + VY**2

# Choose a single depth or loop later
depth = 2.0  # meters
L = depth

# === Buoyant force ===
F_buoy = b_per_meter * L  # constant for fixed depth

# === Drag force components ===
F_drag_mag = 0.5 * rho_water * Cd * diameter * L * speed_squared
F_drag_x = -F_drag_mag * VX / np.sqrt(speed_squared + 1e-6)
F_drag_y = -F_drag_mag * VY / np.sqrt(speed_squared + 1e-6)

# === Total force vector components ===
F_total_x = F_drag_x
F_total_y = F_drag_y
F_total_z = F_buoy  # constant (upward)

# === Total force magnitude ===
F_total_mag = np.sqrt(F_total_x**2 + F_total_y**2 + F_total_z**2)

# === Plotting ===
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# Force magnitude
c1 = axs[0].contourf(VX, VY, F_total_mag, 20, cmap='viridis')
axs[0].set_title("Total Force Magnitude (N)")
axs[0].set_xlabel("Forward Velocity (vx) [m/s]")
axs[0].set_ylabel("Side Velocity (vy) [m/s]")
fig.colorbar(c1, ax=axs[0])

# X drag component
c2 = axs[1].contourf(VX, VY, F_drag_x, 20, cmap='coolwarm')
axs[1].set_title("X-Drag Force (N)")
axs[1].set_xlabel("vx [m/s]")
axs[1].set_ylabel("vy [m/s]")
fig.colorbar(c2, ax=axs[1])

# Y drag component
c3 = axs[2].contourf(VX, VY, F_drag_y, 20, cmap='coolwarm')
axs[2].set_title("Y-Drag Force (N)")
axs[2].set_xlabel("vx [m/s]")
axs[2].set_ylabel("vy [m/s]")
fig.colorbar(c3, ax=axs[2])

plt.tight_layout()
plt.show()

