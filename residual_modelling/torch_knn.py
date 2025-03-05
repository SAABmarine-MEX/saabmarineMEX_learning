import torch
import torch.nn as nn
import numpy as np
import pickle

# --- CONFIGURATION ---
NUM_NEIGHBORS = 5  # K in KNN

# --- SIMULATED TRAINING DATA ---
# Split into position mismatch, velocity mismatch, and force input

# Position Mismatch (6DOF): [px, py, pz, roll, pitch, yaw]
pos_mismatch = np.array([
    [0.1, 0, 0, 0, 0, 0],  
    [0.3, 0, 0, 0, 0, 0],  
    [0.6, 0, 0, 0, 0, 0],  
    [1.0, 0, 0, 0, 0, 0],  
    [1.5, 0, 0, 0, 0, 0]
])

# Velocity Mismatch (6DOF): [vx, vy, vz, ωx, ωy, ωz]
vel_mismatch = np.array([
    [0.2, 0, 0, 0, 0, 0],  
    [0.4, 0, 0, 0, 0, 0],  
    [0.6, 0, 0, 0, 0, 0],  
    [0.8, 0, 0, 0, 0, 0],  
    [1.0, 0, 0, 0, 0, 0]
])

# 🚀 Applied Force Input (6DOF): [fx, fy, fz, τx, τy, τz]
force_input = np.array([
    [1, 0, 0, 0, 0, 0],  
    [1, 0, 0, 0, 0, 0],  
    [1, 0, 0, 0, 0, 0],  
    [1, 0, 0, 0, 0, 0],  
    [1, 0, 0, 0, 0, 0]
])

# 🔥 Combine all input features into `data_x`
data_x = np.hstack([pos_mismatch, vel_mismatch, force_input])

# 🔥 Compute acceleration mismatch (random example)
sim_accel = np.array([
    [2.1, 0, 0, 0, 0, 0],  
    [2.1, 0, 0, 0, 0, 0],  
    [2.1, 0, 0, 0, 0, 0],  
    [2.1, 0, 0, 0, 0, 0],  
    [2.1, 0, 0, 0, 0, 0]
])  # Simulated acceleration

real_accel = np.array([
    [2, 0, 0, 0, 0, 0],  
    [2, 0, 0, 0, 0, 0],  
    [2, 0, 0, 0, 0, 0],  
    [2, 0, 0, 0, 0, 0],  
    [2, 0, 0, 0, 0, 0]
])  # Real-world acceleration

eps = 1e-10
force_scaling = real_accel / (sim_accel+ eps)


# Convert to PyTorch tensors
X_train = torch.tensor(data_x, dtype=torch.float32)
Y_train = torch.tensor(force_scaling, dtype=torch.float32)

# --- KNN PREDICTION FUNCTION ---
def knn_predict(query, data_x, data_y, k=NUM_NEIGHBORS):
    """
    Predicts force scaling factors for a given query using KNN.
    """
    # Compute Euclidean distance to all training samples
    distances = np.linalg.norm(data_x - query, axis=1)
    
    # Find the indices of the k-nearest neighbors
    knn_indices = np.argsort(distances)[:k]
    
    # Get corresponding distances and target values
    knn_distances = distances[knn_indices]
    knn_targets = data_y[knn_indices]
    
    # Compute inverse distance weights (higher weight for closer neighbors)
    weights = 1 / (knn_distances + eps)  # Avoid division by zero
    weights /= weights.sum()  # Normalize weights
    
    # Compute weighted average of force scaling factors
    weighted_prediction = np.sum(knn_targets * weights[:, np.newaxis], axis=0)

    return weighted_prediction

# --- TESTING KNN PREDICTION ---
test_pos_mismatch = np.array([1.2, 0, 0, 0, 0, 0])  # Position mismatch (6DOF)
test_vel_mismatch = np.array([0.7, 0, 0.5, 0, 0, 0])  # Velocity mismatch (6DOF)
test_force_input = np.array([1, 0, 0, 0, 0, 0])  # Force input (6DOF)

test_input = np.hstack([test_pos_mismatch, test_vel_mismatch, test_force_input])  # Combine

predicted_scaling = knn_predict(test_input, data_x, force_scaling)

print(f"🔧 KNN-Predicted Force Scaling: {predicted_scaling}")