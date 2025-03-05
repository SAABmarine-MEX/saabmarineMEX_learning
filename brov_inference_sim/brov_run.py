"""
The purpose of this script is to demonstrate how the DRL will run for the actual brov. 
Eventually this will be ros intregrated, but this is just like a simple demo.

setup:
1. load .onnx file from the DRL (about the model: INPUT: two latest positions and next gate?, OUTPUT: 6DoF normalized control (-1-1: x,y,z,roll,pitch,yaw))
2. 

loop:
1. get mocap state
2. input to the model
3. send the output from the model to mavros rcin
"""
import torch
import onnx

## Setup
# Load ONNX model
onnx_model_path = "model.onnx"  # Replace with your ONNX file
onnx_model = onnx.load(onnx_model_path)

# Print model characteristics
print("Model IR Version:", onnx_model.ir_version)
print("Producer Name:", onnx_model.producer_name)
print("Opset Version:", onnx_model.opset_import[0].version)

# Model input and output
print("\nModel Inputs:")
for input in onnx_model.graph.input:
    print(f"- {input.name}: {input.type}")

print("\nModel Outputs:")
for output in onnx_model.graph.output:
    print(f"- {output.name}: {output.type}")


