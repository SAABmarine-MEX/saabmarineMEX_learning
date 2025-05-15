
# OBS OUTDATED

# README Python <-> Unity 

# install in correct vm with your paths (may be overkill)
pip install \
  fastapi \
  uvicorn[standard] \
  protobuf==3.20.* \
  requests


# Proto model definition in: /home/michael-exjobb/mex/protos/model.proto

syntax = "proto3";
package mex;

// Input vector: [sim_vel, sim_acc, control] length = F
message InputFeatures {
  repeated float features = 1;
}
// Output vector: predicted residuals length = R
message Prediction {
  repeated float residuals = 1;
}

# Create protomodel
protoc \
  --proto_path=/home/michael-exjobb/mex/protos \
  --python_out=/home/michael-exjobb/mex/DRL-Python/residual_modelling \
  /home/michael-exjobb/mex/protos/model.proto

# Gives: /home/michael-exjobb/mex/DRL-Python/residual_modelling/model_pb2.py


# To run server (in this case: /home/michael-exjobb/mex/DRL-Python/residual_modelling/server_test.py)
python -m uvicorn server_test:app --host 0.0.0.0 --port 8000
# This is --workers 1


# To run client atm (in this case: /home/michael-exjobb/mex/unity/SMARCUnityAssets/Runtime/Scripts/BlueROV2/SAABmarineMEX/BlueROV2/client_test.py)

export PYTHONPATH=/home/michael-exjobb/mex/DRL-Python/residual_modelling
python /home/michael-exjobb/mex/unity/SMARCUnityAssets/Runtime/Scripts/BlueROV2/SAABmarineMEX/BlueROV2/client_test.py
