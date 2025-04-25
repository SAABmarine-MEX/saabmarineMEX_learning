import socket
import pickle
import struct
import numpy as np

# Load your model here
with open("knn_model.pkl", "rb") as f:
    knn = pickle.load(f)

def handle_request(data):
    # Deserialize input
    input_vector = np.array(data).reshape(1, -1)

    # Predict with KNN or GP
    pred = knn.predict(input_vector)[0]

    return pred.tolist()

def start_server(host="127.0.0.1", port=5005):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((host, port))
    server.listen(1)
    print(f"[Python] Server listening on {host}:{port}")

    conn, addr = server.accept()
    print(f"[Python] Connection from {addr}")

    while True:
        # Receive length of payload
        raw_msglen = conn.recv(4)
        if not raw_msglen:
            break
        msglen = struct.unpack('>I', raw_msglen)[0]

        # Receive the actual payload
        data = b''
        while len(data) < msglen:
            packet = conn.recv(msglen - len(data))
            if not packet:
                return None
            data += packet

        # Deserialize the input
        input_data = pickle.loads(data)
        print(f"[Python] Received: {input_data}")

        # Predict
        response = handle_request(input_data)

        # Serialize response
        response_bytes = pickle.dumps(response)
        response_len = struct.pack('>I', len(response_bytes))
        conn.sendall(response_len + response_bytes)

start_server()
