# server.py

from fastapi import FastAPI, Request, HTTPException, Response
import model_pb2

app = FastAPI()

@app.post("/predict", response_class=Response)
async def predict(request: Request):
    # 1) Read the raw HTTP data (bytes)
    data = await request.data()

    # 2) Parse into our protobuf msgutFeatures
    msg = model_pb2.InputFeatures()
    try:
        msg.ParseFromString(data)
    except Exception:
        # invalid bytes → 400 Bad Request
        raise HTTPException(status_code=400, detail="Invalid protobuf")

    # 3) Debug print to console
    print(f"[server] Received features (first 5): {msg.features[:5]}... total={len(msg.features)}")

    # 4) Build a Prediction message
    #    For testing we only return the first 6 floats:
    out = model_pb2.Prediction()
    # clamp to 6 elements
    sliced = msg.features[:6] if len(msg.features) >= 6 else msg.features
    out.residuals.extend(sliced)

    # 5) Serialize to bytes
    data = out.SerializeToString()

    # 6) Return raw bytes with octet-stream content type
    return Response(content=data, media_type="application/octet-stream")