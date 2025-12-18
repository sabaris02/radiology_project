from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uuid
import os
import shutil
import cv2

from model import generate_heatmap
from xray_to_3d import xray_to_3d

app = FastAPI(title="X-ray Visualizer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    ext = file.filename.split(".")[-1]
    image_id = str(uuid.uuid4())

    original_path = f"{UPLOAD_DIR}/{image_id}_original.{ext}"
    heatmap_path = f"{UPLOAD_DIR}/{image_id}_heatmap.jpg"
    depth3d_path = f"{UPLOAD_DIR}/{image_id}_3d.jpg"

    # Save original
    with open(original_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Heatmap
    heatmap_img = generate_heatmap(original_path)
    cv2.imwrite(heatmap_path, heatmap_img)

    # 3D render
    depth_img = xray_to_3d(original_path)
    cv2.imwrite(depth3d_path, depth_img)

    return {
        "message": "Analysis complete",
        "original": f"http://127.0.0.1:8000/image/{os.path.basename(original_path)}",
        "heatmap": f"http://127.0.0.1:8000/image/{os.path.basename(heatmap_path)}",
        "depth_3d": f"http://127.0.0.1:8000/image/{os.path.basename(depth3d_path)}"
    }


@app.get("/image/{filename}")
def get_image(filename: str):
    return FileResponse(os.path.join(UPLOAD_DIR, filename))
