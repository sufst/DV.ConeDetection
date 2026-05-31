from ultralytics import YOLO

# Path to data.yaml (edit if you move the file)
DATA_YAML = "data.yaml"

# Model choice: yolov8n.pt is the smallest, fastest; yolov8s.pt is more accurate
# You can use yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
MODEL_NAME = "yolov8l.pt"  # Changed to yolov8l.pt for better accuracy

# Training parameters
EPOCHS = 50
IMG_SIZE = 640

if __name__ == "__main__":
    # Download model if not present
    model = YOLO(MODEL_NAME)

    # Train
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        project="runs",
        name="yolov8_cone",
        exist_ok=True,
    )

    print("Training complete. Best weights are in the runs/yolov8_cone/ folder.")
