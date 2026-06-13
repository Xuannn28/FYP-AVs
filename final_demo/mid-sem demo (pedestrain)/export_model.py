# ============================================================
# export_model.py — Run this on your PC / laptop (NOT on Pi)
#
# Exports YOLOv8n to NCNN format, which runs ~2x faster
# on the Raspberry Pi 4 ARM CPU compared to the .pt file.
#
# After running, copy the output folder to both Pis:
#   scp -r yolov8n_ncnn_model/ pi@<PI_IP>:~/YOLO/
#
# Then update config.py:
#   MODEL_PATH = "yolov8n_ncnn_model"
# ============================================================

from ultralytics import YOLO

def main():
    print("Downloading YOLOv8n weights (if not cached)...")
    model = YOLO("yolov8n.pt")

    print("Exporting to NCNN format (optimised for ARM CPU)...")
    model.export(
        format="ncnn",
        imgsz=640,     # keep at 640 — smaller = faster but less accurate
    )

    print()
    print("Export complete.")
    print("Output folder: ./yolov8n_ncnn_model/")
    print()
    print("Next steps:")
    print("  1. Copy folder to both Pis:")
    print("       scp -r yolov8n_ncnn_model/ pi@<EGO_PI_IP>:~/YOLO/")
    print("       scp -r yolov8n_ncnn_model/ pi@<RSU_PI_IP>:~/YOLO/")
    print("  2. In config.py set:  MODEL_PATH = 'yolov8n_ncnn_model'")


if __name__ == "__main__":
    main()
