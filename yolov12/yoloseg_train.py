from ultralytics import YOLO

def main():
    model = YOLO("yolov12m-seg.pt")  # or "yolov12m-seg.pt" if available
    results = model.train(
        data=r"C:/Users/20235050/Downloads/BDS_Y3/DC3/yolov12/data/coral_seg.yaml", #path to coral_seg.yaml
        imgsz=640,
        epochs=10,
        batch=4,        # adjust for your RTX A1000 VRAM
        device=0,
        workers=2,
        amp=1
    )

if __name__ == "__main__":
    main()