from ultralytics import YOLO

def main():
    model = YOLO("yolov12m-seg.pt")  # or "yolov12m-seg.pt" if available
    results = model.train(
        data=r"C:/Users/20235050/Downloads/BDS_Y3/DC3/yolov12/data/coral_seg.yaml", #path to coral_seg.yaml
        imgsz=640,
        epochs=10,
        batch=4,        
        device=0,
        workers=2,
        amp=1,
        # Disasbeling buit-in augmentation
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.0,
        erasing=0.0,
        rect=True,
        multi_scale=False
    )

if __name__ == "__main__":
    main()
