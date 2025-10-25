# coral-reef-detection

### setup

Download all data from [the drive](https://drive.google.com/drive/folders/1mOuhlo0y-b65eo8QzlyUYLQMpwmvJYXF) and put it in a data folder according to this structure:
```
data
├── coral_bleaching
│   ├── others
│   ├── reef_support
├── benthic_datasets
│   ├── point_labels
│   ├── mask_labels 
```

## Training YOLOv12 on Coral Bleaching Segmentation

Follow these steps to train a YOLOv12 segmentation model using the provided pipeline:

1. **Clone YOLOv12**
    ```bash
    git clone https://github.com/sunsmarterjie/yolov12.git
    ```


2. Install the required dependencies by running:
    ```bash
    pip install -r requirements.txt
    ```
    in the `yolov12` directory/environment.

   Note: If your OS is not Linux, remove the Linux-only wheel entry from requirements.txt before installing. Specifically, delete the line:
   ```bash
   flash_attn-2.7.3+cu11torch2.2cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
   ```


4. **Download Pretrained Weights**  
   Download the YOLOv12 segmentation model weights:
    - [yolov12m-seg.pt](https://github.com/sunsmarterjie/yolov12/releases/download/seg/yolov12m-seg.pt)
    - Place the file inside the `yolov12` directory.


5. **Prepare the Pipeline Code**
    - Copy your preprocessing and training scripts (such as `image_structure.py`, `masks_to_yolo_polygons.py`, and `yoloseg_train.py`) into the `yolov12` directory.

    
6. **Edit Paths in Scripts**
    - Open each `.py` file (`image_structure.py`, `masks_to_yolo_polygons.py`, etc.) and update the dataset paths to point to your local data.  
      _For example:_  
      ```python
      ROOT = Path("C:/Users/yourname/path/to/data/coral_bleaching/reef_support")
      ```


7. **Organize Images for YOLO**
    - Run the following to create the YOLO folder structure (`yolo_seg`) inside `reef_support`:
      ```bash
      python image_structure.py
      ```
    - This will create `yolo_seg/images/train` and `yolo_seg/images/val`.


8. **Update Dataset YAML**
    - Edit `coral_seg.yaml` and set the `train` and `val` paths to your new YOLO structure.  
      _Example:_  
      ```yaml
      train: DC3/coral_bleaching/reef_support/yolo_seg/images/train
      val: DC3/coral_bleaching/reef_support/yolo_seg/images/val
      ```
    - Make sure to update the `nc` (number of classes) and `names` fields if needed.


9. **Convert Masks to YOLO Polygon Labels**
    - Run:
      ```bash
      python masks_to_yolo_polygons.py
      ```
    - This will generate YOLO format label files in `yolo_seg/labels/train` and `yolo_seg/labels/val`.


10. **Train the YOLOv12 Segmentation Model**
    - Start training using your custom data:
      ```bash
      python yoloseg_train.py
      ```

**Tip:**  
- Adjust `epochs`, `batch`, and `imgsz` in the training command as needed.
- For troubleshooting or visualizing results, check the outputs under the `runs/` directory.

---

_This section guides you through setting up and training a YOLOv12 segmentation model using your coral bleaching dataset and custom preprocessing pipeline._
