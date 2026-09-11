from ultralytics import YOLO

def main():
    model = YOLO("yolo11s.pt")

    #Config YAML file path
    model.train(
        data="C:/Users/chakk/Desktop/CSODYOLO/ConstructionSiteObjectDetection/data.yaml",
        epochs=60,
        imgsz=1200,
        batch=4,
        workers=4,
        optimizer="SGD",
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.2
    )


if __name__ == "__main__":
    main()
