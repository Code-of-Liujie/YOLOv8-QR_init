from ultralytics import YOLO
if __name__ == '__main__':
    # Load a model
    # model = YOLO('yolov8m.pt')  # load an official model
    model = YOLO(r"E:\DCNV3\ultralytics-main\ultralytics\models\yolo\detect\runs\detect\train383\weights\best.pt")  # load a custom model

    # Validate the model22
    metrics = model.val(split='val')  # no arguments needed, dataset and settings remembered
    print(metrics)

6
6