from ultralytics import YOLO

# Load a model
model = YOLO("E:\\DCNV3\\ultralytics-main\\ultralytics\\models\\yolo\\detect\\runs\\detect\\train293\weights\\best.pt")  # load an official model
# Validate the model
metrics = model.val(data='E:\\DCNV3\\ultralytics-main\\ultralytics\\cfg\\datasets\\coco128.yaml', iou=0.7, conf=0.001, half=False, device=0, save_json=True)

print(metrics.box.map)  # map50-95
print(metrics.box.map50)  # map50
print(metrics.box.map75)  # map75
print(metrics.box.maps)  # 包含每个类别的map50-95列表

