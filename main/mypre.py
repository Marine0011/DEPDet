from ultralytics import YOLO


if __name__ == '__main__':

    # 自己训练好的模型
    model = YOLO("/root/my-tmp/project/DEPDet/main/runs/train/exp/weights/best.pt")

    # 填自己的单个图片，或路径
    results = model.predict(source="/root/my-tmp/project/DEPDet/dataset_SSDD/images/test/000762.jpg", save=True, augment=True) # Display preds. Accepts all YOLO predict arguments
