import torch
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights



from dataset import BDD100KDetectionDataset
from utils import collate_fn

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = BDD100KDetectionDataset(
        image_dir=Path("/data/bdd100k_images_100k/bdd100k/images/100k/train"),
        label_json=Path("/data/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json"),
        transforms=transforms.ToTensor(),
    )

    # Small subset for demo 
    subset = torch.utils.data.Subset(dataset, range(100))

    dataloader = DataLoader(
        subset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

#-----------------------------------------------------------------------------------#
    #Recommended Code


    # Load COCO-pretrained model
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)

    # Replace classification head
    num_classes = 11  # 10 classes + background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        num_classes
    )

    model.to(device)
    model.train()

#-----------------------------------------------------------------------------------#

    # model = fasterrcnn_resnet50_fpn(
    #     pretrained=True,
    #     num_classes=11  # 10 classes + background
    # )
    # model.to(device)
    # model.train()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005,
    )

    print("Starting 1-epoch training...")
    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        print(f"Loss: {losses.item():.4f}")

    print("Training completed.")


if __name__ == "__main__":
    main()
