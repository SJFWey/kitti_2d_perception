import os
import torch
import cv2
from torch.utils.data import Dataset, DataLoader


class KittiDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms

        self.image_dir = os.path.join(root_dir, "images")
        self.label_dir = os.path.join(root_dir, "labels")

        self.imgs = list(sorted(os.listdir(self.image_dir)))

        self.class_to_id = {
            "Car": 1,
            "Pedestrian": 2,
            "Cyclist": 3,
        }

        self.id_to_class = {v: k for k, v in self.class_to_id.items()}

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.imgs[idx])

        img = cv2.imread(img_path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        label_path = os.path.join(
            self.label_dir, self.imgs[idx].replace(".png", ".txt")
        )

        boxes = []
        labels = []

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split(" ")
                    obj_type = parts[0]
                    if obj_type in self.class_to_id:
                        x1 = float(parts[4])
                        y1 = float(parts[5])
                        x2 = float(parts[6])
                        y2 = float(parts[7])
                        boxes.append([x1, y1, x2, y2])
                        labels.append(self.class_to_id[obj_type])

        num_objs = len(boxes)

        if num_objs > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        image_id = torch.tensor([idx])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id

        if self.transforms is not None:
            # TODO apply transforms
            pass

        return img_tensor, target


def main():
    data_root = "../data/kitti_detection"

    if not os.path.exists(data_root):
        print(f"Data root {data_root} does not exist.")
    else:
        dataset = KittiDataset(data_root)
        print(f"Dataset size: {len(dataset)}")
        img, target = dataset[10]
        print(f"Image shape: {img.shape}")
        print(f"Target keys: {list(target.keys())}")
        print(f"Boxes shape: {target['boxes'].shape}")
        print(f"Labels shape: {target['labels'].shape}")

        def collate_fn(batch):
            return tuple(zip(*batch))

        dataloader = DataLoader(
            dataset, batch_size=4, shuffle=True, collate_fn=collate_fn
        )

        print("\nTesting DataLoader (Batch size 4)...")
        images, targets = next(iter(dataloader))
        print(f"Batch received. Number of images: {len(images)}")
        print(f"Labels in first image: {targets[0]['labels']}")
        print("Success!")


if __name__ == "__main__":
    main()
