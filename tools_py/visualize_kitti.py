import cv2
import random
from pathlib import Path

COLORS = {
    "Car": (0, 255, 0),
    "Pedestrian": (0, 0, 255),
    "Cyclist": (255, 255, 0),
    "DontCare": (128, 128, 128),
}


def parse_label_file(label_path):
    objects = []

    with open(label_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split(" ")
            obj_type = parts[0]

            x1 = float(parts[4])
            y1 = float(parts[5])
            x2 = float(parts[6])
            y2 = float(parts[7])

            objects.append({"type": obj_type, "bbox": [x1, y1, x2, y2]})

    return objects


def main():
    project_root = Path("..")
    image_dir = project_root / "data/kitti/images"
    label_dir = project_root / "data/kitti/labels"
    output_dir = project_root / "output/debug_vis"

    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(list(image_dir.glob("*.png")))
    if not image_files:
        print("No images found in the specified directory.")
        return

    print(f"Found {len(image_files)} images, processing 5 random samples.")

    samples = random.sample(image_files, min(5, len(image_files)))

    for img_path in samples:
        file_id = img_path.stem
        label_path = label_dir / f"{file_id}.txt"

        if not label_path.exists():
            print(f"Label file not found for image {file_id}, skipping.")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Failed to read image {file_id}, skipping.")
            continue

        objects = parse_label_file(label_path)

        for obj in objects:
            cls = obj["type"]
            x1, y1, x2, y2 = obj["bbox"]

            color = COLORS.get(cls, (255, 255, 255))

            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            text = f"{cls}"
            cv2.putText(
                img,
                text,
                (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

        output_path = output_dir / f"vis_{file_id}.png"
        cv2.imwrite(str(output_path), img)


if __name__ == "__main__":
    main()
