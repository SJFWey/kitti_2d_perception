import argparse
import random
import sys
from pathlib import Path

import cv2

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from configs.public.config_utils import cfg_get, load_public_config, resolve_path

COLORS = {
    "Car": (0, 255, 0),
    "Pedestrian": (0, 0, 255),
    "Cyclist": (255, 255, 0),
    "DontCare": (128, 128, 128),
}


def parse_label_file(label_path):
    objects = []

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            obj_type = parts[0]

            try:
                x1 = float(parts[4])
                y1 = float(parts[5])
                x2 = float(parts[6])
                y2 = float(parts[7])
            except ValueError:
                continue

            objects.append({"type": obj_type, "bbox": [x1, y1, x2, y2]})

    return objects


def collect_images(image_dir: Path):
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    return sorted(
        [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    )


def main():
    parser = argparse.ArgumentParser(
        description="Visualize KITTI detection labels on sample images."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to INI config file.",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="KITTI detection root containing images/ and labels/.",
    )
    parser.add_argument(
        "--image-dir",
        default=None,
        help="Optional override for images directory.",
    )
    parser.add_argument(
        "--label-dir",
        default=None,
        help="Optional override for labels directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for visualizations.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of random samples to visualize (<=0 uses all).",
    )
    args = parser.parse_args()

    config_path = args.config if args.config else None
    config_required = args.config is not None
    cfg = load_public_config(repo_root, config_path=config_path, required=config_required)

    section = "visualize_kitti"
    paths_section = "paths"

    errors = []
    data_root_value = args.data_root or cfg_get(
        cfg,
        section,
        "data_root",
        str,
        "",
        fallback_sections=[paths_section],
    )
    if not data_root_value:
        errors.append(
            "Missing required parameter: data_root. "
            "Set --data-root, [visualize_kitti].data_root, or [paths].kitti_detection_root."
        )

    image_dir_value = args.image_dir or cfg_get(cfg, section, "image_dir", str, "")
    label_dir_value = args.label_dir or cfg_get(cfg, section, "label_dir", str, "")

    output_dir_value = args.output_dir or cfg_get(cfg, section, "output_dir", str, "")
    if not output_dir_value:
        output_root = cfg_get(
            cfg,
            paths_section,
            "output_root",
            str,
            "",
        )
        if not output_root:
            errors.append(
                "Missing config value: [paths].output_root is required to build output_dir."
            )
        else:
            output_dir_value = str(Path(output_root) / "debug_vis")

    if args.num_samples is not None:
        num_samples = args.num_samples
    else:
        num_samples = cfg_get(cfg, section, "num_samples", int, None)
        if num_samples is None:
            errors.append(
                "Missing business parameter: num_samples. Set [visualize_kitti].num_samples."
            )

    if errors:
        message = "Config validation failed:\n" + "\n".join(
            f"  - {err}" for err in errors
        )
        raise SystemExit(message)

    data_root = resolve_path(repo_root, data_root_value)
    output_dir = resolve_path(repo_root, output_dir_value)

    image_dir = resolve_path(repo_root, image_dir_value) if image_dir_value else data_root / "images"
    label_dir = resolve_path(repo_root, label_dir_value) if label_dir_value else data_root / "labels"

    if not image_dir.exists():
        raise SystemExit(f"Image directory not found: {image_dir}")
    if not label_dir.exists():
        raise SystemExit(f"Label directory not found: {label_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = collect_images(image_dir)
    if not image_files:
        raise SystemExit(f"No images found in: {image_dir}")

    if num_samples <= 0:
        samples = image_files
    else:
        samples = random.sample(image_files, min(num_samples, len(image_files)))

    print(f"Found {len(image_files)} images, processing {len(samples)} sample(s).")

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
        if not cv2.imwrite(str(output_path), img):
            print(f"Failed to write image: {output_path}")


if __name__ == "__main__":
    main()
