import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from configs.public.config_utils import cfg_get, cfg_get_bool, load_public_config, resolve_path

NUM_CLASSES = 4  # 0:BG, 1:Car, 2:Ped, 3:Cyc


def get_model(num_classes: int, img_height: int, img_width: int) -> torch.nn.Module:
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(
        weights=weights,
        min_size=img_height,
        max_size=img_width,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


def resolve_weights_path(weights_dir: Path) -> Path:
    weights_path = weights_dir / "best_model.pth"
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    return weights_path


def load_checkpoint(weights_path: Path, device: torch.device, allow_unsafe_load: bool):
    load_kwargs = {"map_location": device}
    try:
        return torch.load(weights_path, weights_only=True, **load_kwargs)
    except TypeError:
        if not allow_unsafe_load:
            raise RuntimeError(
                "weights_only is not supported by this torch version. "
                "Rerun with --allow-unsafe-load to enable full pickle loading."
            )
        return torch.load(weights_path, **load_kwargs)
    except (pickle.UnpicklingError, RuntimeError) as exc:
        if not allow_unsafe_load:
            raise RuntimeError(
                "Refusing to perform full pickle load. "
                "Rerun with --allow-unsafe-load for trusted checkpoints."
            ) from exc
        print(
            "Warning: weights_only load failed; "
            f"falling back to full load. Reason: {exc}"
        )
        return torch.load(weights_path, **load_kwargs)


def main():
    parser = argparse.ArgumentParser(
        description="Export a Faster R-CNN model to ONNX and verify outputs."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to INI config file.",
    )
    parser.add_argument(
        "--weights-dir",
        default=None,
        help="Directory containing best_model.pth.",
    )
    parser.add_argument(
        "--output-onnx",
        default=None,
        help="Destination path for the ONNX model.",
    )
    parser.add_argument(
        "--allow-unsafe-load",
        action="store_true",
        default=None,
        help="Allow full pickle load for trusted checkpoints.",
    )
    args = parser.parse_args()

    config_path = args.config if args.config else None
    config_required = args.config is not None
    cfg = load_public_config(repo_root, config_path=config_path, required=config_required)

    paths_section = "paths"
    section = "export_and_verify"

    models_dir_value = cfg_get(cfg, paths_section, "models_dir", str, "")
    model_name = cfg_get(cfg, paths_section, "model_name", str, "")

    errors = []

    weights_dir_value = args.weights_dir or cfg_get(
        cfg,
        section,
        "weights_dir",
        str,
        "",
        fallback_sections=[paths_section],
    )
    if not weights_dir_value:
        errors.append(
            "Missing required parameter: weights_dir. "
            "Set --weights-dir, [export_and_verify].weights_dir, or [paths].weights_dir."
        )

    output_onnx_value = args.output_onnx or cfg_get(
        cfg,
        section,
        "output_onnx",
        str,
        "",
    )
    if not output_onnx_value:
        if not models_dir_value:
            errors.append("Missing config value: [paths].models_dir is required.")
        if not model_name:
            errors.append("Missing config value: [paths].model_name is required.")
        if models_dir_value and model_name:
            output_onnx_value = str(Path(models_dir_value) / model_name)
    if not output_onnx_value:
        errors.append(
            "Missing required parameter: output_onnx. "
            "Set --output-onnx or [export_and_verify].output_onnx."
        )
    allow_unsafe_load = (
        args.allow_unsafe_load
        if args.allow_unsafe_load is not None
        else cfg_get_bool(cfg, section, "allow_unsafe_load", False)
    )

    input_height = cfg_get(
        cfg,
        section,
        "input_height",
        int,
        -1,
        fallback_sections=["perception2d_app"],
    )
    input_width = cfg_get(
        cfg,
        section,
        "input_width",
        int,
        -1,
        fallback_sections=["perception2d_app"],
    )
    if input_height <= 0:
        errors.append(
            "Missing or invalid input_height. "
            "Set [export_and_verify].input_height or [perception2d_app].input_height."
        )
    if input_width <= 0:
        errors.append(
            "Missing or invalid input_width. "
            "Set [export_and_verify].input_width or [perception2d_app].input_width."
        )

    if errors:
        print("Config validation failed:")
        for message in errors:
            print(f"  - {message}")
        return

    weights_dir = (
        Path(args.weights_dir)
        if args.weights_dir
        else resolve_path(repo_root, weights_dir_value)
    )
    output_onnx_path = (
        Path(args.output_onnx)
        if args.output_onnx
        else resolve_path(repo_root, output_onnx_value)
    )

    try:
        weights_path = resolve_weights_path(weights_dir)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return

    device = torch.device("cpu")

    print(f"Loading checkpoint from {weights_path}...")
    model = get_model(NUM_CLASSES, input_height, input_width)

    try:
        checkpoint = load_checkpoint(weights_path, device, allow_unsafe_load)
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        model = checkpoint

    model.eval()
    model.to(device)

    output_onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.rand(1, 3, input_height, input_width, device=device)

    print("Exporting to ONNX...")

    torch.onnx.export(
        model,
        dummy_input,
        output_onnx_path,
        opset_version=11,
        input_names=["input"],
        output_names=["boxes", "labels", "scores"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "boxes": {0: "num_detections"},
            "labels": {0: "num_detections"},
            "scores": {0: "num_detections"},
        },
    )
    print(f"ONNX model exported to {output_onnx_path}")

    print("\n--- Verifying ONNX Runtime Consistency ---")
    with torch.inference_mode():
        torch_out = model(dummy_input)
        torch_boxes = to_numpy(torch_out[0]["boxes"])
        torch_labels = to_numpy(torch_out[0]["labels"])
        torch_scores = to_numpy(torch_out[0]["scores"])

    ort_session = ort.InferenceSession(
        output_onnx_path, providers=["CPUExecutionProvider"]
    )

    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
    ort_outs = ort_session.run(None, ort_inputs)

    onnx_boxes = np.asarray(ort_outs[0])
    onnx_labels = np.asarray(ort_outs[1])
    onnx_scores = np.asarray(ort_outs[2])

    try:
        np.testing.assert_allclose(torch_boxes, onnx_boxes, rtol=1e-03, atol=1e-05)
        np.testing.assert_allclose(torch_scores, onnx_scores, rtol=1e-03, atol=1e-05)
        np.testing.assert_array_equal(torch_labels, onnx_labels)

        print("SUCCESS: PyTorch and ONNX Runtime outputs match!")
        print(f"Detections count: {len(onnx_boxes)}")
        sample_box = onnx_boxes[0] if len(onnx_boxes) > 0 else "None"
        print(f"Box Sample (ONNX): {sample_box}")
    except AssertionError as e:
        print("FAILURE: Outputs do not match.")
        print(e)


if __name__ == "__main__":
    main()
