from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

IMG_HEIGHT = 384
IMG_WIDTH = 1248
NUM_CLASSES = 4  # 0:BG, 1:Car, 2:Ped, 3:Cyc


def get_model(num_classes: int) -> torch.nn.Module:
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(
        weights=weights,
        min_size=IMG_HEIGHT,
        max_size=IMG_WIDTH,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


def resolve_weights_path(weights_dir: Path) -> Path:
    candidates = [
        weights_dir / "best_model.pth",
        weights_dir / "last.pth",
        weights_dir / "model_v2.pth",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Weights not found. Expected one of: "
        + ", ".join(str(path) for path in candidates)
    )


def main():
    repo_root = Path(__file__).resolve().parent.parent
    weights_dir = repo_root / "weights"
    output_onnx_path = repo_root / "models" / "model_v2.onnx"

    try:
        weights_path = resolve_weights_path(weights_dir)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return

    device = torch.device("cpu")

    print(f"Loading checkpoint from {weights_path}...")
    model = get_model(NUM_CLASSES)

    checkpoint = torch.load(weights_path, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        model = checkpoint

    model.eval()
    model.to(device)

    output_onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.rand(1, 3, IMG_HEIGHT, IMG_WIDTH)

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
    with torch.no_grad():
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
        print(f"Box Sample (ONNX): {onnx_boxes[0] if len(onnx_boxes) > 0 else 'None'}")
    except AssertionError as e:
        print("FAILURE: Outputs do not match.")
        print(e)


if __name__ == "__main__":
    main()
