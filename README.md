## Perception2D

End-to-end 2D perception pipeline for KITTI-style data:
- **Training** via Faster R-CNN with Torchvision
- **Detection** via exported ONNX model
- **Tracking** via SORT (C++)
- **Visualization** utilities

This repo is designed for experimentation. Datasets and model weights are
**not** included and must be provided by you.

## Project Layout
- `src_cpp/`: C++ inference + tracking app
- `src_python/`: training notebook
- `tools_py/`: export/verification and visualization scripts
- `configs/`: public defaults
- `result_examples/`: example results

## Prerequisites
- C++17 compiler, CMake >= 3.20
- OpenCV (C++ libs)
- ONNX Runtime (C++ libs)
- CLI11 (header-only library)
- Python 3.10+ for scripts

### Setting up ONNX Runtime on Linux

ONNX Runtime provides pre-built C++ binaries. Download and extract them to a location of your choice.

1. **Download the release** (CPU version example):
   ```bash
   # Check https://github.com/microsoft/onnxruntime/releases for latest version
   wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz
   tar -xzf onnxruntime-linux-x64-1.16.3.tgz
   ```

2. **Option A: Place in `third_party/` (auto-detected by CMake)**
   ```bash
   mkdir -p third_party
   mv onnxruntime-linux-x64-1.16.3 third_party/onnxruntime
   ```

3. **Option B: Keep it elsewhere and pass `-DORT_ROOT`**
   ```bash
   # Example: keep it in ~/libs
   mv onnxruntime-linux-x64-1.16.3 ~/libs/onnxruntime
   # Then at configure time:
   cmake -S . -B build -DORT_ROOT=~/libs/onnxruntime
   ```

4. **Verify the layout** — CMake expects:
   ```
   <ORT_ROOT>/
     include/
       onnxruntime_cxx_api.h
       ...
     lib/
       libonnxruntime.so
       ...
   ```

> **GPU builds**: If you need CUDA/TensorRT support, download the corresponding
> `onnxruntime-linux-x64-gpu-*` archive instead and ensure your CUDA toolkit is installed.

### Setting up CLI11 on Linux

CLI11 is a header-only library, so no compilation is needed.

1. **Clone the repository**:
   ```bash
   git clone --depth 1 https://github.com/CLIUtils/CLI11.git
   ```

2. **Option A: Place in `third_party/` (auto-detected by CMake)**
   ```bash
   mkdir -p third_party
   mv CLI11 third_party/cli11
   ```

3. **Option B: Keep it elsewhere and pass `-DCLI11_ROOT`**
   ```bash
   mv CLI11 ~/libs/cli11
   # Then at configure time:
   cmake -S . -B build -DCLI11_ROOT=~/libs/cli11
   ```

4. **Verify the layout** — CMake looks for either:
   ```
   <CLI11_ROOT>/include/CLI/CLI.hpp   # standard layout
   <CLI11_ROOT>/CLI/CLI.hpp           # flat layout
   <CLI11_ROOT>/include/CLI11.hpp     # single-header variant
   ```

> **Tip**: You can also use the single-header release from
> https://github.com/CLIUtils/CLI11/releases (download `CLI11.hpp` and place it
> in `third_party/cli11/include/`).

## Configuration
Defaults live in `configs/public/default.ini` with sectioned settings per tool.
CLI flags take priority over config values.

## C++ Build (Detection + Tracking)
This repo expects ONNX Runtime headers and shared library to be available.

### Option A: Use a local ONNX Runtime install
```bash
cmake -S . -B build -DORT_ROOT=/path/to/onnxruntime
cmake --build build -j
```

### Option B: Use the bundled layout (if you provide it)
If you keep ONNX Runtime under `third_party/onnxruntime`, CMake will auto-detect it.

### CMake Presets (optional)
If you vendor dependencies under `third_party/`, you can use the default preset:
```bash
cmake --preset default
cmake --build --preset default -j
```

### Required header-only deps (CLI11)
CLI11 is not vendored in this repo. Provide it via `third_party/cli11`
or point CMake to a local checkout:
- `third_party/cli11` (from https://github.com/CLIUtils/CLI11)

You can also set:
```bash
cmake -S . -B build \
  -DCLI11_ROOT=/path/to/cli11 \
  -DORT_ROOT=/path/to/onnxruntime
```

## Running the C++ App
```bash
./build/perception2d_app
```
Required parameter: `sequence_id` (set via CLI flags or config).
Defaults for `input_path` and `model_path` are derived from `[paths]` in `configs/public/default.ini`.

Override via config:
```bash
./build/perception2d_app --config /path/to/config.ini
```

Override defaults if you use a different layout:
```bash
./build/perception2d_app \
  --input /path/to/{kitti_tracking_root}/{sequence_id}/images \
  --sequence 0000 \
  --model-path /path/to/{models_dir}/{model_name} \
  --output /path/to/{output_root}/{sequence_id}
```

## Python Utilities

Install dependencies (conda):
```bash
conda env create -f environment.yml
conda activate perception2d
```

### Export ONNX Model
Place weights at `{weights_dir}/best_model.pth`, then:
```bash
python tools_py/export_and_verify.py
```
This will write `{models_dir}/best_model.onnx`.
For trusted checkpoints that require full pickle loading, add `--allow-unsafe-load`.
You can also supply `--config` to pull defaults from a custom INI file.

### Visualize KITTI Labels
```bash
python tools_py/visualize_kitti.py --data-root /path/to/{kitti_detection_root}
```
Defaults can be configured via `configs/public/default.ini`.

## Dataset Expectations
- **Tracking** input: a folder of images or a single image.
- **Detection dataset (training)** expects the official KITTI 2D Object Detection layout:
```
data/kitti_detection/
  data_object_image_2/training/image_2/000001.png
  data_object_label_2/training/label_2/000001.txt
```
- **visualize_kitti.py** expects simplified layout:
```
data/kitti_detection/
  images/000001.png
  labels/000001.txt
```

### Training (Python)
Open `src_python/train.ipynb` in Jupyter/Colab and follow the notebook cells.

## Security Note
`tools_py/export_and_verify.py` uses `torch.load`, which relies on Python pickle.
By default it refuses full pickle loads unless you pass `--allow-unsafe-load`.
**Never** enable unsafe loading for untrusted weights.

## License
MIT (see `LICENSE`).
