# SAM2 Two-Rat Tracking

This workflow tracks two animals in a video with Meta SAM2/SAM2.1. It saves full-frame masks, centroid tracks, and an overlay video for visual checking.

## Files

- `setup_sam2.ps1`: creates a Python environment, installs PyTorch/SAM2, and downloads a SAM2.1 checkpoint.
- `sam2_two_rats.py`: end-to-end video tracking script.
- `sam2_video_utils.py`: shared video, mask, centroid, and overlay helpers.
- `sam2_export_centroids.py`: re-export centroid CSV files from saved mask `.npz` files.
- `sam2_tracking_config.yaml`: editable tracking configuration.
- `requirements-sam2.txt`: non-PyTorch Python dependencies plus SAM2 from GitHub.

## Setup

Run from the repository root in PowerShell:

```powershell
.\setup_sam2.ps1
.\.venv-sam2\Scripts\Activate.ps1
```

The default setup installs CUDA 12.1 PyTorch wheels and downloads `checkpoints/sam2.1_hiera_large.pt`.

For a smaller model:

```powershell
.\setup_sam2.ps1 -Model small
python sam2_two_rats.py --video test_100.avi --init-frame 20 --checkpoint checkpoints\sam2.1_hiera_small.pt
```

The tracker infers the matching SAM2 config from the checkpoint filename. Use `--model-cfg` only for custom checkpoints.

SAM2 video inference is intended for an NVIDIA CUDA GPU. If native Windows install fails, use WSL2 with CUDA and run the same Python scripts there.

## Track A Video

Interactive box selection:

```powershell
python sam2_two_rats.py
```

The script asks you to select the folder containing the movie. It then looks for `sam2_tracking_config.yaml` in that folder. If the YAML file does not exist, it creates a default one there and fills `video` automatically when one video file is found. Before tracking starts, it asks you to review the YAML file and confirm that it should continue.

## Run From Visual Studio Code

1. Open this repository folder in VS Code.
2. Install the Microsoft Python extension if it is not already installed.
3. Select the `sam2track` Conda interpreter with `Ctrl+Shift+P`, then `Python: Select Interpreter`.
4. Open `Run and Debug`, choose `SAM2 Track Two Animals`, and press the green run button.
5. Select the folder containing the movie when the folder picker opens.
6. If a default YAML was created, edit it for your frame choices and run again.

The launch configuration is stored in `.vscode/launch.json`.

Edit `sam2_tracking_config.yaml` before running. At minimum, set `video`, `num_animals`, and the frame choices:

```yaml
video: test_100.avi
num_animals: 2
init_frame: 20
rat1_frame:
rat2_frame:
```

Track only one animal:

```yaml
video: test_100.avi
num_animals: 1
rat1_frame: 20
rat1_box:
```

Use different selection frames when each animal is clearest at a different moment:

```yaml
video: test_100.avi
num_animals: 2
rat1_frame: 20
rat2_frame: 45
```

Headless run with explicit boxes in `x,y,width,height` pixels:

```yaml
video: test_100.avi
num_animals: 2
init_frame: 20
rat1_box: [120, 80, 90, 60]
rat2_box: [300, 110, 85, 65]
```

Headless run with different selection frames:

```yaml
video: test_100.avi
num_animals: 2
rat1_frame: 20
rat1_box: [120, 80, 90, 60]
rat2_frame: 45
rat2_box: [300, 110, 85, 65]
```

Outputs are written to `<video_stem>_sam2/`:

- `<video_stem>_sam2_masks.npz`: boolean masks named `rat1` and `rat2`, plus metadata.
- `<video_stem>_sam2_centroids.csv`: per-frame centroid and area for each rat.
- `<video_stem>_sam2_overlay.mp4`: red/green visual mask overlay.
- `frames/`: extracted JPEG frames used by SAM2.

To reuse already extracted frames:

```powershell
python sam2_two_rats.py
```

Keep this setting in the YAML file:

```yaml
extract_frames_if_missing_only: true
```

With this setting, frames are extracted only when `frames_dir` is missing or contains no JPG frames. If the frame folder already exists, the script uses it.

If the animal-selection picture is cut off on your screen, reduce the display size:

```powershell
python sam2_two_rats.py
```

Set `display_max_width: 1000` and `display_max_height: 700` in the YAML file.

To skip overlay generation for a faster run:

```powershell
python sam2_two_rats.py
```

Set `skip_overlay: true` in the YAML file.

## Large Movies

SAM2's video predictor needs the movie as a folder of JPEG frames. For large movies, use FFmpeg extraction in the YAML file:

```yaml
frame_extraction_method: ffmpeg
ffmpeg_path: ffmpeg
```

If your FFmpeg build supports NVIDIA CUDA decoding, you can try:

```yaml
frame_extraction_method: ffmpeg_cuda
ffmpeg_path: ffmpeg
```

If FFmpeg is not installed, use the slower OpenCV fallback:

```yaml
frame_extraction_method: opencv
```

## CUDA Out Of Memory

For long or high-resolution movies, keep these memory-safe defaults in the YAML file:

```yaml
offload_video_to_cpu: true
offload_state_to_cpu: true
clear_cuda_cache: true
```

If memory is still too high, use a smaller checkpoint:

```yaml
checkpoint: checkpoints/sam2.1_hiera_small.pt
model_cfg:
```

Then download the small checkpoint if needed:

```powershell
.\setup_sam2.ps1 -Model small
```

Also close other Python notebooks, terminals, or programs using the GPU before running.

## Re-Export Centroids

```powershell
python sam2_export_centroids.py --masks test_100_sam2\test_100_sam2_masks.npz
```

## Check Results

Always inspect the overlay video for identity swaps, missing masks, and failures during animal contact or occlusion. If tracking fails, choose clearer `--rat1-frame` and `--rat2-frame` values and draw boxes tightly around each full animal.
