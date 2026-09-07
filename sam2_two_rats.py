"""
SAM2 two-animal video tracking.

This script extracts a video to SAM2-compatible JPEG frames, initializes two
animal identities from bounding boxes, propagates masks through the video in
both directions, and saves masks, centroids, and an overlay video.

Example:
    python sam2_two_rats.py
"""

from pathlib import Path
from types import SimpleNamespace
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import yaml
import numpy as np
import torch
import tkinter as tk
from tkinter import filedialog
from sam2_video_utils import Sam2VideoUtils


CONFIG_FILENAME = "sam2_tracking_config.yaml"
PROJECT_DIR = Path(__file__).resolve().parent
VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv", ".mpg", ".mpeg", ".wmv"}


MODEL_CONFIGS = {
    "hiera_tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
    "hiera_small": "configs/sam2.1/sam2.1_hiera_s.yaml",
    "hiera_base_plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "hiera_large": "configs/sam2.1/sam2.1_hiera_l.yaml",
}


DEFAULT_CONFIG = {
    "video": None,
    "num_animals": 2,
    "init_frame": 0,
    "rat1_frame": None,
    "rat2_frame": None,
    "checkpoint": "checkpoints/sam2.1_hiera_large.pt",
    "model_cfg": None,
    "output_dir": None,
    "frames_dir": None,
    "reuse_frames": False,
    "extract_frames_if_missing_only": True,
    "frame_extraction_method": "ffmpeg",
    "ffmpeg_path": "ffmpeg",
    "offload_video_to_cpu": True,
    "offload_state_to_cpu": True,
    "clear_cuda_cache": True,
    "rat1_box": None,
    "rat2_box": None,
    "skip_overlay": False,
    "jpeg_quality": 95,
    "display_max_width": 1400,
    "display_max_height": 900,
}


# Input: YAML box value and its config key name.
# Output: (x, y, width, height) tuple, or None when no box is configured.
def parse_box(value, label):
    if value is None:
        return None

    if isinstance(value, str):
        try:
            parts = [float(part.strip()) for part in value.split(",")]
        except ValueError as exc:
            raise ValueError(f"{label} must contain comma-separated numbers") from exc
    else:
        try:
            parts = [float(part) for part in value]
        except TypeError as exc:
            raise ValueError(f"{label} must be a list of 4 numbers or x,y,w,h") from exc

    if len(parts) != 4:
        raise ValueError(f"{label} must have exactly 4 values: x,y,w,h")
    return tuple(parts)


# Input: YAML boolean-like value and its config key name.
# Output: True or False.
def parse_bool(value, label):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1", "on"}:
            return True
        if lowered in {"false", "no", "0", "off"}:
            return False
    raise ValueError(f"{label} must be true or false")


# Input: directory path to search.
# Output: sorted list of video file Path objects found directly in that directory.
def find_video_files(work_dir):
    return sorted(
        path
        for path in Path(work_dir).iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


# Input: directory path containing possible movie files.
# Output: selected video filename, or None if no supported video file is found.
def choose_video_for_default_config(work_dir):
    videos = find_video_files(work_dir)
    if not videos:
        return None
    if len(videos) == 1:
        return videos[0].name

    print("\nMultiple video files were found:")
    for index, video in enumerate(videos, start=1):
        print(f"  {index}. {video.name}")

    while True:
        selection = input("Choose video number for the default config: ").strip()
        try:
            selected_index = int(selection) - 1
        except ValueError:
            print("Please type a number from the list.")
            continue
        if 0 <= selected_index < len(videos):
            return videos[selected_index].name
        print("Selection is outside the list range.")


# Input: user folder selection from dialog, or typed folder path as fallback.
# Output: resolved Path to the working directory containing the movie.
def ask_for_working_directory():
    print("Select the folder that contains the movie you want to track.")
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected_dir = filedialog.askdirectory(
            title="Select folder containing the movie to track"
        )
        root.destroy()
    except Exception:
        selected_dir = input("Folder path containing the movie: ").strip().strip('"')

    if not selected_dir:
        raise RuntimeError("No working directory was selected.")

    work_dir = Path(selected_dir).resolve()
    if not work_dir.is_dir():
        raise NotADirectoryError(work_dir)
    return work_dir


# Input: working directory Path.
# Output: Path to an existing or newly created SAM2 YAML config file.
def find_or_create_config(work_dir):
    config_path = work_dir / CONFIG_FILENAME
    if config_path.exists():
        return config_path

    config = dict(DEFAULT_CONFIG)
    config["video"] = choose_video_for_default_config(work_dir)
    config["checkpoint"] = str((PROJECT_DIR / config["checkpoint"]).resolve())
    with open(config_path, "w", encoding="utf-8") as config_file:
        yaml.safe_dump(config, config_file, sort_keys=False)

    print(f"\nCreated default config: {config_path}")
    if config["video"] is None:
        raise RuntimeError(
            "No video file was found in the selected folder. Put the movie in that "
            "folder, edit the 'video' key in the YAML file, and run again."
        )
    print("Using the default config. Edit the YAML before rerunning if you need different settings.")
    return config_path


# Input: Path to the YAML config file that will be used.
# Output: None if the user confirms; raises RuntimeError if the user stops.
def ask_user_to_review_config(config_path):
    print(f"\nReview the YAML configuration before continuing:")
    print(config_path)
    print("Save any changes in your editor before answering.")

    while True:
        answer = input("Continue with this YAML file? [y/n]: ").strip().lower()
        if answer in {"y", "yes"}:
            return
        if answer in {"n", "no"}:
            raise RuntimeError("Stopped so you can edit the YAML file and run again.")
        print("Please answer y or n.")


# Input: path value from YAML and base directory for relative paths.
# Output: resolved Path, or None when the YAML value is empty.
def resolve_path(value, base_dir):
    if value is None:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


# Input: checkpoint path from YAML and the config directory.
# Output: checkpoint Path resolved from config directory first, then project directory.
def resolve_checkpoint_path(value, config_dir):
    path = Path(value).expanduser()
    if path.is_absolute():
        return path

    config_relative = (config_dir / path).resolve()
    if config_relative.exists():
        return config_relative
    return (PROJECT_DIR / path).resolve()


# Input: Path to a SAM2 tracking YAML file.
# Output: resolved config Path and SimpleNamespace with validated config values.
def load_config(config_path):
    config_path = Path(config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as config_file:
        loaded = yaml.safe_load(config_file) or {}

    if not isinstance(loaded, dict):
        raise ValueError("Config file must contain YAML key/value pairs.")

    unknown_keys = sorted(set(loaded) - set(DEFAULT_CONFIG))
    if unknown_keys:
        raise ValueError("Unknown config keys: " + ", ".join(unknown_keys))

    config = dict(DEFAULT_CONFIG)
    config.update(loaded)

    if not config["video"]:
        raise ValueError("Config key 'video' is required.")

    config["num_animals"] = int(config["num_animals"])
    if config["num_animals"] not in {1, 2}:
        raise ValueError("num_animals must be 1 or 2")
    config["init_frame"] = int(config["init_frame"])
    config["rat1_frame"] = (
        None if config["rat1_frame"] is None else int(config["rat1_frame"])
    )
    config["rat2_frame"] = (
        None if config["rat2_frame"] is None else int(config["rat2_frame"])
    )
    config["reuse_frames"] = parse_bool(config["reuse_frames"], "reuse_frames")
    config["extract_frames_if_missing_only"] = parse_bool(
        config["extract_frames_if_missing_only"], "extract_frames_if_missing_only"
    )
    config["skip_overlay"] = parse_bool(config["skip_overlay"], "skip_overlay")
    config["offload_video_to_cpu"] = parse_bool(
        config["offload_video_to_cpu"], "offload_video_to_cpu"
    )
    config["offload_state_to_cpu"] = parse_bool(
        config["offload_state_to_cpu"], "offload_state_to_cpu"
    )
    config["clear_cuda_cache"] = parse_bool(config["clear_cuda_cache"], "clear_cuda_cache")
    config["frame_extraction_method"] = str(config["frame_extraction_method"]).lower()
    if config["frame_extraction_method"] not in {"opencv", "ffmpeg", "ffmpeg_cuda"}:
        raise ValueError("frame_extraction_method must be opencv, ffmpeg, or ffmpeg_cuda")
    config["jpeg_quality"] = int(config["jpeg_quality"])
    config["display_max_width"] = int(config["display_max_width"])
    config["display_max_height"] = int(config["display_max_height"])
    config["rat1_box"] = parse_box(config["rat1_box"], "rat1_box")
    config["rat2_box"] = parse_box(config["rat2_box"], "rat2_box")

    return config_path, SimpleNamespace(**config)


# Input: SAM2 model config name/path, checkpoint path, and torch device.
# Output: initialized SAM2 video predictor.
def build_predictor(model_cfg, checkpoint, device):
    try:
        from sam2.build_sam import build_sam2_video_predictor
    except ImportError as exc:
        raise RuntimeError(
            "Could not import SAM2. Run setup_sam2.ps1 first, or install "
            "facebookresearch/sam2 in this Python environment."
        ) from exc

    return build_sam2_video_predictor(model_cfg, checkpoint, device=device)


# Input: SAM2 checkpoint Path or filename.
# Output: matching SAM2 model config path string inferred from the checkpoint name.
def infer_model_cfg(checkpoint):
    checkpoint_name = Path(checkpoint).name
    for model_name, model_cfg in MODEL_CONFIGS.items():
        if model_name in checkpoint_name:
            return model_cfg

    raise ValueError(
        "Could not infer SAM2 model config from checkpoint name. "
        "Pass --model-cfg explicitly."
    )


# Input: none; uses the active PyTorch/CUDA installation.
# Output: None if CUDA works; raises RuntimeError if no working GPU is available.
def check_gpu_is_working():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU was not detected. SAM2 video inference is intended to run "
            "on an NVIDIA GPU. On Windows, WSL2 + CUDA is usually the most reliable setup."
        )

    try:
        device_name = torch.cuda.get_device_name(0)
        test_tensor = torch.ones((1,), device="cuda")
        torch.cuda.synchronize()
    except Exception as exc:
        raise RuntimeError("CUDA GPU was detected but failed a basic tensor test.") from exc

    print(f"GPU is working: {device_name}")


# Input: SAM2 output for one frame and mask storage arrays.
# Output: None; updates rat mask arrays and seen flags in place.
def store_masks(frame_idx, obj_ids, mask_logits, rat1_masks, rat2_masks, seen):
    seen[frame_idx] = True
    for idx, obj_id in enumerate(obj_ids):
        mask = (mask_logits[idx] > 0.0).detach().cpu().numpy()
        mask = np.squeeze(mask).astype(bool)
        if int(obj_id) == 1:
            rat1_masks[frame_idx] = mask
        elif int(obj_id) == 2:
            rat2_masks[frame_idx] = mask


# Input: SAM2 predictor state, start frame, output mask arrays, and direction flag.
# Output: None; propagates masks and stores results in place.
def propagate(predictor, inference_state, init_frame, rat1_masks, rat2_masks, seen, reverse):
    direction = "backward" if reverse else "forward"
    print(f"Propagating {direction}...")

    kwargs = {"start_frame_idx": init_frame, "reverse": reverse}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        inference_state, **kwargs
    ):
        store_masks(
            int(out_frame_idx), out_obj_ids, out_mask_logits, rat1_masks, rat2_masks, seen
        )
        print(f"\r{direction}: frame {int(out_frame_idx)}", end="", flush=True)
    print()

#--------------------------------MAIN--------------------------------#
# Input: user-selected movie folder and YAML configuration from that folder.
# Output: saved mask NPZ, centroid CSV, and optional overlay video.
def main():
    work_dir = ask_for_working_directory()
    config_path = find_or_create_config(work_dir)
    ask_user_to_review_config(config_path)
    config_path, args = load_config(config_path)

    check_gpu_is_working()

    config_dir = config_path.parent
    video_path = resolve_path(args.video, config_dir)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    checkpoint = resolve_checkpoint_path(args.checkpoint, config_dir)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"SAM2 checkpoint not found: {checkpoint}. Run setup_sam2.ps1 first."
        )

    model_cfg = args.model_cfg or infer_model_cfg(checkpoint)

    video_info = Sam2VideoUtils.read_video_info(video_path)
    total_frames = video_info["total_frames"]
    fps = video_info["fps"]
    width = video_info["width"]
    height = video_info["height"]
    num_animals = args.num_animals
    rat1_frame = args.rat1_frame if args.rat1_frame is not None else args.init_frame #to define different frame for each animal
    rat2_frame = args.rat2_frame if args.rat2_frame is not None else args.init_frame
    frame_checks = [("rat1_frame", rat1_frame)]
    if num_animals == 2:
        frame_checks.append(("rat2_frame", rat2_frame))
    for label, frame_idx in frame_checks:
        if frame_idx < 0 or frame_idx >= total_frames:
            raise ValueError(f"{label}={frame_idx} is outside video range 0..{total_frames - 1}")

    output_dir = Sam2VideoUtils.default_output_dir(
        video_path, resolve_path(args.output_dir, config_dir)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = resolve_path(args.frames_dir, config_dir) if args.frames_dir else output_dir / "frames"

    print(f"Config: {config_path}")
    print(f"Video: {video_path}")
    print(f"Frames: {total_frames}")
    print(f"Size: {width} x {height}")
    print(f"FPS: {fps:.3f}")
    print(f"Animals to track: {num_animals}")
    print(f"Rat 1 initialization frame: {rat1_frame}")
    if num_animals == 2:
        print(f"Rat 2 initialization frame: {rat2_frame}")
    print(f"SAM2 config: {model_cfg}")
    print(f"Output directory: {output_dir}")
    print(f"Offload video to CPU: {args.offload_video_to_cpu}")
    print(f"Offload state to CPU: {args.offload_state_to_cpu}")

    print("\nPreparing SAM2 frames...")
    existing_frames = sorted(frames_dir.glob("*.jpg")) if frames_dir.exists() else []
    if args.extract_frames_if_missing_only and existing_frames:
        extracted = len(existing_frames)
        print(f"Using existing frame folder: {frames_dir}")
    else:
        if args.extract_frames_if_missing_only:
            print(f"Frame folder is missing or empty; extracting frames to: {frames_dir}")
        extracted = Sam2VideoUtils.extract_frames(
            video_path,
            frames_dir,
            overwrite=not args.reuse_frames,
            jpeg_quality=args.jpeg_quality,
            method=args.frame_extraction_method,
            ffmpeg_path=args.ffmpeg_path,
        )
    if extracted != total_frames:
        print(f"Warning: extracted/reused {extracted} frames, video metadata says {total_frames}.")

    if args.rat1_box is None:
        rat1_image = Sam2VideoUtils.read_frame(video_path, rat1_frame)
        print(f"\nSelecting Rat 1 on frame {rat1_frame}.")
        rat1_roi = Sam2VideoUtils.select_box(
            rat1_image, "Rat 1", args.display_max_width, args.display_max_height
        )
    else:
        rat1_roi = args.rat1_box
        Sam2VideoUtils.validate_roi(rat1_roi, "Rat 1")

    if num_animals == 1:
        rat2_roi = None
    else:
        if args.rat2_box is None:
            rat2_image = Sam2VideoUtils.read_frame(video_path, rat2_frame)
            print(f"\nSelecting Rat 2 on frame {rat2_frame}.")
            rat2_roi = Sam2VideoUtils.select_box(
                rat2_image, "Rat 2", args.display_max_width, args.display_max_height
            )
        else:
            rat2_roi = args.rat2_box
            Sam2VideoUtils.validate_roi(rat2_roi, "Rat 2")

    rat1_box = Sam2VideoUtils.roi_xywh_to_xyxy(rat1_roi)
    print("\nRat 1 box xyxy:", rat1_box.tolist())
    if num_animals == 2:
        rat2_box = Sam2VideoUtils.roi_xywh_to_xyxy(rat2_roi)
        print("Rat 2 box xyxy:", rat2_box.tolist())

    device = torch.device("cuda")
    if args.clear_cuda_cache:
        torch.cuda.empty_cache()

    print("\nLoading SAM2...")
    predictor = build_predictor(model_cfg, str(checkpoint), device)

    print("Starting SAM2 video state...")
    inference_state = predictor.init_state(
        video_path=str(frames_dir),
        offload_video_to_cpu=args.offload_video_to_cpu,
        offload_state_to_cpu=args.offload_state_to_cpu,
    )
    predictor.reset_state(inference_state)

    rat1_masks = np.zeros((total_frames, height, width), dtype=bool)
    rat2_masks = np.zeros((total_frames, height, width), dtype=bool)
    seen = np.zeros(total_frames, dtype=bool)

    autocast_context = torch.autocast("cuda", dtype=torch.bfloat16)
    with torch.inference_mode(), autocast_context:
        print("Adding Rat 1 prompt...")
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=rat1_frame,
            obj_id=1,
            box=rat1_box,
        )
        store_masks(rat1_frame, out_obj_ids, out_mask_logits, rat1_masks, rat2_masks, seen)

        if num_animals == 2:
            print("Adding Rat 2 prompt...")
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=rat2_frame,
                obj_id=2,
                box=rat2_box,
            )
            store_masks(rat2_frame, out_obj_ids, out_mask_logits, rat1_masks, rat2_masks, seen)

        print("\nPropagating masks through the video...")
        forward_start = min(rat1_frame, rat2_frame) if num_animals == 2 else rat1_frame
        backward_start = max(rat1_frame, rat2_frame) if num_animals == 2 else rat1_frame
        propagate(predictor, inference_state, forward_start, rat1_masks, rat2_masks, seen, reverse=False)
        propagate(predictor, inference_state, backward_start, rat1_masks, rat2_masks, seen, reverse=True)

    output_masks = output_dir / f"{video_path.stem}_sam2_masks.npz"
    output_csv = output_dir / f"{video_path.stem}_sam2_centroids.csv"
    output_video = output_dir / f"{video_path.stem}_sam2_overlay.mp4"

    Sam2VideoUtils.save_masks(
        output_masks,
        num_animals,
        rat1_masks,
        rat2_masks,
        seen,
        rat1_frame,
        rat2_frame,
        rat1_roi,
        rat2_roi,
        fps,
    )
    Sam2VideoUtils.write_centroids_csv(output_csv, rat1_masks, rat2_masks, seen, num_animals)

    print(f"Saved masks: {output_masks}")
    print(f"Saved centroids: {output_csv}")

    if not args.skip_overlay:
        print("Writing overlay video...")
        Sam2VideoUtils.write_overlay_video(
            video_path,
            output_video,
            num_animals,
            rat1_masks,
            rat2_masks,
            fps,
            width,
            height,
            rat1_frame,
            rat2_frame,
        )
        print(f"Saved overlay video: {output_video}")

    print("\nFinished. Inspect the overlay for identity swaps, missed masks, and occlusion errors.")


if __name__ == "__main__":
    main()
