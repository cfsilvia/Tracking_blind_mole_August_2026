from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np


class Sam2VideoUtils:
    """Utility methods for two-animal SAM2 video tracking."""

    @staticmethod
    def read_video_info(video_path):
        """
        Input: path to an input video file.
        Output: dictionary with total_frames, fps, width, and height.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        info = {
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": float(cap.get(cv2.CAP_PROP_FPS) or 30.0),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
        cap.release()

        if info["total_frames"] <= 0 or info["width"] <= 0 or info["height"] <= 0:
            raise RuntimeError(f"Video has invalid metadata: {video_path}")

        return info

    @staticmethod
    def read_frame(video_path, frame_index):
        """
        Input: video path and zero-based frame index.
        Output: one OpenCV BGR image frame as a NumPy array.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_index < 0 or frame_index >= total:
            cap.release()
            raise ValueError(
                f"init-frame={frame_index} is outside video range 0..{total - 1}"
            )

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        cap.release()

        if not ok:
            raise RuntimeError(f"Could not read frame {frame_index}")

        return frame

    @staticmethod
    def extract_frames(
        video_path,
        frames_dir,
        overwrite=False,
        jpeg_quality=95,
        method="ffmpeg",
        ffmpeg_path="ffmpeg",
    ):
        """
        Input: video path, output frame directory, overwrite flag, JPEG quality, and extraction method.
        Output: number of JPEG frames extracted or reused.
        """
        frames_dir = Path(frames_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)

        existing_frames = sorted(frames_dir.glob("*.jpg"))
        if existing_frames and not overwrite:
            return len(existing_frames)

        if overwrite:
            for frame_path in existing_frames:
                frame_path.unlink()

        if method in {"ffmpeg", "ffmpeg_cuda"}:
            return Sam2VideoUtils.extract_frames_ffmpeg(
                video_path, frames_dir, jpeg_quality, method, ffmpeg_path
            )
        if method == "opencv":
            return Sam2VideoUtils.extract_frames_opencv(
                video_path, frames_dir, jpeg_quality
            )
        raise ValueError("frame_extraction_method must be opencv, ffmpeg, or ffmpeg_cuda")

    @staticmethod
    def extract_frames_opencv(video_path, frames_dir, jpeg_quality=95):
        """
        Input: video path, output frame directory, and JPEG quality.
        Output: number of frames extracted using OpenCV on CPU.
        """
        frames_dir = Path(frames_dir)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        frame_count = 0
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_path = frames_dir / f"{frame_count:06d}.jpg"
            if not cv2.imwrite(str(frame_path), frame, encode_params):
                cap.release()
                raise RuntimeError(f"Could not write frame: {frame_path}")
            frame_count += 1

        cap.release()
        if frame_count == 0:
            raise RuntimeError(f"No frames were extracted from: {video_path}")
        return frame_count

    @staticmethod
    def extract_frames_ffmpeg(video_path, frames_dir, jpeg_quality=95, method="ffmpeg", ffmpeg_path="ffmpeg"):
        """
        Input: video path, output frame directory, JPEG quality, FFmpeg mode, and FFmpeg executable.
        Output: number of frames extracted using FFmpeg, optionally with CUDA decode.
        """
        if shutil.which(ffmpeg_path) is None and not Path(ffmpeg_path).exists():
            raise RuntimeError(
                f"FFmpeg was not found: {ffmpeg_path}. Install FFmpeg or set "
                "frame_extraction_method: opencv in the YAML file."
            )

        frames_dir = Path(frames_dir)
        output_pattern = str(frames_dir / "%06d.jpg")
        quality = Sam2VideoUtils.jpeg_quality_to_ffmpeg_qscale(jpeg_quality)
        command = [ffmpeg_path, "-hide_banner", "-loglevel", "error", "-y"]
        if method == "ffmpeg_cuda":
            command.extend(["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"])
        command.extend(
            [
                "-i",
                str(video_path),
                "-start_number",
                "0",
                "-q:v",
                str(quality),
                output_pattern,
            ]
        )

        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            details = exc.stderr.strip() or exc.stdout.strip() or str(exc)
            raise RuntimeError(f"FFmpeg frame extraction failed: {details}") from exc

        frame_count = len(list(frames_dir.glob("*.jpg")))
        if frame_count == 0:
            raise RuntimeError(f"No frames were extracted from: {video_path}")
        return frame_count

    @staticmethod
    def jpeg_quality_to_ffmpeg_qscale(jpeg_quality):
        """
        Input: JPEG quality on OpenCV's 1-100 scale.
        Output: FFmpeg qscale value where lower means better quality.
        """
        jpeg_quality = max(1, min(100, int(jpeg_quality)))
        return max(2, min(31, round(31 - (jpeg_quality / 100.0) * 29)))

    @staticmethod
    def select_two_boxes(frame, max_display_width=1400, max_display_height=900):
        """
        Input: image frame and maximum display window size.
        Output: two ROI tuples in original image coordinates: (x, y, width, height).
        """
        return (
            Sam2VideoUtils.select_box(
                frame, "Animal 1", max_display_width, max_display_height
            ),
            Sam2VideoUtils.select_box(
                frame, "Animal 2", max_display_width, max_display_height
            ),
        )

    @staticmethod
    def select_box(frame, label, max_display_width=1400, max_display_height=900):
        """
        Input: image frame, animal label, and maximum display window size.
        Output: selected ROI tuple in original image coordinates: (x, y, width, height).
        """
        display, scale = Sam2VideoUtils.resize_for_display(
            frame, max_display_width, max_display_height
        )

        print(f"\nDraw a box around {label.upper()}.")
        print("Press ENTER or SPACE when finished.")
        window_name = f"Select {label}"
        roi = cv2.selectROI(window_name, display, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow(window_name)
        cv2.destroyAllWindows()

        Sam2VideoUtils.validate_roi(roi, label)
        return Sam2VideoUtils.scale_roi_to_original(roi, scale)

    @staticmethod
    def resize_for_display(frame, max_width, max_height):
        """
        Input: image frame and maximum display width/height.
        Output: resized display frame and scale factor back to original coordinates.
        """
        height, width = frame.shape[:2]
        scale = min(float(max_width) / width, float(max_height) / height, 1.0)
        if scale >= 1.0:
            return frame.copy(), 1.0

        display = cv2.resize(
            frame,
            (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
        print(
            f"Frame resized for selection window: {width}x{height} -> "
            f"{display.shape[1]}x{display.shape[0]}"
        )
        return display, scale

    @staticmethod
    def scale_roi_to_original(roi, scale):
        """
        Input: ROI selected on the display image and the display scale factor.
        Output: ROI converted to original image coordinates.
        """
        if scale == 1.0:
            return roi
        x, y, w, h = roi
        return (
            int(round(x / scale)),
            int(round(y / scale)),
            int(round(w / scale)),
            int(round(h / scale)),
        )

    @staticmethod
    def validate_roi(roi, label):
        """
        Input: ROI tuple and animal label for error messages.
        Output: None if the ROI is valid; raises RuntimeError for zero-area boxes.
        """
        if roi[2] <= 0 or roi[3] <= 0:
            raise RuntimeError(f"{label} box was not selected or has zero area.")

    @staticmethod
    def roi_xywh_to_xyxy(roi):
        """
        Input: ROI tuple as (x, y, width, height).
        Output: NumPy array as (x_min, y_min, x_max, y_max) for SAM2.
        """
        x, y, w, h = roi
        return np.array([x, y, x + w, y + h], dtype=np.float32)

    @staticmethod
    def default_output_dir(video_path, output_dir=None):
        """
        Input: input video path and optional output directory.
        Output: resolved output directory Path.
        """
        if output_dir:
            return Path(output_dir).resolve()
        return video_path.with_name(video_path.stem + "_sam2")

    @staticmethod
    def save_masks(
        output_masks,
        num_animals,
        rat1_masks,
        rat2_masks,
        seen,
        rat1_init_frame,
        rat2_init_frame,
        rat1_roi,
        rat2_roi,
        fps,
    ):
        """
        Input: output path, tracking metadata, mask arrays, prompt frames, boxes, and fps.
        Output: None; writes a compressed NPZ mask file to disk.
        """
        init_frame = rat1_init_frame if rat1_init_frame == rat2_init_frame else -1
        rat2_box = (
            np.asarray(rat2_roi, dtype=np.float32)
            if rat2_roi is not None
            else np.array([], dtype=np.float32)
        )
        np.savez_compressed(
            output_masks,
            num_animals=np.array(num_animals),
            rat1=rat1_masks,
            rat2=rat2_masks,
            seen=seen,
            init_frame=np.array(init_frame),
            rat1_init_frame=np.array(rat1_init_frame),
            rat2_init_frame=np.array(rat2_init_frame),
            rat1_box_xywh=np.asarray(rat1_roi, dtype=np.float32),
            rat2_box_xywh=rat2_box,
            fps=np.array(fps, dtype=np.float32),
        )

    @staticmethod
    def mask_centroid(mask):
        """
        Input: boolean mask for one animal in one frame.
        Output: centroid x, centroid y, and mask area in pixels.
        """
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return np.nan, np.nan, 0
        return float(xs.mean()), float(ys.mean()), int(len(xs))

    @staticmethod
    def write_centroids_csv(output_csv, rat1_masks, rat2_masks, seen, num_animals=2):
        """
        Input: output CSV path, rat mask arrays, seen flags, and number of animals.
        Output: None; writes per-frame centroid coordinates and mask areas to CSV.
        """
        import csv

        with open(output_csv, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            header = ["frame", "seen", "rat1_x", "rat1_y", "rat1_area_px"]
            if num_animals == 2:
                header.extend(["rat2_x", "rat2_y", "rat2_area_px"])
            writer.writerow(header)
            for frame_idx in range(len(seen)):
                rat1_x, rat1_y, rat1_area = Sam2VideoUtils.mask_centroid(rat1_masks[frame_idx])
                row = [frame_idx, bool(seen[frame_idx]), rat1_x, rat1_y, rat1_area]
                if num_animals == 2:
                    rat2_x, rat2_y, rat2_area = Sam2VideoUtils.mask_centroid(rat2_masks[frame_idx])
                    row.extend([rat2_x, rat2_y, rat2_area])
                writer.writerow(row)

    @staticmethod
    def overlay_mask(frame, mask, color, alpha=0.45):
        """
        Input: image frame, boolean mask, BGR color tuple, and overlay opacity.
        Output: frame copy with the mask overlay and contour drawn.
        """
        if mask is None:
            return frame

        mask = mask.astype(bool)
        if mask.shape[:2] != frame.shape[:2]:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        out = frame.copy()
        color_layer = np.zeros_like(frame)
        color_layer[:] = color
        blended = cv2.addWeighted(frame, 1.0 - alpha, color_layer, alpha, 0)
        out[mask] = blended[mask]

        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(out, contours, -1, color, 2)
        return out

    @staticmethod
    def add_label(frame, mask, text, color):
        """
        Input: image frame, boolean mask, label text, and BGR color tuple.
        Output: frame with label text drawn near the mask centroid.
        """
        if mask is None or not np.any(mask):
            return frame

        ys, xs = np.where(mask)
        x = int(np.median(xs))
        y = int(np.median(ys))
        cv2.putText(
            frame,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )
        return frame

    @classmethod
    def write_overlay_video(
        cls,
        video_path,
        output_video,
        num_animals,
        rat1_masks,
        rat2_masks,
        fps,
        width,
        height,
        rat1_init_frame,
        rat2_init_frame,
    ):
        """
        Input: video path, output path, animal count, masks, video metadata, and prompt frames.
        Output: None; writes an MP4 overlay video to disk.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Could not create output video: {output_video}")

        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx >= len(rat1_masks):
                break

            frame = cls.overlay_mask(frame, rat1_masks[frame_idx], (0, 0, 255))
            frame = cls.add_label(frame, rat1_masks[frame_idx], "RAT 1", (0, 0, 255))
            if num_animals == 2:
                frame = cls.overlay_mask(frame, rat2_masks[frame_idx], (0, 255, 0))
                frame = cls.add_label(frame, rat2_masks[frame_idx], "RAT 2", (0, 255, 0))

            cv2.putText(
                frame,
                f"frame {frame_idx}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            if num_animals == 2 and frame_idx == rat1_init_frame == rat2_init_frame:
                cv2.putText(
                    frame,
                    "INITIALIZATION FRAME",
                    (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            else:
                if frame_idx == rat1_init_frame:
                    cv2.putText(
                        frame,
                        "RAT 1 INIT FRAME",
                        (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
                if num_animals == 2 and frame_idx == rat2_init_frame:
                    cv2.putText(
                        frame,
                        "RAT 2 INIT FRAME",
                        (30, 130 if frame_idx == rat1_init_frame else 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

            writer.write(frame)
            frame_idx += 1

        cap.release()
        writer.release()
