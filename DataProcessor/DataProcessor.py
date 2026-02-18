#!/usr/bin/env python3
"""
DataProcessor - Convert KeyRecorder logs + video to Lumine training format

Usage:
    python DataProcessor.py --log input_log.txt --video input.mkv --output dataset/
"""

import argparse
import os
import re
import subprocess
import json
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import shutil


@dataclass
class KeyChunkEvent:
    timestamp: int  # 100-nanosecond intervals (FILETIME)
    keys: List[str]


@dataclass
class MouseEvent:
    timestamp: int
    event_type: str  # ABS, REL, WHEEL
    x: int = 0
    y: int = 0
    dx: int = 0
    dy: int = 0
    delta: int = 0


@dataclass
class ActionChunk:
    """One 33ms chunk of actions"""

    dx: int = 0  # mouse X movement
    dy: int = 0  # mouse Y movement
    scroll: int = 0  # scroll amount
    keys: List[str] = field(default_factory=list)


@dataclass
class ActionFrame:
    """200ms (6 chunks) of actions at 5fps"""

    chunks: List[ActionChunk]

    def to_lumine_format(self) -> str:
        # Format: <|action_start|>X Y Z ; k1 k2 k3 ; k4 k5 ; k6 ; k7 ; k8 ; k9 k10<|action_end|>

        # Mouse Movement: First, specify the relative displacement X, Y and scroll amount Z
        # The paper says: "discretize mouse movement values using units of 5 pixels along the X-axis and 4 pixels along the Y-axis."
        total_dx = sum(c.dx for c in self.chunks)
        total_dy = sum(c.dy for c in self.chunks)
        total_scroll = sum(c.scroll for c in self.chunks)

        dx = int(round(total_dx / 5) * 5)
        dy = int(round(total_dy / 4) * 4)
        z = int(total_scroll)

        key_parts = []
        for chunk in self.chunks:
            # Each chunk can contain up to 4 keys.
            keys = chunk.keys[:4]
            key_parts.append(" ".join(keys))

        return f"<|action_start|>{dx} {dy} {z} ; {' ; '.join(key_parts)}<|action_end|>"


class KeyRecorderParser:
    """Parse KeyRecorder log file"""

    def __init__(self, log_path: str):
        self.log_path = log_path
        self.key_chunks: List[KeyChunkEvent] = []
        self.mouse_events: List[MouseEvent] = []
        self.start_timestamp: Optional[int] = None

    def parse(self):
        """Parse the log file"""
        if not os.path.exists(self.log_path):
            print(f"Error: Log file {self.log_path} not found")
            return self

        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split(",")
                if len(parts) < 2:
                    continue

                try:
                    timestamp = int(parts[0])
                except ValueError:
                    continue

                # Set start timestamp
                if self.start_timestamp is None:
                    self.start_timestamp = timestamp

                event_type = parts[1]

                if event_type == "KEY_CHUNK":
                    keys = parts[2].split() if len(parts) > 2 else []
                    self.key_chunks.append(KeyChunkEvent(timestamp, keys))

                elif event_type == "MOUSE_ABS":
                    x = int(parts[2]) if len(parts) > 2 else 0
                    y = int(parts[3]) if len(parts) > 3 else 0
                    self.mouse_events.append(MouseEvent(timestamp, "ABS", x=x, y=y))

                elif event_type == "MOUSE_REL":
                    dx = int(parts[2]) if len(parts) > 2 else 0
                    dy = int(parts[3]) if len(parts) > 3 else 0
                    self.mouse_events.append(MouseEvent(timestamp, "REL", dx=dx, dy=dy))

                elif event_type == "MOUSE":
                    if len(parts) > 2 and parts[2] == "WHEEL":
                        delta = int(parts[3]) if len(parts) > 3 else 0
                        self.mouse_events.append(
                            MouseEvent(timestamp, "WHEEL", delta=delta)
                        )

        # Sort by timestamp
        self.key_chunks.sort(key=lambda x: x.timestamp)
        self.mouse_events.sort(key=lambda x: x.timestamp)

        print(
            f"Parsed {len(self.key_chunks)} key chunks, {len(self.mouse_events)} mouse events"
        )
        return self

    def get_actions_at_time(
        self, start_time: int, duration_ms: int = 200
    ) -> ActionFrame:
        """Get actions for a time window (default 200ms for 1 frame at 5fps)"""
        end_time = start_time + (duration_ms * 10000)  # Convert ms to 100ns units

        # Split into 6 chunks of 33.33ms each
        chunks = []
        chunk_duration_100ns = int((duration_ms / 6) * 10000)

        for i in range(6):
            c_start = start_time + (i * chunk_duration_100ns)
            c_end = c_start + chunk_duration_100ns

            # Get mouse events in this chunk
            chunk_mouse = [
                e for e in self.mouse_events if c_start <= e.timestamp < c_end
            ]

            dx = sum(e.dx for e in chunk_mouse if e.event_type == "REL")
            dy = sum(e.dy for e in chunk_mouse if e.event_type == "REL")
            scroll = sum(e.delta for e in chunk_mouse if e.event_type == "WHEEL")

            # For KEY_CHUNK, we take the last one in the interval if any,
            # or the one closest to the end of the interval
            chunk_keys_events = [
                e for e in self.key_chunks if c_start <= e.timestamp < c_end
            ]

            if chunk_keys_events:
                # Use the latest state in this 33ms window
                keys = chunk_keys_events[-1].keys
            else:
                # Find the most recent state before this window
                past_events = [e for e in self.key_chunks if e.timestamp < c_start]
                keys = past_events[-1].keys if past_events else []

            chunks.append(ActionChunk(dx=dx, dy=dy, scroll=scroll, keys=keys))

        return ActionFrame(chunks=chunks)


class VideoProcessor:
    """Extract frames from video using FFmpeg"""

    def __init__(
        self,
        video_path: str,
        output_dir: str,
        fps: int = 5,
        width: int = 1280,
        height: int = 720,
    ):
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.fps = fps
        self.width = width
        self.height = height

    def extract_frames(self) -> List[Path]:
        """Extract frames at specified fps and resolution"""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg",
            "-i",
            self.video_path,
            "-vf",
            f"fps={self.fps},scale={self.width}:{self.height}",
            "-q:v",
            "2",
            "-start_number",
            "0",
            "-y",
            str(self.output_dir / "frame_%05d.jpg"),
        ]

        print(f"Extracting frames: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"FFmpeg error: {result.stderr}")
                return []
        except FileNotFoundError:
            print("Error: ffmpeg not found.")
            return []

        frames = sorted(self.output_dir.glob("frame_*.jpg"))
        print(f"Extracted {len(frames)} frames")

        return frames

    def get_video_info(self) -> Tuple[int, int]:
        """Get video duration in 100ns units"""
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            self.video_path,
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            duration_sec = float(result.stdout.strip())
            duration_100ns = int(duration_sec * 10000000)
            return 0, duration_100ns
        except (FileNotFoundError, ValueError):
            return 0, 0


class LumineDataset:
    """Create Lumine-compatible dataset"""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.samples: List[Dict] = []

    def add_sample(
        self,
        frame_idx: int,
        frame_path: Path,
        action: str,
        instruction: Optional[str] = None,
        thought: Optional[str] = None,
    ):
        """Add a training sample"""
        sample = {"frame_idx": frame_idx, "image": frame_path.name, "action": action}

        if instruction:
            sample["instruction"] = instruction
        if thought:
            sample["thought"] = thought

        self.samples.append(sample)

    def save(self):
        """Save dataset metadata"""
        with open(self.output_dir / "metadata.jsonl", "w", encoding="utf-8") as f:
            for sample in self.samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        # Stage 1: Pretrain (image -> action)
        with open(
            self.output_dir / "stage1_pretrain.jsonl", "w", encoding="utf-8"
        ) as f:
            for sample in self.samples:
                out = {"image": sample["image"], "text": sample["action"]}
                f.write(json.dumps(out, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Convert KeyRecorder logs to Lumine training format"
    )
    parser.add_argument("--log", required=True, help="Path to log file or directory")
    parser.add_argument(
        "--video", required=True, help="Path to video file or directory"
    )
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--fps", type=int, default=5, help="FPS (default: 5)")
    parser.add_argument("--width", type=int, default=1280, help="Width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Height (default: 720)")
    parser.add_argument(
        "--offset", type=int, default=0, help="Timestamp offset (100ns)"
    )
    parser.add_argument("--skip-video", action="store_true", help="Skip extraction")

    args = parser.parse_args()

    log_path = Path(args.log)
    video_path = Path(args.video)
    output_base = Path(args.output)

    # Simple single-pair logic for brevity in this tool call
    if not log_path.is_dir():
        log_files = [log_path]
        video_files = [video_path]
    else:
        log_files = sorted(log_path.glob("*.txt"))
        video_files = [video_path / f"{f.stem}.mkv" for f in log_files]

    for log_f, video_f in zip(log_files, video_files):
        if not video_f.exists():
            continue

        print(f"\nProcessing: {log_f.name}")
        output_dir = output_base / log_f.stem

        parser_obj = KeyRecorderParser(str(log_f)).parse()

        video_proc = VideoProcessor(
            str(video_f),
            str(output_dir / "frames"),
            fps=args.fps,
            width=args.width,
            height=args.height,
        )

        frames = []
        if not args.skip_video:
            frames = video_proc.extract_frames()
        else:
            frames = sorted((output_dir / "frames").glob("frame_*.jpg"))

        if not frames:
            continue

        video_start, video_end = video_proc.get_video_info()

        # Sync: log starts at parser_obj.start_timestamp
        # Video starts at 0 (relative to its own clock)
        # We need to find the absolute time of the first frame.
        # Assuming the log start and video start are aligned (or using manual offset)

        dataset = LumineDataset(str(output_dir))

        frame_interval_100ns = 10000000 // args.fps

        for i, frame in enumerate(frames):
            # Frame time relative to video start
            frame_time_rel = i * frame_interval_100ns

            # Map to log time
            # For now, assume log start is video start, plus optional manual offset
            log_time = parser_obj.start_timestamp + frame_time_rel + args.offset

            if log_time > (
                parser_obj.key_chunks[-1].timestamp if parser_obj.key_chunks else 0
            ):
                break

            action_frame = parser_obj.get_actions_at_time(log_time, duration_ms=200)
            dataset.add_sample(
                frame_idx=i, frame_path=frame, action=action_frame.to_lumine_format()
            )

        dataset.save()
        print(f"Done: {log_f.stem}")


if __name__ == "__main__":
    main()
