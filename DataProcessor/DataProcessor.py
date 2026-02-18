#!/usr/bin/env python3
"""
DataProcessor - Convert KeyRecorder logs + video to Lumine training format

Usage:
    python DataProcessor.py --log input_log.txt --video input.mkv --output dataset/
    python DataProcessor.py --log input_log.txt --video input.mkv --output dataset/ --fps 5 --width 1280 --height 720
"""

import argparse
import os
import re
import subprocess
import json
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import shutil


@dataclass
class KeyEvent:
    timestamp: int  # 100-nanosecond intervals (FILETIME)
    key: str
    is_down: bool


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
    keys: Optional[List[str]] = None  # keys pressed during this chunk

    def __post_init__(self):
        if self.keys is None:
            self.keys = []


def convert_filetime_to_unix(filetime: int) -> float:
    """Convert Windows FILETIME (100-ns since 1601) to Unix timestamp"""
    # Unix epoch is 11644473600 seconds after FILETIME epoch
    unix_seconds = (filetime - 116444736000000000) / 10000000
    return unix_seconds


@dataclass
class ActionFrame:
    """200ms (6 chunks) of actions at 5fps"""

    chunks: List[ActionChunk]

    def to_lumine_format(self) -> str:
        # Format: <|action_start|>X Y Z ; k1 k2 ; k3 ; k4 ; k5 ; k6<|action_end|>
        dx = round(self.chunks[0].dx / 5) * 5  # discretize to 5px
        dy = round(self.chunks[0].dy / 4) * 4  # discretize to 4px
        scroll = self.chunks[0].scroll

        key_parts = []
        for chunk in self.chunks:
            if chunk.keys:
                key_parts.append(" ".join(chunk.keys))
            else:
                key_parts.append("")

        return f"<|action_start|>{dx} {dy} {scroll} ; {' ; '.join(key_parts)}<|action_end|>"


class KeyRecorderParser:
    """Parse KeyRecorder log file"""

    def __init__(self, log_path: str):
        self.log_path = log_path
        self.key_events: List[KeyEvent] = []
        self.mouse_events: List[MouseEvent] = []
        self.start_timestamp: Optional[int] = None

    def parse(self):
        """Parse the log file"""
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split(",")
                if len(parts) < 3:
                    continue

                try:
                    timestamp = int(parts[0])
                except ValueError:
                    continue

                # Set start timestamp
                if self.start_timestamp is None:
                    self.start_timestamp = timestamp

                event_type = parts[1]

                if event_type == "KEY":
                    key = parts[2] if len(parts) > 2 else ""
                    is_down = parts[3] == "DOWN" if len(parts) > 3 else True
                    self.key_events.append(KeyEvent(timestamp, key, is_down))

                elif event_type == "MOUSE_ABS":
                    x = int(parts[2]) if len(parts) > 2 else 0
                    y = int(parts[3]) if len(parts) > 3 else 0
                    self.mouse_events.append(MouseEvent(timestamp, "ABS", x=x, y=y))

                elif event_type == "MOUSE_REL":
                    dx = int(parts[2]) if len(parts) > 2 else 0
                    dy = int(parts[3]) if len(parts) > 3 else 0
                    self.mouse_events.append(MouseEvent(timestamp, "REL", dx=dx, dy=dy))

                elif event_type == "MOUSE":
                    if len(parts) > 2:
                        if parts[2].startswith("WHEEL"):
                            delta = int(parts[3]) if len(parts) > 3 else 0
                            self.mouse_events.append(
                                MouseEvent(timestamp, "WHEEL", delta=delta)
                            )
                        elif "LB" in parts[2]:
                            self.mouse_events.append(
                                MouseEvent(
                                    timestamp,
                                    "LB_DOWN" if "DOWN" in parts[2] else "LB_UP",
                                )
                            )
                        elif "RB" in parts[2]:
                            self.mouse_events.append(
                                MouseEvent(
                                    timestamp,
                                    "RB_DOWN" if "DOWN" in parts[2] else "RB_UP",
                                )
                            )
                        elif "MB" in parts[2]:
                            self.mouse_events.append(
                                MouseEvent(
                                    timestamp,
                                    "MB_DOWN" if "DOWN" in parts[2] else "MB_UP",
                                )
                            )

        # Sort by timestamp
        self.key_events.sort(key=lambda x: x.timestamp)
        self.mouse_events.sort(key=lambda x: x.timestamp)

        print(
            f"Parsed {len(self.key_events)} key events, {len(self.mouse_events)} mouse events"
        )
        print(f"Start timestamp: {self.start_timestamp}")

        return self

    def get_actions_at_time(
        self, start_time: int, duration_ms: int = 200
    ) -> ActionFrame:
        """Get actions for a time window (default 200ms for 1 frame at 5fps)"""
        end_time = start_time + (duration_ms * 10000)  # Convert ms to 100ns units

        # Get events in this window
        keys_in_window = [
            e for e in self.key_events if start_time <= e.timestamp < end_time
        ]
        mouse_in_window = [
            e for e in self.mouse_events if start_time <= e.timestamp < end_time
        ]

        # Split into 6 chunks of 33ms each
        chunks = []
        chunk_duration = 33 * 10000  # 33ms in 100ns units

        active_keys = set()  # Track currently pressed keys

        for i in range(6):
            chunk_start = start_time + (i * chunk_duration)
            chunk_end = chunk_start + chunk_duration

            # Get mouse events in this chunk
            chunk_mouse = [
                e for e in mouse_in_window if chunk_start <= e.timestamp < chunk_end
            ]

            dx = sum(e.dx for e in chunk_mouse if e.event_type == "REL")
            dy = sum(e.dy for e in chunk_mouse if e.event_type == "REL")
            scroll = sum(e.delta for e in chunk_mouse if e.event_type == "WHEEL")

            # Get key down events in this chunk
            chunk_keys = [
                e for e in keys_in_window if chunk_start <= e.timestamp < chunk_end
            ]

            # Update active keys
            for e in chunk_keys:
                if e.is_down:
                    active_keys.add(e.key)
                else:
                    active_keys.discard(e.key)

            chunks.append(
                ActionChunk(dx=dx, dy=dy, scroll=scroll, keys=sorted(list(active_keys)))
            )

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

        # FFmpeg command to extract frames
        # -vf "fps=5,scale=1280:720" -q:v 2 for JPEG quality
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
            str(self.output_dir / "frame_%05d.jpg"),
        ]

        print(f"Extracting frames: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"FFmpeg error: {result.stderr}")
                return []
        except FileNotFoundError:
            print("Error: ffmpeg not found. Please install ffmpeg.")
            return []

        # Get list of extracted frames
        frames = sorted(self.output_dir.glob("frame_*.jpg"))
        print(f"Extracted {len(frames)} frames")

        return frames

    def get_frame_timestamp(self, frame_idx: int) -> int:
        """Calculate timestamp for a frame (in 100ns units)"""
        # At fps, each frame is (1/fps) seconds = (10000000/fps) 100ns units
        frame_interval = 10000000 // self.fps
        return frame_idx * frame_interval

    def get_video_info(self) -> Tuple[int, int]:
        """Get video start and end timestamps in 100ns units (start always 0, duration calculated)"""
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
        # Save as JSONL
        with open(self.output_dir / "metadata.jsonl", "w", encoding="utf-8") as f:
            for sample in self.samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        # Save as separate files for each stage
        # Stage 1: image-action pairs
        with open(
            self.output_dir / "stage1_pretrain.jsonl", "w", encoding="utf-8"
        ) as f:
            for sample in self.samples:
                # Format: {"image": "frame_00001.jpg", "action": "<|action_start|>..."}
                out = {"image": sample["image"], "text": sample["action"]}
                f.write(json.dumps(out, ensure_ascii=False) + "\n")

        # Stage 2: instruction-image-action (if available)
        inst_samples = [s for s in self.samples if "instruction" in s]
        if inst_samples:
            with open(
                self.output_dir / "stage2_instruct.jsonl", "w", encoding="utf-8"
            ) as f:
                for sample in inst_samples:
                    out = {
                        "instruction": sample["instruction"],
                        "image": sample["image"],
                        "answer": sample["action"],
                    }
                    f.write(json.dumps(out, ensure_ascii=False) + "\n")

        # Stage 3: thought-image-action (if available)
        thought_samples = [s for s in self.samples if "thought" in s]
        if thought_samples:
            with open(
                self.output_dir / "stage3_reasoning.jsonl", "w", encoding="utf-8"
            ) as f:
                for sample in thought_samples:
                    out = {
                        "thought": sample["thought"],
                        "image": sample["image"],
                        "answer": sample["action"],
                    }
                    f.write(json.dumps(out, ensure_ascii=False) + "\n")

        print(f"Saved {len(self.samples)} samples to {self.output_dir}")
        print(f"  - stage1_pretrain.jsonl: {len(self.samples)}")
        print(f"  - stage2_instruct.jsonl: {len(inst_samples)}")
        print(f"  - stage3_reasoning.jsonl: {len(thought_samples)}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert KeyRecorder logs to Lumine training format"
    )
    parser.add_argument(
        "--log",
        required=True,
        help="Path to KeyRecorder log file OR directory containing .txt logs",
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to video file OR directory containing .mkv videos",
    )
    parser.add_argument("--output", required=True, help="Output directory for dataset")
    parser.add_argument(
        "--fps", type=int, default=5, help="Frames per second (default: 5)"
    )
    parser.add_argument(
        "--width", type=int, default=1280, help="Frame width (default: 1280)"
    )
    parser.add_argument(
        "--height", type=int, default=720, help="Frame height (default: 720)"
    )
    parser.add_argument(
        "--timestamp-offset",
        type=int,
        default=0,
        help="Manual timestamp offset to sync with video",
    )
    parser.add_argument(
        "--skip-video",
        action="store_true",
        help="Skip video extraction (use existing frames)",
    )

    args = parser.parse_args()

    print("=" * 50)
    print("Lumine Data Processor")
    print("=" * 50)

    # Check if input is directory or single file
    log_path = Path(args.log)
    video_path = Path(args.video)

    # Find all pairs
    if log_path.is_dir() and video_path.is_dir():
        # Batch processing: find matching pairs
        log_files = sorted(log_path.glob("*.txt"))

        pairs = []
        for log_file in log_files:
            name = log_file.stem
            video_file = video_path / f"{name}.mkv"
            if video_file.exists():
                pairs.append((log_file, video_file))
            else:
                print(f"Warning: No matching video for {log_file.name}")

        if not pairs:
            print("Error: No matching log/video pairs found")
            return

        print(f"\nFound {len(pairs)} pairs to process:")
        for log_file, video_file in pairs:
            print(f"  - {log_file.name} + {video_file.name}")

        # Process each pair
        all_samples = []

        for i, (log_file, video_file) in enumerate(pairs):
            print(f"\n{'=' * 50}")
            print(f"Processing pair {i + 1}/{len(pairs)}: {log_file.stem}")
            print(f"{'=' * 50}")

            output_dir = Path(args.output) / log_file.stem

            # Parse KeyRecorder log
            print("\n[1/4] Parsing KeyRecorder log...")
            parser_obj = KeyRecorderParser(str(log_file))
            parser_obj.parse()

            # Extract video frames
            if args.skip_video:
                print("\n[2/4] Skipping video extraction...")
                frames_dir = output_dir / "frames"
                frames = sorted(frames_dir.glob("frame_*.jpg"))
                print(f"Found {len(frames)} existing frames")
            else:
                print("\n[2/4] Extracting video frames...")
                video_proc = VideoProcessor(
                    str(video_file),
                    str(output_dir / "frames"),
                    fps=args.fps,
                    width=args.width,
                    height=args.height,
                )
                frames = video_proc.extract_frames()

            if not frames:
                print(f"Warning: No frames extracted for {log_file.stem}")
                continue

            # Get video info for synchronization
            video_proc = VideoProcessor(
                str(video_file),
                str(output_dir / "frames"),
                fps=args.fps,
                width=args.width,
                height=args.height,
            )
            video_start, video_end = video_proc.get_video_info()

            # Get key timestamps
            key_start = parser_obj.start_timestamp + args.timestamp_offset
            all_events = parser_obj.key_events + parser_obj.mouse_events
            last_event_ts = max((e.timestamp for e in all_events), default=key_start)
            key_end = last_event_ts + args.timestamp_offset

            # Calculate overlap range
            overlap_start = max(video_start, key_start)
            overlap_end = min(video_end, key_end)

            if overlap_start >= overlap_end:
                print(f"Warning: No overlapping time range for {log_file.stem}")
                print(
                    f"  Video: {video_start} - {video_end}, Keys: {key_start} - {key_end}"
                )
                continue

            print(
                f"Timestamp sync: Video [{video_start}, {video_end}], Keys [{key_start}, {key_end}]"
            )
            print(f"Overlap range: [{overlap_start}, {overlap_end}]")

            # Generate action labels
            print("\n[3/4] Generating action labels...")
            dataset = LumineDataset(str(output_dir))

            frame_duration = 10000000 // args.fps

            valid_frame_count = 0
            for j, frame in enumerate(frames):
                frame_time = j * frame_duration

                if frame_time < overlap_start or frame_time >= overlap_end:
                    continue

                action_frame = parser_obj.get_actions_at_time(
                    frame_time, duration_ms=200
                )
                action_str = action_frame.to_lumine_format()

                dataset.add_sample(
                    frame_idx=valid_frame_count, frame_path=frame, action=action_str
                )
                all_samples.append(
                    {
                        "source": log_file.stem,
                        "frame_idx": valid_frame_count,
                        "image": frame.name,
                        "action": action_str,
                    }
                )
                valid_frame_count += 1

            print(f"Generated {valid_frame_count} samples (within overlap)")

            # Save dataset
            print("\n[4/4] Saving dataset...")
            dataset.save()

        # Save combined dataset
        print(f"\n{'=' * 50}")
        print("Saving combined dataset...")
        combined_dir = Path(args.output)
        combined_dir.mkdir(parents=True, exist_ok=True)

        # Also save frames in combined directory
        combined_frames_dir = combined_dir / "frames"
        combined_frames_dir.mkdir(parents=True, exist_ok=True)

        # Copy all frames and track renames
        frame_mapping = {}  # old_name -> new_name (with source prefix)

        for sample in all_samples:
            source = sample["source"]
            old_name = sample["image"]
            new_name = f"{source}_{old_name}"
            frame_mapping[old_name] = new_name

            # Find and copy the frame
            source_dir = Path(args.output) / source / "frames"
            if source_dir.exists():
                src_frame = source_dir / old_name
                if src_frame.exists():
                    dst_frame = combined_frames_dir / new_name
                    if not dst_frame.exists():
                        shutil.copy2(src_frame, dst_frame)

        # Save combined JSONL files
        with open(combined_dir / "all_samples.jsonl", "w", encoding="utf-8") as f:
            for sample in all_samples:
                # Update image path to include source prefix
                sample_copy = sample.copy()
                sample_copy["image"] = frame_mapping.get(
                    sample["image"], sample["image"]
                )
                f.write(json.dumps(sample_copy, ensure_ascii=False) + "\n")

        # Save combined stage files
        stage1_samples = []
        stage2_samples = []
        stage3_samples = []

        # Reorganize by stage
        for log_file, video_file in pairs:
            source_dir = Path(args.output) / log_file.stem
            for stage_file, stage_list in [
                (source_dir / "stage1_pretrain.jsonl", stage1_samples),
                (source_dir / "stage2_instruct.jsonl", stage2_samples),
                (source_dir / "stage3_reasoning.jsonl", stage3_samples),
            ]:
                if stage_file.exists():
                    with open(stage_file, "r", encoding="utf-8") as sf:
                        for line in sf:
                            data = json.loads(line)
                            # Update image path
                            if "image" in data:
                                data["image"] = f"{log_file.stem}_{data['image']}"
                            stage_list.append(data)

        # Write combined stage files
        if stage1_samples:
            with open(
                combined_dir / "stage1_pretrain.jsonl", "w", encoding="utf-8"
            ) as f:
                for sample in stage1_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            print(f"  - stage1_pretrain.jsonl: {len(stage1_samples)} samples")

        if stage2_samples:
            with open(
                combined_dir / "stage2_instruct.jsonl", "w", encoding="utf-8"
            ) as f:
                for sample in stage2_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            print(f"  - stage2_instruct.jsonl: {len(stage2_samples)} samples")

        if stage3_samples:
            with open(
                combined_dir / "stage3_reasoning.jsonl", "w", encoding="utf-8"
            ) as f:
                for sample in stage3_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            print(f"  - stage3_reasoning.jsonl: {len(stage3_samples)} samples")

        print(f"\nTotal samples: {len(all_samples)}")
        print(f"\n{'=' * 50}")
        print("Done!")
        print(f"Output: {args.output}")

    else:
        # Single file processing
        _process_single(args.log, args.video, args.output, args)


def _process_single(log_file: str, video_file: str, output_dir: str, args):
    """Process a single log/video pair"""
    print("=" * 50)
    print("Lumine Data Processor - Single File Mode")
    print("=" * 50)

    # Parse KeyRecorder log
    print("\n[1/4] Parsing KeyRecorder log...")
    parser_obj = KeyRecorderParser(log_file)
    parser_obj.parse()

    # Extract video frames
    if args.skip_video:
        print("\n[2/4] Skipping video extraction...")
        frames_dir = Path(output_dir) / "frames"
        frames = sorted(frames_dir.glob("frame_*.jpg"))
        print(f"Found {len(frames)} existing frames")
    else:
        print("\n[2/4] Extracting video frames...")
        video_proc = VideoProcessor(
            video_file,
            os.path.join(output_dir, "frames"),
            fps=args.fps,
            width=args.width,
            height=args.height,
        )
        frames = video_proc.extract_frames()

    if not frames:
        print("Error: No frames extracted")
        return

    # Get video info for synchronization
    video_proc = VideoProcessor(
        video_file,
        os.path.join(output_dir, "frames"),
        fps=args.fps,
        width=args.width,
        height=args.height,
    )
    video_start, video_end = video_proc.get_video_info()

    # Get key timestamps
    key_start = parser_obj.start_timestamp + args.timestamp_offset
    all_events = parser_obj.key_events + parser_obj.mouse_events
    last_event_ts = max((e.timestamp for e in all_events), default=key_start)
    key_end = last_event_ts + args.timestamp_offset

    # Calculate overlap range
    overlap_start = max(video_start, key_start)
    overlap_end = min(video_end, key_end)

    if overlap_start >= overlap_end:
        print(f"Warning: No overlapping time range")
        print(f"  Video: {video_start} - {video_end}, Keys: {key_start} - {key_end}")
        return

    print(
        f"Timestamp sync: Video [{video_start}, {video_end}], Keys [{key_start}, {key_end}]"
    )
    print(f"Overlap range: [{overlap_start}, {overlap_end}]")

    # Generate action labels
    print("\n[3/4] Generating action labels...")
    dataset = LumineDataset(output_dir)

    frame_duration = 10000000 // args.fps
    valid_frame_count = 0

    for i, frame in enumerate(frames):
        frame_time = i * frame_duration

        if frame_time < overlap_start or frame_time >= overlap_end:
            continue

        action_frame = parser_obj.get_actions_at_time(frame_time, duration_ms=200)
        action_str = action_frame.to_lumine_format()

        dataset.add_sample(
            frame_idx=valid_frame_count, frame_path=frame, action=action_str
        )
        valid_frame_count += 1

    print(f"Generated {valid_frame_count} samples (within overlap)")

    # Save dataset
    print("\n[4/4] Saving dataset...")
    dataset.save()

    print("\n" + "=" * 50)
    print("Done!")
    print(f"Output: {output_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()
