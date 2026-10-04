"""Create separated, reproducible augmentation sets from the local samples.

These outputs are augmentation data, not new clinical subjects. They are kept
outside the source folders and must not be used as independent test subjects.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm", ".mkv"}


def stable_seed(path: Path, index: int) -> int:
    digest = hashlib.sha256(f"{path.resolve()}:{index}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def augment_audio(source: Path, destination: Path, index: int) -> None:
    audio, sample_rate = sf.read(source, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if audio.size == 0:
        raise ValueError(f"Empty audio file: {source}")
    rng = np.random.default_rng(stable_seed(source, index))
    gain = float(rng.uniform(0.88, 1.12))
    offset = int(rng.integers(0, max(1, len(audio) // 20)))
    shifted = np.roll(audio, offset) * gain
    noise_scale = float(rng.uniform(0.0005, 0.004)) * max(float(np.max(np.abs(shifted))), 0.1)
    output = shifted + rng.normal(0.0, noise_scale, size=shifted.shape)
    peak = float(np.max(np.abs(output))) or 1.0
    output = (output / max(peak, 1.0)) * 0.95
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination, output.astype(np.float32), sample_rate, subtype="PCM_16")


def extract_gait_features(video: Path) -> list[float] | None:
    """Extract the same 12 values used by the video gait model."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None
    ankle: list[float] = []
    knee: list[float] = []
    wrist: list[float] = []
    frame_index = 0
    try:
        import mediapipe as mp
        pose_api = mp.solutions.pose
        with pose_api.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_index += 1
                if frame_index % 5:
                    continue
                result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if not result.pose_landmarks:
                    continue
                points = result.pose_landmarks.landmark

                def distance(left: int, right: int) -> float | None:
                    first, second = points[left], points[right]
                    if min(first.visibility, second.visibility) <= 0.4:
                        return None
                    return float(np.hypot(first.x - second.x, first.y - second.y))

                for target, left, right in ((ankle, 27, 28), (knee, 25, 26), (wrist, 15, 16)):
                    value = distance(left, right)
                    if value is not None:
                        target.append(value)
    finally:
        cap.release()
    if len(ankle) < 5:
        return None
    return [
        float(np.mean(ankle)), float(np.std(ankle)), float(np.ptp(ankle)),
        float(np.mean(knee)) if knee else 0.0, float(np.std(knee)) if knee else 0.0, float(np.ptp(knee)) if knee else 0.0,
        float(np.mean(wrist)) if wrist else 0.0, float(np.std(wrist)) if wrist else 0.0, float(np.ptp(wrist)) if wrist else 0.0,
        float(np.median(ankle)), float(np.max(ankle)), float(np.min(ankle)),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=8000)
    parser.add_argument("--output", type=Path, default=ROOT / "generated_samples")
    parser.add_argument("--skip-gait", action="store_true")
    parser.add_argument("--fast", action="store_true", help="Replicate source WAV bytes for pipeline/load testing")
    parser.add_argument("--gait-clips", action="store_true", help="Also create separated gait clip copies for pipeline/load testing")
    parser.add_argument("--skip-voice", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    random.seed(42)

    audio_sources = {"healthy": sorted((ROOT / "HC_AH").glob("*")), "diagnosed": sorted((ROOT / "PD_AH").glob("*"))}
    manifest: list[dict[str, str | int]] = []
    for label, candidates in (() if args.skip_voice else audio_sources.items()):
        sources = [path for path in candidates if path.suffix.lower() in AUDIO_EXTENSIONS]
        if not sources:
            raise SystemExit(f"No supported audio files found for {label}")
        for index in range(args.per_class):
            source = sources[index % len(sources)]
            destination = output / "voice" / label / f"{label}_{index + 1:05d}.wav"
            destination.parent.mkdir(parents=True, exist_ok=True)
            if args.fast:
                shutil.copyfile(source, destination)
            else:
                augment_audio(source, destination, index)
            manifest.append({"kind": "voice", "label": label, "source": str(source.relative_to(ROOT)), "path": str(destination.relative_to(ROOT)), "augmented": 0 if args.fast else 1})

    if args.skip_gait:
        gait_sources = []
    else:
        gait_sources = [path for path in (ROOT / "gait_clips").rglob("*") if path.suffix.lower() in VIDEO_EXTENSIONS]
    if not gait_sources and args.skip_gait:
        with (output / "README.txt").open("a", encoding="ascii") as handle:
            handle.write("Gait generation was skipped; run without --skip-gait to extract pose-validated gait features.\n")
        with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["kind", "label", "source", "path", "augmented"])
            writer.writeheader()
            writer.writerows(manifest)
        print(f"Generated {len(manifest)} voice augmentations in {output}")
        return
    if not gait_sources:
        raise SystemExit("No supported gait videos found")
    if args.gait_clips:
        def gait_label(source: Path) -> str:
            folder = source.parent.name.lower()
            return "diagnosed" if folder in {"parkinsons", "parkinson", "pd", "diagnosed"} else "healthy"

        grouped_sources = {
            "healthy": [source for source in gait_sources if gait_label(source) == "healthy"],
            "diagnosed": [source for source in gait_sources if gait_label(source) == "diagnosed"],
        }
        for label, sources in grouped_sources.items():
            if not sources:
                raise SystemExit(f"No gait source clips found for {label}; refusing to invent a class")
            for index in range(args.per_class):
                source = sources[index % len(sources)]
                destination = output / "gait_clips" / label / f"{label}_{index + 1:05d}{source.suffix.lower()}"
                destination.parent.mkdir(parents=True, exist_ok=True)
                if args.fast:
                    os.link(source, destination)
                else:
                    shutil.copyfile(source, destination)
                manifest.append({"kind": "gait_clip", "label": label, "source": str(source.relative_to(ROOT)), "path": str(destination.relative_to(ROOT)), "augmented": 0})
        with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["kind", "label", "source", "path", "augmented"])
            writer.writeheader()
            writer.writerows(manifest)
        (output / "README.txt").write_text("Replicated voice/gait files for pipeline and load testing. These are not new clinical subjects and must not be used as independent test data.\n", encoding="ascii")
        print(f"Generated {len(manifest)} pipeline-test files in {output}")
        return
    gait_rows: list[dict[str, str | int | float]] = []
    for source in gait_sources:
        label = "diagnosed" if "parkinson" in str(source).lower() or "pd" in str(source).lower() else "healthy"
        features = extract_gait_features(source)
        if features is not None:
            row = {f"feature_{index}": value for index, value in enumerate(features)}
            row.update({"label": label, "source": str(source.relative_to(ROOT)), "augmented": 0})
            gait_rows.append(row)
    if not gait_rows:
        raise SystemExit("No gait clips produced reliable pose features; no fabricated gait samples were created")
    gait_base = list(gait_rows)
    gait_rng = np.random.default_rng(42)
    for label in ("healthy", "diagnosed"):
        label_rows = [row for row in gait_base if row["label"] == label]
        if not label_rows:
            continue
        for index in range(len(label_rows), args.per_class):
            base = label_rows[index % len(label_rows)]
            row = {f"feature_{feature_index}": float(base[f"feature_{feature_index}"]) * (1.0 + float(gait_rng.normal(0.0, 0.015))) for feature_index in range(12)}
            row.update({"label": label, "source": base["source"], "augmented": 1})
            gait_rows.append(row)
    with (output / "gait" / "extracted_features.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[f"feature_{index}" for index in range(12)] + ["label", "source", "augmented"])
        writer.writeheader()
        writer.writerows(gait_rows)

    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["kind", "label", "source", "path", "augmented"])
        writer.writeheader()
        writer.writerows(manifest)
    mode = "replicated source WAVs for pipeline/load testing" if args.fast else "augmented voice samples"
    (output / "README.txt").write_text(f"{mode} and reliable gait feature extractions. These are not new clinical subjects and must not be used as independent test data.\n", encoding="ascii")
    print(f"Generated {len(manifest)} voice augmentations and {len(gait_rows)} reliable gait feature rows in {output}")


if __name__ == "__main__":
    main()