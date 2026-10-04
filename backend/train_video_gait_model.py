import os
import glob
import numpy as np
import pandas as pd
import joblib
import json
import warnings
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import cv2
import mediapipe as mp

warnings.filterwarnings('ignore')

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GAIT_CLIPS_DIR = os.path.join(os.path.dirname(BASE_DIR), 'gait_clips')
HEALTHY_DIR = os.path.join(GAIT_CLIPS_DIR, 'healthy')
PD_DIR = os.path.join(GAIT_CLIPS_DIR, 'parkinsons')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

POSE_MODEL = os.path.join(MODELS_DIR, 'pose_landmarker_full.task')


def _pose_detector():
    """Create a pose detector for both legacy and current MediaPipe builds."""
    if hasattr(mp, 'solutions'):
        return mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    if not os.path.exists(POSE_MODEL):
        raise RuntimeError(f"Pose model not found: {POSE_MODEL}")
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=POSE_MODEL),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    return mp.tasks.vision.PoseLandmarker.create_from_options(options)


def _pose_landmarks(detector, frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if hasattr(mp, 'solutions'):
        result = detector.process(image)
        return result.pose_landmarks.landmark if result.pose_landmarks else None
    result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=image))
    return result.pose_landmarks[0] if result.pose_landmarks else None

def extract_features_from_video(video_path, skip_frames=5, max_frames=120):
    """
    Extract basic gait features using MediaPipe Pose.
    To be fast, we skip frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    ankle_distances = []
    knee_distances = []
    wrist_distances = []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return None
    # Live phone clips can be 4K; use an early bounded window after upload
    # normalization so decoding cannot consume the entire request budget.
    total_frames = min(total_frames, int((cap.get(cv2.CAP_PROP_FPS) or 24.0) * 2))
    sample_count = min(max_frames, total_frames)
    sample_indices = np.unique(np.linspace(0, total_frames - 1, sample_count, dtype=int))
    sample_index_set = set(int(index) for index in sample_indices)
    detector = _pose_detector()
    try:
        frame_index = 0
        while cap.isOpened() and frame_index <= int(sample_indices[-1]):
            ret, frame = cap.read()
            if not ret:
                break
            if frame_index not in sample_index_set:
                frame_index += 1
                continue

            # Pose landmarks are scale-normalized; limiting the working size
            # keeps live phone analysis responsive without changing features.
            height, width = frame.shape[:2]
            if max(height, width) > 640:
                scale = 640.0 / max(height, width)
                frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
                
            # Convert BGR to RGB
            landmarks = _pose_landmarks(detector, frame)
            if landmarks:
                
                # Extract keypoints we care about
                left_ankle = landmarks[27]
                right_ankle = landmarks[28]
                left_knee = landmarks[25]
                right_knee = landmarks[26]
                left_wrist = landmarks[15]
                right_wrist = landmarks[16]
                
                # Calculate 3D distances if visibility is decent
                if min(left_ankle.visibility, right_ankle.visibility) > 0.25:
                    dist = np.sqrt((left_ankle.x - right_ankle.x)**2 + (left_ankle.y - right_ankle.y)**2)
                    ankle_distances.append(dist)
                    
                if min(left_knee.visibility, right_knee.visibility) > 0.25:
                    dist = np.sqrt((left_knee.x - right_knee.x)**2 + (left_knee.y - right_knee.y)**2)
                    knee_distances.append(dist)
                    
                if min(left_wrist.visibility, right_wrist.visibility) > 0.25:
                    dist = np.sqrt((left_wrist.x - right_wrist.x)**2 + (left_wrist.y - right_wrist.y)**2)
                    wrist_distances.append(dist)
                    frame_index += 1

    finally:
        if hasattr(detector, 'close'):
            detector.close()
        cap.release()
    
    if len(ankle_distances) < 5:
        # Fallback if pose estimation failed on most frames
        return [0.0]*12
        
    # Calculate aggregate features
    features = [
        np.mean(ankle_distances), np.std(ankle_distances), np.ptp(ankle_distances),
        np.mean(knee_distances) if knee_distances else 0,
        np.std(knee_distances) if knee_distances else 0,
        np.ptp(knee_distances) if knee_distances else 0,
        np.mean(wrist_distances) if wrist_distances else 0,
        np.std(wrist_distances) if wrist_distances else 0,
        np.ptp(wrist_distances) if wrist_distances else 0,
        np.median(ankle_distances),
        np.max(ankle_distances) if ankle_distances else 0,
        np.min(ankle_distances) if ankle_distances else 0
    ]
    
    return features

def main():
    print("Starting Video Gait Model Training...")
    
    healthy_clips = glob.glob(os.path.join(HEALTHY_DIR, '*.mp4'))
    pd_clips = glob.glob(os.path.join(PD_DIR, '*.mp4'))
    
    print(f"Found {len(healthy_clips)} healthy clips and {len(pd_clips)} PD clips.")
    
    X = []
    y = []
    
    # We will sample a subset if there are too many to keep it within free credits/time
    MAX_SAMPLES = 40
    if len(healthy_clips) > MAX_SAMPLES: healthy_clips = healthy_clips[:MAX_SAMPLES]
    if len(pd_clips) > MAX_SAMPLES: pd_clips = pd_clips[:MAX_SAMPLES]
    
    print(f"Processing up to {MAX_SAMPLES} clips from each class to save time/credits...")
    
    for i, clip in enumerate(healthy_clips):
        if i % 5 == 0: print(f"  Healthy: {i}/{len(healthy_clips)}...")
        features = extract_features_from_video(clip)
        if features is not None and len(features) == 12 and np.isfinite(features).all() and not np.allclose(features, 0.0):
            X.append(features)
            y.append(0)
            
    for i, clip in enumerate(pd_clips):
        if i % 5 == 0: print(f"  PD: {i}/{len(pd_clips)}...")
        features = extract_features_from_video(clip)
        if features is not None and len(features) == 12 and np.isfinite(features).all() and not np.allclose(features, 0.0):
            X.append(features)
            y.append(1)
            
    X = np.array(X)
    y = np.array(y)
    
    print(f"Extracted features for {len(X)} samples.")
    
    if len(X) < 10:
        print("Not enough samples extracted successfully.")
        return
        
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=42)
    rf.fit(X, y)
    
    preds = rf.predict(X)
    print("Training Accuracy:", accuracy_score(y, preds))
    print(classification_report(y, preds))
    
    # Save Model
    model_path = os.path.join(MODELS_DIR, 'video_gait_rf.pkl')
    joblib.dump(rf, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
