"""
FastAPI Backend Server for Pose Evaluation
Uses the same ML logic as PoseEvaluator.py
"""
from fastapi import FastAPI, WebSocket, UploadFile, File, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import pickle
import json
import base64
from typing import Dict, List
import os

# ---------------- CONFIG ----------------
MODEL_FILE = "backend_model.pkl"
SCALER_FILE = "backend_scaler.pkl"
CSV_FILE = "armwrestling_data.csv"
VISIBILITY_THRESHOLD = 0.9
NUM_POINTS = 48

# ---------------- LOAD MODEL ----------------
print("Loading trained model...")
with open(MODEL_FILE, 'rb') as f:
    model = pickle.load(f)
with open(SCALER_FILE, 'rb') as f:
    scaler = pickle.load(f)

print(f"Model loaded successfully!")

# Calculate auto_threshold from training data
df = pd.read_csv(CSV_FILE)
print(f"Training samples: {len(df)}")

# ---------------- MEDIAPIPE SETUP ----------------
mp_holistic = mp.solutions.holistic

# ---------------- HELPER FUNCTIONS ----------------
def calculate_angle(a, b, c):
    """Calculate angle at point b given three points a, b, c."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
    
    return angle

def calculate_distance(a, b):
    """Calculate Euclidean distance between two points."""
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

FEATURE_NAMES = [
    'Left Elbow Angle', 'Right Elbow Angle',
    'Left Hand Closure', 'Right Hand Closure',
    'Left Finger Curl', 'Right Finger Curl',
    'Shoulder Level Diff'
]

def extract_features(row):
    """
    Convert 144-point raw landmark data into meaningful biomechanical features.
    Input: row (numpy array of 144 values)
    Output: numpy array of features
    """
    features = []
    
    # Indices based on DataCollection.py structure:
    # Left Hand: 0-62 (21*3)
    # Left Arm: 63-71 (3*3) -> Shoulder(63-65), Elbow(66-68), Wrist(69-71)
    # Right Hand: 72-134 (21*3)
    # Right Arm: 135-143 (3*3) -> Shoulder(135-137), Elbow(138-140), Wrist(141-143)
    
    # Extract Arm Points (x, y only for 2D angles)
    l_sh = row[63:65]
    l_el = row[66:68]
    l_wr = row[69:71]
    
    r_sh = row[135:137]
    r_el = row[138:140]
    r_wr = row[141:143]
    
    # 1. Elbow Angles
    features.append(calculate_angle(l_sh, l_el, l_wr))
    features.append(calculate_angle(r_sh, r_el, r_wr))
    
    # Extract Hand Points
    l_hand = row[0:63].reshape(21, 3)
    r_hand = row[72:135].reshape(21, 3)
    
    # 2. Hand Closure (Thumb tip to Pinky tip distance)
    # Index 4=Thumb Tip, 20=Pinky Tip
    features.append(calculate_distance(l_hand[4][:2], l_hand[20][:2]))
    features.append(calculate_distance(r_hand[4][:2], r_hand[20][:2]))
    
    # 3. Finger Curl (Average distance from fingertips to wrist)
    # Tips: 4, 8, 12, 16, 20. Wrist: 0
    l_wrist_pt = l_hand[0][:2]
    l_tips = [l_hand[i][:2] for i in [4, 8, 12, 16, 20]]
    features.append(np.mean([calculate_distance(p, l_wrist_pt) for p in l_tips]))
    
    r_wrist_pt = r_hand[0][:2]
    r_tips = [r_hand[i][:2] for i in [4, 8, 12, 16, 20]]
    features.append(np.mean([calculate_distance(p, r_wrist_pt) for p in r_tips]))
    
    # 4. Shoulder Level Difference (Vertical distance)
    # y is at index 1
    features.append(abs(l_sh[1] - r_sh[1]))
    
    return np.array(features)

# Calculate auto_threshold
raw_data = df.drop(columns=['label']).values
feature_list = []
for row in raw_data:
    if len(row) == 144:
        try:
            feats = extract_features(row)
            feature_list.append(feats)
        except:
            pass

X_features = np.array(feature_list)
X_scaled = scaler.transform(X_features)
scores = model.score_samples(X_scaled)
auto_threshold = scores.mean() - 3 * scores.std()
print(f"Auto threshold: {auto_threshold:.3f}")

def extract_data_row(results):
    """Extracts the same data structure as PoseEvaluator.py recording script."""
    data_row = []
    lm = results.pose_landmarks.landmark if results.pose_landmarks else None

    # Left hand + arm
    if results.left_hand_landmarks and results.pose_landmarks:
        left_arm_indices = [11, 13, 15]  # shoulder, elbow, wrist
        if all(lm[i].visibility > VISIBILITY_THRESHOLD for i in left_arm_indices):
            for h in results.left_hand_landmarks.landmark:
                data_row.extend([h.x * 1920, h.y * 1080, h.z])
            for idx in left_arm_indices:
                data_row.extend([lm[idx].x * 1920, lm[idx].y * 1080, lm[idx].z])

    # Right hand + arm
    if results.right_hand_landmarks and results.pose_landmarks:
        right_arm_indices = [12, 14, 16]
        if all(lm[i].visibility > VISIBILITY_THRESHOLD for i in right_arm_indices):
            for h in results.right_hand_landmarks.landmark:
                data_row.extend([h.x * 1920, h.y * 1080, h.z])
            for idx in right_arm_indices:
                data_row.extend([lm[idx].x * 1920, lm[idx].y * 1080, lm[idx].z])

    return np.array(data_row) if len(data_row) == NUM_POINTS * 3 else None

def analyze_pose_ml(row, results):
    """
    Analyze pose using the ML model and Feature Deviation (EXACT same logic as PoseEvaluator.py).
    Returns: dict with feedback, scores, joint_colors, ml_score, landmarks
    """
    feedback = []
    scores = {}
    joint_colors = {}
    
    # Extract landmarks for drawing
    landmarks = {}
    if results.pose_landmarks:
        pose_lms = results.pose_landmarks.landmark
        landmarks['pose'] = {
            'left_shoulder': {'x': pose_lms[11].x, 'y': pose_lms[11].y, 'visibility': pose_lms[11].visibility},
            'right_shoulder': {'x': pose_lms[12].x, 'y': pose_lms[12].y, 'visibility': pose_lms[12].visibility},
            'left_elbow': {'x': pose_lms[13].x, 'y': pose_lms[13].y, 'visibility': pose_lms[13].visibility},
            'right_elbow': {'x': pose_lms[14].x, 'y': pose_lms[14].y, 'visibility': pose_lms[14].visibility},
            'left_wrist': {'x': pose_lms[15].x, 'y': pose_lms[15].y, 'visibility': pose_lms[15].visibility},
            'right_wrist': {'x': pose_lms[16].x, 'y': pose_lms[16].y, 'visibility': pose_lms[16].visibility},
        }
    
    if results.left_hand_landmarks:
        landmarks['left_hand'] = [{'x': lm.x, 'y': lm.y} for lm in results.left_hand_landmarks.landmark]
    
    if results.right_hand_landmarks:
        landmarks['right_hand'] = [{'x': lm.x, 'y': lm.y} for lm in results.right_hand_landmarks.landmark]
    
    # 1. Extract Features
    try:
        features = extract_features(row)
    except:
        return {
            'status': 'error',
            'message': 'Feature extraction failed',
            'ml_score': -100,
            'feedback': [],
            'feature_scores': {},
            'joint_colors': {},
            'landmarks': landmarks
        }
        
    # 2. Scale Features (Z-scores)
    features_reshaped = features.reshape(1, -1)
    features_scaled = scaler.transform(features_reshaped)
    
    # 3. Get ML Score
    ml_score = model.score_samples(features_scaled)[0]
    
    # 4. Analyze Deviations (Z-scores)
    # A Z-score > 2 means the value is 2 standard deviations away from the professional mean
    z_scores = features_scaled[0]
    
    # Map features to joints for coloring
    # 0: L Elbow, 1: R Elbow, 2: L Hand, 3: R Hand, 4: L Curl, 5: R Curl, 6: Shoulders
    feature_map = {
        0: 'left_elbow', 1: 'right_elbow',
        2: 'left_hand', 3: 'right_hand',
        4: 'left_hand', 5: 'right_hand',
        6: 'shoulders'
    }
    
    for i, z in enumerate(z_scores):
        feat_name = FEATURE_NAMES[i]
        abs_z = abs(z)
        
        # Score (0-100) based on deviation
        # 0 deviation = 100%, 3 deviation = 0%
        score = max(0, 100 - (abs_z * 33))
        scores[feat_name] = round(score, 2)
        
        # Determine Status
        status = "OK"
        if abs_z > 3.0: status = "CRITICAL"
        elif abs_z > 2.0: status = "WARNING"
        
        # Generate Feedback
        if status != "OK":
            direction = "High" if z > 0 else "Low"
            
            # Custom messages based on feature type
            if "Elbow Angle" in feat_name:
                msg = "Straighten" if direction == "Low" else "Bend"
                feedback.append(f"⚠️ {feat_name}: {msg} ({features[i]:.0f}°)")
                joint_colors[feature_map[i]] = 'red' if status == "CRITICAL" else 'yellow'
                
            elif "Hand Closure" in feat_name:
                msg = "Grip Tighter" if direction == "High" else "Relax Grip"
                feedback.append(f"⚠️ {feat_name}: {msg}")
                joint_colors[feature_map[i]] = 'yellow'
                
            elif "Finger Curl" in feat_name:
                msg = "Curl Fingers" if direction == "High" else "Extend Fingers" 
                feedback.append(f"⚠️ {feat_name}: {msg}")
                
            elif "Shoulder" in feat_name:
                feedback.append(f"⚠️ Level Shoulders")
                joint_colors['shoulders'] = 'yellow'
    
    # Set green for joints without issues
    for joint in ['left_elbow', 'right_elbow', 'left_hand', 'right_hand', 'shoulders']:
        if joint not in joint_colors:
            joint_colors[joint] = 'green'
    
    # Overall Score (normalized ML score)
    # ML score is usually negative (log likelihood). 
    # We map it: threshold -> 50%, max -> 100%
    score_norm = max(0, min(100, 50 + (ml_score - auto_threshold) * 10))
    
    return {
        'status': 'success',
        'ml_score': round(score_norm, 2),
        'feature_scores': scores,
        'feedback': feedback,
        'joint_colors': joint_colors,
        'landmarks': landmarks
    }

# ---------------- FASTAPI APP ----------------
app = FastAPI(title="Pose Evaluation API")

# Enable CORS for mobile apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "Pose Evaluation API is running",
        "endpoints": {
            "health": "GET /",
            "model_info": "GET /api/model-info",
            "evaluate_image": "POST /api/evaluate-image",
            "websocket": "WS /ws/pose-evaluation"
        }
    }

@app.get("/api/model-info")
async def model_info():
    return {
        "model_type": "OneClassSVM",
        "features": FEATURE_NAMES,
        "training_samples": len(df),
        "threshold": float(auto_threshold)
    }

@app.post("/api/evaluate-image")
async def evaluate_image(file: UploadFile = File(...)):
    """Evaluate a single image."""
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return {"status": "error", "message": "Invalid image"}
        
        # Process with MediaPipe Holistic
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        with mp_holistic.Holistic(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.5
        ) as holistic:
            results = holistic.process(rgb)
        
        # Extract landmarks for drawing (even if incomplete)
        landmarks = {}
        if results.pose_landmarks:
            pose_lms = results.pose_landmarks.landmark
            landmarks['pose'] = {
                'left_shoulder': {'x': pose_lms[11].x, 'y': pose_lms[11].y, 'visibility': pose_lms[11].visibility},
                'right_shoulder': {'x': pose_lms[12].x, 'y': pose_lms[12].y, 'visibility': pose_lms[12].visibility},
                'left_elbow': {'x': pose_lms[13].x, 'y': pose_lms[13].y, 'visibility': pose_lms[13].visibility},
                'right_elbow': {'x': pose_lms[14].x, 'y': pose_lms[14].y, 'visibility': pose_lms[14].visibility},
                'left_wrist': {'x': pose_lms[15].x, 'y': pose_lms[15].y, 'visibility': pose_lms[15].visibility},
                'right_wrist': {'x': pose_lms[16].x, 'y': pose_lms[16].y, 'visibility': pose_lms[16].visibility},
            }
        
        if results.left_hand_landmarks:
            landmarks['left_hand'] = [{'x': lm.x, 'y': lm.y} for lm in results.left_hand_landmarks.landmark]
        
        if results.right_hand_landmarks:
            landmarks['right_hand'] = [{'x': lm.x, 'y': lm.y} for lm in results.right_hand_landmarks.landmark]
        
        row = extract_data_row(results)
        if row is None:
            return {
                "status": "incomplete_pose",
                "message": "Incomplete pose detected. Ensure both hands and arms are visible.",
                "landmarks": landmarks
            }
        
        # Analyze using ML
        result = analyze_pose_ml(row, results)
        return result
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.websocket("/ws/pose-evaluation")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time video evaluation."""
    await websocket.accept()
    print("WebSocket client connected")
    
    try:
        with mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as holistic:
            while True:
                # Receive frame
                data = await websocket.receive_text()
                frame_data = json.loads(data)
                
                # Decode base64 image
                img_bytes = base64.b64decode(frame_data['frame'])
                nparr = np.frombuffer(img_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if image is None:
                    await websocket.send_json({
                        "status": "error",
                        "message": "Invalid frame"
                    })
                    continue
                
                # Process with MediaPipe
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb)
                
                # Extract landmarks for drawing (even if incomplete)
                landmarks = {}
                if results.pose_landmarks:
                    pose_lms = results.pose_landmarks.landmark
                    landmarks['pose'] = {
                        'left_shoulder': {'x': pose_lms[11].x, 'y': pose_lms[11].y, 'visibility': pose_lms[11].visibility},
                        'right_shoulder': {'x': pose_lms[12].x, 'y': pose_lms[12].y, 'visibility': pose_lms[12].visibility},
                        'left_elbow': {'x': pose_lms[13].x, 'y': pose_lms[13].y, 'visibility': pose_lms[13].visibility},
                        'right_elbow': {'x': pose_lms[14].x, 'y': pose_lms[14].y, 'visibility': pose_lms[14].visibility},
                        'left_wrist': {'x': pose_lms[15].x, 'y': pose_lms[15].y, 'visibility': pose_lms[15].visibility},
                        'right_wrist': {'x': pose_lms[16].x, 'y': pose_lms[16].y, 'visibility': pose_lms[16].visibility},
                    }
                
                if results.left_hand_landmarks:
                    landmarks['left_hand'] = [{'x': lm.x, 'y': lm.y} for lm in results.left_hand_landmarks.landmark]
                
                if results.right_hand_landmarks:
                    landmarks['right_hand'] = [{'x': lm.x, 'y': lm.y} for lm in results.right_hand_landmarks.landmark]
                
                # Extract and analyze
                row = extract_data_row(results)
                if row is None:
                    await websocket.send_json({
                        "status": "incomplete_pose",
                        "message": "Incomplete pose detected",
                        "landmarks": landmarks
                    })
                    continue
                
                result = analyze_pose_ml(row, results)
                await websocket.send_json(result)
            
    except WebSocketDisconnect:
        print("WebSocket client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")

if __name__ == "__main__":
    print("\n" + "="*50)
    print("Starting Pose Evaluation Backend Server")
    print("="*50)
    print(f"Server will run on: http://0.0.0.0:8000")
    print(f"Local access: http://localhost:8000")
    print(f"Network access: http://192.168.0.237:8000")
    print("="*50 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
