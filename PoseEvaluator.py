import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
import os

# ---------------- CONFIG ----------------
CSV_FILE = "armwrestling_data.csv"
VISIBILITY_THRESHOLD = 0.9
NUM_POINTS = 48  # 48 landmarks (21 left hand + 3 arm + 21 right hand + 3 arm)

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

# ---------------- TRAIN MODEL ----------------
if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(f"{CSV_FILE} not found. Record data first.")

print(f"Loading data from {CSV_FILE} ...")
df = pd.read_csv(CSV_FILE)

# Extract features from raw data
print("Extracting biomechanical features...")
raw_data = df.drop(columns=['label']).values
feature_list = []
valid_indices = []

for i, row in enumerate(raw_data):
    if len(row) == 144:
        try:
            feats = extract_features(row)
            feature_list.append(feats)
            valid_indices.append(i)
        except Exception as e:
            pass

X_features = np.array(feature_list)
print(f"Training data shape: {X_features.shape} (Samples, Features)")

# Train Scaler and Model on FEATURES
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features)

# Train unsupervised model on "correct" feature vectors
model = OneClassSVM(kernel='rbf', gamma='auto', nu=0.05)
model.fit(X_scaled)

# Compute adaptive threshold
scores = model.score_samples(X_scaled)
auto_threshold = scores.mean() - 3 * scores.std() # Statistical threshold

print(f"Training complete ✅")
print(f"Score range: [{scores.min():.3f}, {scores.max():.3f}]")
print(f"Auto threshold: {auto_threshold:.3f}")

# Print learned baselines
print(f"\n{'='*60}")
print(f"LEARNED PROFESSIONAL BASELINES (Mean ± Std Dev):")
print(f"{'='*60}")
for i, name in enumerate(FEATURE_NAMES):
    mean_val = scaler.mean_[i]
    std_val = np.sqrt(scaler.var_[i])
    print(f"{name:<20}: {mean_val:.1f} ± {std_val:.1f}")
print(f"{'='*60}\n")

# ---------------- REAL-TIME TEST ----------------
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(3, 1080)
cap.set(4, 720)

def extract_data_row(results):
    """Extracts the same data structure as your recording script."""
    data_row = []
    lm = results.pose_landmarks.landmark if results.pose_landmarks else None

    # Left hand + arm
    if results.left_hand_landmarks and results.pose_landmarks:
        left_arm_indices = [11, 13, 15]  # shoulder, elbow, wrist
        if all(lm[i].visibility > VISIBILITY_THRESHOLD for i in left_arm_indices):
            for h in results.left_hand_landmarks.landmark:
                data_row.extend([(1 - h.x) * 1920, (1 - h.y) * 1080, h.z])
            for idx in left_arm_indices:
                data_row.extend([(1 - lm[idx].x) * 1920, (1 - lm[idx].y) * 1080, lm[idx].z])

    # Right hand + arm
    if results.right_hand_landmarks and results.pose_landmarks:
        right_arm_indices = [12, 14, 16]
        if all(lm[i].visibility > VISIBILITY_THRESHOLD for i in right_arm_indices):
            for h in results.right_hand_landmarks.landmark:
                data_row.extend([(1 - h.x) * 1920, (1 - h.y) * 1080, h.z])
            for idx in right_arm_indices:
                data_row.extend([(1 - lm[idx].x) * 1920, (1 - lm[idx].y) * 1080, lm[idx].z])

    return np.array(data_row) if len(data_row) == NUM_POINTS * 3 else None

def analyze_pose_ml(row, scaler, model, feature_names):
    """
    Analyze pose using the ML model and Feature Deviation.
    Returns: feedback list, scores dict, joint colors dict
    """
    feedback = []
    scores = {}
    joint_colors = {}
    
    # 1. Extract Features
    try:
        features = extract_features(row)
    except:
        return feedback, scores, joint_colors, -100
        
    # 2. Scale Features (Z-scores)
    # Reshape to (1, n_features)
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
        feat_name = feature_names[i]
        abs_z = abs(z)
        
        # Score (0-100) based on deviation
        # 0 deviation = 100%, 3 deviation = 0%
        score = max(0, 100 - (abs_z * 33))
        scores[feat_name] = score
        
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
                joint_colors[feature_map[i]] = (0, 0, 255) if status == "CRITICAL" else (0, 255, 255)
                
            elif "Hand Closure" in feat_name:
                # High distance = Open. Low distance = Closed.
                # If Z is positive (High distance) -> Too Open -> "Grip Tighter"
                msg = "Grip Tighter" if direction == "High" else "Relax Grip"
                feedback.append(f"⚠️ {feat_name}: {msg}")
                
            elif "Finger Curl" in feat_name:
                # High distance = Extended. Low distance = Curled.
                msg = "Curl Fingers" if direction == "High" else "Extend Fingers" 
                feedback.append(f"⚠️ {feat_name}: {msg}")
                
            elif "Shoulder" in feat_name:
                feedback.append(f"⚠️ Level Shoulders")
                
    return feedback, scores, joint_colors, ml_score

def draw_joint_circles(img, results, joint_colors):
    if not results.pose_landmarks: return
    lm = results.pose_landmarks.landmark
    h, w, _ = img.shape
    joints = {'left_elbow': 13, 'right_elbow': 14, 'left_wrist': 15, 'right_wrist': 16, 'left_shoulder': 11, 'right_shoulder': 12}
    for name, idx in joints.items():
        if lm[idx].visibility > VISIBILITY_THRESHOLD:
            x, y = int(lm[idx].x * w), int(lm[idx].y * h)
            color = joint_colors.get(name, (0, 255, 0)) # Default Green
            cv2.circle(img, (x, y), 12, color, -1)
            cv2.circle(img, (x, y), 14, (255, 255, 255), 2)

with mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    refine_face_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:
    print("Press ESC to quit.")
    while True:
        ret, img = cap.read()
        if not ret: continue

        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = holistic.process(img_rgb)
        h, w, _ = img.shape

        data_row = extract_data_row(results)
        
        if data_row is None:
            cv2.putText(img, "Incomplete pose ❌", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        else:
            # Analyze using ML Pipeline
            feedback, scores, joint_colors, ml_score = analyze_pose_ml(data_row, scaler, model, FEATURE_NAMES)
            
            # Overall Score (normalized ML score)
            # ML score is usually negative (log likelihood). 
            # We map it: threshold -> 50%, max -> 100%
            # This is a heuristic mapping
            score_norm = max(0, min(100, 50 + (ml_score - auto_threshold) * 10))
            
            color = (0, 255, 0) if score_norm > 80 else (0, 255, 255) if score_norm > 50 else (0, 0, 255)
            cv2.putText(img, f"ML Score: {score_norm:.0f}%", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Display Individual Scores
            y = 80
            for feat_name, score in scores.items():
                s_color = (0, 255, 0) if score > 80 else (0, 255, 255) if score > 50 else (0, 0, 255)
                cv2.putText(img, f"{feat_name}: {score:.0f}%", (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, s_color, 1)
                y += 25

            # Feedback
            y += 10
            for msg in feedback[:4]:
                cv2.putText(img, msg, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                y += 30
                
            # Draw Joints
            draw_joint_circles(img, results, joint_colors)

        # Draw landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                     landmark_drawing_spec=mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                     connection_drawing_spec=mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=2))
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                     landmark_drawing_spec=mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=1, circle_radius=1),
                                     connection_drawing_spec=mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2))
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                     landmark_drawing_spec=mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=1, circle_radius=1),
                                     connection_drawing_spec=mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2))
            
        cv2.imshow("Arm Wrestling Form Coach (ML Powered)", img)
        if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()
