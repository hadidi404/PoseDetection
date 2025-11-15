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

# ---------------- TRAIN MODEL ----------------
if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(f"{CSV_FILE} not found. Record data first.")

print(f"Loading data from {CSV_FILE} ...")
df = pd.read_csv(CSV_FILE)

# Extract only numeric features (ignore label)
X = df.drop(columns=['label']).values

print(f"Training data shape: {X.shape}")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train unsupervised model on "correct" poses
model = OneClassSVM(kernel='rbf', gamma='auto', nu=0.05)
model.fit(X_scaled)

# Compute adaptive threshold
scores = model.score_samples(X_scaled)
auto_threshold = scores.mean()  # instead of using min or median
threshold = auto_threshold - 3  # lower by 2 points to allow more flexibility

print(f"Training complete ✅")
print(f"Score range: [{scores.min():.3f}, {scores.max():.3f}]")
print(f"Auto threshold: {threshold:.3f}")

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
        if not ret:
            continue

        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = holistic.process(img_rgb)

        data_row = extract_data_row(results)

        if data_row is None:
            cv2.putText(img, "Incomplete pose ❌", (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        else:
            features = data_row.reshape(1, -1)
            scaled = scaler.transform(features)
            score = model.score_samples(scaled)[0]
            status = "Correct ✅" if score >= threshold else "Incorrect ⚠️"
            color = (0, 255, 0) if score >= threshold else (0, 0, 255)
            cv2.putText(img, f"Score: {score:.3f}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(img, status, (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        # Draw landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        cv2.imshow("Form Checker", img)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
