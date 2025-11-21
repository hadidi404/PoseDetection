"""
Export model for backend server use
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
import pickle
import os

# Same logic as PoseEvaluator.py
CSV_FILE = "armwrestling_data.csv"

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

def extract_features(row):
    """
    Convert 144-point raw landmark data into meaningful biomechanical features.
    """
    features = []
    
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
    features.append(calculate_distance(l_hand[4][:2], l_hand[20][:2]))
    features.append(calculate_distance(r_hand[4][:2], r_hand[20][:2]))
    
    # 3. Finger Curl (Average distance from fingertips to wrist)
    l_wrist_pt = l_hand[0][:2]
    l_tips = [l_hand[i][:2] for i in [4, 8, 12, 16, 20]]
    features.append(np.mean([calculate_distance(p, l_wrist_pt) for p in l_tips]))
    
    r_wrist_pt = r_hand[0][:2]
    r_tips = [r_hand[i][:2] for i in [4, 8, 12, 16, 20]]
    features.append(np.mean([calculate_distance(p, r_wrist_pt) for p in r_tips]))
    
    # 4. Shoulder Level Difference (Vertical distance)
    features.append(abs(l_sh[1] - r_sh[1]))
    
    return np.array(features)

print(f"Loading data from {CSV_FILE} ...")
df = pd.read_csv(CSV_FILE)

# Extract features from raw data
print("Extracting biomechanical features...")
raw_data = df.drop(columns=['label']).values
feature_list = []

for i, row in enumerate(raw_data):
    if len(row) == 144:
        try:
            feats = extract_features(row)
            feature_list.append(feats)
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

print(f"Training complete ✅")

# Save for backend
print("\nSaving model for backend...")
with open('backend_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('backend_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("✅ Model saved as backend_model.pkl and backend_scaler.pkl")
print("Backend server can now use these files!")
