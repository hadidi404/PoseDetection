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

# Ideal arm wrestling pose parameters
IDEAL_ELBOW_ANGLE = 90
ELBOW_TOLERANCE = 15
IDEAL_WRIST_STRAIGHTNESS = 170  # Close to 180° = straight
WRIST_TOLERANCE = 20

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

# Calculate reference pose metrics from training data
print("Calculating reference pose metrics...")
reference_metrics = {
    'left_elbow_angles': [],
    'right_elbow_angles': [],
    'left_shoulder_heights': [],
    'right_shoulder_heights': [],
    'left_hand_closure': [],  # Distance from thumb to pinky
    'right_hand_closure': [],
    'left_finger_curl': [],  # Average curl of fingers
    'right_finger_curl': []
}

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

# Compute reference metrics from training data
for idx, row in df.iterrows():
    features = row.drop('label').values
    # Left hand: 0-62 (21 landmarks * 3), Left arm: 63-71 (3 landmarks * 3)
    # Right hand: 72-134 (21 landmarks * 3), Right arm: 135-143 (3 landmarks * 3)
    
    if len(features) == 144:
        # Left hand landmarks (21 points * 3 coords = 63 values)
        left_hand = features[0:63].reshape(21, 3)
        # Left arm: shoulder(63-65), elbow(66-68), wrist(69-71)
        left_shoulder = features[63:66]
        left_elbow = features[66:69]
        left_wrist = features[69:72]
        
        # Right hand landmarks
        right_hand = features[72:135].reshape(21, 3)
        # Right arm: shoulder(135-137), elbow(138-140), wrist(141-143)
        right_shoulder = features[135:138]
        right_elbow = features[138:141]
        right_wrist = features[141:144]
        
        # Calculate arm angles
        left_angle = calculate_angle(left_shoulder[:2], left_elbow[:2], left_wrist[:2])
        right_angle = calculate_angle(right_shoulder[:2], right_elbow[:2], right_wrist[:2])
        
        reference_metrics['left_elbow_angles'].append(left_angle)
        reference_metrics['right_elbow_angles'].append(right_angle)
        reference_metrics['left_shoulder_heights'].append(left_shoulder[1])
        reference_metrics['right_shoulder_heights'].append(right_shoulder[1])
        
        # Calculate hand metrics
        # Hand landmarks: 0=wrist, 4=thumb_tip, 8=index_tip, 12=middle_tip, 16=ring_tip, 20=pinky_tip
        
        # Left hand: closure (thumb to pinky distance)
        left_thumb_tip = left_hand[4][:2]
        left_pinky_tip = left_hand[20][:2]
        left_closure = calculate_distance(left_thumb_tip, left_pinky_tip)
        reference_metrics['left_hand_closure'].append(left_closure)
        
        # Left hand: finger curl (average distance from fingertips to wrist)
        left_wrist_pos = left_hand[0][:2]
        left_finger_distances = []
        for tip_idx in [4, 8, 12, 16, 20]:  # All fingertips
            left_finger_distances.append(calculate_distance(left_hand[tip_idx][:2], left_wrist_pos))
        left_avg_curl = np.mean(left_finger_distances)
        reference_metrics['left_finger_curl'].append(left_avg_curl)
        
        # Right hand metrics
        right_thumb_tip = right_hand[4][:2]
        right_pinky_tip = right_hand[20][:2]
        right_closure = calculate_distance(right_thumb_tip, right_pinky_tip)
        reference_metrics['right_hand_closure'].append(right_closure)
        
        right_wrist_pos = right_hand[0][:2]
        right_finger_distances = []
        for tip_idx in [4, 8, 12, 16, 20]:
            right_finger_distances.append(calculate_distance(right_hand[tip_idx][:2], right_wrist_pos))
        right_avg_curl = np.mean(right_finger_distances)
        reference_metrics['right_finger_curl'].append(right_avg_curl)

# Calculate ideal values and acceptable ranges from professional data
ideal_left_elbow = np.median(reference_metrics['left_elbow_angles']) if reference_metrics['left_elbow_angles'] else IDEAL_ELBOW_ANGLE
ideal_right_elbow = np.median(reference_metrics['right_elbow_angles']) if reference_metrics['right_elbow_angles'] else IDEAL_ELBOW_ANGLE
ideal_left_shoulder_height = np.median(reference_metrics['left_shoulder_heights']) if reference_metrics['left_shoulder_heights'] else 0
ideal_right_shoulder_height = np.median(reference_metrics['right_shoulder_heights']) if reference_metrics['right_shoulder_heights'] else 0

# Hand metrics
ideal_left_hand_closure = np.median(reference_metrics['left_hand_closure']) if reference_metrics['left_hand_closure'] else 0
ideal_right_hand_closure = np.median(reference_metrics['right_hand_closure']) if reference_metrics['right_hand_closure'] else 0
ideal_left_finger_curl = np.median(reference_metrics['left_finger_curl']) if reference_metrics['left_finger_curl'] else 0
ideal_right_finger_curl = np.median(reference_metrics['right_finger_curl']) if reference_metrics['right_finger_curl'] else 0

# Calculate acceptable ranges from professional variation (data-driven thresholds)
left_elbow_std = np.std(reference_metrics['left_elbow_angles']) if reference_metrics['left_elbow_angles'] else ELBOW_TOLERANCE
right_elbow_std = np.std(reference_metrics['right_elbow_angles']) if reference_metrics['right_elbow_angles'] else ELBOW_TOLERANCE
elbow_tolerance_left = max(10, left_elbow_std * 1.5)  # 1.5 std deviation as tolerance
elbow_tolerance_right = max(10, right_elbow_std * 1.5)

# Hand tolerances
left_closure_std = np.std(reference_metrics['left_hand_closure']) if reference_metrics['left_hand_closure'] else 50
right_closure_std = np.std(reference_metrics['right_hand_closure']) if reference_metrics['right_hand_closure'] else 50
hand_closure_tolerance_left = max(30, left_closure_std * 1.5)
hand_closure_tolerance_right = max(30, right_closure_std * 1.5)

left_curl_std = np.std(reference_metrics['left_finger_curl']) if reference_metrics['left_finger_curl'] else 30
right_curl_std = np.std(reference_metrics['right_finger_curl']) if reference_metrics['right_finger_curl'] else 30
finger_curl_tolerance_left = max(20, left_curl_std * 1.5)
finger_curl_tolerance_right = max(20, right_curl_std * 1.5)

print(f"\n{'='*60}")
print(f"Reference metrics from {len(reference_metrics['left_elbow_angles'])} professional samples:")
print(f"{'='*60}")
print(f"ARM ANGLES:")
print(f"  Left elbow angle:  {ideal_left_elbow:.1f}° ± {elbow_tolerance_left:.1f}°")
print(f"  Right elbow angle: {ideal_right_elbow:.1f}° ± {elbow_tolerance_right:.1f}°")
print(f"\nHAND METRICS:")
print(f"  Left hand closure:  {ideal_left_hand_closure:.1f}px ± {hand_closure_tolerance_left:.1f}px")
print(f"  Right hand closure: {ideal_right_hand_closure:.1f}px ± {hand_closure_tolerance_right:.1f}px")
print(f"  Left finger curl:   {ideal_left_finger_curl:.1f}px ± {finger_curl_tolerance_left:.1f}px")
print(f"  Right finger curl:  {ideal_right_finger_curl:.1f}px ± {finger_curl_tolerance_right:.1f}px")
print(f"\nAcceptable ranges (learned from professionals):")
print(f"  ✅ Left elbow:  {ideal_left_elbow - elbow_tolerance_left:.1f}° - {ideal_left_elbow + elbow_tolerance_left:.1f}°")
print(f"  ✅ Right elbow: {ideal_right_elbow - elbow_tolerance_right:.1f}° - {ideal_right_elbow + elbow_tolerance_right:.1f}°")
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

def analyze_pose(results, img_width, img_height):
    """Analyze pose and provide specific feedback."""
    feedback = []
    scores = {}
    joint_colors = {}
    
    if not results.pose_landmarks:
        return feedback, scores, joint_colors
    
    lm = results.pose_landmarks.landmark
    
    # Analyze both arms
    arms = {
        'Left': {'shoulder': 11, 'elbow': 13, 'wrist': 15, 'ideal_angle': ideal_left_elbow},
        'Right': {'shoulder': 12, 'elbow': 14, 'wrist': 16, 'ideal_angle': ideal_right_elbow}
    }
    
    for side, indices in arms.items():
        # Check visibility
        if all(lm[i].visibility > VISIBILITY_THRESHOLD for i in [indices['shoulder'], indices['elbow'], indices['wrist']]):
            # Get coordinates
            shoulder = [lm[indices['shoulder']].x * img_width, lm[indices['shoulder']].y * img_height]
            elbow = [lm[indices['elbow']].x * img_width, lm[indices['elbow']].y * img_height]
            wrist = [lm[indices['wrist']].x * img_width, lm[indices['wrist']].y * img_height]
            
            # Calculate elbow angle
            elbow_angle = calculate_angle(shoulder, elbow, wrist)
            ideal_angle = indices['ideal_angle']
            tolerance = elbow_tolerance_left if side == 'Left' else elbow_tolerance_right
            angle_diff = abs(elbow_angle - ideal_angle)
            
            # Score the elbow (0-100) based on data-driven tolerance
            elbow_score = max(0, 100 - (angle_diff / tolerance) * 100)
            scores[f'{side.lower()}_elbow'] = elbow_score
            
            # Determine color and feedback based on data-driven tolerance
            if angle_diff <= tolerance:
                joint_colors[f'{side.lower()}_elbow'] = (0, 255, 0)  # Green
            elif angle_diff <= tolerance * 2:
                joint_colors[f'{side.lower()}_elbow'] = (0, 255, 255)  # Yellow
                if elbow_angle < ideal_angle:
                    feedback.append(f"{side} elbow: Straighten more ({elbow_angle:.0f}° → {ideal_angle:.0f}°)")
                else:
                    feedback.append(f"{side} elbow: Bend more ({elbow_angle:.0f}° → {ideal_angle:.0f}°)")
            else:
                joint_colors[f'{side.lower()}_elbow'] = (0, 0, 255)  # Red
                if elbow_angle < ideal_angle:
                    feedback.append(f"❌ {side} elbow: Too bent! Straighten ({elbow_angle:.0f}° → {ideal_angle:.0f}°)")
                else:
                    feedback.append(f"❌ {side} elbow: Too straight! Bend ({elbow_angle:.0f}° → {ideal_angle:.0f}°)")
            
            # Check wrist alignment (should be in line with forearm)
            forearm_angle = np.arctan2(wrist[1] - elbow[1], wrist[0] - elbow[0]) * 180 / np.pi
            
            # Check shoulder height alignment
            shoulder_height_diff = abs(lm[11].y - lm[12].y) * img_height
            if shoulder_height_diff > 50:  # More than 50 pixels difference
                feedback.append(f"⚠️ Keep shoulders level (diff: {shoulder_height_diff:.0f}px)")
                scores['shoulder_alignment'] = max(0, 100 - shoulder_height_diff)
            else:
                scores['shoulder_alignment'] = 100
    
    # Analyze hand positions and grip
    hands_config = {
        'Left': {
            'landmarks': results.left_hand_landmarks,
            'ideal_closure': ideal_left_hand_closure,
            'ideal_curl': ideal_left_finger_curl,
            'closure_tolerance': hand_closure_tolerance_left,
            'curl_tolerance': finger_curl_tolerance_left
        },
        'Right': {
            'landmarks': results.right_hand_landmarks,
            'ideal_closure': ideal_right_hand_closure,
            'ideal_curl': ideal_right_finger_curl,
            'closure_tolerance': hand_closure_tolerance_right,
            'curl_tolerance': finger_curl_tolerance_right
        }
    }
    
    for side, config in hands_config.items():
        if config['landmarks']:
            hand_lm = config['landmarks'].landmark
            
            # Get key hand points in pixel coordinates
            thumb_tip = [hand_lm[4].x * img_width, hand_lm[4].y * img_height]
            index_tip = [hand_lm[8].x * img_width, hand_lm[8].y * img_height]
            middle_tip = [hand_lm[12].x * img_width, hand_lm[12].y * img_height]
            ring_tip = [hand_lm[16].x * img_width, hand_lm[16].y * img_height]
            pinky_tip = [hand_lm[20].x * img_width, hand_lm[20].y * img_height]
            wrist_pos = [hand_lm[0].x * img_width, hand_lm[0].y * img_height]
            
            # Calculate hand closure (thumb to pinky distance)
            hand_closure = calculate_distance(thumb_tip, pinky_tip)
            closure_diff = abs(hand_closure - config['ideal_closure'])
            closure_score = max(0, 100 - (closure_diff / config['closure_tolerance']) * 100)
            scores[f'{side.lower()}_hand_grip'] = closure_score
            
            # Calculate finger curl (average fingertip to wrist distance)
            finger_distances = [
                calculate_distance(thumb_tip, wrist_pos),
                calculate_distance(index_tip, wrist_pos),
                calculate_distance(middle_tip, wrist_pos),
                calculate_distance(ring_tip, wrist_pos),
                calculate_distance(pinky_tip, wrist_pos)
            ]
            avg_curl = np.mean(finger_distances)
            curl_diff = abs(avg_curl - config['ideal_curl'])
            curl_score = max(0, 100 - (curl_diff / config['curl_tolerance']) * 100)
            
            # Combine hand metrics
            hand_score = (closure_score + curl_score) / 2
            scores[f'{side.lower()}_hand'] = hand_score
            
            # Provide feedback
            if closure_diff > config['closure_tolerance']:
                if hand_closure > config['ideal_closure']:
                    feedback.append(f"⚠️ {side} hand: Grip tighter (fingers too spread)")
                else:
                    feedback.append(f"⚠️ {side} hand: Relax grip slightly")
            
            if curl_diff > config['curl_tolerance']:
                if avg_curl < config['ideal_curl']:
                    feedback.append(f"⚠️ {side} hand: Extend fingers more")
                else:
                    feedback.append(f"⚠️ {side} hand: Curl fingers more")
    
    return feedback, scores, joint_colors

def draw_joint_circles(img, results, joint_colors):
    """Draw colored circles on joints based on correctness."""
    if not results.pose_landmarks:
        return
    
    lm = results.pose_landmarks.landmark
    h, w, _ = img.shape
    
    joints = {
        'left_elbow': 13,
        'right_elbow': 14,
        'left_wrist': 15,
        'right_wrist': 16,
        'left_shoulder': 11,
        'right_shoulder': 12
    }
    
    for joint_name, idx in joints.items():
        if lm[idx].visibility > VISIBILITY_THRESHOLD:
            x = int(lm[idx].x * w)
            y = int(lm[idx].y * h)
            
            # Get color for this joint (default to blue if not analyzed)
            color = joint_colors.get(joint_name, (255, 0, 0))
            
            # Draw circle
            cv2.circle(img, (x, y), 12, color, -1)
            cv2.circle(img, (x, y), 14, (255, 255, 255), 2)  # White border

def draw_legend(img):
    """Draw color legend for joint indicators."""
    h, w, _ = img.shape
    legend_x = w - 200
    legend_y = 30
    
    # Background for legend
    cv2.rectangle(img, (legend_x - 10, legend_y - 10), (w - 10, legend_y + 90), (0, 0, 0), -1)
    cv2.rectangle(img, (legend_x - 10, legend_y - 10), (w - 10, legend_y + 90), (255, 255, 255), 2)
    
    # Title
    cv2.putText(img, "Joint Status:", (legend_x, legend_y + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Green - Good
    cv2.circle(img, (legend_x + 10, legend_y + 35), 6, (0, 255, 0), -1)
    cv2.putText(img, "Correct", (legend_x + 25, legend_y + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    
    # Yellow - Warning
    cv2.circle(img, (legend_x + 10, legend_y + 55), 6, (0, 255, 255), -1)
    cv2.putText(img, "Adjust", (legend_x + 25, legend_y + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    
    # Red - Wrong
    cv2.circle(img, (legend_x + 10, legend_y + 75), 6, (0, 0, 255), -1)
    cv2.putText(img, "Fix", (legend_x + 25, legend_y + 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)


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
        
        h, w, _ = img.shape

        data_row = extract_data_row(results)
        
        # Analyze pose for specific feedback
        feedback, scores, joint_colors = analyze_pose(results, w, h)

        if data_row is None:
            cv2.putText(img, "Incomplete pose ❌", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        else:
            # ML Model scoring
            features = data_row.reshape(1, -1)
            scaled = scaler.transform(features)
            ml_score = model.score_samples(scaled)[0]
            
            # Calculate overall score (combine ML and rule-based)
            if scores:
                rule_score = np.mean(list(scores.values()))
                overall_score = (rule_score * 0.6) + (min(100, max(0, (ml_score + 5) * 10)) * 0.4)
            else:
                overall_score = min(100, max(0, (ml_score + 5) * 10))
            
            # Display overall score
            score_color = (0, 255, 0) if overall_score >= 80 else (0, 255, 255) if overall_score >= 60 else (0, 0, 255)
            cv2.putText(img, f"Overall: {overall_score:.0f}%", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, score_color, 2)
            
            # Display component scores
            y_offset = 80
            if scores:
                for component, score in scores.items():
                    comp_color = (0, 255, 0) if score >= 80 else (0, 255, 255) if score >= 60 else (0, 0, 255)
                    display_name = component.replace('_', ' ').title()
                    cv2.putText(img, f"{display_name}: {score:.0f}%", (30, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, comp_color, 2)
                    y_offset += 30
            
            # Display feedback messages
            feedback_y = h - 120
            for i, msg in enumerate(feedback[:3]):  # Show top 3 feedback items
                cv2.putText(img, msg, (30, feedback_y + i * 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            
            # If no issues, show success message
            if overall_score >= 85 and not feedback:
                cv2.putText(img, "Perfect Form! ✅", (30, h - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        # Draw colored joint circles
        draw_joint_circles(img, results, joint_colors)
        
        # Draw legend
        draw_legend(img)
        
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

        cv2.imshow("Arm Wrestling Form Coach", img)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
