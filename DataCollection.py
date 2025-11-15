import cv2
import mediapipe as mp
import socket
import os 
import time as t
import csv
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

filename = "armwrestling_data.csv"
if not os.path.exists(filename):
    num_points = 48  # 48 landmarks total (hands + arms)
    headers = []

    # Create headers: x1, y1, z1, x2, y2, z2, ..., x48, y48, z48
    for i in range(1, num_points + 1):
        headers.extend([f'x{i}', f'y{i}', f'z{i}'])

    # Add the label column at the end
    headers.append('label')

    # Write to CSV
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)

recording = False

first = True
label = "correct"

# Communication  
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
severAdressPort = ("127.0.0.1", 5052)

cap = cv2.VideoCapture(0)
cap.set(3, 1080)  # Width
cap.set(4, 720) 
with mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    refine_face_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:

    while True:
        data_row = []
        success, img = cap.read()
        if not success:
            continue

        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = holistic.process(img_rgb)

        VISIBILITY_THRESHOLD = 0.9 # Adjust as needed

        # Left hand + arm
        if results.left_hand_landmarks and results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            left_arm_indices = [11, 13, 15]  # shoulder, elbow, wrist
            # Check if all landmarks are visible enough
            if all(lm[i].visibility > VISIBILITY_THRESHOLD for i in left_arm_indices):
                # Record hand points
                for h in results.left_hand_landmarks.landmark:
                    data_row.extend([(1 - h.x) * 1920, (1 - h.y) * 1080, h.z])
                # data_row.extend([" "])
                # Record arm points
                for idx in left_arm_indices:
                    data_row.extend([(1 - lm[idx].x) * 1920, (1 - lm[idx].y) * 1080, lm[idx].z])
                # data_row.extend([" "])
                
        # Right hand + arm
        if results.right_hand_landmarks and results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            right_arm_indices = [12, 14, 16]
            if all(lm[i].visibility > VISIBILITY_THRESHOLD for i in right_arm_indices):
                # Record hand points
                for h in results.right_hand_landmarks.landmark:
                    data_row.extend([(1 - h.x) * 1920, (1 - h.y) * 1080, h.z])
                # data_row.extend([" "])
                # Record arm points
                for idx in right_arm_indices:
                    data_row.extend([(1 - lm[idx].x) * 1920, (1 - lm[idx].y) * 1080, lm[idx].z])
                # data_row.extend([" "]) 
        


        if data_row:  # only send if we actually have data
            message = ",".join(map(str, data_row))
            sock.sendto(message.encode(), severAdressPort)

        
        # Draw landmarks for everything
        mp_drawing.draw_landmarks(
            img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(
            img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(
            img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        
# -------------------- RECORD DATA --------------------
        if recording and len(data_row) == 144:
            if first:
                t.sleep(1)
                first = False
            with open(filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(data_row + [label])
            cv2.putText(img, "Recording...", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.imshow("Holistic Tracking", img)
        # -------------------- KEY CONTROLS --------------------
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('r') or key == ord('R'):
            recording = not recording
            print("Recording:", recording)
cap.release()
cv2.destroyAllWindows()
