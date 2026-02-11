import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
cap = cv2.VideoCapture(0)

# A–Z labels only (26)
labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

data = []
target = []

print("Press 's' to save frame, 'n' to go next label, 'q' to quit")
current_label = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    h, w, _ = frame.shape
    all_landmarks = [0.0] * 84  # 2 hands * 21 landmarks * 2 coords

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):

            label = handedness.classification[0].label  # Left / Right
            coords = []

            for lm in hand_landmarks.landmark:
                coords.extend([lm.x, lm.y])

            if label == "Left":
                all_landmarks[:42] = coords
                color = (255, 0, 0)
            else:
                all_landmarks[42:] = coords
                color = (0, 255, 0)

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=3)
            )

        cv2.putText(frame, f"Collecting: {labels[current_label]}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            data.append(all_landmarks)
            target.append(current_label)
            print(f"Saved sample for {labels[current_label]}")
        elif key == ord('n'):
            current_label += 1
            if current_label >= len(labels):
                break
        elif key == ord('q'):
            break

    else:
        cv2.putText(frame, "Show both hands clearly!", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Two-Hand Data Collection", frame)

cap.release()
cv2.destroyAllWindows()

np.save("X.npy", np.array(data))
np.save("y.npy", np.array(target))
print("Dataset saved (A–Z only)")
