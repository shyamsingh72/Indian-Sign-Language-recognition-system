import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import threading
from tensorflow.keras.models import load_model

# -----------------------------
# BACKGROUND SPEAK FUNCTION
# -----------------------------
def speak(text):
    threading.Thread(target=engine_speak, args=(text,)).start()

def engine_speak(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1.0)
    engine.say(text)
    engine.runAndWait()

# -----------------------------
# LOAD MODEL
# -----------------------------
model = load_model('isl_mediapipe_AZ.keras')

# A–Z labels
labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Mediapipe Initialization
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

cap = cv2.VideoCapture(0)
last_prediction = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # Always initialize with zeros (84 values)
    left_hand = [0.0] * 42
    right_hand = [0.0] * 42

    # If any hand detected
    if results.multi_hand_landmarks and results.multi_handedness:

        for lm_data, handed in zip(results.multi_hand_landmarks, results.multi_handedness):

            hand_label = handed.classification[0].label  # Left or Right
            coords = []

            # Extract 21 landmarks → x,y only → 42 values
            for lm in lm_data.landmark:
                coords.extend([lm.x, lm.y])

            if hand_label == "Left":
                left_hand = coords
                color = (0, 0, 255)
            else:
                right_hand = coords
                color = (0, 255, 0)

            # Draw hand
            mp_draw.draw_landmarks(
                frame,
                lm_data,
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=color, thickness=2, circle_radius=3)
            )

    # -------------------------------------
    # ALWAYS 84 values (missing hand = zeros)
    # -------------------------------------
    all_landmarks = left_hand + right_hand

    # Prediction
    pred = model.predict(np.array([all_landmarks]), verbose=0)
    pred_idx = np.argmax(pred)

    if pred_idx < len(labels):
        predicted_letter = labels[pred_idx]

        # Display on screen
        cv2.putText(frame, f"Prediction: {predicted_letter}",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

        # Speak only when prediction changes
        if predicted_letter != last_prediction:
            speak(predicted_letter)
            last_prediction = predicted_letter

    # Show output window
    cv2.imshow("Two-Hand ISL Detection + Voice", frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
