import cv2
import mediapipe as mp
import numpy as np
import os
import joblib

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("[WARN] pyttsx3 not installed. Text-to-speech disabled.")

MODEL_PATH = os.path.join("models", "hand_sign_model.pkl")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def extract_landmarks(hand_landmarks):
    coords = []
    for lm in hand_landmarks.landmark:
        coords.append(lm.x)
        coords.append(lm.y)
    return coords

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found at {MODEL_PATH}. Train it using train_model.py first.")
        return

    bundle = joblib.load(MODEL_PATH)
    clf = bundle["classifier"]
    le = bundle["label_encoder"]

    if TTS_AVAILABLE:
        engine = pyttsx3.init()
    else:
        engine = None

    sentence = ""

    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        current_prediction = ""

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            h, w, _ = frame.shape

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )
                    features = extract_landmarks(hand_landmarks)
                    X = np.array(features).reshape(1, -1)
                    pred_idx = clf.predict(X)[0]
                    pred_label = le.inverse_transform([pred_idx])[0]
                    current_prediction = pred_label
                    cv2.putText(frame, f"Prediction: {pred_label}", (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                current_prediction = ""
                cv2.putText(frame, "No hand detected", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.putText(frame, f"Sentence: {sentence}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.putText(frame, "SPACE:Add  S:Speak  C:Clear  Q:Quit", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            cv2.imshow("Hand Sign Recognition", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            if key == ord('c'):
                sentence = ""

            if key == ord(' '):  # SPACE
                if current_prediction:
                    if sentence:
                        sentence += " "
                    sentence += current_prediction
                    print(f"[INFO] Updated sentence: {sentence}")

            if key == ord('s'):
                if TTS_AVAILABLE and sentence.strip():
                    engine.say(sentence)
                    engine.runAndWait()
                elif not TTS_AVAILABLE:
                    print("[WARN] pyttsx3 not installed. Cannot speak sentence.")
                else:
                    print("[INFO] Sentence is empty.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
