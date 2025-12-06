import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd

# ---------- Settings ----------
LABELS = ["A", "B", "C", "Hello", "Yes", "No"]  # you can edit this
DATA_DIR = "data"
CSV_PATH = os.path.join(DATA_DIR, "hand_signs.csv")

os.makedirs(DATA_DIR, exist_ok=True)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def extract_landmarks(hand_landmarks):
    coords = []
    for lm in hand_landmarks.landmark:
        coords.append(lm.x)
        coords.append(lm.y)
    return coords  # length = 42 (21 points * 2)

def main():
    cap = cv2.VideoCapture(0)
    samples = []

    current_label_index = 0
    current_label = LABELS[current_label_index]

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

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

            cv2.putText(frame, f"Label: {current_label}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "C: Capture | N: Next Label | Q: Quit", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("Collect Hand Sign Data", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            if key == ord('n'):
                current_label_index = (current_label_index + 1) % len(LABELS)
                current_label = LABELS[current_label_index]

            if key == ord('c'):
                # Capture sample
                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    features = extract_landmarks(hand_landmarks)
                    row = [current_label] + features
                    samples.append(row)
                    print(f"[INFO] Captured sample for label: {current_label}")
                else:
                    print("[WARN] No hand detected, try again.")

    cap.release()
    cv2.destroyAllWindows()

    if samples:
        num_features = len(samples[0]) - 1
        columns = ["label"] + [f"x{i//2}" if i % 2 == 0 else f"y{i//2}"
                               for i in range(num_features)]
        df = pd.DataFrame(samples, columns=columns)
        if os.path.exists(CSV_PATH):
            # append
            df_existing = pd.read_csv(CSV_PATH)
            df = pd.concat([df_existing, df], ignore_index=True)
        df.to_csv(CSV_PATH, index=False)
        print(f"[INFO] Saved {len(samples)} samples to {CSV_PATH}")
    else:
        print("[INFO] No samples captured.")

if __name__ == "__main__":
    main()
