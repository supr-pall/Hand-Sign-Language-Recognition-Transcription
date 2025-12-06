# Hand Sign Language Recognition & Transcription 👋

This project recognizes simple hand signs from a webcam using **MediaPipe + Machine Learning** and converts them into a **text transcription** (and optionally speech).

## 🔧 Features
- Real-time hand detection using **MediaPipe**
- Classifies hand signs using a **Random Forest** model
- Shows **live prediction** on webcam feed
- Press **SPACE** to add the current predicted sign to a sentence
- Press **S** to speak the sentence (text-to-speech, optional)
- Press **C** to clear the sentence
- Press **Q** to quit

---

## 🧰 Tech Stack
- Python
- OpenCV
- MediaPipe
- Scikit-Learn
- Pandas, NumPy
- pyttsx3 (optional Text-to-Speech)

---

## 📦 Installation

```bash
git clone https://github.com/your-username/hand-sign-recognition.git
cd hand-sign-recognition

pip install -r requirements.txt
