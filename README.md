🤟 Indian Sign Language Recognition System (ISL)

A real-time Indian Sign Language (ISL) recognition system that detects hand gestures using a webcam and converts them into text and voice output. This project aims to reduce the communication gap between hearing-impaired individuals and the general public using Computer Vision and Deep Learning.

📌 Project Overview

Indian Sign Language (ISL) is a primary mode of communication for many hearing-impaired individuals. However, most people are not familiar with ISL, creating communication barriers.
This project uses MediaPipe Hand Landmarks and a Deep Learning model to recognize ISL alphabets (A–Z) in real time and provide spoken output for better accessibility.


✨ Features

✅ Real-time ISL alphabet recognition (A–Z)

🤲 Supports single-hand and double-hand gestures

🎥 Live webcam-based detection

🧠 Deep Learning–based classification model

🔊 Text-to-Speech voice output

🎨 Color-coded landmarks

🔴 Left hand

🟢 Right hand

💻 Low-cost system (only webcam required)



🛠️ Technologies Used

Programming Language: Python

Computer Vision: OpenCV

Hand Tracking: MediaPipe

Deep Learning: TensorFlow / Keras

Text-to-Speech: pyttsx3

Model Format: .h5 / .keras

📂 Project Structure

├── collect_data.py # Collect ISL hand landmark data
├── train_model.py           # Train deep learning model
├── real_time_detection.py   # Real-time ISL recognition + voice
├── X.npy                    # Feature dataset (hand landmarks)
├── y.npy                    # Labels dataset
├── isl_mediapipe_AZ.keras   # Trained model
├── README.md



⚙️ How It Works

Data Collection

Hand landmarks are captured using MediaPipe.

Landmarks are stored in X.npy and labels in y.npy.

Model Training

A neural network is trained on the collected landmark data.

The trained model is saved for real-time prediction.

Real-Time Detection

Webcam input is processed.

Hand landmarks are extracted and passed to the model.

The predicted ISL alphabet is displayed and spoken aloud.



▶️ How to Run

1️⃣ Install Dependencies
pip install opencv-python mediapipe tensorflow pyttsx3 numpy

2️⃣ Collect Data
python collect_data.py

3️⃣ Train Model
python train_model.py

4️⃣ Run Real-Time Detection
python real_time_detection.py

Press q to exit.



📊 Dataset Details

X.npy → Hand landmark coordinates (84 values for two hands)

y.npy → Corresponding alphabet labels

Each sample represents one ISL gesture



🎯 Applications

Assistive technology for hearing-impaired individuals

Educational tools for learning ISL

Human–Computer Interaction (HCI)

Accessibility systems

Gesture-based interfaces



🚀 Future Enhancements

Word and sentence-level recognition

Mobile app using TensorFlow Lite

Cloud-based ISL recognition API

Support for dynamic gestures

Integration with smart devices
