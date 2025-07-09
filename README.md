🧠 Real-Time Live Face Morphing using OpenCV + MediaPipe
This project performs live face morphing from a target image onto your face using your webcam, blending it smoothly in real-time with cv2.seamlessClone(), powered by MediaPipe for facial landmark detection and OpenCV for processing.

https://github.com/maruthivelaga


🚀 Features
Real-time face detection using MediaPipe FaceMesh

Delaunay triangulation for stable morphing

Histogram matching for color consistency

Seamless cloning for natural blending

Adjustable blending with a trackbar

Works on any webcam-enabled system

🧰 Technologies Used
Python 3.8+

OpenCV

MediaPipe

NumPy

Scikit-Image (match_histograms)

📦 Installation
bash
Copy
Edit
git clone https://github.com/maruthivelaga/Face_morph
cd live-face-morph
python -m venv .venv
.venv\Scripts\activate  # On Windows
pip install -r requirements.txt
📂 Project Structure
bash
Copy
Edit
live-face-morph/
│
├── main.py              # Main app logic
├── target.jpg           # Target face to morph (place your image here)
├── README.md
└── requirements.txt
📸 Add a Target Face
Replace the included target.jpg with the face you'd like to morph onto your webcam in real-time.

📏 Recommended: Use a frontal, clear, 640x480 face image with good lighting.

▶️ Usage
bash
Copy
Edit
python main.py
Make sure your webcam is connected

Face the camera, and you’ll see the morph effect live

Press q to exit

🧪 Requirements
Python 3.8+

Webcam

Windows / macOS / Linux

⚠️ Known Issues
Face must be clearly visible to avoid landmark detection errors.

May fail if webcam lighting is too dark or if the face is tilted heavily.

Face may not move perfectly with your head — tracking improvement WIP.

🛠️ To Do
✅ Make morph move with your head position

🔄 Add face registration GUI

🎨 Improve blending at face boundaries

🧬 Add multiple face morph presets

📱 Build a mobile-compatible version (using MediaPipe + Flutter?)

📝 License
MIT License – use it freely with attribution.

🤝 Contributing
Pull requests, issues, and forks are welcome!
Star ⭐ the repo if you like it – it helps more people discover it.

👤 Author
Maruthi Velaga
GitHub • LinkedIn

