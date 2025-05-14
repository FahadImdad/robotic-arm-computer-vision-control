# 🤖 Robotic Arm Control via Computer Vision

An interactive 3D simulation of a robotic arm controlled by real-time hand tracking using a webcam. This project leverages **MediaPipe** and **OpenCV** to detect directional hand gestures and translate them into robotic arm movements. The arm is visualized in 3D using **Matplotlib** within a **Tkinter** interface.

> 🚀 Developed by [Muhammad Fahad Imdad](https://fahadimdad.com)

---

## 📸 Key Features

* 🖐️ **Real-Time Hand Tracking**: Uses MediaPipe to detect hand landmarks with high precision.
* 🧠 **Zone-Based Gesture Recognition**: Divides the screen into four intuitive zones (up, down, left, right) for gesture-based control.
* 🎮 **Animated Arm Movement**: Smooth arm transitions using forward kinematics for natural animation.
* 🖼️ **3D Visualization**: Real-time graphical simulation with labeled axes and camera view adjustments.
* 💾 **Command Logging**: Automatically logs all directional commands with timestamps and joint angles to `movement_log.json`.

---

## 🛠️ Tech Stack

| Tool/Library    | Purpose                              |
| --------------- | ------------------------------------ |
| Python 3.8+     | Main programming language            |
| OpenCV          | Real-time webcam video processing    |
| MediaPipe       | Hand tracking and landmark detection |
| Tkinter         | GUI for embedding Matplotlib plots   |
| Matplotlib (3D) | 3D plotting of robotic arm           |
| NumPy           | Mathematical calculations            |
| JSON            | Logging user commands                |

---

## 🎥 Demo Video

[![Robotic Arm Control via Computer Vision](https://img.youtube.com/vi/UxEhhBWcOTg/0.jpg)](https://youtu.be/UxEhhBWcOTg)

---

## 🚀 Getting Started

### 📂 Folder Structure

```
robotic_arm_computer_vision/
│
├── main.py                  # Main application script
├── movement_log.json        # Logs all movements
├── .venv/                   # (Optional) Python virtual environment
├── README.md                # Project documentation
```

---

### 🧪 Installation

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/robotic_arm_computer_vision.git
cd robotic_arm_computer_vision
```

2. **Set up virtual environment (recommended):**

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install required libraries:**

```bash
pip install -r requirements.txt
```

> If `requirements.txt` doesn't exist, use:

```bash
pip install opencv-python mediapipe matplotlib numpy
```

---

### ▶️ Run the Application

```bash
python main.py
```

> Make sure your webcam is enabled and no other app is using it.

---

## 💡 How It Works

1. **Hand Detection**: The webcam captures your hand, and MediaPipe tracks its landmarks.
2. **Zone Mapping**: The screen is divided into four zones—Up, Down, Left, Right.
3. **Gesture Interpretation**: The position of your hand relative to these zones triggers movement.
4. **Forward Kinematics**: The arm’s 3D joint angles are calculated and animated smoothly.
5. **Visualization**: Tkinter + Matplotlib renders the robotic arm in real-time.
6. **Logging**: All movements are saved in `movement_log.json`.

---

## 🧠 Control Guide

| Gesture/Zone        | Command Triggered    |
| ------------------- | -------------------- |
| Move hand **left**  | Rotate left by 150°  |
| Move hand **right** | Rotate right by 30°  |
| Move hand **up**    | Extend arm upward    |
| Move hand **down**  | Retract arm downward |

---

## 📝 Future Improvements

* Add **inverse kinematics** for better control.
* Integrate **voice commands** for hybrid control.
* Export simulation as a video/gif.
* Add **"Extend"** and **"Retract"** arm length options via separate gestures or buttons.

---

## 👨‍💻 Author

**Muhammad Fahad Imdad**
🌐 [fahadimdad.com](https://fahadimdad.com)
📧 [fahadimdad966@gmail.com](mailto:fahadimdad966@gmail.com)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
