import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
import matplotlib.backends.backend_tkagg as tkagg
from mpl_toolkits.mplot3d import Axes3D
import threading
import time
import json

# Arm parameters
L1, L2 = 5, 3

def forward_kinematics(theta1, theta2, elevation=np.radians(0)):
    x0, y0, z0 = 0, 0, 0
    x1 = L1 * np.cos(theta1) * np.cos(elevation)
    y1 = L1 * np.sin(theta1) * np.cos(elevation)
    z1 = L1 * np.sin(elevation)
    x2 = x1 + L2 * np.cos(theta1 + theta2) * np.cos(elevation)
    y2 = y1 + L2 * np.sin(theta1 + theta2) * np.cos(elevation)
    z2 = z1 + L2 * np.sin(elevation)
    return [(x0, y0, z0), (x1, y1, z1), (x2, y2, z2)]

def command_to_angles(direction):
    if direction == "right":
        return np.radians(30), np.radians(30), np.radians(0)
    elif direction == "left":
        return np.radians(150), np.radians(-60), np.radians(0)
    elif direction == "up":
        return np.radians(90), np.radians(0), np.radians(35)
    elif direction == "down":
        return np.radians(90), np.radians(0), np.radians(-30)
    else:
        raise ValueError("Unknown direction")

class RoboticArmApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Zone-Based Robotic Arm")
        self.root.geometry("1300x900")

        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.canvas = tkagg.FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.status = tk.Label(root, text="🖐️ Move hand into zones", font=("Arial", 24), fg="green")
        self.status.pack(pady=10)

        self.current_joints = forward_kinematics(0, 0, 0)
        self.draw_arm(self.current_joints)

        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not self.capture.isOpened():
            raise RuntimeError("❌ Could not open webcam.")

        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.command_log = []
        self.last_action_time = time.time()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        threading.Thread(target=self.cv_loop, daemon=True).start()

    def draw_arm(self, joints):
        self.ax.clear()

        # Plot arm
        x_vals, y_vals, z_vals = zip(*joints)
        self.ax.plot(x_vals, y_vals, z_vals, 'o-', linewidth=8, markersize=12, color='gray', label='Arm')
        self.ax.scatter(*joints[0], color='red', s=200, label='Base')  # Bigger base dot

        # Ground plane
        x_plane, y_plane = np.meshgrid(np.linspace(-10, 10, 2), np.linspace(-10, 10, 2))
        z_plane = np.zeros_like(x_plane)
        self.ax.plot_surface(x_plane, y_plane, z_plane, color='lightgray', alpha=0.3, zorder=0)

        # Coordinate axes (thicker, labeled)
        self.ax.quiver(0, 0, 0, 5, 0, 0, color='blue', linewidth=3, label='X (Right)')
        self.ax.quiver(0, 0, 0, 0, 5, 0, color='green', linewidth=3, label='Y (Forward)')
        self.ax.quiver(0, 0, 0, 0, 0, 5, color='purple', linewidth=3, label='Z (Up)')

        # Axis labels with giant font
        self.ax.set_xlabel("X Axis (Left-Right)", fontsize=24, labelpad=20, fontweight='bold')
        self.ax.set_ylabel("Y Axis (Forward-Back)", fontsize=24, labelpad=20, fontweight='bold')
        self.ax.set_zlabel("Z Axis (Up-Down)", fontsize=24, labelpad=20, fontweight='bold')

        # Axis ticks
        self.ax.tick_params(labelsize=20)

        # Legend
        self.ax.legend(loc='upper right', fontsize=20)

        # Limits and view
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        self.ax.set_zlim(-5, 10)
        self.ax.view_init(elev=30, azim=45)

        self.canvas.draw()


    def animate_arm(self, target_joints, steps=10, delay=0.01):
        for i in range(1, steps + 1):
            intermediate = [
                (
                    c[0] + (t[0] - c[0]) * i / steps,
                    c[1] + (t[1] - c[1]) * i / steps,
                    c[2] + (t[2] - c[2]) * i / steps
                ) for c, t in zip(self.current_joints, target_joints)
            ]
            self.draw_arm(intermediate)
            self.root.update()
            time.sleep(delay)
        self.current_joints = target_joints

    def cv_loop(self):
        while True:
            ret, frame = self.capture.read()
            frame = cv2.flip(frame, 1)  # 👈 Flip the image horizontally

            if not ret:
                continue

            h, w, _ = frame.shape
            zones = {
                'up': (int(w * 0.2), 0, int(w * 0.8), int(h * 0.3)),
                'down': (int(w * 0.2), int(h * 0.7), int(w * 0.8), h),
                'left': (0, int(h * 0.2), int(w * 0.3), int(h * 0.8)),
                'right': (int(w * 0.7), int(h * 0.2), w, int(h * 0.8)),
            }

            direction = None
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                lm8 = hand.landmark[8]  # Index fingertip
                lm0 = hand.landmark[0]  # Wrist
                cx = int((lm0.x + lm8.x) / 2 * w)
                cy = int((lm0.y + lm8.y) / 2 * h)
                cv2.circle(frame, (cx, cy), 25, (255, 255, 255), -1)

                for d, (x1, y1, x2, y2) in zones.items():
                    if x1 <= cx <= x2 and y1 <= cy <= y2:
                        direction = d
                        break

            for d, (x1, y1, x2, y2) in zones.items():
                color = (0, 255, 0) if d == direction else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, d.upper(), (x1 + 10, y1 + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

            if direction and time.time() - self.last_action_time > 1:
                self.status.config(text=f"✅ Direction: {direction.upper()}")
                try:
                    theta1, theta2, elev = command_to_angles(direction)
                    joints = forward_kinematics(theta1, theta2, elev)
                    self.animate_arm(joints)
                    self.command_log.append({
                        'time': time.time(),
                        'direction': direction,
                        'theta1': float(theta1),
                        'theta2': float(theta2),
                        'elevation': float(elev)
                    })
                except Exception as e:
                    self.status.config(text=f"⚠️ {str(e)}")
                self.last_action_time = time.time()

            cv2.imshow("Webcam Zones", frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break

    def on_close(self):
        with open('movement_log.json', 'w') as f:
            json.dump(self.command_log, f, indent=2)
        self.capture.release()
        cv2.destroyAllWindows()
        self.root.destroy()

if __name__ == "__main__":
    print("🚀 Starting Robotic Arm Using Computer vision")
    root = tk.Tk()
    app = RoboticArmApp(root)
    root.mainloop()
