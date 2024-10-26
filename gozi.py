import sys
import cv2
import mediapipe as mp
import random
import numpy as np
import time
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer

class GameWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hand Shooter Game")
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet("background-color: #2C3E50;")

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        self.title_label = QLabel("Hand Shooter Game")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setFont(QFont("Arial", 36, QFont.Bold))
        self.title_label.setStyleSheet("color: #ECF0F1;")
        self.layout.addWidget(self.title_label)

        self.start_button = QPushButton("Start Game")
        self.start_button.setFont(QFont("Arial", 18))
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #27AE60;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #2ECC71;
            }
        """)
        self.start_button.clicked.connect(self.start_game)
        self.layout.addWidget(self.start_button, alignment=Qt.AlignCenter)

        self.quit_button = QPushButton("Quit")
        self.quit_button.setFont(QFont("Arial", 18))
        self.quit_button.setStyleSheet("""
            QPushButton {
                background-color: #C0392B;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #E74C3C;
            }
        """)
        self.quit_button.clicked.connect(self.close)
        self.layout.addWidget(self.quit_button, alignment=Qt.AlignCenter)

        self.game_widget = None

    def start_game(self):
        self.game_widget = GameWidget()
        self.setCentralWidget(self.game_widget)
        self.game_widget.start_game()

class GameWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.video_label = QLabel()
        self.layout.addWidget(self.video_label)

        self.score_label = QLabel("Score: 0")
        self.score_label.setAlignment(Qt.AlignCenter)
        self.score_label.setFont(QFont("Arial", 24, QFont.Bold))
        self.score_label.setStyleSheet("color: #ECF0F1;")
        self.layout.addWidget(self.score_label)

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=2)
        self.mp_draw = mp.solutions.drawing_utils
        self.cap = cv2.VideoCapture(-1)

        self.num_balls = 5
        self.balls = []
        self.ball_radius = 20
        self.ball_speed_min, self.ball_speed_max = 7, 10
        for _ in range(self.num_balls):
            ball_x = random.randint(50, 600)
            ball_y = random.randint(-500, -50)
            ball_speed = random.randint(self.ball_speed_min, self.ball_speed_max)
            self.balls.append([ball_x, ball_y, ball_speed])

        self.score = 0
        self.bullet_fired = False
        self.bullet_x, self.bullet_y = 0, 0
        self.bullet_radius = 10
        self.bullet_speed = 10
        self.bullet_dir_x, self.bullet_dir_y = 0, 0
        self.last_bullet_time = time.time()
        self.bullet_delay = 0.2

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

    def start_game(self):
        self.timer.start(30)  # Update every 30 ms (approx. 33 FPS)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(img_rgb)

        # Update balls
        for ball in self.balls:
            ball[1] += ball[2]
            if ball[1] > frame.shape[0]:
                ball[0], ball[1] = random.randint(50, frame.shape[1] - 50), random.randint(-500, -50)
            cv2.circle(frame, (ball[0], ball[1]), self.ball_radius, (0, 0, 255), -1)

        # Detect hands
        if result.multi_hand_landmarks:
            for hand_lms in result.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_lms, self.mp_hands.HAND_CONNECTIONS)
                wrist = hand_lms.landmark[0]
                middle_finger_mcp = hand_lms.landmark[9]
                
                h, w, c = frame.shape
                wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)
                middle_x, middle_y = int(middle_finger_mcp.x * w), int(middle_finger_mcp.y * h)

                # Bullet effect
                for id, lm in enumerate(hand_lms.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    if id == 8 and not self.bullet_fired and time.time() - self.last_bullet_time >= self.bullet_delay:
                        self.bullet_fired = True
                        self.bullet_x, self.bullet_y = cx, cy
                        self.bullet_dir_x, self.bullet_dir_y = cx - wrist_x, cy - wrist_y
                        
                        magnitude = np.sqrt(self.bullet_dir_x**2 + self.bullet_dir_y**2)
                        if magnitude > 0:
                            self.bullet_dir_x /= magnitude
                            self.bullet_dir_y /= magnitude
                        self.last_bullet_time = time.time()

        # Update bullet
        if self.bullet_fired:
            self.bullet_x += int(self.bullet_speed * self.bullet_dir_x)
            self.bullet_y += int(self.bullet_speed * self.bullet_dir_y)
            cv2.circle(frame, (self.bullet_x, self.bullet_y), self.bullet_radius, (255, 0, 255), -1)
            if self.bullet_x < 0 or self.bullet_x > w or self.bullet_y < 0 or self.bullet_y > h:
                self.bullet_fired = False

            for ball in self.balls:
                if (ball[0] - self.ball_radius < self.bullet_x < ball[0] + self.ball_radius) and (ball[1] - self.ball_radius < self.bullet_y < ball[1] + self.ball_radius):
                    self.score += 1
                    ball[0], ball[1] = random.randint(50, frame.shape[1] - 50), random.randint(-500, -50)
                    self.bullet_fired = False

        self.score_label.setText(f"Score: {self.score}")

        # Convert frame to QPixmap and display
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        convert_to_qt_format = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(convert_to_qt_format)
        self.video_label.setPixmap(pixmap)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GameWindow()
    window.show()
    sys.exit(app.exec_())
