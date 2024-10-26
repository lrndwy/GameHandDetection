import json
import os
import random
import time

import cv2
import mediapipe as mp
import numpy as np


class GameMenu:
    def __init__(self):
        # Set ukuran window ke fullscreen
        self.screen = cv2.namedWindow("Menu", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Menu", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # Dapatkan ukuran layar
        self.width = 3142 # Lebar layar fullscreen untuk MacBook Pro M1 2021
        self.height = 1964  # Tinggi layar fullscreen untuk MacBook Pro M1 2021

        # Background dengan gradient
        self.menu_bg = self.create_gradient_background()
        self.highscore = self.load_highscore()

        # Load gambar background (jika ada)
        self.particles = []
        self.create_particles(30)  # Membuat 30 partikel bergerak

    def create_gradient_background(self):
        bg = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        for i in range(self.height):
            # Buat gradient dari biru tua ke ungu
            blue = int(255 * (1 - i / self.height))
            red = int(100 * (i / self.height))
            bg[i, :] = [blue, 0, red]
        return bg

    def create_particles(self, num_particles):
        for _ in range(num_particles):
            self.particles.append(
                {
                    "x": random.randint(0, self.width),
                    "y": random.randint(0, self.height),
                    "speed": random.randint(1, 3),
                    "size": random.randint(2, 5),
                }
            )

    def update_particles(self, menu):
        for particle in self.particles:
            # Gambar partikel
            cv2.circle(
                menu,
                (particle["x"], particle["y"]),
                particle["size"],
                (255, 255, 255),
                -1,
            )

            # Update posisi
            particle["y"] += particle["speed"]

            # Reset posisi jika keluar layar
            if particle["y"] > self.height:
                particle["y"] = 0
                particle["x"] = random.randint(0, self.width)

    def load_highscore(self):
        if os.path.exists("highscore.json"):
            with open("highscore.json", "r") as f:
                return json.load(f)["highscore"]
        return 0

    def save_highscore(self, score):
        if score > self.highscore:
            self.highscore = score
            with open("highscore.json", "w") as f:
                json.dump({"highscore": score}, f)

    def draw_button(self, img, text, position, size, hover=False):
        x, y = position
        w, h = size

        # Efek hover
        color = (0, 255, 0) if hover else (0, 200, 0)

        # Button background dengan alpha blending
        button_bg = img[y : y + h, x : x + w].copy()
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.rectangle(overlay, (0, 0), (w, h), color, -1)
        cv2.addWeighted(overlay, 0.5, button_bg, 0.5, 0, button_bg)
        img[y : y + h, x : x + w] = button_bg

        # Button border
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

        # Text dengan shadow
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = x + (w - text_size[0]) // 2
        text_y = y + (h + text_size[1]) // 2

        # Shadow
        cv2.putText(
            img,
            text,
            (text_x + 2, text_y + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2,
        )
        # Text
        cv2.putText(
            img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
        )

        return (x, y, w, h)

    def show(self):
        mouse_pos = (0, 0)

        def on_mouse_move(event, x, y, flags, param):
            nonlocal mouse_pos
            mouse_pos = (x, y)

        cv2.setMouseCallback("Menu", on_mouse_move)

        while True:
            menu = self.menu_bg.copy()

            # Update dan gambar partikel
            self.update_particles(menu)

            # Judul dengan efek glow
            title = "Hand Detection Game"
            title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, 2.5, 3)[0]
            title_x = (self.width - title_size[0]) // 2

            # Glow effect
            for offset in range(3, 0, -1):
                cv2.putText(
                    menu,
                    title,
                    (title_x, 150),
                    cv2.FONT_HERSHEY_DUPLEX,
                    2.5,
                    (0, 0, 50 * offset),
                    3 + offset,
                )

            # Main title
            cv2.putText(
                menu,
                title,
                (title_x, 150),
                cv2.FONT_HERSHEY_DUPLEX,
                2.5,
                (255, 255, 255),
                3,
            )

            # Highscore dengan efek glow
            score_text = f"Highscore: {self.highscore}"
            score_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)[
                0
            ]
            score_x = (self.width - score_size[0]) // 2

            # Glow effect for score
            for offset in range(3, 0, -1):
                cv2.putText(
                    menu,
                    score_text,
                    (score_x, 250),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 0, 50 * offset),
                    2 + offset,
                )

            # Main score text
            cv2.putText(
                menu,
                score_text,
                (score_x, 250),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (255, 255, 255),
                2,
            )

            # Buttons dengan efek hover
            start_pos = ((self.width - 250) // 2, 400)
            quit_pos = ((self.width - 250) // 2, 500)

            # Check hover untuk setiap button
            start_hover = (
                start_pos[0] < mouse_pos[0] < start_pos[0] + 250
                and start_pos[1] < mouse_pos[1] < start_pos[1] + 60
            )
            quit_hover = (
                quit_pos[0] < mouse_pos[0] < quit_pos[0] + 250
                and quit_pos[1] < mouse_pos[1] < quit_pos[1] + 60
            )

            start_btn = self.draw_button(
                menu, "Start Game", start_pos, (250, 60), start_hover
            )
            quit_btn = self.draw_button(
                menu, "Quit Game", quit_pos, (250, 60), quit_hover
            )

            # Instructions
            cv2.putText(
                menu,
                "How to Play:",
                (start_pos[0], 650),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                menu,
                "- Use your index finger to hit falling objects",
                (start_pos[0], 690),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (200, 200, 200),
                2,
            )
            cv2.putText(
                menu,
                "- Don't let objects reach the bottom",
                (start_pos[0], 720),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (200, 200, 200),
                2,
            )
            cv2.putText(
                menu,
                "- Press 'Q' to quit, 'R' to return to menu",
                (start_pos[0], 750),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (200, 200, 200),
                2,
            )

            cv2.imshow("Menu", menu)
            key = cv2.waitKey(1) & 0xFF

            # Handle button clicks
            if cv2.getWindowProperty("Menu", cv2.WND_PROP_VISIBLE) < 1:
                break

            def on_mouse(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    if (
                        start_btn[0] < x < start_btn[0] + start_btn[2]
                        and start_btn[1] < y < start_btn[1] + start_btn[3]
                    ):
                        cv2.destroyWindow("Menu")
                        game = FallingObjectGame(self)
                        game.run()
                    elif (
                        quit_btn[0] < x < quit_btn[0] + quit_btn[2]
                        and quit_btn[1] < y < quit_btn[1] + quit_btn[3]
                    ):
                        cv2.destroyWindow("Menu")

            cv2.setMouseCallback("Menu", on_mouse)

            if key == ord("q"):
                break

        cv2.destroyAllWindows()


class FallingObjectGame:
    def __init__(self, menu):
        self.menu = menu

        # Inisialisasi MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Inisialisasi webcam
        self.cap = cv2.VideoCapture(0)

        # Set fullscreen untuk game window
        cv2.namedWindow("Game", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Game", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # Setting ukuran layar ke fullscreen
        self.width = 1920  # Default fullscreen width
        self.height = 1080  # Default fullscreen height

        # Set webcam resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # Variabel untuk objek yang jatuh
        self.objects = []
        self.score = 0
        self.game_over = False

    def create_falling_object(self):
        x = random.randint(50, self.width - 50)
        return {
            "x": x,
            "y": 0,
            "radius": 20,
            "speed": random.randint(5, 10),
            "color": (
                random.randint(50, 255),  # Warna random untuk setiap objek
                random.randint(50, 255),
                random.randint(50, 255),
            ),
        }

    def draw_objects(self, frame):
        for obj in self.objects:
            # Gambar objek dengan glow effect
            cv2.circle(
                frame, (obj["x"], int(obj["y"])), obj["radius"] + 4, (255, 255, 255), -1
            )  # Outer glow
            cv2.circle(
                frame, (obj["x"], int(obj["y"])), obj["radius"], obj["color"], -1
            )  # Main object

    def update_objects(self):
        new_objects = []
        for obj in self.objects:
            obj["y"] += obj["speed"]
            if obj["y"] < self.height:
                new_objects.append(obj)
            else:
                self.game_over = True
        self.objects = new_objects

    def check_collision(self, left_fist, right_fist):
        objects_to_remove = []
        for obj in self.objects:
            if left_fist:
                left_distance = np.sqrt((left_fist[0] - obj["x"]) ** 2 + (left_fist[1] - obj["y"]) ** 2)
                if left_distance < obj["radius"] + 20:  # Tambahkan toleransi untuk ukuran kepalan
                    self.create_particle_effect(obj["x"], obj["y"], obj["color"])
                    objects_to_remove.append(obj)
                    self.score += 1
                    continue
            
            if right_fist:
                right_distance = np.sqrt((right_fist[0] - obj["x"]) ** 2 + (right_fist[1] - obj["y"]) ** 2)
                if right_distance < obj["radius"] + 20:  # Tambahkan toleransi untuk ukuran kepalan
                    self.create_particle_effect(obj["x"], obj["y"], obj["color"])
                    objects_to_remove.append(obj)
                    self.score += 1

        # Hapus objek yang terkena pukulan
        for obj in objects_to_remove:
            self.objects.remove(obj)

    def create_particle_effect(self, x, y, color):
        # TODO: Implementasi efek partikel saat objek dihancurkan
        pass

    def show_game_over(self, frame):
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.width, self.height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Game Over text dengan efek glow
        text = "GAME OVER!"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 3, 3)[0]
        text_x = (self.width - text_size[0]) // 2

        # Glow effect
        for offset in range(5, 0, -1):
            cv2.putText(
                frame,
                text,
                (text_x, self.height // 2),
                cv2.FONT_HERSHEY_DUPLEX,
                3,
                (0, 0, 50 * offset),
                3 + offset,
            )

        # Main text
        cv2.putText(
            frame,
            text,
            (text_x, self.height // 2),
            cv2.FONT_HERSHEY_DUPLEX,
            3,
            (0, 0, 255),
            3,
        )

        # Score
        score_text = f"Final Score: {self.score}"
        score_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 2)[0]
        score_x = (self.width - score_size[0]) // 2

        cv2.putText(
            frame,
            score_text,
            (score_x, self.height // 2 + 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 255, 255),
            2,
        )

        # Return to menu instruction
        return_text = "Press 'R' to Return to Menu"
        return_size = cv2.getTextSize(return_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        return_x = (self.width - return_size[0]) // 2

        cv2.putText(
            frame,
            return_text,
            (return_x, self.height // 2 + 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

    def detect_fists(self, hand_landmarks):
        if not hand_landmarks:
            return None, None

        left_fist = None
        right_fist = None

        for hand_lms in hand_landmarks:
            # Hitung jarak antara ujung jari dan pangkal telapak tangan
            thumb_tip = hand_lms.landmark[4]
            index_tip = hand_lms.landmark[8]
            middle_tip = hand_lms.landmark[12]
            ring_tip = hand_lms.landmark[16]
            pinky_tip = hand_lms.landmark[20]
            wrist = hand_lms.landmark[0]

            fingers_folded = all([
                self.is_finger_folded(thumb_tip, wrist),
                self.is_finger_folded(index_tip, wrist),
                self.is_finger_folded(middle_tip, wrist),
                self.is_finger_folded(ring_tip, wrist),
                self.is_finger_folded(pinky_tip, wrist)
            ])

            if fingers_folded:
                fist_x = int(wrist.x * self.width)
                fist_y = int(wrist.y * self.height)
                
                # Tentukan apakah ini tangan kiri atau kanan
                if wrist.x < 0.5:
                    left_fist = (fist_x, fist_y)
                else:
                    right_fist = (fist_x, fist_y)

        return left_fist, right_fist

    def is_finger_folded(self, finger_tip, wrist):
        return finger_tip.y > wrist.y

    def run(self):
        last_object_time = time.time()

        while True:
            success, frame = self.cap.read()
            if not success:
                break

            # Resize frame ke fullscreen
            frame = cv2.resize(frame, (self.width, self.height))
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            if not self.game_over:
                # Game logic
                current_time = time.time()
                if current_time - last_object_time > 2:
                    self.objects.append(self.create_falling_object())
                    last_object_time = current_time

                self.update_objects()
                self.draw_objects(frame)

                left_fist = None
                right_fist = None

                if results.multi_hand_landmarks:
                    left_fist, right_fist = self.detect_fists(results.multi_hand_landmarks)
                    
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw hand landmarks with custom style
                        for connection in self.mp_hands.HAND_CONNECTIONS:
                            start_point = hand_landmarks.landmark[connection[0]]
                            end_point = hand_landmarks.landmark[connection[1]]

                            start_x = int(start_point.x * self.width)
                            start_y = int(start_point.y * self.height)
                            end_x = int(end_point.x * self.width)
                            end_y = int(end_point.y * self.height)

                            cv2.line(
                                frame,
                                (start_x, start_y),
                                (end_x, end_y),
                                (0, 255, 0),
                                2,
                            )

                        for landmark in hand_landmarks.landmark:
                            x = int(landmark.x * self.width)
                            y = int(landmark.y * self.height)
                            cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)

                        # Highlight index finger
                        index_finger = hand_landmarks.landmark[8]
                        finger_x = int(index_finger.x * self.width)
                        finger_y = int(index_finger.y * self.height)

                        # Draw glowing circle at index finger
                        for r in range(15, 5, -3):
                            cv2.circle(
                                frame, (finger_x, finger_y), r, (0, 0, 255 - r * 10), -1
                            )
                        cv2.circle(frame, (finger_x, finger_y), 5, (0, 0, 255), -1)

                    # Gambar kepalan tangan
                    if left_fist:
                        cv2.circle(frame, left_fist, 20, (0, 0, 255), -1)
                    if right_fist:
                        cv2.circle(frame, right_fist, 20, (255, 0, 0), -1)

                    self.check_collision(left_fist, right_fist)

                # Tampilkan score dengan efek glow
                score_text = f"Score: {self.score}"
                # Shadow
                cv2.putText(
                    frame,
                    score_text,
                    (52, 52),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 0),
                    2,
                )
                # Main text
                cv2.putText(
                    frame,
                    score_text,
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )
            else:
                # Update highscore jika perlu
                self.menu.save_highscore(self.score)
                self.show_game_over(frame)

            cv2.imshow("Hand Detection Game", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("r") and self.game_over:
                cv2.destroyWindow("Hand Detection Game")
                self.menu.show()
                break

        self.cap.release()
        cv2.destroyAllWindows()
        self.hands.close()


# Jalankan game
if __name__ == "__main__":
    menu = GameMenu()
    menu.show()

