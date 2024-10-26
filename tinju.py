import cv2
import mediapipe as mp
import numpy as np
import random
import time

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Inisialisasi webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Variabel game
score = 0
lives = 3
game_objects = []
last_object_time = time.time()
game_start_time = time.time()
object_interval = 2.0
min_object_interval = 0.5
is_game_over = False
base_speed = 4
difficulty_multiplier = 1.0

class GameObject:
    def __init__(self, difficulty_mult):
        self.x = random.randint(100, 1180)
        self.y = 0
        self.size = 35
        self.speed = random.randint(4, 8) * difficulty_mult
        self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        self.passed_bottom = False

    def move(self):
        self.y += self.speed
        # Konversi ke integer untuk posisi y
        self.y = int(self.y)
        if self.y > 720 and not self.passed_bottom:
            self.passed_bottom = True
            return True
        return False

    def draw(self, frame):
        # Pastikan koordinat adalah integer
        center = (int(self.x), int(self.y))
        cv2.circle(frame, center, self.size, self.color, -1)

def is_fist(hand_landmarks):
    wrist = hand_landmarks.landmark[0]
    fingertips = [hand_landmarks.landmark[8], hand_landmarks.landmark[12],
                 hand_landmarks.landmark[16], hand_landmarks.landmark[20]]
    palm_center = hand_landmarks.landmark[9]
    
    total_ratio = 0
    for tip in fingertips:
        tip_to_wrist_dist = np.sqrt((tip.x - wrist.x)**2 + (tip.y - wrist.y)**2)
        palm_to_wrist_dist = np.sqrt((palm_center.x - wrist.x)**2 + (palm_center.y - wrist.y)**2)
        if palm_to_wrist_dist > 0:
            ratio = tip_to_wrist_dist / palm_to_wrist_dist
            total_ratio += ratio
    
    avg_ratio = total_ratio / 4
    return avg_ratio < 1.3

def check_collision(hand_x, hand_y, obj):
    distance = np.sqrt((hand_x - obj.x)**2 + (hand_y - obj.y)**2)
    return distance < obj.size + 40

def draw_hand_detection_area(frame, hand_x, hand_y):
    cv2.circle(frame, (int(hand_x), int(hand_y)), 40, (0, 255, 0), 2)

def draw_game_over(frame):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (1280, 720), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    
    cv2.putText(frame, "GAME OVER", (400, 300), 
                cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5)
    cv2.putText(frame, f"Final Score: {score}", (450, 400), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    cv2.putText(frame, "Press 'R' to Restart", (460, 500), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return frame

def reset_game():
    global score, lives, game_objects, last_object_time, game_start_time
    global is_game_over, difficulty_multiplier, object_interval
    score = 0
    lives = 3
    game_objects.clear()
    game_start_time = time.time()
    last_object_time = time.time()
    is_game_over = False
    difficulty_multiplier = 1.0
    object_interval = 2.0

while True:
    success, frame = cap.read()
    if not success:
        break
    
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if not is_game_over:
        # Update difficulty
        elapsed_time = time.time() - game_start_time
        difficulty_multiplier = 1.0 + (elapsed_time / 30.0)
        object_interval = max(min_object_interval, 2.0 - (elapsed_time / 60.0))
        
        # Tambah objek baru
        current_time = time.time()
        if current_time - last_object_time > object_interval:
            game_objects.append(GameObject(difficulty_multiplier))
            last_object_time = current_time
        
        # Update dan gambar objek
        objects_to_remove = []
        for obj in game_objects:
            if obj.move():
                lives -= 1
                if lives <= 0:
                    is_game_over = True
                objects_to_remove.append(obj)
            obj.draw(frame)
        
        # Proses deteksi tangan
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                if is_fist(hand_landmarks):
                    hand_x = int(hand_landmarks.landmark[9].x * frame.shape[1])
                    hand_y = int(hand_landmarks.landmark[9].y * frame.shape[0])
                    
                    draw_hand_detection_area(frame, hand_x, hand_y)
                    
                    for obj in game_objects:
                        if check_collision(hand_x, hand_y, obj):
                            objects_to_remove.append(obj)
                            score += 1
                            cv2.circle(frame, (hand_x, hand_y), 45, (0, 255, 255), -1, cv2.LINE_AA)
        
        # Hapus objek
        for obj in objects_to_remove:
            if obj in game_objects:
                game_objects.remove(obj)
        
        # Tampilkan informasi game
        cv2.rectangle(frame, (30, 20), (300, 60), (0, 0, 0), -1)
        cv2.putText(frame, f'Score: {score} | Lives: {lives}', (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Tampilkan level
        level = int(difficulty_multiplier)
        cv2.putText(frame, f'Level: {level}', (50, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    else:
        frame = draw_game_over(frame)
    
    cv2.imshow('Hand Detection Game', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r') and is_game_over:
        reset_game()

cap.release()
cv2.destroyAllWindows()
