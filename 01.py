import cv2
import mediapipe as mp
import math
import pygetwindow as gw
import pyautogui

#  MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils


# def beetween two landmark
def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# def for changing window size
def resize_window(distance):
    # find open window
    windows = gw.getWindowsWithTitle("This PC")  # for example my computer
    if windows:
        window = windows[0]
        new_width = int(distance * 2)
        new_height = window.height
        window.resizeTo(new_width, new_height)


cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # to rgb
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    height, width, _ = frame.shape
    hand_positions = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x = int(hand_landmarks.landmark[0].x * width)
            y = int(hand_landmarks.landmark[0].y * height)
            hand_positions.append((x, y))
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if len(hand_positions) == 2:  # if detected
            x1, y1 = hand_positions[0]
            x2, y2 = hand_positions[1]
            distance = calculate_distance((x1, y1), (x2, y2))

            # drawing box
            if distance > 100:  # distance beetween hands
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            resize_window(distance)

    cv2.imshow("Hand Detection and Window Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to end the code
        break

cap.release()
cv2.destroyAllWindows()
