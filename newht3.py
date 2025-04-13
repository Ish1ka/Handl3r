import cv2
import time
import math
import autopy
import threading
import mediapipe as mp
from pynput.keyboard import Controller, Key

keyboard = Controller()

frame = None
running = True
lock = threading.Lock()

def distCalc(lm1, lm2):
    return math.hypot(lm2.x - lm1.x, lm2.y - lm1.y)

def update_mode(new_mode, mode):
    if new_mode != mode:
        return new_mode
    return mode

def grab_frames():
    global frame, running
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    while running:
        success, img = cap.read()
        if success:
            with lock:
                frame = img.copy()

    cap.release()

def main():
    global frame, running

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5
    )

    thread = threading.Thread(target=grab_frames)
    thread.start()

    screen_w, screen_h = autopy.screen.size()
    smoothening = 2
    clickTolerance = 0.05
    modeTolerance = 0.24
    px, py = 0, 0
    cx, cy = 0, 0
    mode = "mouse"

    while True:
        if frame is None:
            continue

        with lock:
            img = frame.copy()

        img = cv2.flip(img, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        #print(mode)

        if results.multi_hand_landmarks and results.multi_handedness:
            left_hand = None
            right_hand = None

            for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = hand_handedness.classification[0].label
                if label == 'Left':
                    left_hand = hand_landmarks
                elif label == 'Right':
                    right_hand = hand_landmarks

            # Agora, após o loop, usa as mãos
            if left_hand:
                lm0 = left_hand.landmark[0]
                lm8 = left_hand.landmark[8]
                lm12 = left_hand.landmark[12]

                if distCalc(lm0, lm8) < modeTolerance and distCalc(lm0, lm12) < modeTolerance:
                    mode = update_mode("grab", mode)
                else:
                    mode = update_mode("mouse", mode)
                    #print(distCalc(lm0, lm8))
            
            if left_hand == None:
                mode = update_mode("mouse", mode)

            if right_hand:
                margin_x, margin_y = 0.1, 0.1
                handBase = right_hand.landmark[0]
                index = right_hand.landmark[8]
                middle = right_hand.landmark[12]
                thumb = right_hand.landmark[4]
                lm13 = right_hand.landmark[13]
                x, y = lm13.x, lm13.y
                #print(x)

                x = min(max(x, margin_x), 1 - margin_x)
                y = min(max(y, margin_y), 1 - margin_y)

                x = (x - margin_x) / (1 - 2 * margin_x)
                y = (y - margin_y) / (1 - 2 * margin_y)

                rawmouse_x, rawmouse_y = int(x * screen_w), int(y * screen_h)
                mouse_x = min(max(0, rawmouse_x), screen_w - 1)
                mouse_y = min(max(0, rawmouse_y), screen_h - 1)

                cx = px + (mouse_x - px) / smoothening
                cy = py + (mouse_y - py) / smoothening
                px, py = cx, cy
                #print(distCalc(index, thumb))
                autopy.mouse.move(cx, cy)

                if mode == "mouse":
                    if distCalc(index, thumb) < clickTolerance:
                        autopy.mouse.toggle(autopy.mouse.Button.LEFT, True)
                    else:
                        autopy.mouse.toggle(autopy.mouse.Button.LEFT, False)
                    if distCalc(middle, thumb) < clickTolerance:
                        autopy.mouse.toggle(autopy.mouse.Button.RIGHT, True)
                    else:
                        autopy.mouse.toggle(autopy.mouse.Button.RIGHT, False)


                if mode == "grab":
                    if distCalc(index, thumb) < clickTolerance:
                        autopy.mouse.toggle(autopy.mouse.Button.RIGHT, True)
                        keyboard.press(Key.shift)
                    else:
                        autopy.mouse.toggle(autopy.mouse.Button.RIGHT, False)
                        keyboard.release(Key.shift)

                    if distCalc(handBase, index) < modeTolerance:
                        autopy.mouse.toggle(autopy.mouse.Button.MIDDLE, True)
                    else:
                        autopy.mouse.toggle(autopy.mouse.Button.MIDDLE, False)

        # Display
        cv2.imshow("Hand Tracking", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    running = False
    thread.join()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
