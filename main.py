

# 원본 코드 https://google.github.io/mediapipe/solutions/hands.html


"""The 21 hand landmarks."""
# 손가락 위치 정의 참고 https://google.github.io/mediapipe/images/mobile/hand_landmarks.png
#
# WRIST = 0
# THUMB_CMC = 1
# THUMB_MCP = 2
# THUMB_IP = 3
# THUMB_TIP = 4    엄지
# INDEX_FINGER_MCP = 5
# INDEX_FINGER_PIP = 6
# INDEX_FINGER_DIP = 7
# INDEX_FINGER_TIP = 8  검지
# MIDDLE_FINGER_MCP = 9
# MIDDLE_FINGER_PIP = 10
# MIDDLE_FINGER_DIP = 11
# MIDDLE_FINGER_TIP = 12  중지
# RING_FINGER_MCP = 13
# RING_FINGER_PIP = 14
# RING_FINGER_DIP = 15
# RING_FINGER_TIP = 16  약지
# PINKY_MCP = 17
# PINKY_PIP = 18
# PINKY_DIP = 19
# PINKY_TIP = 20  새끼


# 필요한 라이브러리
# pip install opencv-python mediapipe pillow numpy

#
# WRIST = 0
# THUMB_CMC = 1
# THUMB_MCP = 2
# THUMB_IP = 3
# THUMB_TIP = 4    엄지
# INDEX_FINGER_MCP = 5
# INDEX_FINGER_PIP = 6
# INDEX_FINGER_DIP = 7
# INDEX_FINGER_TIP = 8  검지
# MIDDLE_FINGER_MCP = 9
# MIDDLE_FINGER_PIP = 10
# MIDDLE_FINGER_DIP = 11
# MIDDLE_FINGER_TIP = 12  중지
# RING_FINGER_MCP = 13
# RING_FINGER_PIP = 14
# RING_FINGER_DIP = 15
# RING_FINGER_TIP = 16  약지
# PINKY_MCP = 17
# PINKY_PIP = 18
# PINKY_DIP = 19
# PINKY_TIP = 20  새끼


# 필요한 라이브러리
# pip install opencv-python mediapipe pillow numpy

import cv2
import mediapipe as mp
from PIL import ImageFont, ImageDraw, Image
import numpy as np

#드로잉 객체
mp_drawing = mp.solutions.drawing_utils

#손 객체
mp_hands = mp.solutions.hands

#드로잉 스타일
mp_drawing_styles = mp.solutions.drawing_styles

# For webcam input:
# video capture 한장. 순간 프레임이라고 생각하자.
cap = cv2.VideoCapture(0) # 카메라 순번 0은 노트북이었음, 1은 usb 외부 카메라

#미디어파이프 손 인스턴스를 쓰고 버리도록 with문을 쓴다.
with mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()

        if not success:
            print("Ignoring empty camera frame.")

            # If loading a video, use 'break' instead of 'continue'.
            continue

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_height, image_width, _ = image.shape

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                # 엄지를 제외한 나머지 4개 손가락의 마디 위치 관계를 확인하여 플래그 변수를 설정합니다. 손가락을 일자로 편 상태인지 확인합니다.
                thumb_finger_state = 0
                if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y * image_height > hand_landmarks.landmark[
                    mp_hands.HandLandmark.THUMB_MCP].y * image_height:
                    if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * image_height > \
                            hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y * image_height:
                        if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y * image_height > \
                                hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_height:
                            thumb_finger_state = 1

                index_finger_state = 0
                if hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * image_height > \
                        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y * image_height:
                    if hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y * image_height > \
                            hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y * image_height:
                        if hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y * image_height > \
                                hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height:
                            index_finger_state = 1

                middle_finger_state = 0
                if hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * image_height > \
                        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y * image_height:
                    if hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y * image_height > \
                            hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y * image_height:
                        if hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y * image_height > \
                                hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * image_height:
                            middle_finger_state = 1

                ring_finger_state = 0
                if hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y * image_height > \
                        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y * image_height:
                    if hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y * image_height > \
                            hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y * image_height:
                        if hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y * image_height > \
                                hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * image_height:
                            ring_finger_state = 1

                pinky_finger_state = 0
                if hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y * image_height > hand_landmarks.landmark[
                    mp_hands.HandLandmark.PINKY_PIP].y * image_height:
                    if hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y * image_height > \
                            hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y * image_height:
                        if hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y * image_height > \
                                hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * image_height:
                            pinky_finger_state = 1

                # 손가락 위치 확인한 값을 사용하여 가위,바위,보 중 하나를 출력 해줍니다.
                # fontpath = "/fonts/gulim.ttc"
                fontpath = "fonts/MaruBuri-Bold.ttf"

                font = ImageFont.truetype(fontpath, 80)
                image = Image.fromarray(image)
                draw = ImageDraw.Draw(image)

                text = ""
                if thumb_finger_state == 1 and index_finger_state == 1 and middle_finger_state == 1 and ring_finger_state == 1 and pinky_finger_state == 1:
                    text = "paper"
                elif thumb_finger_state == 1 and index_finger_state == 1 and middle_finger_state == 0 and ring_finger_state == 0 and pinky_finger_state == 0:
                    text = "scissors"
                elif thumb_finger_state == 0 and index_finger_state == 1 and middle_finger_state == 1 and ring_finger_state == 0 and pinky_finger_state == 0:
                    text = "scissors"
                # 간섭 때문인지 잘 안된다.
                # elif thumb_finger_state == 0 and index_finger_state == 0 and middle_finger_state == 0 and ring_finger_state == 1 and pinky_finger_state == 1:
                #     text = "scissors"
                elif thumb_finger_state == 1 and index_finger_state == 0 and middle_finger_state == 0 and ring_finger_state == 0 and pinky_finger_state == 0:
                    text = "Thumbs Up"
                elif thumb_finger_state == 0 and index_finger_state == 1 and middle_finger_state == 0 and ring_finger_state == 1 and pinky_finger_state == 0:
                    text = "Rock"
                elif thumb_finger_state == 0 and index_finger_state == 0 and middle_finger_state == 1 and ring_finger_state == 0 and pinky_finger_state == 0:
                    text = "Fuck"
                elif thumb_finger_state == 1 and index_finger_state == 0 and middle_finger_state == 0 and ring_finger_state == 0 and pinky_finger_state == 1:
                    text = "Call Me"
                elif thumb_finger_state == 0 and index_finger_state == 0 and middle_finger_state == 0 and ring_finger_state == 0 and pinky_finger_state == 1:
                    text = "약속해줘~"
                elif index_finger_state == 0 and middle_finger_state == 0 and ring_finger_state == 0 and pinky_finger_state == 0:
                    text = "Fist"
                #--------------Error----------------
                # w, h = font.getsize(text)
                #
                # x = 50
                # y = 50
                #
                # draw.rectangle((x, y, x + w, y + h), fill='black')
                # draw.text((x, y), text, font=font, fill=(255, 255, 255))
                #-------------------------------------------------
                color = (255, 255, 0)
                width = 3
                bbox = (50, 50, 300, 300)
                text_pos = (bbox[0] + width, bbox[1])
                font_size = 15
                font = ImageFont.truetype("fonts/MaruBuri-Bold.ttf", font_size)  # arial.ttf 글씨체, font_size=15
                draw = ImageDraw.Draw(image)
                draw.rectangle(bbox, outline=color, width=width)
                # draw.text(text_pos, 'Ground Truth',color,font=font) # font 설정
                txt = '\n손이 인식되었을때만 나오는 \n박스입니다.\n\n \n\n제스쳐:\n\n\n'
                txt = txt + text
                draw.text(text_pos, txt, color, font=font)  # font 설정

                image = np.array(image)




                # 손가락 뼈대를 그려줍니다.
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

        cv2.imshow('MediaPipe Hands', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()















