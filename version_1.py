import copy
import itertools
import pickle

import cv2 as cv
import numpy as np
import mediapipe as mp

import RPi.GPIO as GPIO
import speech_recognition as sr
import time
import collections

import pygame
import os
from pygame.locals import *
import time

#设置 GPIO 模式为 BCM
GPIO.setmode(GPIO.BCM)
  
#定义 GPIO 引脚
GPIO_TRIGGER = 26
GPIO_ECHO = 19
  
#设置 GPIO 的工作方式 (IN / OUT)
GPIO.setup(GPIO_TRIGGER, GPIO.OUT)
GPIO.setup(GPIO_ECHO, GPIO.IN)
stk_voice = collections.deque(['move','left','right'])

def main(hand_control, voice_control):
    stk = collections.deque(['stop', 'stop', 'stop', 'stop'])
    #GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    AIN1 = 5
    AIN2 = 6
    PWMA = 13
    BIN1 = 20
    BIN2 = 21
    PWMB = 12
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(PWMA, GPIO.OUT)
    GPIO.setup(AIN1, GPIO.OUT)
    GPIO.setup(AIN2, GPIO.OUT)
    GPIO.setup(PWMB, GPIO.OUT)
    GPIO.setup(BIN1, GPIO.OUT)
    GPIO.setup(BIN2, GPIO.OUT)
    frequency = 50
    stop_dc = 0
    half_dc = 25
    full_dc = 50
    pA = GPIO.PWM(PWMA, frequency)
    pA.start(0)
    pB = GPIO.PWM(PWMB, frequency)
    pB.start(0)

    cap = cv.VideoCapture(0)
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    filename = 'rand_forest.sav'
    loaded_model = pickle.load(open(filename, 'rb'))

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )

    gesture_label = 'stop'
    previous_label = 'stop'
    code_run = True
    
    while hand_control:

        key = cv.waitKey(1)
        if key == 27:
            break

        # camera capture
        ret, image = cap.read()

        if not ret:
            continue

        image = cv.flip(image, -1)
        debug_image = copy.deepcopy(image)

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                brect = cal_bounding_rect(debug_image, hand_landmarks)
                landmark_list = cal_landmark_list(debug_image, hand_landmarks)

                # normalize coordinates
                # preprocess_landmark_list = preprocess_landmark(landmark_list)
                # preprocess_landmark_list = np.array(preprocess_landmark_list).reshape(1,-1)
                # predict_list = loaded_model.predict_proba(preprocess_landmark_list)[0]
                #
                # max_index = np.argmax(predict_list)
                #
                # if(max_index==0):
                #     gesture_label = "stop"
                # elif(max_index==1):
                #     gesture_label = "move"
                # elif(max_index==2):
                #     gesture_label = "left"
                # elif(max_index==3):
                #     gesture_label = "right"
                array = []
                array_x = []
                array_y = []
                for i, lm in enumerate(hand_landmarks.landmark):
                    yPos = int(lm.y * image.shape[0])
                    xPos = int(lm.x * image.shape[1])
                    array_x.append(xPos)
                    array_y.append(yPos)
                    array.append(xPos)
                    array.append(yPos)

                if len(array_y) == 0:
                    continue
                # print(array_x)
                # print(array_y)
                if min(array_x[8], array_x[12], array_x[16]) == min(array_x) and max(array_x[0], array_x[1],
                                                                                     array_x[2]) == max(array_x):
                    print("Right")
                    gesture_label = "right"
                elif array_y[0] == max(array_y) and array_y[8] <= array_y[6] and array_y[12] <= array_y[10] and array_y[
                    16] <= array_y[14]:
                    print("Move")
                    gesture_label = "move"
                elif array_y[0] == max(array_y) and array_y[8] >= array_y[6] and array_y[12] >= array_y[10] and array_y[
                    16] >= array_y[14]:
                    print("Stop")
                    gesture_label = "stop"
                elif array_y[0] != max(array_y) and array_x[8] > array_x[5]:
                    print("Left")
                    gesture_label = "left"

                debug_image = cv.rectangle(debug_image, (brect[0], brect[1]), (brect[2], brect[3]),
                                           (0, 255, 0), 4)

                text_position = (brect[0] + 50, brect[1] + 10)
                debug_image = cv.putText(debug_image, gesture_label, text_position, cv.FONT_HERSHEY_SIMPLEX, 1,
                                         (255, 0, 0), 3)

                mp_drawing.draw_landmarks(
                    debug_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

        # print(gesture_label)

        cv.imshow('Hand Gesture Recognition', debug_image)
        stk.popleft()
        stk.append(gesture_label)
        if stk[0] == stk[1] and stk[1] == stk[2] and stk[2] == stk[3]:
            if stk[0] == 'stop':
                stop(pA)
                stop(pB)
                print('stop')

            elif (stk[0] == 'move'):
                dist = distance()
                print("Measured Distance = {:.2f} cm".format(dist))
                if (dist < 40):
                    counterclockwise(pA, full_dc, AIN1, AIN2)
                    counterclockwise(pB, full_dc, BIN1, BIN2)
                    print('distance less than 10, back')
                if (dist > 70):
                    clockwise(pA, full_dc, AIN1, AIN2)
                    clockwise(pB, full_dc, BIN1, BIN2)
                    print('distance more than 70, move')

            elif (stk[0] == 'left'):
                clockwise(pA, full_dc, AIN1, AIN2)
                clockwise(pB, half_dc, BIN1, BIN2)
                print('left')

            elif (stk[0] == 'right'):
                clockwise(pA, half_dc, AIN1, AIN2)
                clockwise(pB, full_dc, BIN1, BIN2)
                print('right')
                
    while voice_control:

        while True:
            
            key = cv.waitKey(1)
            if key==27:
                break

            # camera capture
            
            r = sr.Recognizer()
            speech = sr.Microphone(device_index=2)
            with speech as source:
                print("say something!…")
                audio = r.adjust_for_ambient_noise(source)
                audio = r.listen(source)
            try:
                recog = r.recognize_google(audio)
                print("You said: " + recog)
            except sr.UnknownValueError:
                print("Google Speech Recognition could not understand audio")
            except sr.RequestError as e:
                print("Could not request results from Google Speech Recognition service; {0}".format(e))
            stk.append(recog)
            

            if( stk[0] =='move'):

                clockwise(pA, full_dc, AIN1, AIN2)
                clockwise(pB, full_dc, BIN1, BIN2)
                time.sleep(1)
                clockwise(pA, stop_dc, AIN1, AIN2)
                clockwise(pB, stop_dc, BIN1, BIN2)
                
            elif( stk[0] == 'left'):
                clockwise(pA, 2*full_dc, AIN1, AIN2)
                counterclockwise(pB, 2*full_dc, BIN1, BIN2)
                print('left')
                time.sleep(1)
                clockwise(pA, stop_dc, AIN1, AIN2)
                clockwise(pB, stop_dc, BIN1, BIN2)         
            elif( stk[0] == 'right'):
                counterclockwise(pA, 2*full_dc, AIN1, AIN2)
                clockwise(pB, 2*full_dc, BIN1, BIN2)
                print('right')
                time.sleep(1)
                clockwise(pA, stop_dc, AIN1, AIN2)
                clockwise(pB, stop_dc, BIN1, BIN2)
                    
            stk.popleft()
        
    cap.release()

    pA.stop()
    pB.stop()

    GPIO.output(AIN2, GPIO.LOW)
    GPIO.output(AIN1, GPIO.LOW)
    GPIO.output(BIN2, GPIO.LOW)
    GPIO.output(BIN1, GPIO.LOW)

    GPIO.cleanup()


def distance():
    # 发送高电平信号到 Trig 引脚
    GPIO.output(GPIO_TRIGGER, True)

    # 持续 10 us
    time.sleep(0.00001)
    GPIO.output(GPIO_TRIGGER, False)

    start_time = time.time()
    stop_time = time.time()

    # 记录发送超声波的时刻1
    while GPIO.input(GPIO_ECHO) == 0:
        start_time = time.time()

    # 记录接收到返回超声波的时刻2
    while GPIO.input(GPIO_ECHO) == 1:
        stop_time = time.time()
    # 计算超声波的往返时间 = 时刻2 - 时刻1
    time_elapsed = stop_time - start_time
    # 声波的速度为 343m/s， 转化为 34300cm/s。
    distance = (time_elapsed * 34300) / 2

    return distance


def cal_bounding_rect(image, landmarks):
    img_width = image.shape[1]
    img_height = image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * img_width), img_width - 1)
        landmark_y = min(int(landmark.y * img_height), img_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]
        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def cal_landmark_list(image, landmarks):
    img_width = image.shape[1]
    img_height = image.shape[0]
    landmark_point = []
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * img_width), img_width - 1)
        landmark_y = min(int(landmark.y * img_height), img_height - 1)
        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def preprocess_landmark(landmark_list):
    temp_lm_list = copy.deepcopy(landmark_list)

    base_x = 0
    base_y = 0

    for index, landmark_point in enumerate(temp_lm_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_lm_list[index][0] = temp_lm_list[index][0] - base_x
        temp_lm_list[index][1] = temp_lm_list[index][1] - base_y

    temp_lm_list = list(itertools.chain.from_iterable(temp_lm_list))

    max_value = max(list(map(abs, temp_lm_list)))

    def normalize_(n):
        return n / max_value

    temp_lm_list = list(map(normalize_, temp_lm_list))

    return temp_lm_list


def clockwise(p, dc, in1, in2):
    p.ChangeDutyCycle(dc)
    GPIO.output(in1, GPIO.HIGH)
    GPIO.output(in2, GPIO.LOW)
    return None


def counterclockwise(p, dc, in1, in2):
    p.ChangeDutyCycle(dc)
    GPIO.output(in2, GPIO.HIGH)
    GPIO.output(in1, GPIO.LOW)
    return None


def stop(p):
    p.ChangeDutyCycle(0)
    return None


if __name__ == '__main__':
    # 设置 GPIO 模式为 BCM
    GPIO.setmode(GPIO.BCM)
    # 定义 GPIO 引脚
    GPIO_TRIGGER = 26
    GPIO_ECHO = 19

    # 设置 GPIO 的工作方式 (IN / OUT)
    GPIO.setup(GPIO_TRIGGER, GPIO.OUT)
    GPIO.setup(GPIO_ECHO, GPIO.IN)

    os.putenv('SDL_VIDEODRIVER', 'fbcon')
    os.putenv('SDL_FBDEV', '/dev/fb1')
    os.putenv('SDL_MOUSEDRV', 'TSLIB')
    os.putenv('SDL_MOUSEDEV', '/dev/input/touchscreen')

    GPIO.setup(27, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    start_time = time.time()
    pygame.init()

    # Control Mode
    hand_control = False
    voice_control = False

    # pygame.mouse.set_visible(True)

    WHITE = 255, 255, 255
    BLACK = 0, 0, 0
    screen = pygame.display.set_mode((320, 240))

    my_font = pygame.font.Font(None, 50)
    my_buttons = {'Hand Control': (150, 200), 'Voice Control': (150, 100)}
    screen.fill(BLACK)

    my_button_rect = {}
    for my_text, text_pos in my_buttons.items():
        text_surface = my_font.render(my_text, True, WHITE)
        rect = text_surface.get_rect(center=text_pos)
        screen.blit(text_surface, rect)
        my_button_rect[my_text] = rect

    pygame.display.flip()

    code_run = True

    while code_run:

        for event in pygame.event.get():
            if (event.type is MOUSEBUTTONDOWN):
                pos = pygame.mouse.get_pos()
            elif (event.type is MOUSEBUTTONUP):
                pos = pygame.mouse.get_pos()

                for (my_text, rect) in my_button_rect.items():
                    if (rect.collidepoint(pos)):
                        if (my_text == 'Hand Control'):
                            print("Hand Control Begins")
                            hand_control = True
                            voice_control = False
                        if (my_text == 'Voice Control'):
                            voice_control = True
                            hand_control = False
                            print("voice")


        main(hand_control, voice_control)

       # if not GPIO.input(27):
           # code_run = False

    pygame.quit()

