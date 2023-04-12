# Auto-driving-along-the-line
노랑선 따라가는 자율주행 로봇 코딩(출처:챗GPT)
import cv2
import numpy as np
import RPi.GPIO as GPIO
import time

# GPIO 핀 번호 설정
GPIO.setmode(GPIO.BOARD)
GPIO.setup(11, GPIO.OUT)
GPIO.setup(12, GPIO.OUT)
GPIO.setup(13, GPIO.OUT)
GPIO.setup(15, GPIO.OUT)

# 모터 제어 함수
def set_motor(left, right):
    if left > 0:
        GPIO.output(11, GPIO.HIGH)
        GPIO.output(12, GPIO.LOW)
        pwm1.ChangeDutyCycle(left)
    elif left == 0:
        GPIO.output(11, GPIO.LOW)
        GPIO.output(12, GPIO.LOW)
        pwm1.ChangeDutyCycle(left)
    else:
        GPIO.output(11, GPIO.LOW)
        GPIO.output(12, GPIO.HIGH)
        pwm1.ChangeDutyCycle(-left)

    if right > 0:
        GPIO.output(13, GPIO.HIGH)
        GPIO.output(15, GPIO.LOW)
        pwm2.ChangeDutyCycle(right)
    elif right == 0:
        GPIO.output(13, GPIO.LOW)
        GPIO.output(15, GPIO.LOW)
        pwm2.ChangeDutyCycle(right)
    else:
        GPIO.output(13, GPIO.LOW)
        GPIO.output(15, GPIO.HIGH)
        pwm2.ChangeDutyCycle(-right)

# PWM 제어 설정
GPIO.setup(16, GPIO.OUT)
GPIO.setup(18, GPIO.OUT)
pwm1 = GPIO.PWM(16, 100)
pwm1.start(0)
pwm2 = GPIO.PWM(18, 100)
pwm2.start(0)

# 카메라 설정
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

while True:
    _, frame = cap.read()

    # 이미지 전처리
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    edges = cv2.Canny(mask, 50, 150)

    # 선 검출
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 20, minLineLength=30, maxLineGap=5)

    # 선 정보 분석
    left_lines = []
    right_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x1 == x2:
            continue
        k = (y2-y1) / (x2-x1)
        if k < -0.5:
            left_lines.append(line)
        elif k > 0.5:
            right_lines.append(line)

# 왼쪽 방향, 오른쪽 방향 선 추출
    left_lines = np.array(left_lines)
    right_lines = np.array(right_lines)

    # 왼쪽 방향, 오른쪽 방향 선 평균 계산
    left_mean = np.mean(left_lines, axis=0)
    right_mean = np.mean(right_lines, axis=0)

    # 선 중앙 위치 계산
    x1, y1, x2, y2 = left_mean[0]
    left_k = (y2-y1) / (x2-x1)
    left_b = y1 - left_k * x1
    x1, y1, x2, y2 = right_mean[0]
    right_k = (y2-y1) / (x2-x1)
    right_b = y1 - right_k * x1
    center_x = int((320 - right_b + left_b) / (left_k - right_k))
    center_y = int(left_k * center_x + left_b)

    # 중앙 위치에 따른 제어 신호 생성
    err = center_x - 160
    k_p = 0.5
    left = 50 + k_p * err
    right = 50 - k_p * err
    left = np.clip(left, 0, 100)
    right = np.clip(right, 0, 100)

    # 모터 제어
    set_motor(left, right)

    # 영상 출력
    cv2.line(frame, (int(left_mean[0,0]), int(left_mean[0,1])), (int(left_mean[0,2]), int(left_mean[0,3])), (0, 255, 0), 2)
    cv2.line(frame, (int(right_mean[0,0]), int(right_mean[0,1])), (int(right_mean[0,2]), int(right_mean[0,3])), (0, 0, 255), 2)
    cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)
    cv2.imshow("frame", frame)
    cv2.imshow("mask", mask)
    cv2.imshow("edges", edges)

    if cv2.waitKey(1) == 27:
        break
