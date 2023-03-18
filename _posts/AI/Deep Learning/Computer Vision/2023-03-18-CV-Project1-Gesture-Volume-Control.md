---
title: "[CV] Gesture Volume Control"
date: 2023-03-18

categories:
  - AI
  - Deep Learning
tags:
  - Computer Vision
  - OpenCV
  - Mediapipe
---

## Gesture Volume Control Project

### HandTrackingModule.py


```python
import cv2
import mediapipe as mp
import time

# class로 모듈화
class handDetector():
    # 초기화 part class
    def __init__(self, mode=False, maxHands=2, modelComplexity=1, detectionConf=0.5, trackConf=0.5):
        # mpHands.Hands()의 기본 파라미터 설정
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.detectionConf = detectionConf
        self.trackConf = trackConf
        
        # 모델을 사용하기 전에 수행해야하는 형식
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplexity, self.detectionConf, self.trackConf)
        self.mpDraw = mp.solutions.drawing_utils

    # Detection part class
    # draw=False로 생성하면 hand detection 되지 않음
    def findHands(self, frame, draw=True):
        # hands 객체가 RGB Image만을 사용하기 때문에 BGR -> RGB 변환
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # result는 hands와 같다
        self.results = self.hands.process(imgRGB) 

        # 여러 손을 가질 수 있도록 내부 정보 추출
        if self.results.multi_hand_landmarks:
            # handLms는 각 손의 정보
            for handLms in self.results.multi_hand_landmarks:
                # draw=True : 그리기를 원하는지의 여부
                # draw=True일 때만 그리게 됨
                if draw:
                    # 실제로 그릴 수 있도록 mpDraw 정보 작성 후 랜드마크 작성 -> 점 생성
                    # mpHands.HAND_CONNECTIONS -> 점들을 연결 -> 선 생성
                    self.mpDraw.draw_landmarks(frame, handLms, self.mpHands.HAND_CONNECTIONS)

        return frame
    
    # 원하는 랜드마크 포인트에 점 찍기
    # draw=False로 
    def findPosition(self, frame, handNo=0, draw=True):
        # 모든 랜드마크에 대한 정보
        landmarkList = []

        # 여러 손을 가질 수 있도록 내부 정보 추출
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

        # 랜드마크 정보 (x, y), id 번호
            for id, landmark in enumerate(myHand.landmark):
                h, w, c = frame.shape
                cx, cy = int(landmark.x*w), int(landmark.y*h)

                landmarkList.append([id, cx, cy])

                if draw:
                    cv2.circle(frame, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

        return landmarkList
```

### VolumeHandControl.py


```python
import cv2
import mediapipe as mp
import time
import numpy as np
import math
```


```python
# pip install pycaw
```


```python
# 볼륨 조절을 위한 라이브러리
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
```

- detectionCont=0.5으로 설정되어 있음(기본값)
- 손인지 확인하고 볼륨을 변경할 때 이 방법으로만 손을 감지하기를 원함
- 해당 프로젝트에 사용할 신뢰도를 0.7로 변경


```python
# Webcam width, height
wCam, hCam = 640, 480

cap = cv2.VideoCapture(0) # device
cap.set(3, wCam) # Webcam width
cap.set(4, hCam) # Webcam height

# fps 계산을 위한 변수 초기화
cTime = 0
pTime = 0

# HandTrackingModule 불러오기
detector = handDetector(detectionConf=0.7)

################ Volume Control을 위한 Code ################
# Initialization
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# volume.GetMute() : 볼륨 음소거
# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange() # (-65.25, 0.0, 0.03125) -> 최소(-65), 최대값(0)
minVol = volRange[0]
maxVol = volRange[1]

vol = 0
volBar = 400
###############################################################

# Webcam이 제대로 작동하지 않으면 멈춤
if not cap.isOpened():
    exit()

while True:
    success, frame = cap.read()

    # HandTrackingModule - findHands 함수 사용
    frame = detector.findHands(frame, draw=True)

    # HandTrackingModule - findPosition 함수 사용 (엄지 4번, 검지 8번)
    landmarkList = detector.findPosition(frame, draw=False)

    if len(landmarkList) != 0:
        # [4, 531, 150], [8, 340, 35] 형식으로 출력됨
        # print(landmarkList[4], landmarkList[8])

        x1, y1 = landmarkList[4][1], landmarkList[4][2] # 4에 대한 값
        x2, y2 = landmarkList[8][1], landmarkList[8][2] # 8에 대한 값

        # 4, 8번 점과 점들을 이은 선의 중심 지점 찾기
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # 4, 8번 원으로 포인트 그리기
        cv2.circle(frame, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
        cv2.circle(frame, (x2, y2), 10, (255, 0, 255), cv2.FILLED)

        # 4, 8번 점을 선으로 잇기
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

        # 4번, 8번을 이은 선의 중심 지점 포인트 그리기
        cv2.circle(frame, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

        # 볼륨을 조절하기 위한 직선의 길이 계산
        length = math.hypot(x2 - x1, y2 - y1)
        # print(length) 출력을 통해 최대, 최소값 확인

        # Hand range 50 ~ 300
        # Volume range -65 ~ 0
        # 두 범위 변환 [50, 300] -> [minVol, maxVol]으로 변환 후 MasterVolume
        vol = np.interp(length, [50, 300], [minVol, maxVol])

        # Volume 직사각형 사용을 위한 범위 변환(Volume이 최소 때 400, Volume이 최대일 때 150)
        volBar = np.interp(length, [50, 300], [400, 150])

        volume.SetMasterVolumeLevel(vol, None)

        # length < 50일 경우 볼륨 줄이는 제스쳐 활용을 위한 중심점 색상 변경
        if length < 50:
            cv2.circle(frame, (cx, cy), 10, (0, 250, 255), cv2.FILLED)

    # Volume Bar 직사각형 생성
    cv2.rectangle(frame, (50, 150), (85, 400), (0, 255, 0))
    # Volume 크기만큼 직사각형 채우기
    cv2.rectangle(frame, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)

    # # fps 계산 및 text 입력
    # cTime = time.time()
    # fps = 1 / (cTime - pTime)
    # pTime = cTime

    # cv2.putText(frame, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX,
    #             1, (255, 0, 0), 2)
    
    cv2.imshow('Img', frame)

    # q 버튼 누르면 종료
    if cv2.waitKey(1) == ord('q'):
        break

cap.release() 
cv2.destroyAllWindows()
```

## Test Video

https://user-images.githubusercontent.com/100760303/226105559-6da40005-d3b0-462c-9c7e-56764c9f1783.mp4
