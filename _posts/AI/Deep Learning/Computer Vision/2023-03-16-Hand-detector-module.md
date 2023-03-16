---
title: "[CV] Hand Tracking Module"
date: 2023-03-16

categories:
  - AI
  - Deep Learning
tags:
  - Computer Vision
  - OpenCV
  - Mediapipe
---

## HandTrackingModule
- HandTrackingMin을 모듈화 시킨 코드
- 트래킹을 실행하는데 필요한 최소한의 코드
- 각 Hand의 21개의 값 목록을 요청
- 다른 프로젝트에서 활용할 수 있도록 모듈화

## mpHands.Hands()의 파라미터
- static_image_mode=False
- max_num_hands=2
- min_detection_confidence=0.5
- min_tracking_confidence=0.5
- `modelComplexity=1`를 기본 파라미터를 추가 해주어야 모듈로 만들 때 오류가 발생하지 않음

## HandTrackingModule Code
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
def main():
    pTime = 0
    cTime = 0
    
    cap = cv2.VideoCapture(0)

    # 위에서 생성한 handDetector 클래스 불러오기
    detrctor = handDetector()

    # 카메라가 열리지 않은 경우 프로그램 종료
    if not cap.isOpened(): 
        exit() 

    while True:
        success, frame = cap.read()
        frame = detrctor.findHands(frame)

        landmarkList = detrctor.findPosition(frame)
        ## 아래 주석을 풀고 실행하면 id 4에 대한 cx, xy 정보가 
        # if len(landmarkList) != 0:
        #     print(landmarkList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime) # 1/(현재 시간-이전 시간)
        pTime = cTime

        cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN,
                    3, (255, 0, 255), 3)

        cv2.imshow('Image', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release() 
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

```

## HandTrackingMin
- 모듈화 하기 이전 전체 코드

```python
import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0) # device

# 모델을 사용하기 전에 수행해야하는 형식
mpHands = mp.solutions.hands
hands = mpHands.Hands()

# 손을 그릴 수 있게 해줌
mpDraw = mp.solutions.drawing_utils

# fps 프레임 속도 계산을 위한 변수 초기화 (이전 시간, 현재 시간)
pTime = 0
cTime = 0

# 카메라가 열리지 않은 경우 프로그램 종료
if not cap.isOpened(): 
    exit() 

while True:
    success, frame = cap.read()

    # hands 객체가 RGB Image만을 사용하기 때문에 BGR -> RGB 변환
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # result는 hands와 같다
    results = hands.process(imgRGB)

    # 여러 손을 가질 수 있도록 내부 정보 추출
    if results.multi_hand_landmarks:
        # handLms는 각 손의 정보
        for handLms in results.multi_hand_landmarks:
            # 랜드마크 정보 (x, y), id 번호
            for id, landmark in enumerate(handLms.landmark):
                h, w, c = frame.shape
                cx, cy = int(landmark.x*w), int(landmark.y*h)

                if id == 4: # 엄지 (id 번호는 이미지 참초)
                    cv2.circle(frame, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

            # 실제로 그릴 수 있도록 mpDraw 정보 작성 후 랜드마크 작성 -> 점 생성
            # mpHands.HAND_CONNECTIONS -> 점들을 연결 -> 선 생성
            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

    # fps 계산 후 text 삽입
    cTime = time.time()
    fps = 1 / (cTime - pTime) # 1/(현재 시간-이전 시간)
    pTime = cTime

    cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 255), 3)

    cv2.imshow('Image', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release() # 자원 해제
cv2.destroyAllWindows() # 모든 창 닫기
```

## Results
![캡처](https://user-images.githubusercontent.com/100760303/225493139-399d6693-50b3-421c-88b3-443516b2f61a.PNG)

![캡처1](https://user-images.githubusercontent.com/100760303/225493167-fbd8da5e-efe1-4ec2-9664-a69a80aa9e1f.PNG)

![캡처2](https://user-images.githubusercontent.com/100760303/225493178-f228d242-3878-44da-9976-7f9a5d9d76a5.PNG)


