---
title: "[CV] OpenCV Tutorial"
date: 2023-03-15

categories:
  - AI
  - Deep Learning
tags:
  - Computer Vision
  - OpenCV
---

## Read Images, Videos, Webcam

### Image


```python
import cv2

img = cv2.imread('C:/Users/User/Desktop/img.jpg')

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
![image](https://user-images.githubusercontent.com/100760303/225216092-783897e6-4ae8-467f-bec1-20c2a0ca4fa6.png)


### Video

- `frame` : 하나하나 가져오는 이미지
- `success` : 성공적으로 완료되었는지의 여부(True/False)


```python
import cv2

cap = cv2.VideoCapture('C:/Users/User/Desktop/video.mp4')

frameWidth = 550 # 가로
frameHeight = 400 # 높이

while cap.isOpened():
    success, frame = cap.read()

    # 성공적으로 불러와지지 않으면 종료
    if not success:
        break
    
    # 이미지 사이즈 조정
    frame = cv2.resize(frame, (frameWidth, frameHeight))
    cv2.imshow('video', frame)

    # 'q' 버튼 누르면 비디오 종료
    if cv2.waitKey(1) == ord('q'):
        break

cap.release() # 자원 해제
cv2.destroyAllWindows() # 모든 창 닫기
```

### Webcam


```python
import cv2

cap = cv2.VideoCapture(0) # 0번째 카메라 장치 [Device ID]

cap.set(3, 500) # ID 3 Width
cap.set(4, 500) # ID 4 Height
cap.set(10, 100) # ID 10 Light

# 카메라가 열리지 않은 경우 프로그램 종료
if not cap.isOpened(): 
    exit() 

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    cv2.imshow('Camera', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## Basic Functions
- 일반적으로 RGB이지만, OpenCV는 `BGR`


```python
import cv2
import numpy as np

img = cv2.imread('C:/Users/USER/Desktop/img.jpg')
kernel = np.ones((5, 5), np.uint8)

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Gray
imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 0) # Blur
imgCanny = cv2.Canny(img, threshold1=150, threshold2=200) # Edge
imgDialation = cv2.dilate(imgCanny, kernel=kernel, iterations=1)
imgEroded = cv2.erode(imgDialation, kernel=kernel, iterations=1)

cv2.imshow('Gray Img', imgGray)
cv2.imshow('Blur Img', imgBlur)
cv2.imshow('Canny Img', imgCanny)
cv2.imshow('Dialation Img', imgDialation)
cv2.imshow('Eroded Img', imgEroded)

cv2.waitKey(0)
cv2.destroyAllWindows()

```

## Resizing and Cropping

### Resizing


```python
import cv2

img = cv2.imread('C:/Users/User/Desktop/img.jpg')
print(img.shape) # (391, 640, 3)

# Resize
imgResize = cv2.resize(img, (200, 200))
print(imgResize.shape) # (200, 200, 3)

cv2.imshow('img', img)
cv2.imshow('Resize img', imgResize)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

    (391, 640, 3)
    (200, 200, 3)
    
![image](https://user-images.githubusercontent.com/100760303/225216225-a0f77b70-7dca-4edf-833e-cd5011feb419.png)


### Crop


```python
import cv2

img = cv2.imread('C:/Users/User/Desktop/img.jpg')

# Crop (Height, Width)
imgCropped = img[50:200, 200:350]

cv2.imshow('img', img)
cv2.imshow('Crop img', imgCropped)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

![image](https://user-images.githubusercontent.com/100760303/225216323-3c881a44-fa37-4caf-ad8f-9b7425ad2eec.png)


## Shapers and Texts


```python
import cv2
import numpy as np

# 0으로 채워진 행렬 (검은색)
img = np.zeros([512, 512, 3], dtype=np.uint8)
print(img.shape)

# 이미지 색상
img[:] = (255, 0, 0) # 전체를 파란색으로
img[0:100, 0:300] = (255, 0, 0) # 일부만 파란색으로
img[:] = (0, 0, 0) # 전체를 검은색으로

# 이미지 도형
# 시작점, 끝점, 색상, 두께
cv2.line(img, (0, 0), (300, 300), (0, 255, 0), 3)
cv2.rectangle(img, (0, 0), (250, 350), (0, 0, 255), cv2.FILLED) # cv2.FILLED : 채우기

# 중심점, 반지름, 색상, 두께
cv2.circle(img, (400, 50), 30, (255, 255, 0), cv2.FILLED)

# Text
cv2.putText(img, 'OpenCV', (300, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 150, 0), 2)

cv2.imshow('img', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

    (512, 512, 3)
    
![image](https://user-images.githubusercontent.com/100760303/225216364-424b0cba-9ac9-456a-853d-7b73b11824c7.png)


## Warp Prespective


```python
import cv2

img = cv2.imread('C:/Users/User/Desktop/poker.jpg')

width, height = 250, 350
pts1 = np.float32([[513, 140], [613, 241], [472, 381], [373, 281]])
pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

matrix = cv2.getPerspectiveTransform(pts1, pts2) # matrix를 얻어옴
result = cv2.warpPerspective(img, matrix, (width, height)) # matrix대로 변환

cv2.imshow('img', img)
cv2.imshow('result img', result)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

![image](https://user-images.githubusercontent.com/100760303/225216435-e85fa7ba-1b17-4e65-ba10-dc9a87e753fb.png)


## Joining Images


```python
import cv2
import numpy as np

img = cv2.imread('C:/Users/User/Desktop/img.jpg')

# 수평(좌우)로 이미지 합치기
imgHor = np.hstack((img, img))
# 수직(위아래)로 이미지 합치기
imgVer = np.vstack((img, img))

cv2.imshow('Horizontal', imgHor)
cv2.imshow('Vertical', imgVer)

cv2.waitKey(0)
cv2.destroyAllWindows()
```


```python
def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver
```


```python
img = cv2.imread('C:/Users/User/Desktop/img.jpg')

imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgStack = stackImages(0.5,([img, imgGray, img],[img, img, img]))

cv2.imshow("ImageStack",imgStack)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

![image](https://user-images.githubusercontent.com/100760303/225216490-ef7e4166-9e86-44e5-ada7-4878156cac1b.png)
