import math

import cv2
from PIL.ImageChops import offset
from cvzone.HandTrackingModule import HandDetector
import numpy as np

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset=20
imgSize=300
while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite=np.ones((imgSize,imgSize,3),np.uint8)*255
        imgCrop= img[y-offset:y+h+offset, x-offset:x+w+offset]

        imgCropShape = imgCrop.shape

        aspectRatio = h/w
        if aspectRatio > 1:
            k=imgSize/h
            wCal = math.ceil(w*k)
            imgResize = cv2.resize(imgCrop,(wCal,imgSize))
            imgResizeShape = imgResize.shape
            wGap= math.ceil((imgSize-wCal)/2)
            imgWhite[:,wGap:wGap+wCal] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(h * k)
            imgResize = cv2.resize(imgCrop, (imgSize,hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hCal,:] = imgResize
        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)


    cv2.imshow("image", img)
    cv2.waitKey(1)