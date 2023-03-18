import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import enchant


cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("keras_model.h5", "labels.txt")

offset = 20
imgSize = 300
d = enchant.Dict("en_US")


labels = ["A", "B", "C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]

prev_index = None
word = ""
word_freq = 0

while True:
    success, img = cap.read()
    if not success:
        continue  
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
            else:
                continue

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
            else:
                continue

        if prev_index is not None and prev_index != index:
            if word_freq >= 10:
                word += labels[prev_index]
            word_freq = 0

        prev_index = index
        word_freq += 1
        if len(word) >= 2:
            suggestions = d.suggest(word)
            if len(suggestions) > 0:
                suggested_words = suggestions[:3]
                word_str = " ".join(suggested_words)
                cv2.putText(imgOutput, word_str, (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 255), 2)
            
   
       

        cv2.rectangle(imgOutput, (x - offset, y - offset-50),
                      (x - offset+90, y - offset-50+50), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y -26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x-offset, y-offset),
                      (x + w+offset, y + h+offset), (255, 0, 255), 4)

    cv2.putText(imgOutput, word, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 255), 2)
       

    cv2.imshow("Hand Gesture Recognition", imgOutput)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

