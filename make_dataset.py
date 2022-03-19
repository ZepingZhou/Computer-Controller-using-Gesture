import os
import cv2
import numpy as np
import pandas as pd
import HandTrackingModule as htm
import json

def func():

    wCam, hCam = 1280, 720
    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)

    detector = htm.handDetector(maxHands=2, maxFaces=1, detectionCon=0.7)
    i = 100
    dir = "./my_data/9/"
    os.makedirs(dir, exist_ok=True)

    while cap.isOpened():
        success, img = cap.read()
        img = cv2.flip(img, flipCode=1)
        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)

        if len(lmList) != 0:
            with open(dir + f"{i}.txt", "a") as writer:
                lmList = json.dumps(lmList)
                writer.write(lmList)

            cv2.imshow("Img", img)
            cv2.imwrite(dir + f"{i}.png", img)
            i += 1
        print(i)

        if cv2.waitKey(500) == 27 or i == 150:
            break

def insert_csv(data):
    df = pd.DataFrame(data)
    df.to_csv('./my_res.csv',encoding = "GBK", header=0, index=0)

def save_csv():
    lines = []
    for i in range(10):
        path = f"./my_data/{i}/"
        for file in os.listdir(path):
            if file.endswith("txt"):
                with open(path + file, "r") as f:
                    line = f.read()
                    line = json.loads(line)
                    line = np.array(line)[:, 1:]
                    line = line.reshape(1, -1)
                    line = np.c_[line, np.ones(1) * (i)]
                    lines.append(line)
    lines = np.array(lines).squeeze(1)
    print(lines)
    print(lines.shape)
    insert_csv(lines)

if __name__ == '__main__':
    # func()
    save_csv()