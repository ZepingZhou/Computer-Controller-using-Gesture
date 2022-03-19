import cv2
import numpy as np
import HandTrackingModule as htm
import math
import qrcode
from pynput.keyboard import Key,Controller
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    """
    定义了一个简单的三层全连接神经网络，每一层都是线性的
    """

    def __init__(self, in_dim, n_hidden1, out_dim):
        super().__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden1)
        self.layer2 = nn.Linear(n_hidden1, n_hidden1)
        self.layer3 = nn.Linear(n_hidden1, out_dim)
        self.bn1 = nn.LayerNorm(n_hidden1)
        self.bn2 = nn.LayerNorm(n_hidden1)

    def forward(self, x):
        x = F.relu(self.bn1(self.layer1(x)))
        x = F.relu(self.bn2(self.layer2(x)))
        cls = self.layer3(x)

        return cls

def func():

    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    min_audio = volume.GetVolumeRange()[0]

    datadir = './my_res.csv'
    df = pd.read_csv(datadir)

    X = torch.FloatTensor(np.array(df.iloc[:, :-1]))
    ss = MinMaxScaler()
    ss.fit_transform(X)

    keyboard = Controller()
    wCam, hCam = 1280, 720

    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)

    detector = htm.handDetector(maxHands=2, maxFaces=1, detectionCon=0.7)

    length = last_length = 0

    qr_text = "Produced by ZePing Zhou; \n QQ: 773552508; \n Wechat: z773552508"

    mode = 0

    while cap.isOpened():

        success, img = cap.read()
        img = cv2.flip(img, flipCode=1)
        img = detector.findHands(img)
        img = detector.findFaces(img)
        lmList = detector.findPosition(img, draw=False)

        if mode != 4 and len(lmList) != 0:

            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]
            x0, y0 = lmList[9][1], lmList[9][2]

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            cv2.circle(img, (x1, y1), 15, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (0, 0, 255), cv2.FILLED)

            length = math.hypot(x2 - x1, y2 - y1)

            if length < 50:
                cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

            if 85 < x0 < 500 and 30 < y0 < 280:
                mode = 1
            elif length < 75 and 800 < x0 < 1200 and 30 < y0 < 280:
                mode = 2
            elif length < 75 and 85 < x0 < 500 and 380 < y0 < 630:
                mode = 3
            elif length < 75 and 800 < x0 < 1200 and 380 < y0 < 630:
                mode = 4

            # gesture
            data = []
            for item in lmList:
                data.append(item[1])
                data.append(item[2])

        # module 1
        if mode == 1 and len(lmList) != 0:
            if length < 50:
                keyboard.press(Key.media_volume_down)
                keyboard.release(Key.media_volume_down)
            elif length > 100:
                keyboard.press(Key.media_volume_up)
                keyboard.release(Key.media_volume_up)
            elif length > last_length + 5:
                keyboard.press(Key.media_volume_up)
                keyboard.release(Key.media_volume_up)
            elif length < last_length - 5:
                keyboard.press(Key.media_volume_down)
                keyboard.release(Key.media_volume_down)
            last_length = length
            cv2.rectangle(img, (85, 30), (500, 280), (0, 0, 255), 3)
            cv2.putText(img, 'volume:', (100, 100), cv2.FONT_HERSHEY_COMPLEX,
                        1, (30,105,210), 2)
            cv2.rectangle(img, (100, 150), (480, 180), (255, 255, 255), 2)
            cur_audio = volume.GetMasterVolumeLevel()
            cv2.rectangle(img, (100, 150), (int(100+(480-100)*(1-cur_audio/min_audio)), 180), (130, 0, 75), cv2.FILLED)
        elif mode == 2 and len(lmList) != 0:
            cv2.rectangle(img, (800, 30), (1200, 280), (0, 0, 255), 3)
            cv2.putText(img, 'shooting!', (850, 160), cv2.FONT_HERSHEY_COMPLEX,
                        2, (0, 0, 255), 2)
            keyboard.press(Key.print_screen)
            keyboard.release(Key.print_screen)
        elif mode == 3 and len(lmList) != 0:
            qr_img = qrcode.make(qr_text)
            qr_img = np.asarray(qr_img, dtype=np.uint8)[..., np.newaxis] * 255
            qr_img = np.repeat(qr_img, repeats=3, axis=2)
            w, h = qr_img.shape[:-1]
            img[150:150+w, 450:450+h, :] = qr_img
            cv2.rectangle(img, (85, 380), (500, 630), (0, 0, 255), 3)
        elif mode == 4:
            if len(lmList) != 0:
                data = []
                for item in lmList:
                    data.append(item[1])
                    data.append(item[2])

                vectors = torch.FloatTensor(ss.transform(np.array(data).reshape(1, -1)))

                if torch.cuda.is_available():
                    device = torch.device("cuda:0")
                    vectors = vectors.to(device)

                modelname = "./gesture.pth"
                model = torch.load(modelname)
                model.eval()
                prediction = model(vectors)
                prediction = F.softmax(prediction, dim=-1)
                idx, predicted = torch.max(prediction, 1)
                if idx > 0.6:
                    cv2.putText(img, f'Prediction: {predicted.item()}', (10, 80), cv2.FONT_HERSHEY_COMPLEX,
                                1.5, (0, 255, 0), 2)
                    cv2.putText(img, f'Confidence: {idx.item():.3f}', (10, 160), cv2.FONT_HERSHEY_COMPLEX,
                                1.5, (0, 255, 0), 2)
                    cv2.putText(img, "Enter eight to quit!", (10, 240), cv2.FONT_HERSHEY_COMPLEX,
                                1.5, (0, 0, 255), 2)
                else:
                    cv2.putText(img, 'Prediction: None', (10, 80), cv2.FONT_HERSHEY_COMPLEX,
                                1.5, (0, 255, 0), 2)
                if predicted.item() == 8:
                    mode = 0
        else:
            # mode 1
            cv2.rectangle(img, (85, 30), (500, 280), (144,238,144), 3)
            cv2.putText(img, 'volume:', (100, 100), cv2.FONT_HERSHEY_COMPLEX,
                        1, (30,105,210), 2)
            cv2.rectangle(img, (100, 150), (480, 180), (255, 255, 255), 2)
            cur_audio = volume.GetMasterVolumeLevel()
            cv2.rectangle(img, (100, 150), (int(100 + (480 - 100) * (1 - cur_audio / min_audio)), 180), (130, 0, 75),
                          cv2.FILLED)
            # mode 2
            cv2.rectangle(img, (800, 30), (1200, 280), (144,238,144), 3)
            cv2.putText(img, 'screen shoot!', (815, 100), cv2.FONT_HERSHEY_COMPLEX,
                        1, (30,105,210), 2)
            # mode 3
            cv2.putText(img, 'QR Code', (95, 450), cv2.FONT_HERSHEY_COMPLEX,
                        1, (30, 105, 210), 2)
            cv2.rectangle(img, (85, 380), (500, 630), (144,238,144), 3)
            # mode 4
            cv2.rectangle(img, (800, 380), (1200, 630), (144,238,144), 3)
            cv2.putText(img, 'gesture recognition!', (815, 450), cv2.FONT_HERSHEY_COMPLEX,
                        1, (30, 105, 210), 2)

        cv2.imshow("Img", img)
        if cv2.waitKey(1) == 27:
            break

if __name__ == '__main__':
    func()