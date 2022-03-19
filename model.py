import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import joblib
import cv2
import math
import copy
import random
import HandTrackingModule as htm

class MyRotate(object):
    def __init__(self, theta=None):
        self.theta = theta

    def __call__(self, src: torch.Tensor):

        if not self.theta:
            self.theta = random.randint(0, 360)

        self.ctr_x, self.ctr_y = src[0, 0], src[0, 1]
        self.sx, self.sy = src[1:, 0], src[1:, 1]

        res = copy.deepcopy(src)

        for idx, (src_x, src_y) in enumerate(zip(self.sx, self.sy)):
            res[idx+1, 0] = (src_x - self.ctr_x) * math.cos(self.theta) + (src_y - self.ctr_y) * math.sin(self.theta) + self.ctr_x
            res[idx+1, 1] = (src_y - self.ctr_y) * math.cos(self.theta) - (src_x - self.ctr_x) * math.sin(self.theta) + self.ctr_y

        return res

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.length = label.shape[0]

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return self.length

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

def add_features(X):
    thumb_x = X[:, 8] - X[:, 0]
    thumb_y = X[:, 9] - X[:, 1]
    finger1_x = X[:, 16] - X[:, 0]
    finger1_y = X[:, 17] - X[:, 1]
    finger2_x = X[:, 24] - X[:, 0]
    finger2_y = X[:, 25] - X[:, 1]
    finger3_x = X[:, 32] - X[:, 0]
    finger3_y = X[:, 33] - X[:, 1]
    pinky_x = X[:, 40] - X[:, 0]
    pinky_y = X[:, 41] - X[:, 1]

    X = torch.cat([X, thumb_x.unsqueeze(1)], dim=1)
    X = torch.cat([X, thumb_y.unsqueeze(1)], dim=1)
    X = torch.cat([X, finger1_x.unsqueeze(1)], dim=1)
    X = torch.cat([X, finger1_y.unsqueeze(1)], dim=1)
    X = torch.cat([X, finger2_x.unsqueeze(1)], dim=1)
    X = torch.cat([X, finger2_y.unsqueeze(1)], dim=1)
    X = torch.cat([X, finger3_x.unsqueeze(1)], dim=1)
    X = torch.cat([X, finger3_y.unsqueeze(1)], dim=1)
    X = torch.cat([X, pinky_x.unsqueeze(1)], dim=1)
    X = torch.cat([X, pinky_y.unsqueeze(1)], dim=1)

    return X


def train(epoch, batch_size, lr):
    datadir = './my_res.csv'
    df = pd.read_csv(datadir)

    Y = df.iloc[:, -1]
    X = torch.FloatTensor(np.array(df.iloc[:, :-1]))
    Y = torch.LongTensor(Y)

    MyTransform = MyRotate()

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)

    ss = MinMaxScaler()
    X_train = torch.FloatTensor(ss.fit_transform(X_train))
    X_test = torch.FloatTensor(ss.transform(X_test))

    max_acc = 0

    print("data loading...")
    train_data = MyDataset(X_train, y_train)
    test_data = MyDataset(X_test, y_test)

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
    print("model loading...")

    model = MyModel(42, 36, 10)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    for i in range(epoch):
        print(f"Epoch{i+1}")

        model.train()
        for b, (x, y) in enumerate(train_dataloader):
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
                x = x.to(device)
                y = y.to(device)

            x = MyTransform(x)
            # x = add_features(x)
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            scheduler.step()

            if b % 50 == 0:
                print(f"train_loss: {loss:.5f}, lr: {optimizer.state_dict()['param_groups'][0]['lr']:.5f}")

        # test
        model.eval()
        test_loss = correct = total = 0
        with torch.no_grad():
            for n, (x, y) in enumerate(test_dataloader):
                if torch.cuda.is_available():
                    device = torch.device("cuda:0")
                    x = x.to(device)
                    y = y.to(device)
                x = MyTransform(x)
                # x = add_features(x)
                pred = model(x)
                test_loss += loss_fn(pred, y).item()
                _, predicted = torch.max(pred.data, 1)
                correct += predicted.eq(y.data).cpu().sum()
                total += y.size(0)
            print(f"test_loss: {test_loss / (n + 1):.5f}")
            print(f"acc: {100. * float(correct) / total:.5f}")
            if (float(correct) / total) >= max_acc:
                save_dir = "gesture.pth"
                print(f"Saving model to {save_dir}")
                torch.save(model, save_dir)
                max_acc = float(correct) / total
        model.train()

    print("Done.")

def test():
    datadir = './my_res.csv'
    df = pd.read_csv(datadir)

    Y = df.iloc[:, -1]
    X = torch.FloatTensor(np.array(df.iloc[:, :-1]))
    Y = torch.LongTensor(Y)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)

    ss = MinMaxScaler()
    X_train = torch.FloatTensor(ss.fit_transform(X_train))

    wCam, hCam = 1280, 720

    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)

    detector = htm.handDetector(maxHands=2, maxFaces=1, detectionCon=0.7)

    while cap.isOpened():
        success, img = cap.read()
        img = cv2.flip(img, flipCode=1)
        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)

        if len(lmList) != 0:
            # gesture
            data = []
            for item in lmList:
                data.append(item[1])
                data.append(item[2])

            vectors = torch.FloatTensor(ss.transform(np.array(data).reshape(1, -1)))

            if torch.cuda.is_available():
                device = torch.device("cuda:0")
                vectors = vectors.to(device)

            modelname = "./gesture.pth"
            if not os.path.exists(modelname):
                print('还未训练' + modelname)
                return
            else:
                model = torch.load(modelname)
                model.eval()
                prediction = model(vectors)
                cv2.putText(img, f'Prediction: {prediction.argmax()}', (10, 50), cv2.FONT_HERSHEY_COMPLEX,
                        1, (0, 255, 0), 2)
                cv2.putText(img, f'Confidence: {prediction}', (10, 100), cv2.FONT_HERSHEY_COMPLEX,
                        0.5, (0, 255, 0), 2)

        cv2.imshow("Img", img)
        if cv2.waitKey(1) == 27:
            break

def test2():

    wCam, hCam = 1280, 720

    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)

    detector = htm.handDetector(maxHands=2, maxFaces=1, detectionCon=0.7)

    while cap.isOpened():
        success, img = cap.read()
        img = cv2.flip(img, flipCode=1)
        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)

        if len(lmList) != 0:
            # gesture
            data = []
            for item in lmList:
                data.append(item[1])
                data.append(item[2])

            vectors = np.array(data).reshape(1, -1)
            model = joblib.load('lr.model')
            prediction = model.predict(vectors)
            cv2.putText(img, f'Prediction: {prediction}', (10, 50), cv2.FONT_HERSHEY_COMPLEX,
                    1, (0, 255, 0), 2)

        cv2.imshow("Img", img)
        if cv2.waitKey(1) == 27:
            break

if __name__ == '__main__':
    # train(epoch=200, batch_size=16, lr=0.001)
    test()
    # test2()
