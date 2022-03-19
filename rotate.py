import math
import torch
import os
import json
import random
import copy

import matplotlib.pyplot as plt

def Srotate(angle,valuex,valuey,pointx,pointy):

    sRotatex = (valuex-pointx)*math.cos(angle) + (valuey-pointy)*math.sin(angle) + pointx

    sRotatey = (valuey-pointy)*math.cos(angle) - (valuex-pointx)*math.sin(angle) + pointy

    return sRotatex, sRotatey

def func(angle, sx, sy, ctr_x, ctr_y):
    for idx, (src_x, src_y) in enumerate(zip(sx, sy)):
        sPointx ,sPointy = Srotate(math.radians(angle), src_x, src_y, ctr_x, ctr_y)
        if idx in [3, 7, 11, 15, 19]:
            plt.plot([ctr_x, src_x], [ctr_y, src_y], color="lightgreen")
            plt.plot([ctr_x, sPointx], [ctr_y, sPointy], color="orangered")
    plt.xlim(0., 1000)
    plt.ylim(0., 1000)
    plt.xticks(torch.arange(0, 1000, 50))
    plt.yticks(torch.arange(0, 1000, 50))
    plt.draw()
    plt.pause(1)
    plt.close()

class MyRotate(object):
    def __init__(self, theta=None):
        self.theta = theta

    def __call__(self, src: torch.Tensor):

        if not self.theta:
            self.theta = random.randint(0, 180)

        self.ctr_x, self.ctr_y = src[0, 0], src[0, 1]
        self.sx, self.sy = src[1:, 0], src[1:, 1]

        res = copy.deepcopy(src)

        for idx, (src_x, src_y) in enumerate(zip(self.sx, self.sy)):
            res[idx+1, 0] = (src_x - self.ctr_x) * math.cos(self.theta) + (src_y - self.ctr_y) * math.sin(self.theta) + self.ctr_x
            res[idx+1, 1] = (src_y - self.ctr_y) * math.cos(self.theta) - (src_x - self.ctr_x) * math.sin(self.theta) + self.ctr_y

        return res



if __name__ == '__main__':
    path = "./my_data/5/"
    for file in os.listdir(path):
        if file.endswith("txt"):
            with open(path + file, "r") as f:
                line = f.read()
                line = json.loads(line)
                hand = torch.tensor(line)[:, 1:]

                rot = MyRotate()
                new_hand = rot(hand)

                for idx in [4, 8, 12, 16, 20]:
                    plt.plot([hand[0, 0], hand[idx, 0]], [hand[0, 1], hand[idx, 1]], color="lightgreen")
                    plt.plot([new_hand[0, 0], new_hand[idx, 0]], [new_hand[0, 1], new_hand[idx, 1]], color="orangered")

                plt.xlim(0., 1000)
                plt.ylim(0., 1000)
                plt.xticks(torch.arange(0, 1000, 50))
                plt.yticks(torch.arange(0, 1000, 50))
                plt.draw()
                plt.pause(1)
                plt.close()

                # ctr_x, ctr_y = hand[0, 0], hand[0, 1]
                # sx, sy = hand[1:, 0], hand[1:, 1]
                # func(90, sx, sy, ctr_x, ctr_y)

