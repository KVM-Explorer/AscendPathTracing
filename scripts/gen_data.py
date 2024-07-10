#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright 2022-2023 Huawei Technologies Co., Ltd
import numpy as np

width = 16
height = 16
samples = 1

def gen_rays(w, h, s):

    rays = []
    camera = np.array([50, 52, 295.6]), np.array([0, -0.042612, -1]) / np.linalg.norm([0, -0.042612, -1])
    cx = np.array([w * 0.5135 / h, 0, 0])
    cy = np.cross(cx, camera[1]).reshape(3) * 0.5135 / np.linalg.norm(np.cross(cx, camera[1]))

    for i in range(w):
        for j in range(h):
            for sy in range(2):
                for sx in range(2):
                    for _ in range(s):
                        r1 = 2 * np.random.rand()
                        dx = np.sqrt(r1) - 1 if r1 < 1 else 1 - np.sqrt(2 - r1)
                        r2 = 2 * np.random.rand()
                        dy = np.sqrt(r2) - 1 if r2 < 1 else 1 - np.sqrt(2 - r2)
                        d = cx * ((sx + 0.5 + dx) / 2 + i) / w - 0.5 + \
                            cy * ((sy + 0.5 + dy) / 2 + j) / h - 0.5 + \
                            camera[1]
                        rays.append(np.array([camera[0], (d / np.linalg.norm(d)) * 140]))
    rays = np.array(rays).reshape(-1, 3)
    # 将原有每个ray的xyz—>xyzw，w=0 保证数据是32B的倍数
    rays = np.concatenate([rays[:, :3], np.zeros((rays.shape[0], 1)), rays[:, 3:]], axis=1)
    print(rays.shape)
    rays.astype(np.float32).tofile("./input/rays.bin")



if __name__ == "__main__":
    gen_rays(width, height, samples)
