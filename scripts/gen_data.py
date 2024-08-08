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
    
    #===AoS===
    # rays = np.array(rays).reshape(-1, 3)
    # # 将原有每个ray的xyz—>xyzw，w=0 保证数据是32B的倍数
    # rays = np.concatenate([rays[:, :3], np.zeros((rays.shape[0], 1)), rays[:, 3:]], axis=1)
    
    # print(rays.shape)


    # # rays 的 xyzw 按照SoA的方式存储
    # rays = rays.T


    # ===SOA===
    # ray xyz dx dy dz
    rays = np.array(rays).reshape(-1, 6)
    # print(rays.shape)
    rays = rays.T

    rays.astype(np.float16).tofile("./input/rays.bin")
    # print(rays.shape)

'''
Sphere spheres[] = {//Scene: radius, position, emission, color, material 
   Sphere(1e5, Vec( 1e5+1,40.8,81.6), Vec(),Vec(.75,.25,.25),DIFF),//Left 
   Sphere(1e5, Vec(-1e5+99,40.8,81.6),Vec(),Vec(.25,.25,.75),DIFF),//Rght 
   Sphere(1e5, Vec(50,40.8, 1e5),     Vec(),Vec(.75,.75,.75),DIFF),//Back 
   Sphere(1e5, Vec(50,40.8,-1e5+170), Vec(),Vec(),           DIFF),//Frnt 
   Sphere(1e5, Vec(50, 1e5, 81.6),    Vec(),Vec(.75,.75,.75),DIFF),//Botm 
   Sphere(1e5, Vec(50,-1e5+81.6,81.6),Vec(),Vec(.75,.75,.75),DIFF),//Top 
   Sphere(16.5,Vec(27,16.5,47),       Vec(),Vec(1,1,1)*.999, SPEC),//Mirr 
   Sphere(16.5,Vec(73,16.5,78),       Vec(),Vec(1,1,1)*.999, REFR),//Glas 
   Sphere(600, Vec(50,681.6-.27,81.6),Vec(12,12,12),  Vec(), DIFF) //Lite 
 }; 
'''
def gen_spheres():
    spheres = np.array([])
    spheres = np.append(spheres, [1e5, 1e5+1, 40.8, 81.6,   0,0, 0,   0.75, 0.25, 0.25])  
    spheres = np.append(spheres, [1e5, -1e5+99, 40.8, 81.6,   0,0, 0,   0.25, 0.25, 0.75])
    spheres = np.append(spheres, [1e5, 50, 40.8, 1e5,   0,0, 0,   0.75, 0.75, 0.75])
    spheres = np.append(spheres, [1e5, 50, 40.8, -1e5+170,   0,0, 0,   0, 0, 0])
    spheres = np.append(spheres, [1e5, 50, 1e5, 81.6,   0,0, 0,   0.75, 0.75, 0.75])
    spheres = np.append(spheres, [1e5, 50, -1e5+81.6, 81.6,   0,0, 0,   0.75, 0.75, 0.75])
    spheres = np.append(spheres, [16.5, 27, 16.5, 47,   0,0, 0,   0.999, 0.999, 0.999])
    # spheres = np.append(spheres, [16.5, 73, 16.5, 78,   0,0, 0,   0.999, 0.999, 0.999])
    spheres = np.append(spheres, [600, 50, 681.6-0.27, 81.6,   12, 12, 12,   0, 0, 0])

    # change spheres r -> r^2
    num = spheres.shape[0]
    spheres[0::num] = np.power(spheres[0::num], 2)


    spheres = spheres.reshape(-1, num)
    spheres = spheres.T
    # print(spheres.shape)
    # print(spheres)
    spheres.astype(np.float16).tofile("./input/spheres.bin")

    
if __name__ == "__main__":
    gen_rays(width, height, samples)
    gen_spheres()
