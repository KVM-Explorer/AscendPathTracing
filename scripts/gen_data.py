#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright 2022-2023 Huawei Technologies Co., Ltd
import numpy as np

width = 16
height = 16
samples = 1
eps = 1e-4



def gen_rays(w, h, s):

    rays = []
    camera_pos = np.array([50, 52, 295.6])
    camera_dir = np.array([0, -0.042612, -1]) / np.linalg.norm([0, -0.042612, -1])
    camera = np.array([camera_pos, camera_dir])

    cx = np.array([w * 0.5135 / h, 0, 0])
    cy = np.cross(cx, camera[1]) / np.linalg.norm(np.cross(cx, camera[1])) * 0.5135
    print("camera: ", camera.shape)

    for i in range(w):
        for j in range(h):
            for sy in range(2):
                for sx in range(2):
                    for _ in range(s):
                        r1 = 2 * np.random.rand()
                        dx = np.sqrt(r1) - 1 if r1 < 1 else 1 - np.sqrt(2 - r1)
                        r2 = 2 * np.random.rand()
                        dy = np.sqrt(r2) - 1 if r2 < 1 else 1 - np.sqrt(2 - r2)
                        d = cx * (((sx + 0.5 + dx) / 2 + i) / w - 0.5) + \
                            cy * (((sy + 0.5 + dy) / 2 + j) / h - 0.5) + \
                            camera[1]
                        
                        ray_pos = camera[0] + d * 140
                        ray_dir = d / np.linalg.norm(d)
                        rays.append(np.concatenate([ray_pos, ray_dir]))
                        # if i < 10 and j < 10:
                        #     print("ray", rays[-1],"| d ",d)
    
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
    # for i in range(5):
    #     print("1.rays: ",rays[i])
    # print(rays.shape)
    rays = rays.T

    rays.astype(np.float32).tofile("./input/rays.bin")
    # print(rays.shape)
    rays = rays.T
    rays = rays.reshape(-1, 2, 3)
    return rays

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
    spheres = np.array([]) # 8 spheres
    spheres = np.append(spheres, [1e5, 1e5+1, 40.8, 81.6,   0,0, 0,   0.75, 0.25, 0.25])  # radius, x, y, z, emission xyz, color xyz
    spheres = np.append(spheres, [1e5, -1e5+99, 40.8, 81.6,   0,0, 0,   0.25, 0.25, 0.75])
    spheres = np.append(spheres, [1e5, 50, 40.8, 1e5,   0,0, 0,   0.75, 0.75, 0.75])
    spheres = np.append(spheres, [1e5, 50, 40.8, -1e5+170,   0,0, 0,   0, 0, 0])
    spheres = np.append(spheres, [1e5, 50, 1e5, 81.6,   0,0, 0,   0.75, 0.75, 0.75])
    spheres = np.append(spheres, [1e5, 50, -1e5+81.6, 81.6,   0,0, 0,   0.75, 0.75, 0.75])
    spheres = np.append(spheres, [16.5, 27, 16.5, 47,   0,0, 0,   0.999, 0.999, 0.999])
    # spheres = np.append(spheres, [16.5, 73, 16.5, 78,   0,0, 0,   0.999, 0.999, 0.999])
    spheres = np.append(spheres, [600, 50, 681.6-0.27, 81.6,   12, 12, 12,   0, 0, 0])

    # change spheres r -> r^2
    # print("sphere gen shape",spheres.shape)
    spheres = spheres.reshape(-1, 10)
    # print(spheres.shape)
    # print("spheres before", spheres)
    spheres[:, 0] = np.square(spheres[:, 0])
    # print("spheres after", spheres)


    spheres = spheres.T
    # print("sphere shape:",spheres.shape)
    # print(spheres)
    spheres.astype(np.float32).tofile("./input/spheres.bin")
    return spheres.T

def test_scene(rays, spheres):
    print("test scene ray shape",rays.shape)
    print("test scene sphere shape: ",spheres.shape)

    ret = np.zeros((rays.shape[0], 3), dtype=np.float32)
    print("ret", ret.shape) 

    rays = rays.astype(np.float32)
    spheres = spheres.astype(np.float32)

    # for i in range(3):
    #     print("sphere content:",np.round(spheres[i],2))

    for i,ray in enumerate(rays):
        min_distance = 1e20  
        sphere_id = -1
        for k,sphere in enumerate(spheres):
            # print(ray, sphere)
            op = sphere[1:4] - ray[0] # L = O - C
            # print("op",op)
            b = np.dot(op, ray[1]) # b = L * D
            b2 = b * b # TODO: remove
            c = np.dot(op, op) - sphere[0] # c = L^2 - r^2 TODO: remove
            det = b * b - np.dot(op, op) + sphere[0] # det = b^2 - L^2 + r^2
            if det < 0:
                continue
            # print(det)
            det = np.sqrt(det)
            t0 = b - det
            t1 = b + det
            #print("t0: ", t0, "t1: ", t1)

            if t0 > eps and t0 < min_distance:
                min_distance = t0
                sphere_id = k
            elif t1 > eps and t1 < min_distance:
                min_distance = t1
                sphere_id = k
            else:
                continue
        
        if sphere_id == -1:
            ret[i] = np.array([0, 0, 0])
        else:
            if sphere_id == 7:
                ret[i] = spheres[sphere_id, 4:7]
                
            else:
                ret[i] = spheres[sphere_id, 7:10]
        
        # if i % 100 == 0:
        #     print("hit sphere: ", sphere_id," ,min_dis: ", min_distance)
        # print()

    ret = ret.T
    #print("ret", ret.shape)
    ret.astype(np.float32).tofile("./output/test_scene.bin")

def sim_npu(rays,spheres,k,j,block_len,tilings_len):
    for i in range(0,spheres.shape[1]):
        cur_rays = rays[:, 
                        k * block_len + j * tilings_len + i: 
                        k * block_len + j * tilings_len + i + tilings_len]                
        ocX = spheres[1,i] - cur_rays[0,:]
        ocY = spheres[2,i] - cur_rays[1,:]
        ocZ = spheres[3,i] - cur_rays[2,:]

        b = ocX * cur_rays[3,:] + ocY * cur_rays[4,:] + ocZ * cur_rays[5,:]
        c = ocX * ocX + ocY * ocY + ocZ * ocZ - spheres[0,i]

        det = b * b - c

        detsqrt = np.sqrt(det)

        t0 = b - detsqrt
        t1 = b + detsqrt

        bitmask = (t0 > eps)
        # convert to uint8 使用bit位来表示是否命中
        
        return

def test_soa(rays,spheres):
    rays = rays.astype(np.float32)
    spheres = spheres.astype(np.float32)

    rays = rays.reshape(-1,6).T
    spheres = spheres.reshape(-1,10).T
    print("rays shape",rays.shape)


    core_num = 8
    tilings_len = 64
    block_len = rays.shape[1] // core_num
    tilings_num = block_len // tilings_len
    print("block_len",block_len)
    print("tilings_num",tilings_num)

    for k in range(0,core_num):
        for j in range(0,tilings_num):
            sim_npu(rays,spheres,k,j,block_len,tilings_len)




if __name__ == "__main__":
    np.set_printoptions(formatter={'float_kind': lambda x: f"{x:.5f}"}, precision=2)
    np.random.seed(0)
    rays = gen_rays(width, height, samples)
    spheres = gen_spheres()
    # test_scene(rays, spheres)
    # test_soa(rays, spheres)
