#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright 2022-2023 Huawei Technologies Co., Ltd
import numpy as np

width = 16
height = 16
samples = 1
eps = 1e-4


def PrintTensor(name,tensor , itemPerLine=8):
    print(name, "shape: ", tensor.shape)
    for i in range(0, tensor.shape[0], itemPerLine):
        print("\t",tensor[i:i+itemPerLine])


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
    spheres = np.array([],dtype=np.float32) # 8 spheres
    spheres = np.append(spheres, [1e5, 1e5+1, 40.8, 81.6,   0,0, 0,   0.435, 0.376, 0.667])  # radius, x, y, z, emission xyz, color xyz
    spheres = np.append(spheres, [1e5, -1e5+99, 40.8, 81.6,   0,0, 0,   0.667, 0.129, 0.086])
    spheres = np.append(spheres, [1e5, 50, 40.8, 1e5,   0,0, 0,   0.270, 0.725, 0.486])
    spheres = np.append(spheres, [1e5, 50, 40.8, -1e5+170,   0,0, 0,   0, 0, 0]) # Front Dark
    spheres = np.append(spheres, [1e5, 50, 1e5, 81.6,   0,0, 0,  1.0, 0.902, 0.00])
    spheres = np.append(spheres, [1e5, 50, -1e5+81.6, 81.6,   0,0, 0,   0.141, 0.408, 0.635])
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
    # spheres.astype(np.float32).tofile("./input/spheres.bin")
    # binary_spheres = spheres
    # # 数据类型为Float32，末尾填充0 保证是512B的倍数
    # print("spheres shape: ",spheres.shape)
    current_size =  spheres.size * 4
    # print("spheres size: ",current_size,"sphere itemsize: ", 4,"sphere size: ",spheres.size)
    padding_size = 512 - current_size % 512
    padding_elements =  padding_size // 4  
    # print("padding count: ",padding_elements)
    binary_spheres = spheres.reshape(-1)
    binary_spheres = np.append(binary_spheres, np.zeros(padding_elements))
    binary_spheres.astype(np.float32).tofile("./input/spheres.bin")
    # print("binary_spheres shape: ",binary_spheres.shape)


    
    return spheres.T

def test_scene(rays, spheres):
    # print("test scene ray shape",rays.shape)
    # print("test scene sphere shape: ",spheres.shape)

    ret = np.zeros((rays.shape[0], 3), dtype=np.float32)
    # print("ret", ret.shape) 

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

def sim_npu(rays,spheres,k,j,block_len,tilings_len,depth):
    start = k * block_len + j * tilings_len
    end = k * block_len + j * tilings_len +  tilings_len

    stage1val = np.zeros((spheres.shape[1],tilings_len),dtype=np.float32) # spheres * tilings_len

    # print("sim npu:", "start: ",start," end: ",end," block_len: ",block_len," tilings_len: ",tilings_len,k,j)

    for i in range(0,spheres.shape[1]): # shape(10,8)
        # print("Current Sphere: ",i)
        
        # print("ray range: ",start, " to ", end)
        cur_rays = rays[:,start:end]

        ocX = spheres[1,i] - cur_rays[0,:]
        ocY = spheres[2,i] - cur_rays[1,:]
        ocZ = spheres[3,i] - cur_rays[2,:]

        b = ocX * cur_rays[3,:] + ocY * cur_rays[4,:] + ocZ * cur_rays[5,:]
        c = ocX * ocX + ocY * ocY + ocZ * ocZ - spheres[0,i]
        # if(k ==0 and j==0 and cur_buffer==0 and i == 0 and depth==1):
        #     # print("sphere: ",i," ocX: \n\t",ocX)
        #     # print("sphere: ",i," ocY: \n\t",ocY)
        #     # print("sphere: ",i," ocZ: \n\t",ocZ)
        #     PrintTensor("ray dx: ",cur_rays[3,:])
        #     PrintTensor("ray dy: ",cur_rays[4,:])
        #     PrintTensor("ray dz: ",cur_rays[5,:])
        #     print("sphere: ",i," b: \n\t",b)
            # print("sphere: ",i," c: \n\t",c)

        det = b * b - c

        # kernel0 tiling0 cur buffer0 sphere0 depth1
        # if(k ==0 and j==0 and cur_buffer==0 and i == 0 and depth==1):
        #     print("sphere: ",i," det: \n\t",det)


        detsqrt = np.sqrt(det)

        t0 = b - detsqrt
        t1 = b + detsqrt

        # print("kernal: ",k," tiling: ",j," buffer: ",0," sphere: ",i," depth: ",depth,"start: ",start," end: ",end)
        # if t0 > eps select t0 else select t1

        # if(k ==0 and i == 0 and cur_buffer==0 and depth==1):
        #     print("sphere: ",i," t0: ",t0)
        #     print("sphere: ",i," t1: ",t1)

        stage1val[i] = np.where((t0 > eps), t0, t1)
        # force set stage1val where item < eps to 1e20
        stage1val[i] = np.where((stage1val[i] > eps), stage1val[i],1e20)
    
    return stage1val


def test_soa(rays,spheres):
    rays = rays.astype(np.float32)
    spheres = spheres.astype(np.float32)

    rays = rays.reshape(-1,6).T
    spheres = spheres.reshape(-1,10).T
    # print("rays shape",rays.shape)


    core_num = 8
    tilings_len = 64
    block_len = rays.shape[1] // core_num

    assert rays.shape[1] % core_num == 0

    tilings_num = block_len // tilings_len
    # print("block_len",block_len)
    # print("tilings_num",tilings_num)

    ret_color = np.ones((rays.shape[1],3),dtype=np.float32)
    mask = np.ones((rays.shape[1]),dtype=np.float32)
    # print("mask shape",mask.shape)

    bound = 0
    while bound < 5:
        # print("bound: ===========================",bound)
        stage1val = np.zeros((spheres.shape[1],rays.shape[1]),dtype=np.float32)
        # print("stage1val shape",stage1val.shape)


        for k in range(0,core_num):
            for j in range(0,tilings_num):
                    tmp = sim_npu(rays,spheres,k,j,block_len,tilings_len,bound)

                    # 合并tmp到stage1val
                    start = k * block_len + j * tilings_len  
                    end = k * block_len + j * tilings_len +  tilings_len 

                    stage1val[:,start:end] = tmp
                    # print ("start ",start," end ",end)


        # if bound==1:
        #     print("stage1val shape",stage1val.shape)
        #     print("stage1val sphere0:",stage1val[0,:8])
        #     print("stage1val sphere1:",stage1val[1,64:72])
        #     print("stage1val sphere2:",stage1val[2,128:136])


        stage1val = stage1val.T
        # print("stage1val T",stage1val)
        # print("stage1val shape",stage1val.shape)
        # print("Sphere Nums of Hit Info")
        # for i in range(0,8):
        #     print("\t",end="")
        #     for j in range(0,8):
        #         print(stage1val[i,j],end=" ")
        #     print()



        # stage2 reduce min_distance and sphere_id
        reduce_ret = np.zeros((rays.shape[1],3),dtype=np.float32)
        for i in range(0,rays.shape[1]):
            min_distance = 1e20
            sphere_id = -1
            count = -1
            for j in range(0,spheres.shape[1]):
                if stage1val[i,j] == min_distance:
                    count += 1
                if stage1val[i,j] < min_distance:
                    min_distance = stage1val[i,j]
                    sphere_id = j
                    count = 1
                
            reduce_ret[i] = np.array([min_distance,sphere_id,count])
        
        #print all reduce_ret | but np only print part of it
        # print("reduce_ret shape: ",reduce_ret.shape)
        # print("reduce_ret: (minT)\n\t",reduce_ret[:8,0],"\n\t",reduce_ret[8:16,0])
        # print("reduce_ret: (sphere_id)\n\t",reduce_ret[:8,1],"\n\t",reduce_ret[8:16,1])

        # multi_same = 0
        # for i in range(0,reduce_ret.shape[0]):
        #     if(reduce_ret[i][2] > 1):
        #         multi_same += 1
        #         print("multi_same: ",reduce_ret[i])
        # print("multi_same count: ",multi_same)

        # stage3 compute new ray pos and direction
        new_ray = np.zeros((rays.shape[1],6),dtype=np.float32)

        # normalArray = np.zeros((rays.shape[1],3),dtype=np.float32)
        for i in range(0,reduce_ret.shape[0]):
            new_ray[i,0:3] = rays[0:3,i] + rays[3:6,i] * reduce_ret[i,0] # pos = pos + dir * min_distance

            # compute normal and direction
            normal = new_ray[i,0:3] - spheres[1:4,int(reduce_ret[i,1])] # pos - sphere_pos

            # normalArray[i] = normal

            normalize = normal / np.linalg.norm(normal) # normalize
           
            new_ray[i,3:6] = rays[3:6,i] - 2 * np.dot(rays[3:6,i],normalize) * normalize # dir = dir - 2 * dot(dir,normal) * normal
            

        # print("normalArray shape: ",normalArray.shape)
        # normalizeLen = np.linalg.norm(normalArray,axis=1)

        # normalizeArray = normalArray / normalizeLen[:,None]
        # print("normalizeArray shape: ",normalizeArray.shape)
        # if bound ==0 :
        #     r = 16
        #     PrintTensor("normalX ", normalArray[:r,0])
        #     PrintTensor("normalY ", normalArray[:r,1])
        #     PrintTensor("normalZ ", normalArray[:r,2])
        #     PrintTensor("normalizeX ", normalizeArray[:r,0])
        #     PrintTensor("normalizeY ", normalizeArray[:r,1])
        #     PrintTensor("normalizeZ ", normalizeArray[:r,2])
        #     PrintTensor("normalXYZ2Sum ", normalXYZ2Sum[:r])
        #     PrintTensor("normalizeLen ", normalizeLen[:r])

        # new_ray = new_ray.T
        # print("new_ray shape: ",new_ray.shape)
        # print("new_ray: ")
        # for i in range(0,10):
        #     print("idx: ",i,"xyz:",new_ray[i,0:3],"dir:",new_ray[i,3:6])

        # for i in range(0,10):
        #     print("idx: ",i,"xyz:",rays[0:3,i],"dir:",rays[3:6,i])
        

       
        cur_mask = np.ones((rays.shape[1]),dtype=np.float32)
        for i in range(0,ret_color.shape[0]):
            if reduce_ret[i,1] == 7:
                cur_mask[i] = 0
        # print("cur_mask shape: ",cur_mask.shape)
        # mix with mask
        mask = mask * cur_mask
        # PrintTensor("mask",mask)

        for i in range(0,reduce_ret.shape[0]):
            if mask[i] != 0:
                ret_color[i] = ret_color[i] * spheres[7:10,int(reduce_ret[i,1])]
            


        new_ray = new_ray.T
        rays = new_ray
        bound += 1
    # print("new_ray shape: ",new_ray.shape)
    # print("new_ray x:",new_ray[0,:10])
    # print("new_ray y:",new_ray[1,:10])
    # print("new_ray z:",new_ray[2,:10])
    # print("new_ray dx:",new_ray[3,:10])
    # print("new_ray dy:",new_ray[4,:10])
    # print("new_ray dz:",new_ray[5,:10])

    # stage4 compute color
   
    ret_color = ret_color * np.array([12,12,12],dtype=np.float32)
    # print("diffuse colorX")
    # for i in range(0,64,8):
    #     print(ret_color[i:i+8,0])
    # print("diffuse colorY")
    # for i in range(0,64,8):
    #     print(ret_color[i:i+8,1])
    # print("diffuse colorZ")
    # for i in range(0,64,8):
    #     print(ret_color[i:i+8,2])
    
    


    ret_color = ret_color.T
    ret_color.astype(np.float32).tofile("./output/test_soa.bin")
    





if __name__ == "__main__":
    np.set_printoptions(formatter={'float_kind': lambda x: f"{x:.3f}"}, precision=2)
    np.random.seed(0)
    # np.set_printoptions(threshold=8)  

    rays = gen_rays(width, height, samples)
    spheres = gen_spheres()
    # test_scene(rays, spheres)
    test_soa(rays, spheres)

    print("===========Python Script Done=============")
