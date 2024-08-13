import os
import sys
import numpy as np

width = 16
height = 16
samples = 1

# decode from color bin to PPM image

def write_ppm(w, h,data):
    with open("./output/color.ppm", "w") as f:
        f.write("P3\n{} {}\n255\n".format(w, h))
        for i in range(w):
            for j in range(h):
                f.write("{} {} {} ".format(data[j, i, 0], data[j, i, 1], data[j, i, 2]))
            f.write("\n")


def decode_color(w, h, s):
    # 读取color.bin文件 数据的格式为XYZ的Array
    # colors = np.fromfile("./output/color.bin", dtype=np.float32).reshape(w, h, 4*samples,4)

    # 读取color.bin文件 数据的格式为SoA的XYZ，转化为XYZ的Array
    colors = np.fromfile("./output/color.bin", dtype=np.float32).reshape(3, w, h, 4 * samples)
    # print(colors.shape)
    # print(colors)
    colors = colors.transpose(1, 2,3,0)
    
    # colors = colors[:, :, :,:3] # remove alpha channel


    # 从多个采样点取平均中提取一个像素信息
    # 数据由是w *h* 4 *s生成
    # 现在希望合并4*s的数据到w*h
    # 期望提取w *h的像素信息 rgb
    new_colors = np.zeros((w, h,3))
    for i in range(w):
        for j in range(h):
            u =h - j - 1
            pixel_values = colors[i, j, :, :]
            new_colors[i, j] = np.mean(pixel_values, axis=0)


    # 裁剪到0-1之间
    clips = np.clip(new_colors, 0, 1)
    ret = clips * 255
    ret = ret.astype(np.uint8)
    write_ppm(w, h, ret)
    return ret

def test_decode():
    w = 16
    h = 16
    s = 1
    # (16*16*4*1,4)  
    data = np.random.rand(w * h * 4 * s, 4).astype(np.float32)    
    data.tofile("./output/color.bin")

    output = decode_color(w, h, s)

    #compare data


    data = data[:, :3]
    new_data = np.zeros((w, h,3))
    for i in range(16):
        for j in range(16):
            idx = i * w + j
            st = idx * 4 * s
            ed = st + 4 * s
            pixel_values = data[st:ed]
            avg = np.mean(pixel_values, axis=0)
            new_data[i, j] = avg

    clips = np.clip(new_data, 0, 1)
    ret = clips * 255
    ret = ret.astype(np.uint8)

    assert np.allclose(ret, output)



if __name__ == "__main__":
    decode_color(width, height, samples)
    # test_decode()
