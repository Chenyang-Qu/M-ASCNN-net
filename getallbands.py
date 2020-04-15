import random
import os
import numpy as np
import gdal
from tqdm import tqdm
import cv2
image_sets = ['1.tif']
label_sets = ['1.png'] 
# 12谱段遥感图像

def args_parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("-ch", "--channel", required=Flase,
                    help="please input channel", default = 12)
    args = vars(ap.parse_args())
    return args


img_w = 513
img_h = 513
# 输入图像长宽
def OpenTif(inpath):
    ds=gdal.Open(inpath)
    row=ds.RasterXSize
    col=ds.RasterYSize
    band=ds.RasterCount
    data=np.zeros([col,row, channel])
    for i in range(channel):
        dt=ds.GetRasterBand(i)
        data[:,:,i-1]=dt.ReadAsArray(0,0,row,col)
    return data
# 存成npy格式

def creat_dataset(image_num=10000):
    print('creating dataset...')
  #  image_each = [image_num /len(image_sets), image_num /len(image_sets)]
   # print(image_each)
    g_count = 0
    for i in tqdm(range(len(image_sets))):
        count = 0
        src_img = OpenTif('/home/ubuntu//S1/allbands/' + image_sets[i])  #  channels
        label_img = cv2.imread('/home/ubuntu//S1/allbands/' + label_sets[i],cv2.IMREAD_GRAYSCALE)  # single channel
        X_height, X_width, _ = src_img.shape
        # while count < image_each[i]:
        while count < (image_num):
            random_width = random.randint(0, X_width - img_w - 1)
            random_height = random.randint(0, X_height - img_h - 1)
            src_roi = src_img[random_height: random_height + img_h, random_width: random_width + img_w, :]
            label_roi = label_img[random_height: random_height + img_h, random_width: random_width + img_w]

            visualize = np.zeros((img_w, img_h)).astype(np.uint8)
            visualize = label_roi * 50

            np.save(('/home/ubuntu//S1/train1/visualize/%d.npy' % g_count),visualize)
            np.save(('/home/ubuntu//S1/train1/src/%d.npy' % g_count), src_roi)
            np.save(('/home/ubuntu//S1/train1/label/%d.npy' % g_count), label_roi)
            count += 1
            g_count += 1
            # print(g_count)


# a=np.load('D:/Sentinel2/CLIP/allbands/a.npy')
# print(a)

def gamma_transform(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)

def random_gamma_transform(img, gamma_vari):
    log_gamma_vari = np.log(gamma_vari)
    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
    gamma = np.exp(alpha)
    return gamma_transform(img, gamma)
    

def rotate(xb,yb,angle):
    M_rotate = cv2.getRotationMatrix2D((img_w/2, img_h/2), angle, 1)
    xb = cv2.warpAffine(xb, M_rotate, (img_w, img_h))
    yb = cv2.warpAffine(yb, M_rotate, (img_w, img_h))
    return xb,yb
    
def blur(img):
    img = cv2.blur(img, (3, 3));
    return img

def add_noise(img):
    for i in range(200): #添加点噪声
        temp_x = np.random.randint(0,img.shape[0])
        temp_y = np.random.randint(0,img.shape[1])
        img[temp_x][temp_y] = 255
    return img
    
    
def data_augment(xb,yb):
    if np.random.random() < 0.25:
        xb,yb = rotate(xb,yb,90)
    if np.random.random() < 0.25:
        xb,yb = rotate(xb,yb,180)
    if np.random.random() < 0.25:
        xb,yb = rotate(xb,yb,270)
    if np.random.random() < 0.25:
        xb = cv2.flip(xb, 1)  # flipcode > 0：沿y轴翻转
        yb = cv2.flip(yb, 1)
        
    if np.random.random() < 0.25:
        xb = random_gamma_transform(xb,1.0)
        
    if np.random.random() < 0.25:
        xb = blur(xb)
    
    if np.random.random() < 0.2:
        xb = add_noise(xb)
        
    return xb,yb

def creat_dataset(image_num = 100000, mode = 'original'):
    print('creating dataset...')
    image_each = [ image_num / (3+3) * 3, image_num / (3+3) * 3]
    print(image_each)
    g_count = 0
    for i in tqdm(range(len(image_sets))):
        count = 0
        src_img = cv2.imread('/home/ubuntu//S1/exset/' + image_sets[i])  # 3 channels
        label_img = cv2.imread('/home/ubuntu//S1/exset/lab/' + image_sets[i],cv2.IMREAD_GRAYSCALE)  # single channel
        X_height,X_width,_ = src_img.shape
        while count < image_each[i]:
            random_width = random.randint(0, X_width - img_w - 1)
            random_height = random.randint(0, X_height - img_h - 1)
            src_roi = src_img[random_height: random_height + img_h, random_width: random_width + img_w,:]
            label_roi = label_img[random_height: random_height + img_h, random_width: random_width + img_w]
            if mode == 'augment':
                src_roi,label_roi = data_augment(src_roi,label_roi)
            
            visualize = np.zeros((img_w,img_h)).astype(np.uint8)
            visualize = label_roi *50
            
            cv2.imwrite(('/home/ubuntu//S1/train/visualize/%d.png' % g_count),visualize)
            cv2.imwrite(('/home/ubuntu//S1/train/src/%d.png' % g_count),src_roi)
            #print('ok')
            cv2.imwrite(('/home/ubuntu//S1/train/label/%d.png' % g_count),label_roi)
            count += 1 
            g_count += 1
        print(g_count)


            
    

if __name__=='__main__':  
    creat_dataset(mode='augment')
