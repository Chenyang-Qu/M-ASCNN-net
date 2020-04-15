import cv2
import random
import numpy as np
import os
import argparse
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import gdal

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

TEST_SET = ['input.tif']

image_size = 224

classes = [0.,1, 2, 3, 4, 5, 6, 7]

labelencoder = LabelEncoder()
labelencoder.fit(classes)

def OpenTif(inpath):
    ds=gdal.Open(inpath)
    row=ds.RasterXSize
    col=ds.RasterYSize
    band=ds.RasterCount
    data=np.zeros([col,row,channel])
    for i in range(channel):
        dt=ds.GetRasterBand(i)
        data[:,:,i-1]=dt.ReadAsArray(0,0,row,col)
    print(data.shape)
    return data

def args_parse():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True,
                    help="path to trained model model")
    ap.add_argument("-s", "--stride", required=False,
                    help="crop slide stride", type=int, default=224)
    args = vars(ap.parse_args())
    return args


def predict(args):
    # load the trained convolutional neural network
    print("[INFO] loading network...")
    model = load_model(args["model"])
    stride = args['stride']
    for n in range(len(TEST_SET)):
        path = TEST_SET[n]
        # load the image
        image = OpenTif('/home/ubuntu//S1/' + path)
        h, w, _ = image.shape
       # imagecopy=image
        padding_h = (h // stride + 1) * stride
        padding_w = (w // stride + 1) * stride
        padding_img = np.zeros((padding_h, padding_w, 12), dtype=np.uint8)
        padding_img[0:h, 0:w, :] = image[:, :, :]
        padding_img = padding_img.astype("float") / 255.0
        padding_img = img_to_array(padding_img)
        print('src:', padding_img.shape)
        mask_whole = np.zeros((padding_h, padding_w), dtype=np.uint8)
        #mask_whole[0:h, 0:w] = cv2.cvtColor(imagecopy, cv2.COLOR_RGB2GRAY)
        for i in range(padding_h // stride):
            for j in range(padding_w // stride):
                crop = padding_img[i * stride:i * stride + image_size, j * stride:j * stride + image_size, :12]
                ch, cw, _ = crop.shape
                if ch != image_size or cw != image_size:
                    print('invalid size!')
                    print(crop.shape)
                    continue


                crop = np.expand_dims(crop, axis=0)
                # print 'crop:',crop.shape
                pred = model.predict_classes(crop, verbose=2)
                pred = labelencoder.inverse_transform(pred[0])
                # print (np.unique(pred))
                pred = pred.reshape((image_size,image_size )).astype(np.uint8)
                # print 'pred:',pred.shape
                mask_whole[i * stride:i * stride + image_size, j * stride:j * stride + image_size] = pred[:, :]
            print(crop.shape, i)

        cv2.imwrite('/home/ubuntu//S1/a0.png', mask_whole[0:h, 0:w])

        # color_img = np.zeros((h, w, 3), dtype=np.uint8)
        # for h1 in range(h):
        #     for w1 in range(w):
        #         if (mask_whole[h1, w1] == 29):
        #             color_img[h1, w1, 0] = 138
        #         elif (mask_whole[h1, w1] == 41):
        #             color_img[h1, w1, 1] = 138
        #         elif (mask_whole[h1, w1] == 81):
        #             color_img[h1, w1, 2] = 254
        #         elif (mask_whole[h1, w1] == 139):
        #             color_img[h1, w1, 0] = 254
        #             color_img[h1, w1, 1] = 132
        #         elif (mask_whole[h1, w1] == 225):
        #             color_img[h1, w1, 0] = 254
        #             color_img[h1, w1, 1] = 254
        # cv2.imwrite('/home/ubuntu//S1/a1.png',color_img)




if __name__ == '__main__':
    args = args_parse()
    predict(args)
