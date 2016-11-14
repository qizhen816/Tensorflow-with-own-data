import cv2
import numpy as np
import random
import gzip
import pickle


class DateP(object):
    def __init__(self, in1,in2, in3):
        self.in1 = in1
        self.in2 = in2
        self.in3 = in3

    def date_process(self):
        label = []
        datall = []
        print('reading')
        with open(self.in1) as f:  # 读取txt里面的文件名 read train.txt
            for line in f:
                a = line.strip().split(' ')[0]
                b = line.strip().split(' ')[1]
                label.append(str(int(b)-3)) #一共5类
                dest = 're/'+a  # 路径 plus to get path of images
                data = data_pro(dest, 0, 0, 384, 256)  # 获取数据位置信息
                datall.append(data)
        with open(self.in2) as f:  # 读取txt里面的文件名 read test.txt
            for line in f:
                a = line.strip().split(' ')[0]
                b = line.strip().split(' ')[1]
                label.append(str(int(b)-3)) #一共5类
                dest = 're/'+a  # 路径to get path of images
                data = data_pro(dest, 0, 0, 384, 256)  # 获取数据位置信息 original size
                datall.append(data)

        labels = np.asarray(label, dtype=int)
        img = np.asarray(datall, dtype=float)
        img /= 255
        lens = len(datall)
        img = img.reshape(lens, self.in3)

        toZip = list(zip(img, labels))
        random.shuffle(toZip)
        datas, labelss = map(list, zip(*toZip))
        Data = np.asarray(datas, dtype=float)
        Lable = np.asarray(labelss, dtype=int)
        All = Data, Lable
        return All


def data_pro(src, x1, y1, x2, y2):
    img_ = cv2.imread(src)
    dd = img_[x1:x2, y1:y2]  # 获取roi
    size = (28, 28)  # resize尺寸
    img = cv2.resize(dd, size)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换成灰度图像 rgb2grey
    bb = np.zeros((img.shape[0], img.shape[1]), dtype=img.dtype)
    bb[:, :] = gray[:, :]
    cc = bb.reshape(1, img.shape[0] * img.shape[1])
    return cc


if __name__ == '__main__':
    file1 = "re/train.txt"  # train.txt file path, should have 'imagepath label' as each row in txt
    file2 = "re/test.txt"   #text.txt same row
    Obj1 = DateP(file1,file2, 784) #train test and the total size of one image (28*28)
    O1 = Obj1.date_process()
    d = O1
    p1 = pickle.dumps(d, 2)  # 生成pkl.gz文件就和theano中的一样
    s = gzip.open('cnns.pkl.gz', 'wb')  # save as .gz
    print('writing...')
    s.write(p1)
    s.close()
    print('ok')
"""
    f = open('cnns.pkl', 'wb') #save as .pkl
    cPickle.dump(d, f, 2)
    f.close()
"""