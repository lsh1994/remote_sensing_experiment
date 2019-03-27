# 整合原始数据
from keras.models import load_model
from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from tqdm import tqdm


class Tiff:

    tif_src = r"E:\TeachersExperiment\jx_3\ZY3_GS_jiangxia1.tif"

    label_dict = {'bf': 0, 'br': 1, 'cc': 2, 'cx_b': 3, 'cx_gw': 4, 'cx_rw': 5, 'dp': 6, 'gf': 7, 'gh': 8,
                  'gr': 9, 'greenh': 10, 'lt': 11, 'of': 12, 'ptc': 13, 'rf': 14, 'st': 15, 'wm': 16,
                  'wr': 17, 'wt': 18, 'xkc': 19}


    def __init__(self,fea_src,bands,size):
        """

        :param fea_src: 坐标-标签文件，字符串
        :param bands: 波段，列表
        :param label_dict: 标签，字典
        :param size: 切块大小，整型
        """
        self.dataset = gdal.Open(Tiff.tif_src)  # tif数据

        self.fea=pd.read_csv(fea_src,header=None)

        self.bands=bands

        self.size=size

        # label_dict={}
        # label=list(set(self.fea.iloc[:,-2]))
        # for i in range(len(label)):
        #     label_dict[label[i]]=i
        # self.label_dict=label_dict
        # print(self.label_dict)

    def get_cell(self, pos_x, pos_y):
        try:
            output = []
            for i in self.bands:
                band = self.dataset.GetRasterBand(i)
                if(int(pos_x - self.size / 2)<0 or int(pos_y - self.size / 2)<0
                        or int(pos_x - self.size / 2)+self.size>self.dataset.RasterXSize
                        or int(pos_y - self.size / 2)+self.size>self.dataset.RasterYSize):
                    return None
                t = band.ReadAsArray(int(pos_x - self.size / 2), int(pos_y - self.size / 2), self.size, self.size)
                output.append(t)
            img = np.moveaxis(np.array(output, dtype=np.uint8), 0, 2)
            # print(img2.shape)
        except:
            return None
        return img

    def get_cells(self, pos_start, pos_end):
        imgs = []
        labels=[]

        for i in range(pos_start,pos_end):
            temp=self.fea.iloc[i,:].values
            img = self.get_cell(temp[1], temp[0])
            if img is None:
                continue
            imgs.append(img)
            label=np.zeros(20)
            # print(label)
            label[self.label_dict[temp[2]]]=1
            labels.append(label)
        return imgs,labels


    def get_beatch(self, batch_size=64):
        fea_len=len(self.fea)
        print("len fea:%d"% fea_len)


        while (True):
            for i in range(0, fea_len, batch_size):
                pos_end=min(i+batch_size,fea_len)
                imgs, labels=self.get_cells(i,pos_end)

                imgs = np.array(imgs)
                labels = np.array(labels)

                # print("返回批量数据...")
                yield imgs, labels

    # 在该数据集上测试
    def test_in_test(self,model_src):
        model = load_model(model_src)

        fea_len = len(self.fea)
        # print("test dataset length: %d " % fea_len)

        keys=list(self.label_dict.keys())

        true_labels=[]
        predict_labels=[]


        for i in tqdm(range(fea_len)):
            imgs, labels = self.get_cells(i, i + 1)

            imgs = np.array(imgs)
            if(len(imgs)==0):
                # print("pos: ",i,"out of range.")
                continue

            true_labels.append(self.fea.iloc[i,-1])
            s=np.argmax(model.predict(imgs)[0])
            # print("predict label pos: ",keys[s],"true labels: ",self.fea.iloc[i,-2])
            predict_labels.append(keys[s])

        res=[]
        res.append(true_labels)
        res.append(predict_labels)

        res=pd.DataFrame(np.array(res).T)
        res.to_csv("output/output.txt",header=["tclass2","pclass2"],index=None)


if __name__ == '__main__':
    # tiff=Tiff(fea_src="output/train_pos_label.txt",bands=[3,2,1],size=9)
    # t=tiff.get_beatch(10).__next__()
    # print(t[0].shape,t[1].shape)

    out_validate_tiff = Tiff(fea_src="output/validate_pos_label.txt", bands=[4,3,2,1], size=9)
    out_validate_tiff.test_in_test(model_src="output/four_bands_DLnet_model.h5")

