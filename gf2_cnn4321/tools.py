# 生成坐标-label文件

import seaborn as sns
import pandas as pd
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import gdal

from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score, accuracy_score


def pos_label(pos_src,label_src,target):
    if os.path.exists(target):
        print("target file hava exist.")
        return

    train=pd.read_csv(pos_src,header=None,sep=" ")
    simple=pd.read_csv(label_src,header=None,sep="\t")
    df=pd.concat([train,simple.iloc[:,-2]],axis=1)

    df=pd.DataFrame(df).sample(frac=1) #random_state=2018
    df.to_csv(target,index=None,header=None)
    print("together with pos and label finished with random.")

def get_2_1_labels():
    label_src="e:/TeachersExperiment/JiangXia_simplify/va1.txt"
    simple = pd.read_csv(label_src, header=None, sep="\t")
    dict={}# key: 二级,value: 一级
    for i in range(len(simple)):
        dict[simple.iloc[i,-2]]=simple.iloc[i,-1]
    print(dict)


def eval_res():
    res=pd.read_csv("output/output.txt")
    vec = {'bf': 0, 'br': 1, 'cc': 2, 'cx_b': 3, 'cx_gw': 4, 'cx_rw': 5, 'dp': 6, 'gf': 7, 'gh': 8,
                'gr': 9, 'greenh': 10, 'lt': 11, 'of': 12, 'ptc': 13, 'rf': 14, 'st': 15, 'wm': 16,
                'wr': 17, 'wt': 18, 'xkc': 19}

    # print(output[:5])

    true_label=res.iloc[:,0]
    predict_label=res.iloc[:,1]
    # print(confusion_matrix(true_label, predict_label))
    # print()
    print(classification_report(true_label, predict_label))
    print("kappa: ", cohen_kappa_score(true_label, predict_label))
    print("accuracy: ", accuracy_score(true_label, predict_label))
    mat = confusion_matrix(true_label, predict_label)

    f, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(mat, annot=True, square=True, fmt="d",ax=ax,xticklabels=list(vec.keys()),yticklabels=list(vec.keys()))
    plt.title("cnn4321")
    # sns.heatmap(mat, annot=True, square=True, fmt="d", ax=ax)
    plt.show()

def get_label2():
    dir = r"output\pos_label.txt"
    data = pd.read_csv(dir, header=None)
    vec=(list(set(data.iloc[:, 2])))
    vec.sort()
    dict={}
    for i in range(len(vec)):
        dict[vec[i]]=i
    print(dict)


def eval_res_yiji():
    dict_21={'wr': 'road', 'br': 'road', 'gr': 'road', 'wt': 'water', 'wm': 'water', 'dp': 'gengdi', 'gh': 'gengdi',
     'greenh': 'gengdi', 'st': 'gengdi', 'cx_rw': 'cx', 'cx_gw': 'cx',
     'cx_b': 'cx', 'bf': 'forest', 'rf': 'forest', 'gf': 'forest', 'of': 'forest', 'lt': 'unused', 'cc': 'mine',
     'xkc': 'mine', 'ptc': 'mine'}
    # sorted_x = sorted(dict_21.items(), key=operator.itemgetter(0))
    # for i in range(len(sorted_x)):
    #     print(i+1,sorted_x[i])
    # print(list(set(dict_21.values())))

    t2=[]
    p2=[]
    res = pd.read_csv("output/output.txt")

    for _,row in res.iterrows():
        # print(row[0],row[1])
        t2.append(dict_21[row[0]])
        p2.append(dict_21[row[1]])
    print(classification_report(t2, p2))
    print("kappa: ", cohen_kappa_score(t2, p2))
    print("accuracy: ", accuracy_score(t2, p2))
    # mat = confusion_matrix(t2, p2)
    #
    # f, ax = plt.subplots(figsize=(10, 8))
    # # sns.heatmap(mat, annot=True, square=True, fmt="d", ax=ax, xticklabels=list(vec.keys()),
    # #             yticklabels=list(vec.keys()))
    # sns.heatmap(mat, annot=True, square=True, fmt="d", ax=ax)
    # plt.show()

def ImgSplit():
    tif_src = r"e:\TeachersExperiment\JiangXia_simplify\ZY3_GS_jiangxia1.tif"
    dataset = gdal.Open(tif_src)  # tif数据
    # if not os.path.exists("output/original.png"):
    output = []
    for i in [3,2,1]:
        band = dataset.GetRasterBand(i)
        t = band.ReadAsArray(0, 0, dataset.RasterXSize, dataset.RasterYSize).astype(np.uint8)
        # print(t,type(t))
        print(type(band))
        output.append(t)
    output = np.moveaxis(np.array(output, dtype=np.uint8), 0, 2)
    print(output.shape)
    cv2.imwrite("output/original.png",output)

if __name__ == '__main__':
    # ImgSplit()
    # pos_label(r"E:\TeachersExperiment\jx_3\va_sample_1.txt", r"E:\TeachersExperiment\jx_3\va1.txt",
    #                 "output/validate_pos_label.txt")
    # pos_label(r"E:\TeachersExperiment\jx_3\tr_sample_1.txt", r"E:\TeachersExperiment\jx_3\tr1.txt",
    #                 "output/train_pos_label.txt")

    # eval_res_yiji()
    eval_res()
