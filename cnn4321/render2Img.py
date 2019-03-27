import gdal
from keras.models import load_model
from tqdm import tqdm
import numpy as np
import cv2

def get_cell(dataset,size,pos_x, pos_y):
    try:
        output = []
        for i in [4,3,2,1]:
            band = dataset.GetRasterBand(i)
            if (int(pos_x - size / 2) < 0 or int(pos_y - size / 2) < 0
                    or int(pos_x - size / 2) + size > dataset.RasterXSize
                    or int(pos_y - size / 2) + size > dataset.RasterYSize):
                return None
            t = band.ReadAsArray(int(pos_x - size / 2), int(pos_y - size / 2), size,size)
            output.append(t)
        img2 = np.moveaxis(np.array(output, dtype=np.uint8), 0, 2)
        # print(img2.shape)
    except:
        return None
    return img2


def label2Img(str,colors,label_src):
    # print(str)
    res = np.ones((100, 100*len(str)+40, 3), np.uint8) * 255
    for i, t in enumerate(colors):
        # print(t)
        t = tuple([int(t[0]), int(t[1]), int(t[2])])
        startx = 10 + 100 * i  # 间隔100
        starty=20
        res = cv2.rectangle(res, (startx, starty), (startx + 90, starty+20), t, thickness=-1)
        res = cv2.putText(res, str[i], (startx, starty+10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4,
                          color=(255,255,255),lineType=cv2.LINE_AA)
    cv2.imwrite(label_src, res)

def render_class2(name):
    tif_src = r"e:\TeachersExperiment\JiangXia_simplify\ZY3_GS_jiangxia1.tif"
    dataset = gdal.Open(tif_src)  # tif数据
    model = load_model("output/DLnet_model.h5")

    res=np.zeros(shape=(dataset.RasterXSize,dataset.RasterYSize,3),dtype=np.uint8)

    label=np.random.randint(0,255,size=(20,3))

    step=8
    for i in tqdm(range(16,dataset.RasterXSize,step)):

        for j in range(16,dataset.RasterYSize,step):
            # print(i,j)
            img=get_cell(dataset,32,i,j)
            if img is None:
                continue
            imgs=np.array([img])
            # print(imgs.shape)
            result=model.predict(imgs)[0]
            # if np.max(result)<0.7:#拒判
            #     continue
            s = np.argmax(result)
            # res[i-16:i+16,j-16:j+16]=int(255/21)*s
            res[i-16:i+16,j-16:j+16]=label[s]

    res = np.array([res[:,:,0],res[:,:,1],res[:,:,2]])

    vec = {'bf': 0, 'br': 1, 'cc': 2, 'cx_b': 3, 'cx_gw': 4, 'cx_rw': 5, 'dp': 6, 'gf': 7, 'gh': 8, 'gr': 9,
           'greenh': 10, 'lt': 11, 'of': 12, 'ptc': 13, 'rf': 14, 'st': 15, 'wm': 16, 'wr': 17, 'wt': 18, 'xkc': 19}
    label2Img(list(vec.keys()), label, "stay/render_class2_label_%d_%s.png" % (step,name))
    cv2.imwrite("stay/render_class2_%d_%s.png" % (step,name),res.T)

def render_class2_yiji(name):
    tif_src = r"e:\TeachersExperiment\JiangXia_simplify\ZY3_GS_jiangxia1.tif"
    dataset = gdal.Open(tif_src)  # tif数据
    model = load_model("output/DLnet_model.h5")
    dict_21 = {'wr': 'road', 'br': 'road', 'gr': 'road', 'wt': 'water', 'wm': 'water', 'dp': 'gengdi', 'gh': 'gengdi',
               'greenh': 'gengdi', 'st': 'gengdi', 'cx_rw': 'cx', 'cx_gw': 'cx',
               'cx_b': 'cx', 'bf': 'forest', 'rf': 'forest', 'gf': 'forest', 'of': 'forest', 'lt': 'unused',
               'cc': 'mine','xkc': 'mine', 'ptc': 'mine'}
    dict_21_v=['gengdi', 'mine', 'forest', 'unused', 'cx', 'water', 'road']
    vec = {'bf': 0, 'br': 1, 'cc': 2, 'cx_b': 3, 'cx_gw': 4, 'cx_rw': 5, 'dp': 6, 'gf': 7, 'gh': 8, 'gr': 9,
           'greenh': 10, 'lt': 11, 'of': 12, 'ptc': 13, 'rf': 14, 'st': 15, 'wm': 16, 'wr': 17, 'wt': 18, 'xkc': 19}

    res=np.zeros(shape=(dataset.RasterXSize,dataset.RasterYSize,3),dtype=np.uint8)

    label=np.random.randint(0,255,size=(7,3))

    step=8
    for i in tqdm(range(16,dataset.RasterXSize,step)):

        for j in range(16,dataset.RasterYSize,step):
            # print(i,j)
            img=get_cell(dataset,32,i,j)
            if img is None:
                continue
            imgs=np.array([img])
            # print(imgs.shape)
            result=model.predict(imgs)[0]
            s = np.argmax(result)
            res[i-16:i+16,j-16:j+16]=label[dict_21_v.index(dict_21[list(vec.keys())[s]]) ]

    res = np.array([res[:,:,0],res[:,:,1],res[:,:,2]])

    label2Img(dict_21_v, label, "stay/render_class2_label_%d_%s.png" % (step,name))
    cv2.imwrite("stay/render_class2_%d_%s.png" % (step,name),res.T)

if __name__ == '__main__':
    render_class2("cnn4321")
    render_class2_yiji("cnn4321-yiji")