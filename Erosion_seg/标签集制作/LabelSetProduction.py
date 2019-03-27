"""
@author: LiShiHang
@software: PyCharm
@file: LabelSetProduction.py
@time: 2018/12/7 14:44
"""
import shapefile
import gdal
import pandas as pd
import numpy as np
import cv2


class ExtractGraphics:

    def __init__(self,label_shp,data_tif):

        self.dataset_tif = gdal.Open(data_tif)

        self.rer = shapefile.Reader(label_shp)

    def topos(self,ilabel=0,pos_src=None):
        res = []
        geo = self.dataset_tif.GetGeoTransform()
        for i in range(self.rer.numRecords):  # rer.numRecords
            pos = self.rer.shape(i).points[0]
            label = self.rer.record(i)[ilabel]

            xoffset = int((pos[0] - geo[0]) / geo[1])
            yoffset = int((pos[1] - geo[3]) / geo[5])
            res.append([xoffset, yoffset, label])

        res = pd.DataFrame(res)
        if pos_src is not None:
            res.to_csv(pos_src, header=None, index=None)

        return res


    def get_cell(self,pos_x, pos_y,bands,size):
        try:
            output = []
            for i in range(1,bands+1):
                band = self.dataset_tif.GetRasterBand(i)
                if (int(pos_x - size / 2) < 0 or int(pos_y - size / 2) < 0
                        or int(pos_x - size / 2) + size > self.dataset_tif.RasterXSize
                        or int(pos_y - size / 2) + size > self.dataset_tif.RasterYSize):
                    return None
                t = band.ReadAsArray(int(pos_x - size / 2), int(pos_y - size / 2), size, size)
                output.append(t)
            img = np.moveaxis(np.array(output, dtype=np.uint8), 0, 2)
        except:
            return None
        return img


if __name__ == '__main__':

    # eg=ExtractGraphics(r"e:\Experiment\bishe\suijidian_touying2.shp",r"e:\Experiment\bishe\海伦GF.tif")
    # pos_label=eg.topos()
    # for line,row in pos_label.iterrows():
    #     img_tif = eg.get_cell(row[0], row[1],3,224)
    #     if img_tif is None:
    #         continue
    #     # cv2.imshow("test2",img_tif)
    #     # cv2.waitKey(0)
    #     cv2.imwrite("../output/images/%d.png" % line,img_tif)

    eg = ExtractGraphics(r"e:\Experiment\bishe\suijidian_touying2.shp", r"e:\Experiment\bishe\LabelsTif.tif")
    pos_label = eg.topos()
    for line, row in pos_label.iterrows():
        img_tif = eg.get_cell(row[0], row[1], 1, 224)
        if img_tif is None:
            continue
        # cv2.imshow("test2",img_tif)
        # cv2.waitKey(0)
        cv2.imwrite("../output/annotations_temp/%d.png" % line, img_tif*255)