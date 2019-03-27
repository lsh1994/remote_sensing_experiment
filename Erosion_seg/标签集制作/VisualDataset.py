"""
@author: LiShiHang
@software: PyCharm
@file: VisualDataset.py
@time: 2018/12/9 16:19
"""
import glob
import cv2
from skimage.segmentation import mark_boundaries

def imageSegmentationGenerator(images_path, segs_path):

    assert images_path[-1] == '/'
    assert segs_path[-1] == '/'

    images = sorted(glob.glob(images_path +"*.png") )
    segmentations = sorted(glob.glob(segs_path + "*.png"))

    assert len(images) == len(segmentations)

    for im_fn, seg_fn in zip(images, segmentations):

        img = cv2.imread(im_fn)
        seg = cv2.imread(seg_fn,0)

        out2 = mark_boundaries(img, seg)

        cv2.imshow("img", img)
        cv2.imshow("seg_img", seg*255)
        cv2.imshow("tt",out2)

        print(im_fn,seg_fn)
        cv2.waitKey()


images="../output/images/"
annotations="../output/annotations/"

imageSegmentationGenerator(images, annotations)
