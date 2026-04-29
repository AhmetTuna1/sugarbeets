from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import cv2
import random
from pathlib import Path

HOWMANY = 10000

labelMap = {'soil':   {'id': [0, 1, 97]},
            'crop':  {'id': [10000, 10001, 10002]},
            'weed':  {'id': [2]},
            'dycot': {'id': [20000, 20001, 20002, 20003, 20004, 20005, 20006, 20007, 20008, 20009, 20010, 20011]},
            'grass': {'id': [20100, 20101, 20102, 20103, 20104, 20105]}}


def strechContrast(image, maxVal=255.0, p=0.5):

    oMin = 1e-6
    iMin = np.percentile(image, p)

    oMax = maxVal - 1e-6
    iMax = np.percentile(image, 100 - p)

    out = image - iMin
    out *= oMax / (iMax - iMin)
    out[out < oMin] = oMin
    out[out > oMax] = oMax

    return out.astype(np.uint8)


def getColorImage(iMap):

    iMap = np.expand_dims(iMap, axis=-1)
    ones = np.ones(iMap.shape)
    zeros = np.zeros(iMap.shape)
    soil = np.zeros(iMap.shape)

    # crop
    crop = np.zeros(iMap.shape)
    for label in labelMap['crop']['id']:
        crop += np.where(np.equal(iMap, label),
                         ones,
                         zeros)

    # weed
    weed = np.zeros(iMap.shape)
    for label in labelMap['weed']['id']:
        weed += np.where(np.equal(iMap, label),
                         ones,
                         zeros)
    dycot = np.zeros(iMap.shape)
    for label in labelMap['dycot']['id']:
        dycot += np.where(np.equal(iMap, label),
                          ones,
                          zeros)
    grass = np.zeros(iMap.shape)
    for label in labelMap['grass']['id']:
        grass += np.where(np.equal(iMap, label),
                          ones,
                          zeros)

    weed += grass + dycot
    dlpImg = np.concatenate([soil, crop, weed], axis=-1)

    return (dlpImg * 255).astype(np.uint8)


''' Dataset Folder Structure
'''
# image data
RGBFOLDER = 'images/rgb'
NIRFOLDER = 'images/nir'
# annotaions
IMAPFOLDER = 'annotations/dlp/iMapCleaned'

''' Pick Random Sample
'''
if __name__ == '__main__':

    print('\n  ===== ===== =====\n /\n  ---< Datasets\n\n')

    entries = os.listdir('.')
    dsFolders = [item for item in entries if os.path.isdir(item)]

    annoCount = 0
    print('Found Datasets:\n')
    for ds in dsFolders:
        count = len(os.listdir(os.path.join('.', ds, IMAPFOLDER)))
        print(ds + ' with ' + str(count) + ' annotations')
        annoCount += count

    print('-------------------------------')
    print('Summary: ' + str(len(dsFolders)) +
          ' with ' + str(annoCount) + ' annotations\n')

    print('\n  ===== ===== =====\n')

    for i in range(HOWMANY):

        print('\n  ===== ===== =====\n /\n  ---< Pick Random Sample\n\n')

        dsFolder = random.choice(dsFolders)

        print('Picked : ' + dsFolder)

        # get filesd
        files = os.listdir(os.path.join('.', dsFolder, IMAPFOLDER))
        file = random.choice(files)
        rgbFile = os.path.join(dsFolder, RGBFOLDER, file)
        nirFile = os.path.join(dsFolder, NIRFOLDER, file)
        iMapFile = os.path.join(dsFolder, IMAPFOLDER, file)

        # some files are missing
        if not Path(rgbFile).is_file() or not Path(nirFile).is_file() or not Path(iMapFile).is_file():
            continue

        # load images
        rgb = cv2.imread(rgbFile, cv2.IMREAD_COLOR)
        nir = cv2.imread(nirFile, cv2.IMREAD_COLOR)
        iMap = cv2.imread(iMapFile, cv2.IMREAD_ANYDEPTH)

        # enhance images for better view
        rgb = strechContrast(rgb)
        nir = strechContrast(nir)

        # create color image
        color = getColorImage(iMap)

        # get overlay
        overlay = color // 2 + nir // 2

        # create visualization
        viz = np.concatenate(
            [rgb, nir, color, overlay], axis=1)

        boundary = np.zeros((300, viz.shape[1], 3))
        viz = np.concatenate(
            [boundary, viz, boundary], axis=0)

        viz = cv2.resize(
            viz, (viz.shape[1] // 3, viz.shape[0] // 3)).astype(np.uint8)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(viz, 'Just Released 12340 Labeled Images of the 2016 Sugar Beets Dataset for Semantic Segmentation',
                    (45, 60), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(viz, ' RGB                       NIR                        Label                      Overlay',
                    (150, 485), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('IJRR', viz)
        cv2.waitKey(1000)

        print('\n  ===== ===== =====\n')

    exit()
