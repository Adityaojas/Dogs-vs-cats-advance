import json
import sys
import os

ROOT_DIR = '../..'
sys.path.append(ROOT_DIR)

from mymodule.preprocessing.simplepreprocessor import SimplePreprocessor
from mymodule.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from mymodule.preprocessing.meanpreprocessor import MeanPreprocessor
from mymodule.preprocessing.croppreprocessor import CropPreprocessor
from mymodule.preprocessing.patchpreprocessor import PatchPreprocessor
from mymodule.io.hdf5datasetgenerator import HDF5DatasetGenerator
from mymodule.callbacks.trainingmonitor import TrainingMonitor
from mymodule.conv.alexnet import AlexNet
from keras.models import load_model
from config import dvc_config as config
from imutils import paths
import numpy as np
import pandas as pd
import cv2
import time
import h5py

batch_size = 125

test_paths = list(paths.list_images(config.TEST_PATH))

means = json.loads(open(config.DATASET_MEAN).read())

sp = SimplePreprocessor(227, 227)
mp = MeanPreprocessor(means['R'], means['G'], means['B'])
iap = ImageToArrayPreprocessor()

model = load_model(config.BEST_MODEL_PATH)

p = model.predict(batch)
p.shape
y_pred = np.concatenate((y_pred, p), axis = 0)

img_nb = []
y_pred = []

for i in np.arange(0, len(test_paths), batch_size):
    batch_paths = test_paths[i: i+batch_size]
    i_n = [int(n.split('/')[-1].split('.')[-2]) for n in batch_paths]
    img_nb.extend(i_n)
    batch_images = []
    for j in batch_paths:
        image = cv2.imread(j)
        image = sp.preprocess(image)
        image = mp.preprocess(image)
        image = iap.preprocess(image)
        image = np.expand_dims(image, axis = 0)
        batch_images.append(image)
    
    batch_images = np.vstack(batch_images)
    y_p = model.predict(batch_images)
    y_pred.append(y_p)
    
y_pred = np.concatenate(y_pred, axis = 0)
y_pred = y_pred.argmax(axis = 1)

img_nb, y_pred = zip(*sorted(zip(img_nb, y_pred)))
        
df = pd.DataFrame({'id': img_nb, 'label': y_pred})
df.to_csv(config.RESULT_PATH, index = False)








