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
import numpy as np
import pandas as pd


means = json.loads(open(config.DATASET_MEAN).read())

sp = SimplePreprocessor(227, 227)
cp = CropPreprocessor(227, 227)
mp = MeanPreprocessor(means['R'], means['G'], means['B'])
iap = ImageToArrayPreprocessor()

testGen = HDF5DatasetGenerator(db = config.TEST_HDF5, batch_size = 64, classes = 2,
                            preprocessors = [sp, mp, iap], binarize = True)

(images, labels) = testGen.generator()

model = load_model(config.BEST_MODEL_PATH)


y_pred = model.predict_generator(testGen.generator(), steps = (testGen.num_images // 64) + 1,
                                 max_queue_size = 10)

y_pred = y_pred.argmax(axis=1)

ind = np.arange(1, testGen.num_images+1)
df = pd.DataFrame({'id': ind, 'label': y_pred})
df.to_csv(config.RESULT_PATH, index = False)




