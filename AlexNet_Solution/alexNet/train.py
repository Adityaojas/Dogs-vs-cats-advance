import matplotlib
matplotlib.use('Agg')

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
from config import dvc_config as config
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD, RMSprop

aug = ImageDataGenerator(rotation_range = 15, height_shift_range = 0.2,
                         width_shift_range = 0.2, shear_range = 0.15,
                         zoom_range = 0.2, horizontal_flip = True, fill_mode = 'nearest')

means = json.loads(open(config.DATASET_MEAN).read())

sp = SimplePreprocessor(227, 227)
pp = PatchPreprocessor(227, 227)
mp = MeanPreprocessor(means['R'], means['G'], means['B'])
iap = ImageToArrayPreprocessor()

trainGen = HDF5DatasetGenerator(db = config.TRAIN_HDF5, batch_size = 128, classes = 2,
                              preprocessors = [pp, mp, iap], aug = aug,
                              binarize = True)

valGen = HDF5DatasetGenerator(db = config.VAL_HDF5, batch_size = 128, classes = 2,
                            preprocessors = [sp, mp, iap], binarize = True)

opt = Adam(lr = 0.001)
model = AlexNet.build(227, 227, 3, 2)
model.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = ['accuracy'])

tm_path = config.OUTPUT_PATH + '/{}.png'.format(os.getpid())
cp_path = config.BEST_MODEL_PATH

tm = TrainingMonitor(tm_path)
cp = ModelCheckpoint(cp_path, monitor = 'val_acc', mode = 'max',
                     save_best_only = True, verbose = 1)

callbacks = [tm, cp]

model.fit_generator(trainGen.generator(), steps_per_epoch = trainGen.num_images // 128,
                    validation_data = valGen.generator(),
                    validation_steps = valGen.num_images // 128,
                    epochs = 80, max_queue_size = 10, callbacks = callbacks,
                    verbose = 1)

model.save(config.MODEL_PATH, overwrite = True)

trainGen.close()
valGen.close()