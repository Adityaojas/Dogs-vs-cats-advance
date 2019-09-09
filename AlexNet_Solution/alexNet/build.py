import os
import sys

ROOT_DIR = '../..'
sys.path.append(ROOT_DIR)

from config import dvc_config as config
from mymodule.preprocessing.aspectawarepreprocessor import AspectAwarePreprocessor
from mymodule.io.hdf5datasetwriter import HDF5DatasetWriter
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import progressbar
import json
import cv2

train_paths = list(paths.list_images(config.TRAIN_PATH))
test_paths = list(paths.list_images(config.TEST_PATH))

train_labels = [p.split('/')[-1].split('.')[-3] for p in train_paths]
test_labels = ['cat' if i%2==0 else 'dog' for i in range(len(test_paths))]

le = LabelEncoder()
train_labels = le.fit_transform(train_labels)
test_labels = le.transform(test_labels)

train_paths, val_paths, train_labels, val_labels = train_test_split(train_paths, train_labels,
                                                  test_size = config.NUM_VAL_IMAGES, 
                                                  stratify = train_labels, 
                                                  random_state = 111)
datasets = [
            ('TRAIN', train_paths, train_labels, config.TRAIN_HDF5),
            ('VAL', val_paths, val_labels, config.VAL_HDF5),
            ('TEST', test_paths, test_labels, config.TEST_HDF5)
            ]

aap = AspectAwarePreprocessor(256, 256)
(R, G, B) = ([], [], [])



for (dtype, paths, labels, outpath) in datasets:
    print('... BUILDING {} HDF5'.format(dtype))
    writer = HDF5DatasetWriter((len(paths), 256, 256, 3), outpath)
    
    widgets = ['BUILDING {} HDF5 DATASET: '.format(dtype), progressbar.Percentage(),
               ' ', progressbar.Bar(), ' ', progressbar.ETA(), '\n']
    pbar = progressbar.ProgressBar(maxval = len(paths), widgets = widgets)
    pbar = pbar.start()
    
    for (i, (path, label)) in enumerate(zip(paths, labels)):
        img = cv2.imread(path)
        img = aap.preprocess(img)
        
        if dtype == 'TRAIN':
            (b, g, r) = cv2.mean(img)[:3]
            R.append(r)
            G.append(g)
            B.append(b)
            
        writer.add([img], [label])
        pbar.update(i)
    
    pbar.finish()
    writer.close()
    
"""
print('... BUILDING TEST HDF5')
writer = HDF5DatasetWriter((len(test_paths), 224, 224, 3), config.TEST_HDF5)

widgets = ['Building TEST HDF5 Dataset: ', progressbar.Percentage(),
           ' ', progressbar.Bar(), ' ', progreesbar.ETA()]
pbar = progressbar.ProgressBar(maxval = len(paths), widgets = widgets)
pbar = pbar.start()

for (i, path in enumerate(test_paths):
    img = cv2.imread(path)
    img = aap.preprocess(img)
    
    writer.add([image])
    pbar.update(i)

pbar.finish()
writer.close()

"""
    
print('... SERIALIZING MEANS')
M = {'R': np.mean(R), 'G': np.mean(G), 'B': np.mean(B)}
f = open(config.DATASET_MEAN, 'w')
f.write(json.dumps(M))
print('... DONE')

f.close()



            
        
        




