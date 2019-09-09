import json
import sys
import os

ROOT_DIR = '../..'
sys.path.append(ROOT_DIR)

from mymodule.preprocessing.simplepreprocessor import SimplePreprocessor
from mymodule.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from mymodule.preprocessing.meanpreprocessor import MeanPreprocessor
from mymodule.preprocessing.patchpreprocessor import PatchPreprocessor
from config import dvc_config as config
from keras.models import load_model
import numpy as np
import argparse
import cv2

# input = '../datasets/test/1.jpg'


ap = argparse.ArgumentParser()
ap.add_argument('-i', '--input', help = 'Path to input image', required = True)
args = vars(ap.parse_args())


model = load_model(config.BEST_MODEL_PATH)

means = json.loads(open(config.DATASET_MEAN).read())

image = cv2.imread(args['input']) #input
sp = SimplePreprocessor(227, 227)
mp = MeanPreprocessor(means['R'], means['G'], means['B'])
iap = ImageToArrayPreprocessor()

image_copy = sp.preprocess(image)
image_copy = mp.preprocess(image_copy)
image_copy = iap.preprocess(image_copy)
image_copy = np.expand_dims(image_copy, axis = 0)

(cat, dog) = model.predict(image_copy)[0]
label = 'cat' if cat > dog else 'dog'

cv2.putText(image, label, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
cv2.imshow("image", image)

if cv2.waitKey(1) & 0xFF == ord("q"):
    cv2.destroAllWindows





