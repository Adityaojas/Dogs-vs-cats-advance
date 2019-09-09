from keras.utils import np_utils
import numpy as np
import h5py

class HDF5DatasetGenerator():
    def __init__(self, db, batch_size, classes, preprocessors = None, aug = None, binarize = True):
        
        self.db = h5py.File(db)
        
        self.batch_size = batch_size
        self.preprocessors = preprocessors
        self.aug = aug
        self.binarize = binarize
        self.classes = classes
        self.num_images = self.db['labels'].shape[0]
        
    def generator(self, passes = np.inf):
        epochs = 0
        while epochs < passes:
            for i in np.arange(0, self.num_images, self.batch_size):
                images = self.db['images'][i:i+self.batch_size]
                labels = self.db['labels'][i:i+self.batch_size]
                
                if self.binarize == True:
                    labels = np_utils.to_categorical(labels, self.classes)
                    
                if self.preprocessors != None:
                    pp = []
                    
                    for img in images:
                        for p in self.preprocessors:
                            img = p.preprocess(img)
                        
                        pp.append(img)
                    
                    images = np.array(pp)
                    
                if self.aug != None:
                    (images, labels) = next(self.aug.flow(images, labels, self.batch_size))
                    
                yield (images, labels)
                
            epochs += 1
                          
    def close(self):
        self.db.close
