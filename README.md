# Dogs-vs-cats-advance
Advance Deep Learning Solutions for Dogs vs Cats Challenge

mymodule consists of the custom made classes that I'll be using in this repo

AlexNet_Solution:

The solution makes the use of a custom alexnet architecture that was originally built for the ImageNet Dataset.
Make another directory 'datasets' inside the AlexNet_Solution directory. And another one inside the 'datasets' with the name of hdf5. This would contain of our data in hdf5 format.
Inside the alexNet directory make another directory 'output' which would store our output utilities

The whole structure inside the AlexNet_Solutions would look like:
alexNet/
| . |---config/
| .   . |---__init__.py
| .   . |---dvc_config.py
| . |---outputs/
| . |---build.py
| . |---inspect.py
| . |---kaggle_test.py
| . |---test.py
| . |---train.py
datasets/
| . |---hdf5/
| . |---train/
| . |---test/

Grab the dataset from: https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data and put it inside the datasets/ directory according to the above structure

dvc_config.py is the configuration file for the whole solution
outputs/ would consist of the outout utilities after running the code
The first run would be build.py which would grab the images from ../datasets/train and ../datasets/test and make an hdf5 image file consisting of the train, val and test datasets
Run train.py to extract these images batch wise from ../datasets/hdf5/train.hdf5 to train the alexnet architecture on it, the model would get saved in outputs/
Also, training can be monitored on real time basis using the pid graph inside the outputs/, which would get created after training starts.
inspect.py can be used to check the model on individual images
kaggle_test.py would generate a csv submission file for submission on Kaggle
test.py randomly tests the images on the val dataset and gets an accuracy of about 93%

