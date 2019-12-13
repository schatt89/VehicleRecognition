import warnings
warnings.filterwarnings('ignore')
import os
import tensorflow as tf
import argparse

import pandas as pd
import numpy as np 

import split_utils
import matplotlib.pyplot as plt
from urllib.request import urlopen,urlretrieve

from tensorflow.keras.models import load_model
from sklearn.datasets import load_files   
#from tensorflow.keras.utils import np_utils
from glob import glob
from tensorflow.keras import applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint

from tensorflow.keras.optimizers import SGD, Adam
from classification_models.tfkeras import Classifiers

def main(args):
    
    class_names = sorted(os.listdir(r"/home/nvme/data/train/train"))
    N_classes = len(class_names)

    ResNext101, preprocess_input = Classifiers.get('resnext101')
    
    base_model = ResNext101(input_shape=(224,224,3), weights='imagenet', include_top=False)
    
    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    # let's add a fully-connected layer
    if args.add_dense:
        x = Dense(4096, activation='relu')(x)
    else:
        pass
    
    # and a logistic layer -- let's say we have 17 classes
    predictions = Dense(N_classes, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # compile the model (should be done *after* setting layers to non-trainable)
    adam = Adam(lr = 0.0001)

    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    
    original_dir = "/home/nvme/data/train/train"
    validation_split = 0.2

    batch_size = 16

    # all data in train_dir and val_dir which are alias to original_data. (both dir is temporary directory)
    # don't clear base_dir, because this directory holds on temp directory.
    base_dir, train_dir, val_dir = split_utils.train_valid_split(original_dir, validation_split, seed=1)
    
    # generator for train data
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        class_mode = "categorical",
        target_size = (224, 224),
        batch_size = batch_size,
        shuffle = True,
        seed = 42
    )

    # generator for validation data
    val_datagen = ImageDataGenerator(rescale=1./255)

    val_gen = val_datagen.flow_from_directory(
        val_dir,
        class_mode = "categorical",
        target_size = (224, 224),
        batch_size = batch_size,
        shuffle = True,
        seed = 42
    )
    
    epochs = args.epochs

    class_weights = {0: 65,
                     1: 42,
                     2: 5,
                     3: 1,
                     4: 4,
                     5: 1,
                     6: 169,
                     7: 27,
                     8: 13,
                     9: 115,
                     10: 2,
                     11: 56,
                     12: 70,
                     13: 42,
                     14: 11,
                     15: 4,
                     16: 7}
    
    for layer in model.layers[:2491]:
        layer.trainable = False
    for layer in model.layers[2491:]:
        layer.trainable = True

    model.fit_generator(train_gen,
        steps_per_epoch = train_gen.samples // batch_size,
        validation_data = val_gen, 
        validation_steps = val_gen.samples // batch_size,
        epochs = epochs,
        class_weight = class_weights
                   )
    
    datagen_test = ImageDataGenerator(rescale = 1./255.)

    test_gen = datagen_test.flow_from_directory('/home/nvme/data/test',
                                        target_size = (224, 224),
                                        batch_size = 1,
                                        shuffle = False)

    pred = model.predict_generator(test_gen, verbose = 1)
    
    p = np.argmax(pred, axis = 1)
    predictions = [class_names[k] for k in p]
    a = np.arange(len(predictions))
    d = {'Id': a, 'Category': predictions}

    df = pd.DataFrame(d)
    
    file_name = args.file
    df.to_csv(file_name, index = None, header = True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training experiment. Set epochs (default 10). Set add_dense (default 1) to add an additional dense layer before the top one. Set aug (default 1) to use augumentation. Set file (default submission) to the name of the predicted results file.')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--add_dense', type=int, default=1, choices = [0,1])
    #parser.add_argument('--aug', type=int, default=1, choices = [0,1])
    parser.add_argument('--file', type=str, default='submission.csv')
    args = parser.parse_args()
    print(args)
    
    main(args)