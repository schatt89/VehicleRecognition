import warnings
warnings.filterwarnings('ignore')
import os
import tensorflow as tf
import argparse

import pandas as pd
import numpy as np 
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

def main(args):
    class_names = sorted(os.listdir(r"/home/nvme/data/train/train"))
    N_classes = len(class_names)
    
    base_model = tf.keras.applications.inception_v3.InceptionV3(
       input_shape = (224,224,3),
       include_top = False
    )
    
    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(4096, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(N_classes, activation='softmax')(x)
    
    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False
          
    # compile the model (should be done *after* setting layers to non-trainable)
    adam = Adam(lr = 0.0001)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    
    
    # example of progressively loading images from file
    
    # create generator
    datagen = ImageDataGenerator(
        #featurewise_center=True,
        #featurewise_std_normalization=True,
        #rotation_range=20,
        #width_shift_range=0.2,
        #height_shift_range=0.2,
        rescale = 1./255.,
        #horizontal_flip=True,
        validation_split=0.2)
    
    train_gen = datagen.flow_from_directory('/home/nvme/data/train/train',
                                            class_mode = "categorical",
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            shuffle = True,
                                            subset = "training",
                                            seed = 42)
    val_gen = datagen.flow_from_directory('/home/nvme/data/train/train',
                                            class_mode = "categorical",
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            shuffle = True,
                                            subset = "validation",
                                            seed = 42)
    
    epochs = args.epochs
    
    model.fit_generator(
       train_gen,
       steps_per_epoch = train_gen.samples // 32,
       validation_data = val_gen, 
       validation_steps = val_gen.samples // 32,
       epochs = epochs 
       #,callbacks=[tensorboard_callback]
    )
    
    for layer in model.layers[:249]:
        layer.trainable = False
    for layer in model.layers[249:]:
        layer.trainable = True
    
    model.compile(optimizer=SGD(lr=0.00001, momentum=0.9),
                                loss='categorical_crossentropy',
                                metrics=['accuracy'])

    model.fit_generator(train_gen,
        steps_per_epoch = train_gen.samples // 32,
        validation_data = val_gen, 
        validation_steps = val_gen.samples // 32,
        epochs = epochs)
    
    
    ### Adding another 10 epochs to learn with aguemented data
    
    # create generator
    datagen = ImageDataGenerator(
        #featurewise_center=True,
        #featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale = 1./255.,
        horizontal_flip=True,
        validation_split=0.2)
    
    train_gen = datagen.flow_from_directory('/home/nvme/data/train/train',
                                            class_mode = "categorical",
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            shuffle = True,
                                            subset = "training",
                                            seed = 42)
    val_gen = datagen.flow_from_directory('/home/nvme/data/train/train',
                                            class_mode = "categorical",
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            shuffle = True,
                                            subset = "validation",
                                            seed = 42)   
    
    model.fit_generator(train_gen,
        steps_per_epoch = train_gen.samples // 32,
        validation_data = val_gen, 
        validation_steps = val_gen.samples // 32,
        epochs = epochs) 
    
    ###
    
    datagen_test = ImageDataGenerator(rescale = 1./255.)

    test_gen = datagen_test.flow_from_directory('/home/nvme/data/test',
                                            #class_mode = "categorical",
                                            target_size = (224, 224),
                                            batch_size = 1,
                                            shuffle = False)

    pred = model.predict_generator(test_gen, verbose = 1)
    

    p = np.argmax(pred, axis = 1)
    predictions = [class_names[k] for k in p]
    a = np.arange(len(predictions))
    d = {'Id': a, 'Category': predictions}


    df = pd.DataFrame(d)
    df.to_csv("submission3.csv", index = None, header = True)
    df.head()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training experiment.')
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()
    print(args)
    
    main(args)
    