import os
import tensorflow as tf
from PIL import Image
import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

if __name__ == "__main__":

    print("Hey, I'm working!")

	class_names = sorted(os.listdir(r"/home/nvme/data/train/train"))

	base_model = tf.keras.applications.resnet.ResNet101(
	    input_shape = (224,224,3),
	    include_top = False)


	in_tensor = base_model.inputs[0] # Grab the input of base model
	out_tensor = base_model.outputs[0] # Grab the output of base model

	# Add an average pooling layer (averaging each of the 1024 channels):
	out_tensor = tf.keras.layers.GlobalAveragePooling2D()(out_tensor)

	# Define the full model by the endpoints.
	model = tf.keras.models.Model(inputs = [in_tensor], outputs = [out_tensor])

	# Compile the model for execution. Losses and optimizers
	# can be anything here, since we don’t train the model.
	model.compile(loss = "categorical_crossentropy", optimizer = 'sgd')

	# Find all image files in the data directory.
	X = [] # Feature vectors will go here.
	y = [] # Class ids will go here.
	for root, dirs, files in os.walk(r"/home/nvme/data/train/train"):
	    for name in tqdm(files):
	        if name.endswith(".jpg"):
	            # Load the image:
	            img = Image.open(root + os.sep + name)
	                                 
	            # Resize it to the net input size:
	            img = img.resize((224,224))
	                                 
	            # Convert the data to float, and remove mean:
	            img = np.array(img)
	            img = img.astype(np.float32)
	            img -= 128
	                                 
	            # Push the data through the model:
	            x = model.predict(img[np.newaxis, ...])[0]
	                                 
	            # And append the feature vector to our list.
	            X.append(x)
	                                 
	            # Extract class name from the directory name:
	            label = root.split(os.sep)[-1]
	            y.append(class_names.index(label))
	                                 
	# Cast the python lists to a numpy array.
	X = np.array(X)
	y = np.array(y) 

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	models = [KNeighborsClassifier(), LinearDiscriminantAnalysis(), SVC(), LogisticRegression(), RandomForestClassifier(n_estimators = 100)]
	model_names = ['KNeighborsClassifier', 'LinearDiscriminantAnalysis', 'SVC', 'LogisticRegression', 'RandomForestClassifier']

	for i in range(len(models)):
	    m = models[i]
	    m.fit(X_train, y_train) 

	    pred = m.predict(X_test)

	    acc = accuracy_score(y_test, pred)
	    print( model_names[i], "Accuracy :", np.round(acc, 2))

	svcmodel = SVC()
	svcmodel.fit(X, y)
	print("Model is trained")

	paths = []
	for root, dirs, files in os.walk(r"/home/nvme/data/test/testset"):
	    for i, name in enumerate(tqdm(files)):
	        if name.endswith(".jpg"):
	            path = str(root + os.sep + name)
	            paths.append(path)
	paths = sorted(paths)


	with open("submission.csv", "w") as fp:
	    fp.write("Id,Category\n")
	    for i in range(len(tqdm(paths))):
	        # Load the image:
	        img = Image.open(paths[i])
	        #print(paths[i])
	                
	                # Resize it to the net input size:
	        img = img.resize((224,224))
	                
	                # Convert the data to float, and remove mean:
	        img = np.array(img)
	        img = img.astype(np.float32)
	        img -= 128
	                                 
	                # Push the data through the model:
	        x = model.predict(img[np.newaxis, ...])[0]

	        pred = svcmodel.predict(x[np.newaxis, ...])
	                
	        label = class_names[pred[0]]
	        # 1. load image and resize
	        # 2. vectorize using the net
	        # 3. predict class using the sklearn model
	        # 4. convert class id to name (label = class_names[class_index])
	        fp.write("%d,%s\n" % (i, label))
