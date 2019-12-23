import os
import tensorflow as tf


class_names = sorted(os.listdir(r"C:\TTY\SP and ML\PR and ML\Project assignment\vehicle\train\train"))

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
model.compile(loss = "categorical_crossentropy", optimizer = 'adam')

from PIL import Image
import numpy as np
from tqdm import tqdm

# Find all image files in the data directory.
X = [] # Feature vectors will go here.
y = [] # Class ids will go here.
for root, dirs, files in os.walk(r"C:\TTY\SP and ML\PR and ML\Project assignment\vehicle\train\train"):
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
            if root.split(os.sep)[-1] != '.ipynb_checkpoints':
                label = root.split(os.sep)[-1]
            else:
                pass
            y.append(class_names.index(label))
                                 
# Cast the python lists to a numpy array.
X = np.array(X)
y = np.array(y)                       

X.shape, y.shape

# Save the data and load it to skip the feature extraction process
#np.save('X_ResNet101.npy',X)
#np.save('y_ResNet101.npy',y)
X = np.load('X_ResNet101.npy')
y = np.load('y_ResNet101.npy')


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = [KNeighborsClassifier(), LinearDiscriminantAnalysis(), SVC(kernel='rbf'), SVC(kernel='linear'), LogisticRegression(), RandomForestClassifier(n_estimators = 100)]
model_names = ['KNeighborsClassifier', 'LinearDiscriminantAnalysis', 'SVC (RBF)', 'SVC (linear)', 'LogisticRegression', 'RandomForestClassifier']

# Calculate priors
priors = np.zeros(17)
for i in range(len(y_train)):
    class_index = y_train[i]
    priors[class_index] = priors[class_index] + 1

priors = priors / len(y_train)



for i in range(len(models)):
    m = models[i]
    m.fit(X_train, y_train) 

    pred = m.predict(X_test)

    acc = accuracy_score(y_test, pred)
    print( model_names[i], "Accuracy :", np.round(acc, decimals=6))
    
    
svcmodel_test = SVC(C=8.0,kernel='rbf',probability=True)
#svcmodel_test = SVC(C=6.0,kernel='poly',degree=2,probability=True) # Little bit higher accuracy on solo but not combined
svcmodel_test.fit(X_train,y_train)
svc_pred = svcmodel_test.predict_proba(X_test)

knnmodel_test = KNeighborsClassifier(n_neighbors=15)
knnmodel_test.fit(X_train,y_train)
knn_pred = knnmodel_test.predict_proba(X_test)

#forest_test = RandomForestClassifier(n_estimators = 100)
#forest_test.fit(X_train,y_train)
#forest_pred = forest_test.predict_proba(X_test)

LDA_test = LinearDiscriminantAnalysis()
LDA_test.fit(X_train,y_train)
LDA_pred = LDA_test.predict_proba(X_test)

#LR_test = LogisticRegression(penalty='l1',solver='liblinear',multi_class='auto',max_iter=10000,C=0.18)
#LR_test.fit(X_train,y_train)
#LR_pred = LR_test.predict_proba(X_test)

# The joint probabilities of the classifiers
pred = svc_pred*0.8 + knn_pred*0.1 + LDA_pred*0.2

predicted_labels = np.argmax(pred,axis=1)

acc = accuracy_score(y_test, predicted_labels)
print("Accuracy of ensemble:", np.round(acc, decimals=6))

    
#svcmodel = SVC() # accuracy = 0.87161
svcmodel = SVC(C=8.0,kernel='rbf',probability=True) # accuracy 0.88064
svcmodel.fit(X,y)
print("SVM trained")

knn_model = KNeighborsClassifier(n_neighbors=15)
knn_model.fit(X,y)
print("KNN trained")

LDA_model = LinearDiscriminantAnalysis()
LDA_model.fit(X,y)
print("LDA trained")


paths = []
for root, dirs, files in os.walk(r"C:\TTY\SP and ML\PR and ML\Project assignment\vehicle\test\testset"):
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
                
        # Resize it to the net input size:
        img = img.resize((224,224))
                
        # Convert the data to float, and remove mean:
        img = np.array(img)
        img = img.astype(np.float32)
        img -= 128
                                 
        # Push the data through the model:
        x = model.predict(img[np.newaxis, ...])[0]

        # Convert class id to name (label = class_names[class_index])
        svc_pred = svcmodel.predict_proba(x[np.newaxis, ...])
        knn_pred = knn_model.predict_proba(x[np.newaxis, ...])
        LDA_pred = LDA_model.predict_proba(x[np.newaxis, ...])
        pred = svc_pred*0.8 + knn_pred*0.1 + LDA_pred*0.2
        predicted_label = np.argmax(pred,axis=1)
                
        label = class_names[int(predicted_label)]
        
        fp.write("%d,%s\n" % (i, label))


    