# Combined classifier with sklearn(SVM+kNN+LDA) + resnext101 + inception network
import os
import numpy as np
import csv

class_names = sorted(os.listdir(r"C:\TTY\SP and ML\PR and ML\Project assignment\vehicle\train\train"))

predictions_sklearn = []
with open("submission_combined_sklearn.csv", "r") as fp:
    next(fp)
    csv_reader = csv.reader(fp)
    for row in csv_reader:
        predictions_sklearn.append(class_names.index(row[1]))
        
predictions_sklearn = np.array(predictions_sklearn)


predictions_inception_v3 = []
with open("submission_inception_v3.csv", "r") as fp:
    next(fp)
    csv_reader = csv.reader(fp)
    for row in csv_reader:
        predictions_inception_v3.append(class_names.index(row[1]))
        
predictions_inception_v3 = np.array(predictions_inception_v3)


predictions_resnext101 = []
with open("submission_resnext101.csv", "r") as fp:
    next(fp)
    csv_reader = csv.reader(fp)
    for row in csv_reader:
        predictions_resnext101.append(class_names.index(row[1]))
        
predictions_resnext101 = np.array(predictions_resnext101)


predictions_efficientnet = []
with open("submission_efficientnet.csv", "r") as fp:
    next(fp)
    csv_reader = csv.reader(fp)
    for row in csv_reader:
        predictions_efficientnet.append(class_names.index(row[1]))
        
predictions_efficientnet = np.array(predictions_efficientnet)


predictions_resnet = []
with open("submission_resnet.csv", "r") as fp:
    next(fp)
    csv_reader = csv.reader(fp)
    for row in csv_reader:
        predictions_resnet.append(class_names.index(row[1]))
        
predictions_resnet = np.array(predictions_resnet)


# Submission file
num_of_classes = len(class_names)
with open("submission.csv", "w") as fp:
    fp.write("Id,Category\n")
    for i in range(len(predictions_sklearn)):
        votes = np.zeros(num_of_classes)
        votes[predictions_sklearn[i]] = votes[predictions_sklearn[i]] + 1
        votes[predictions_inception_v3[i]] = votes[predictions_inception_v3[i]] + 1
        votes[predictions_resnext101[i]] = votes[predictions_resnext101[i]] + 1
        votes[predictions_efficientnet[i]] = votes[predictions_efficientnet[i]] + 1
        votes[predictions_resnet[i]] = votes[predictions_resnet[i]] + 1
        votes.astype(int)
        
        # If we have a majority, then it is our prediction. Otherwise we use the
        # prediction made by our best solo classifier.
        max_votes = np.max(votes)
        if max_votes >= 3:
            class_index = np.argmax(votes)
            label = class_names[int(class_index)]
        else:
            number_of_ties = np.count_nonzero(votes == 2)
            # We have only one class with two votes, so we choose it
            if number_of_ties == 1:
                class_index = np.argmax(votes)
                label = class_names[int(class_index)]
            
            # We choose the prediction of the best solo classifier
            else:
                class_index = predictions_resnext101[i]
                label = class_names[int(class_index)]

        
        fp.write("%d,%s\n" % (i, label))









