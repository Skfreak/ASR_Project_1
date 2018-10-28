
# importing necessary packages 
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
import os
import argparse

# command line arguments
parser = argparse.ArgumentParser()

# argument for delta
parser.add_argument('--delta', type=int, default=0, help='Value of delta while computing mfcc features')
# argument for components
parser.add_argument('--components', type = int, default = 4, help = 'How much number of components')
# arguement for energy coefficient
parser.add_argument('--coefficient', type = str, default = 'Yes', help = 'Enter False to not take energy coefficients')

args = parser.parse_args()
# delta value
delta = args.delta

# Number of components
components = args.components

# Coefficient
if(args.coefficient == 'Yes'):
    coefficient = True
else:
    coefficient = False 

print("Delta is: ", delta)
print("Number of components are: ", components)
print("Energy coefficients are included: ", coefficient)

# loading encoder and Scaler
if(coefficient == True):
    file_scalar = ("./Scalar/delta_" + str(delta) + "_with_coefficients_" + ".pkl")
    print(True)
else:
    file_scalar = ("./Scalar/delta_" + str(delta) + "_without_coefficients_" + ".pkl")
    
file_encoder = ("./labelEncoder/delta_" + str(delta) + "" + ".pkl")

#  Load scalar and label encoder objects
scaler = joblib.load(file_scalar)
lb = joblib.load(file_encoder)

# Load data file
timit_testdf = pd.read_hdf("./features_for_PER/timit_test_delta_" + str(delta) + ".hdf")

print("Test data loaded")

# encoding labels
timit_testdf['labels_lb'] = lb.transform(timit_testdf['labels'])

# Take features and label encoded labels
test = timit_testdf.copy()
test = test[['features', 'labels_lb', 'id']]

# Get unique phonemes
unique_labels = np.unique(test.labels_lb)

# print("unique labels are: ", unique_labels)

# Get test feature set
features_test = np.array(test['features'].tolist())

# Filter the co-efficients based on energy coefficients inclusion
if(coefficient == False):
    if(delta == 0):
        features_test = features_test[:,1:]
    elif(delta == 1):
        features_test = np.delete(features_test,[0, 13], axis = 1)
    else:
        features_test = np.delete(features_test, [0, 13, 26], axis = 1)

# print('features shape' + str(features_test.shape))
# Make predictions
for i in unique_labels:
    if(coefficient == True):
        directory = "./models_updated/delta_" + str(delta) + "_with_energy_coefficients" + "/" + str(components)
    else:
        directory = "./models_updated/delta_" + str(delta) + "_without_energy_coefficients" + "/" + str(components)
    if not os.path.exists(directory):
        os.makedirs(directory)

    filename = directory + "/" + str(i) + ".pkl"
    model = joblib.load(filename)
    log_prob = model.score_samples(scaler.transform(features_test))
    col_name = str(i)
    test[col_name] = log_prob

# Get predictions by using argmax
result = test.copy()
result = result.drop(['features', 'labels_lb', 'id'], axis = 1)

# Make predictions
test['predict'] = (result.idxmax(axis = 1))
test['predict'] = test['predict'].astype(int)

# Make groundtruth and prediction files
final = test.copy()
final = final[['id', 'labels_lb', 'predict']]
# final.head()

final.id = final.id.astype(str)

final.id = "sent_" + (final.id)
# final.head()

uniqueid = np.unique(final.id)
# uniqueid

# File for storing ground truth labels
gt = open("./files_for_WER_computation/groundTruth/groundTruth.txt",  "w") 

# File for storing predicted labels
pred = open("./files_for_WER_computation/predicted/predict_delta_" + str(delta) + "_components_" + str(components) + "_coefficient_" + str(coefficient) + ".txt",  "w")

for i in uniqueid:
    # print("sentence id is: ", i)
    df = final[final.id==i]
    gt.write(str(i))
    pred.write(str(i))
    for j in df.index.values:
        gt.write(" " + str(df.labels_lb[j]))
        pred.write(" " + str(df.predict[j]))
    gt.write("\n")
    pred.write("\n")

gt.close()
pred.close()

