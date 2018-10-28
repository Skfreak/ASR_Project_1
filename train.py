# importing necessary packages 
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.mixture import GaussianMixture
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
delta = args.delta
components = args.components
if(args.coefficient == 'Yes'):
	coefficient = True
else:
	coefficient = False 

print("coefficient is: ", coefficient)

# Training code
timit_traindf = pd.read_hdf("./features/mfcc/delta_" + str(delta) + "/train/timit_train_delta_" + str(delta) + ".hdf")

# Label encode the labels
lb = LabelEncoder()
lb.fit(timit_traindf['labels'])
timit_traindf['labels_lb'] = lb.transform(timit_traindf['labels'])

# Take features and label encoded labels
train = timit_traindf[['features', 'labels_lb']]

# modify the features to list
features_train = np.array(train['features'].tolist())

# Select the features if energy features should be included or not
if(coefficient == False):
	if(delta == 0):
		features_train = features_train[:,1:]
	elif(delta == 1):
		features_train = np.delete(features_train,[0, 13], axis = 1)
	else:
		features_train = np.delete(features_train, [0, 13, 26], axis = 1)

# Import StandardScaler object to normalize data
scaler = StandardScaler()
scaler.fit(features_train)

# Get unique phonemes
unique_labels = np.unique(train.labels_lb)

# Normalize features
features_train = scaler.transform(features_train)
labels = np.array(train["labels_lb"].tolist())

# saving label encoder and standard scalar objects
if(coefficient == True):
    file_scalar = ("./Scalar/delta_" + str(delta) + "_with_coefficients_" + ".pkl")
else:
    file_scalar = ("./Scalar/delta_" + str(delta) + "_without_coefficients_" + ".pkl")
    
file_encoder = ("./labelEncoder/delta_" + str(delta) + ".pkl")

# dumping as pickle files
joblib.dump(lb, file_encoder)
joblib.dump(scaler, file_scalar)

print('Features Dimensions: ' + str(features_train.shape))

print("Training started.")
for i in unique_labels:
	print("Training for label: ", i)
	
	# Filter dataset based on each phoneme
	train_filter = train.loc[train['labels_lb'] == i]

	# initializing the Gaussian Mixture
	train_feats = features_train[labels == i]

	model = GaussianMixture(n_components = components, covariance_type = 'diag')

	# fitting the mixture with data
	model.fit(train_feats)

	# print(train_feats.shape)

	# Get the model directory
	if(coefficient == True):
		directory = "./models/delta_" + str(delta) + "_with_energy_coefficients" + "/" + str(components)
	else:
		directory = "./models/delta_" + str(delta) + "_without_energy_coefficients" + "/" + str(components)
	if not os.path.exists(directory):
		os.makedirs(directory)

	filename = directory + "/" + str(i) + ".pkl"
	
	# Saving the model to disk
	joblib.dump(model, filename)

	print("Training completed for label: ", i)
