import torch
import random
import torch.optim
import pandas as pd
import torch.nn as nn

# Load data and remove the junks
df = pd.read_csv('soil_weather_data.csv')
df = df.drop(['state_name','county_name', 'Value_y', 'lon_y','lat_y'], axis=1)

#exclude labels
# exit()

labels = df['Value_x'].tolist()
# labels = (labels - labels.min()) / (labels.max() - labels.min())
# labels = labels.tolist()
df_no_labels = df.drop(['Value_x'], axis=1)

#Normalize the data:
df_no_labels = (df_no_labels - df_no_labels.min()) / (df_no_labels.max() - df_no_labels.min())
aez = df_no_labels['AEZ_classes'].tolist()
mapping = {}
for i, e in enumerate(set(aez)):	
	mapping[e] = i
#splitting train and test sets
random.seed(0)
indices = list(range(15408))
random.shuffle(indices)
test_indices = indices[:5000]
train_indices = indices[5000:]

#Loss function
def RRMSE(output, target):
	nom = (output - target)**2
	denom = target ** 2
	return torch.sqrt(torch.mean(nom / denom))


def one_hot(v):
	a = []
	for i in range(19):
		a.append(0)
	a[mapping[v]] = 1.
	return a


def retrieve_data(ind):
	inputs = []
	lbl = []
	for i in ind:
		inputs.append(df_no_labels.iloc[i].tolist())
		lbl.append(labels[i])
	for i in range(len(inputs)):
		inputs[i] = inputs[i][:-5] + one_hot(inputs[i][-5]) + inputs[i][-4:]
	inputs = torch.tensor(inputs)
	lbl = torch.tensor(lbl)
	return inputs, lbl.reshape(-1, 1)

reg = torch.load('reg-output.pt').reshape(-1)
mlp = torch.load('mlp-output.pt').reshape(-1)
knn = torch.load('knn-output.pt')
outputs = torch.mean(torch.stack([reg, mlp, knn]), dim=0)
print(outputs.shape)

_, lbl = retrieve_data(test_indices)

print(RRMSE(outputs, lbl))