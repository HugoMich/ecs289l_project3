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


#ML model 1: simple regressor
simple_reg = nn.Linear(274, 1)


#ML model 2: MLP
MLP = nn.Sequential(nn.Linear(274, 256), nn.ReLU(), nn.Linear(256,1))



selected_model = MLP
optimizer = torch.optim.Adam(selected_model.parameters(), lr=0.0001)
epochs = 300
batch_size = 256

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

best = 100


all_losses = []

def test():
	global best, all_losses
	with torch.no_grad():
		inputs, lbl = retrieve_data(test_indices)
		out = selected_model(inputs)
		loss = RRMSE(out, lbl)
		all_losses.append(loss)
		if loss < best:
			best = loss
			torch.save(selected_model.state_dict(),'model_best_mlp.pt')
		print("RRMSE: ", loss.item(), "Best:", best)


for e in range(epochs):
	start_ind = 0
	end_ind = batch_size
	batch_cnt = 0
	while(start_ind < len(train_indices)):
		batch_ind = train_indices[start_ind:end_ind]
		start_ind = end_ind
		end_ind = min(len(train_indices), end_ind + batch_size)
		inputs, lbl = retrieve_data(batch_ind)

		out = selected_model(inputs)
		loss = RRMSE(out, lbl)
		if batch_cnt % 10 == 0:
			print(e, batch_cnt, loss.item())
		batch_cnt += 1
		loss.backward()
		optimizer.step()

	test()


torch.save(torch.tensor(all_losses), 'mlp-losses.pt')