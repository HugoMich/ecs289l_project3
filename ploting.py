import torch
import matplotlib.pyplot as plt 

mlp = torch.load('mlp-losses.pt')
reg = torch.load('simple-reg-losses.pt')

plt.grid( )
plt.plot(range(300), mlp,label='MLP')
plt.plot(range(300), reg,label='Regression')
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("RRMSE")
plt.show()