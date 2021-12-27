import torch
from torch import nn
import numpy as np
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
from sklearn import datasets
from sklearn.metrics import accuracy_score

X,Y = datasets.make_moons(200,noise=0.2)

class LogicNet(nn.Module):
    def __init__(self,inputdim,hiddendim,outputdim):
        super(LogicNet,self).__init__()
        self.Linear1 = nn.Linear(inputdim,hiddendim)
        self.Linear2 = nn.Linear(hiddendim,outputdim)
        self.criterion = nn.CrossEntropyLoss()


    def forward(self,x):
        x = self.Linear1(x)
        x = torch.tanh(x)
        x = self.Linear2(x)

        return x

    def predict(self,x):
        pred = torch.softmax(self.forward(x),dim = 1)
        return torch.argmax(pred,dim=1)

    def getloss(self,x,y):
        y_pred = self.forward(x)
        loss = self.criterion(y_pred,y)
        return loss




model = LogicNet(inputdim=2,hiddendim=3,outputdim=2)
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
xt = torch.from_numpy(X).type(torch.FloatTensor)
yt = torch.from_numpy(Y).type(torch.LongTensor)
epochs = 20000
losses = []
for i in range(epochs):
    loss = model.getloss(xt,yt)
    losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print(max(losses))
print(accuracy_score(model.predict(xt),yt))
