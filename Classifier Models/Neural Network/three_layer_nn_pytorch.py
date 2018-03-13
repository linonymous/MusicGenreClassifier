import pandas as pd
from sklearn.cross_validation import train_test_split
import torch
import sys
from torch.autograd import Variable
import numpy
import matplotlib.pyplot as plt

CLEAN_DATA_PATH = 'path to csv'
data = pd.read_csv(CLEAN_DATA_PATH)

X_data = data.iloc[:, :-1].copy()
y_data = data.iloc[:, -1:].copy()

X_data = X_data.values
labels = {
"blues":0,
"classical":1,
"country":2,
"disco":3,
"hiphop":4,
"jazz":5,
"metal":6,
"pop":7,
"reggae":8,
"rock":9
}

y_data [y_data == "blues"] = 0
y_data [y_data == "classical"] = 1
y_data [y_data == "country"] = 2
y_data [y_data == "disco"] = 3
y_data [y_data == "hiphop"] = 4
y_data [y_data == "jazz"] = 5
y_data [y_data == "metal"] = 6
y_data [y_data == "pop"] = 7
y_data [y_data == "reggae"] = 8
y_data [y_data == "rock"] = 9


X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.25, random_state=333, stratify=data.iloc[:, -1:])

a = []
for i in numpy.ndenumerate(y_train):
    j, k = i
    a.append(k)
y_train = torch.LongTensor(numpy.array(a))
x = Variable(torch.Tensor(X_train))
y = Variable(y_train)


class ThreeLayerNet(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(ThreeLayerNet, self).__init__()
        self.layer_1 = torch.nn.Linear(input_dim, hidden_dim1)
        self.layer_2 = torch.nn.Linear(hidden_dim1, hidden_dim2)
        self.layer_3 = torch.nn.Linear(hidden_dim2, output_dim)

    def forward(self, input):
        out1 = self.layer_1(input)
        relu1 = out1.clamp(min=0)
        out2 = self.layer_2(relu1)
        relu2 = out2.clamp(min=0)
        y_pred = self.layer_3(relu2)
        return y_pred


model = ThreeLayerNet(28, 50, 40, 10)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

itr =[]
l = []
for _ in range(1000):
    y_pred = model(x)

    loss = criterion(y_pred, y)

    print(_, loss.data[0])

    # update grads to zero
    optimizer.zero_grad()
    # calculate grads
    loss.backward()
    # update weights
    optimizer.step()
    l.append(loss.data[0])
    itr.append(_)

axes = plt.gca()
axes.set_xlim([0,100])
axes.set_ylim([0,5])
plt.plot(itr, l)
plt.ylabel('loss')
plt.xlabel('iteration')

plt.show()