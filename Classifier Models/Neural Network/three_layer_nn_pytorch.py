import pandas as pd
from sklearn.cross_validation import train_test_split
import torch
import sys
from torch.autograd import Variable
import numpy
import matplotlib.pyplot as plt

# Data formation

CLEAN_DATA_PATH = 'C:\\Users\\Swapnil.Walke\\PycharmProjects\\PytorchExamples\\clean_data.csv'
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

y_data[y_data == "blues"] = 0
y_data[y_data == "classical"] = 1
y_data[y_data == "country"] = 2
y_data[y_data == "disco"] = 3
y_data[y_data == "hiphop"] = 4
y_data[y_data == "jazz"] = 5
y_data[y_data == "metal"] = 6
y_data[y_data == "pop"] = 7
y_data[y_data == "reggae"] = 8
y_data[y_data == "rock"] = 9


X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.25, random_state=333, stratify=data.iloc[:, -1:])

normal_y_test = y_test
normal_y_test = normal_y_test.values
target = []
for i in numpy.ndenumerate(normal_y_test):
    j, k = i
    target.append(k)

a = []
for i in numpy.ndenumerate(y_train):
    j, k = i
    mk = [0] * 10
    mk[k] = 1
    a.append(mk)

y_train = torch.FloatTensor(numpy.array(a))
y = Variable(y_train)
x = Variable(torch.Tensor(X_train))


a = []
for i in numpy.ndenumerate(y_test):
    j, k = i
    mk = [0] * 10
    mk[k] = 1
    a.append(mk)

# Variable creation
y_test = torch.FloatTensor(numpy.array(a))
x_test = Variable(torch.Tensor(X_test))
y_test = Variable(y_test)


# define the model
class ThreeLayerNet(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(ThreeLayerNet, self).__init__()
        self.layer_1 = torch.nn.Linear(input_dim, hidden_dim1)
        self.layer_2 = torch.nn.Linear(hidden_dim1, hidden_dim2)
        self.layer_3 = torch.nn.Linear(hidden_dim2, output_dim)

    def forward(self, input):
        sigmoid = torch.nn.Tanh()
        out1 = self.layer_1(input)
        out2 = self.layer_2(sigmoid(out1))
        y_pred = self.layer_3(sigmoid(out2))
        return y_pred


model = ThreeLayerNet(28, 50, 40, 10)

criterion = torch.nn.MultiLabelSoftMarginLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.004)

itr =[]
l = []
itr1=[]
l1=[]
for _ in range(120):
    y_pred = model(x)

    loss = criterion(y_pred, y)

    print("Train Loss", _, loss.data[0])

    # update grads to zero
    optimizer.zero_grad()
    # calculate grads
    loss.backward()
    # update weights
    optimizer.step()
    l.append(loss.data[0])
    itr.append(_)

    y_test_pred = model(x_test)

    loss = criterion(y_test_pred, y_test)
    print("Test Loss ", _, loss.data[0])
    l1.append(loss.data[0])
    itr1.append(_)

y_test_pred = model(x_test)
a = []
for i in y_test_pred:
    v, m = i.max(0)
    print(i, m)
    a.append(m)
axes = plt.gca()
axes.set_xlim([0,120])
axes.set_ylim([0,5])
plt.plot(itr, l)
plt.plot(itr1, l1)
plt.ylabel('loss')
plt.xlabel('iteration')
plt.show()