import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import pandas as pd
import numpy as np


class MusicGenreTrainSet(Dataset):

    def __init__(self, path):
        Dataset.__init__(self)
        self.path = path
        self.dataset = pd.read_csv(path, delimiter=",")
        self.dataset = self.dataset.sample(frac=1)
        self.dataset.replace(to_replace='blues', value=0, inplace=True)
        self.dataset.replace(to_replace='classical', value=1, inplace=True)
        self.dataset.replace(to_replace='country', value=2, inplace=True)
        self.dataset.replace(to_replace='disco', value=3, inplace=True)
        self.dataset.replace(to_replace='hiphop', value=4, inplace=True)
        self.dataset.replace(to_replace='jazz', value=5, inplace=True)
        self.dataset.replace(to_replace='metal', value=6, inplace=True)
        self.dataset.replace(to_replace='pop', value=7, inplace=True)
        self.dataset.replace(to_replace='reggae', value=8, inplace=True)
        self.dataset.replace(to_replace='rock', value=9, inplace=True)
        self.data = self.dataset.iloc[:-200, :-1].as_matrix()
        self.label = self.dataset.iloc[:-200, -1:].as_matrix()

    def __getitem__(self, item):
        return self.data[item, :], self.label[item]

    def __len__(self):
        return len(self.data[:, :])


class MusicGenreTestSet(Dataset):

    def __init__(self, path):
        Dataset.__init__(self)
        self.path = path
        self.dataset = pd.read_csv(path, delimiter=",")
        self.dataset = self.dataset.sample(frac=1)
        self.dataset.replace(to_replace='blues', value=0, inplace=True)
        self.dataset.replace(to_replace='classical', value=1, inplace=True)
        self.dataset.replace(to_replace='country', value=2, inplace=True)
        self.dataset.replace(to_replace='disco', value=3, inplace=True)
        self.dataset.replace(to_replace='hiphop', value=4, inplace=True)
        self.dataset.replace(to_replace='jazz', value=5, inplace=True)
        self.dataset.replace(to_replace='metal', value=6, inplace=True)
        self.dataset.replace(to_replace='pop', value=7, inplace=True)
        self.dataset.replace(to_replace='reggae', value=8, inplace=True)
        self.dataset.replace(to_replace='rock', value=9, inplace=True)
        self.data = self.dataset.iloc[-200:, :-1].as_matrix()
        self.label = self.dataset.iloc[-200:, -1:].as_matrix()

    def __getitem__(self, item):
        return self.data[item, :], self.label[item]

    def __len__(self):
        return len(self.data[:, :])


class Model(torch.nn.Module):

    def __init__(self, input, hidden, output):
        torch.nn.Module.__init__(self)
        self.layer1 = torch.nn.Linear(input, hidden)
        self.layer2 = torch.nn.Linear(hidden, output)
        self.relu = torch.nn.ReLU()

    def forward(self, input):
        out1 = self.relu(self.layer1(input))
        out2 = self.layer2(out1)
        return out2


if __name__ == "__main__":

    # GPU
    device = torch.device("cuda")

    # seed values
    torch.manual_seed(100)
    np.random.seed(100)

    # Train and Test loader
    train_loader = MusicGenreTrainSet(path='C:\\Users\\Swapnil.Walke\\PycharmProjects\\PytorchExamples\\clean_data_norm.csv')
    test_loader = MusicGenreTestSet(path='C:\\Users\\Swapnil.Walke\\PycharmProjects\\PytorchExamples\\clean_data_norm.csv')
    train_loader = DataLoader(train_loader, batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_loader, batch_size=32, shuffle=True, num_workers=1)

    # Input variables
    input = 28
    hidden = 50
    output = 10

    # criterion and entropy
    model = Model(input, hidden, output).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(504):
        loss = 0.0
        cnt = 1
        for index, data in enumerate(train_loader):
            x, y = data
            x.requires_grad = True
            x = x.float().to(device)
            y = y.squeeze(1).to(device)
            y_pred = model(x)
            loss += criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            cnt = index
        train_loss = loss.data[0]/cnt
        loss = 0.0
        cnt = 1
        for index, data in enumerate(test_loader):
            x, y = data
            x = x.float().to(device)
            y = y.squeeze(1).to(device)
            y_pred = model(x)
            loss += criterion(y_pred, y)
            cnt = index
        test_loss = loss.data[0]/cnt
        print(f'{epoch} : Train Loss : [{train_loss}] Test Loss : [{test_loss}]')

    cnt = 0.0
    tot = 0
    for index, data in enumerate(test_loader):
        x, y = data
        x = x.float().to(device)
        y = y.squeeze(1).to(device)
        y_pred = model(x)
        preds = y_pred.cpu().detach().numpy().tolist()
        expecteds = y.cpu().detach().numpy().tolist()
        for index, pred in enumerate(preds):
            pred_c = pred.index(max(pred))
            if pred_c == expecteds[index]:
                cnt += 1
            tot += 1
    print(f'Accuracy : {cnt/tot}')