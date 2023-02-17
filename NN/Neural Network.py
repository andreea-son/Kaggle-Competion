import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import time


def load_train_data():
    X = np.loadtxt('train_features.csv', delimiter=',', dtype=np.float32, skiprows=1)
    y = np.loadtxt('train_labels.csv', delimiter=',', dtype=np.float32, skiprows=1)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    return (X_train, y_train), (X_val, y_val)


class TrainDataset(Dataset):
    def __init__(self, train):
        X_train, y_train = train
        self.n_samples = len(X_train)
        self.X_train = torch.from_numpy(X_train)
        self.y_train = torch.from_numpy(y_train)

    def __getitem__(self, index):
        return self.X_train[index], self.y_train[index]

    def __len__(self):
        return self.n_samples


class ValidationDataset(Dataset):
    def __init__(self, validation):
        X_validation, y_validation = validation
        self.n_samples = len(X_validation)
        self.X_validation = torch.from_numpy(X_validation)
        self.y_validation = torch.from_numpy(y_validation)

    def __getitem__(self, index):
        return self.X_validation[index], self.y_validation[index]

    def __len__(self):
        return self.n_samples


class TestDataset(Dataset):
    def __init__(self):
        X = np.loadtxt('test_features.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.n_samples = len(X)
        self.X_test = torch.from_numpy(X)

    def __getitem__(self, index):
        return self.X_test[index]

    def __len__(self):
        return self.n_samples


input_layer = 1197
hidden_layer = 60
output_layer = 3
n_iterations = 20
learning_rate = 0.001
batch_size = 64

train, validation = load_train_data()

train_dataset = TrainDataset(train)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)

validation_dataset = ValidationDataset(validation)
validation_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False)

test_dataset = TestDataset()
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


class NeuralNet(nn.Module):
    def __init__(self, input_layer, hidden_layer, output_layer):
        super(NeuralNet, self).__init__()
        self.input_size = input_layer
        self.l1 = nn.Linear(input_layer, hidden_layer)
        self.leakyRelu = nn.LeakyReLU()
        self.l2 = nn.Linear(hidden_layer, output_layer)

    def forward(self, x):
        out = self.l1(x)
        out = self.leakyRelu(out)
        out = self.l2(out)
        return out


def flatten_list(my_list):
    flat_list = [txt for sublist in my_list for txt in sublist]
    return flat_list


model = NeuralNet(input_layer, hidden_layer, output_layer)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)


num_labels = 0
num_correct_labels = 0
time1 = time.time()
for epoch in range(n_iterations):
    for i, (features, labels) in enumerate(train_loader):
        outputs = model(features)
        loss = criterion(outputs, labels.long())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        num_labels += labels.size(0)
        num_correct_labels += torch.sum(torch.eq(predicted, labels).long())
        accuracy = 100.0 * num_correct_labels / num_labels

        if (i + 1) % 100 == 0:
            print(f'Epoch {epoch + 1}, Step {i + 1}, Loss: {loss.item():.4f}, Accuracy: {accuracy}')

time2 = time.time()
print(time2 - time1)

with torch.no_grad():
    num_correct_labels = 0
    num_labels = 0
    for features, labels in validation_loader:
        outputs = model(features)
        _, predicted = torch.max(outputs.data, 1)
        num_labels += labels.size(0)
        num_correct_labels += torch.sum(torch.eq(predicted, labels).long())
    accuracy = 100.0 * num_correct_labels / num_labels
    print(f'Total accuracy on the validation data is: {accuracy} %')

with torch.no_grad():
    predictions = []
    for features in test_loader:
        outputs = model(features)
        _, predicted = torch.max(outputs.data, 1)
        predictions.append(list(predicted.numpy()))

predictions = flatten_list(predictions)
predictions_df = pd.DataFrame(np.array(predictions), columns=['label'])
predictions_df.index += 1
predictions_df = predictions_df.replace(0, 'England')
predictions_df = predictions_df.replace(1, 'Ireland')
predictions_df = predictions_df.replace(2, 'Scotland')
print(predictions_df['label'].value_counts(normalize=True))
predictions_df.to_csv('predictions.csv', index=True, index_label=['id'])

'''
ENGLAND_PERCENTAGE = 0.55104
IRELAND_PERCENTAGE = 0.242151
SCOTLAND_PERCENTAGE = 0.206809
'''
