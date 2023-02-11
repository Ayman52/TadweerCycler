import json
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import NeuralNetwork

# importing and loading our json file
with open('intents.json', 'r') as f:
        intents = json.load(f)

# create arrays to breakdown our intents from the json file we loaded before
all_words = []
tags = []
xy = []

# for loop to go through our intents array to seperate the tags and patterns and collect all the words
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        # we use extend instead of append because we dont want an array of arrays
        all_words.extend(w)
        # this array holds each word with its crossbonding tag
        xy.append((w, tag))

# ignore punctuation marks
ignore_words = ['?', '.', ',', '!']
# apply stemming to the list of words
all_words = [stem(w) for w in all_words if w not in ignore_words]
# sort the words and add it to a set to get rid of the duplicates
all_words = sorted(set(all_words))
tags = sorted(set(tags))

x_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    # x_train array has the patterns 
    x_train.append(bag)

# y_train array has the indecies for tags
    label = tags.index(tag)
    y_train.append(label)

# create numpy arrays based on the x_train and y_train data
x_train = np.array(x_train)
y_train = np.array(y_train)

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train
    

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    

    def __len__(self):
        return self.n_samples
    
# hyperparameters
batch_size = 16
# this is based on the number of words "input" we already have
input_size = len(x_train[0])
#this one can be changed 
hidden_size = 8
# this has the length of the classes "tags"
output_size = len(tags)

Learning_rate = 0.001
num_epochs = 1500

# print(input_size, len(all_words))
# print(output_size, tags)

dataset = ChatDataset()
train_loader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle = True, num_workers = 0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNetwork(input_size, hidden_size, output_size).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = Learning_rate)


for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device, dtype=torch.int64)

        # forward

        outputs = model(words)
        loss = criterion(outputs, labels)

        # backward and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch +1) % 100 == 0:
        print(f'epoch {epoch +1}/{num_epochs}, loss = {loss.item():.4f}')

print(f'final loss, loss = {loss.item():.4f}')

data = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "output_size": output_size,
        "hidden_size": hidden_size,
        "all_words": all_words,
        "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')