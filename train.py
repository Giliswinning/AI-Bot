import json # need json module to acess information

import numpy as np
from model import NeuralNet
from nltk_utils import tokenize, stem, bag_of_words

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

with open("intents.json", "r") as f:
    intents = json.load(f) # load our json file to train the bot
    #  "r" read mode
    # We want to create our training data apply tokenization lowering stemming
    # Exclude punctuation and then we apply bag of words
    # We need to collect all of the words

all_words = [] # create empty lists or arrays
tags = [] # collect different tags
xy = [] # collect different patterns and tags
for intent in intents["intents"]: # loop over intents
    tag = intent["tag"] # now its a dictionary starting with intents key the one array with all the different tags patterns and responses
    tags.append(tag) # append each tag to the tags array
    for pattern in intent["patterns"]: # loop through all different patterns
        w = tokenize(pattern) # apply tokenization to pattern
        all_words.extend(w) # already implemented a utility function in last part from nltk import tokenize, stem, bag_of_words
        xy.append((w, tag)) # extend to put it into the allwords array w is the array
        # put in token pattern and corresponding label from xy list append tuple knows pattern and corresponding tag
    ignore_words = ["?", "!", ".", ","] # exclude punctuation characters
    all_words = [stem(w) for w in all_words if w not in ignore_words] # apply list comprehension apply stemming stem  each word for w in all words and exclude ignore words
    all_words = sorted(set(all_words)) # we only want unique words remove duplicate elements via set sorted function will return list all different words
    tags = sorted(set(tags)) # tags also all unique every label is unique

    X_train = [] # create bag of words list with x trained data put all bag of words in X
    y_train = [] # y trained data emptylist associated number for each tag
    for (pattern_sentence, tag) in xy: # loop over xy array unpack tuple with pattern xy.append(w, tag)
        bag = bag_of_words(pattern_sentence, all_words) # create bag of words by calling function bag of words already implemented definition will get tokenized sentence pattern sentence already tokenized
        X_train.append(bag) # append to our training data

        label = tags.index(tag) # y data use tags .index for example print the tags gives us labels one and zero if pattern is inside
        y_train.append(label) # append numbers of labels to ytrain sometimes use 1 hot encoded vector but we use pytorch and cross entrupilus doesnt want 1 hot only the class labels

    X_train = np.array(X_train) # convert to numpy array import numpy as np based on x train list
    y_train = np.array(y_train) #  create pytorch dataset from this data
# import torch import torch.nn as nn from torch.utils.data import datasets
    class ChatDataset(Dataset): # create new dataset inherit dataset
        def __init__(self): # implement init function which will only get self store self.number of samples = length of xtrain
            self.n_samples = len(X_train)
            self.x_data = X_train # store data queals xtrainarray
            self.y_data = y_train # equals y training array

        def __getitem__(self, index): # implement get item function with self and index
            return self.x_data[index], self.y_data[index] # later acess dataset with an index

        def __len__(self): #  define the len method return self number chat dataset
            return self.n_samples

    batch_size = 8 # define hyperparameters
    hidden_size = 8
    output_size = len(tags)
    input_size = len(X_train[0]) # we can autommatically iterate over this with pytorchdataset and get batch training len of each bag of wrds created
    learning_rate = 0.001
    num_epochs = 1000

    dataset = ChatDataset() # dataset created
    train_loader = DataLoader( # data loader
        dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0 # multithreading multiprocessing will crash at 2
    )

# create model from mmodel importt neural net
    device = torch.device("cpu") #  device = torch.device('cuda', if torch.cuda is_available else 'cpu' )
    model = NeuralNet(input_size, hidden_size, output_size).to(device) #
# push model to device
    criterion = nn.CrossEntropyLoss() # loss and optimizer for python pipeline
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs): # training loop
        for (words, labels) in train_loader: # use training loader unplug words and labels
            words = words.to(device) # push to device
            labels = labels.to(device, dtype=torch.int64) # push to labels

            outputs = model(words) # forward neural net
            loss = criterion(outputs, labels) #  loss equals criterion

            optimizer.zero_grad() # backward part
            loss.backward() # calculate backward
            optimizer.step()

        if (epoch + 1) % 100 == 0: # print( every 100 step
            print(f"epoch {epoch+1}/{num_epochs}, loss={loss.item():.4f}") # print current epoch +1 all epochs and 4 decimal values

    print(f"final loss, loss={loss.item():.4f}") # equals loss at end

    data = {
        "model_state": model.state_dict(), # save modelstate
        "input_size": input_size,     # save data create dictionary
        "output_size": output_size,
        "hidden_size": hidden_size,
        "all_words": all_words,
        "tags": tags,
    }

    FILE = "data.pth" # define a file name pytorch
    torch.save(data, FILE) # this will serialize file and save it to pickeled file

    print(f"training complete. file saved to {FILE}") # print f string training complete file name
