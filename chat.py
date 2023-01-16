import random # random choice from answers
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device("cpu")

with open("intents.json", "r") as f: # open file
    intents = json.load(f)

FILE = "data.pth" # open save file
data = torch.load(FILE) # load our the file

input_size = data["input_size"] # get the same information key the input size
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state) # load state after creation
# knows learned parameters
model.eval()
# set to evaluation mode

bot_name = "Elon" # implement actual chat name
print("Let's talk! type 'quit' to exit") # first message
while True:
    sentence = input("You: ")
    if sentence == "quit": # if not quit chat will continue
        break

    sentence = tokenize(sentence) # tokenize sentence and calculate bag of words same as training data step
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0]) # reshape bag of words 1 row 1 sample 0 for number of columns
    X = torch.from_numpy(X) # convert to torch tensor because bag of words function returns numbpy array

    output = model(X) # use model
    _, predicted = torch.max(output, dim=1) # gives us the prediction
    tag = tags[predicted.item()] # get actual tag of the index class label number and actual tag

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75: # find corresponding intent loop over all intents and check if tag matches
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}") # print random response pattern
    else:
        print(f"{bot_name}: I do not understand what you are saying") # if nothing matches response
