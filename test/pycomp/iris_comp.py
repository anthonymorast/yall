from ann import ANN
import pandas as pd

data = pd.read_csv("../iris.data")
data = data.sample(frac=1).reset_index(drop=True)
trainsize = int(data.shape[0]*0.8)

train = data[:trainsize]
test = data[trainsize:]

trainx = train.loc[:, train.columns != " class"]
trainy = train.loc[:, train.columns == " class"]
testx = test.loc[:, test.columns != " class"]
testy = test.loc[:, test.columns == " class"]

hidden_layers = 1
hidden_layer_sizes = [2]

model = ANN(input_size=trainx.shape[1], num_hidden_layers=hidden_layers,
            hidden_layer_sizes=hidden_layer_sizes, output_size=trainy.shape[1],
            epochs=500, batch_size=1)
model.build_model()
model.train(trainx, trainy)
