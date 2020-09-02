from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.optimizers import SGD

class ANN(object):
    def __init__(self, input_size=None, num_hidden_layers=None,
                 hidden_layer_sizes=None, output_size=None,
                 epochs=50, batch_size=1, fit_verbose=2, variables=None):
        self.input_size = input_size
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_size = output_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = fit_verbose

    def build_model(self):
        self.model = Sequential()
        self.model.add(Dense(self.hidden_layer_sizes[0], input_shape=(self.input_size, ),
                             activation='sigmoid'))
        for i in range(1, self.num_hidden_layers - 1):
            self.model.add(Dense(self.hidden_layer_sizes[i], activation='sigmoid'))
        self.model.add(Dense(self.hidden_layer_sizes[len(self.hidden_layer_sizes) - 1], activation='sigmoid'))
        self.model.add(Dense(self.output_size, activation='linear'))

        sgd = SGD(lr=0.1, clipnorm=1.0)
        self.model.compile(loss='mean_squared_error', optimizer=sgd)

    def predict(self, data):
        """
            Runs the data in the data parameter through the network and
            returns a list of predicted values.

             data - a matrix of data (explanatory variables) to be sent through the LSTM
        """
        return self.model.predict(data)

    def save_weights(self, file_name):
        self.model.save_weights(file_name)

    def load_model(self, filename):
        """
            Loads a Keras model from an HDF5 file.
        """
        self.model = load_model(filename)


    def get_weights(self):
        """
            Returns the weights for each layer in the network (list of arrays).
        """
        return self.model.get_weights()


    def set_weights(self, weights):
        """
            Sets the weights of the network.
        """
        self.model.set_weights(weights)


    def train(self, train_x, train_y, optimzer='adam',  callbacks=[], validation_data=()):
        """
            Trains the model using the Adam optimization algortihm (more to be implemented
            later). Creates a 'history' attr of the LSTM.

            train_x - a matrix of explanatory variables for training
            train_y - a matrix of dependent variables to train on
            optimizer - optimization algorithm (Adam is the only one implemented)
         """
        self.history = self.model.fit(train_x, train_y, epochs=self.epochs, batch_size=self.batch_size,
                                      verbose=self.verbose, shuffle=False, callbacks=callbacks, validation_data=validation_data)
