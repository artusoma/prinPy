'''
This is the global module that contains principal curve and nonlinear
principal component analysis algorithms that work to optimize a line
over an entire dataset. Additionally, these algorithms should work
in space greater than 2-dimensions.
'''

# General libraries
import numpy as np

# ML libraries
import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Dense, Input, LeakyReLU
from keras import optimizers
import keras.backend as k

# Preprocessing
from sklearn.preprocessing import MinMaxScaler


def orth_dist(y_true, y_pred):
    '''
    Loss function for the NLPCA NN. Returns the sum of the orthogonal
    distance from the output tensor to the real tensor.
    '''
    loss = tf.math.reduce_sum((y_true - y_pred)**2)
    return loss


class NLPCA(object):
    '''This is a global solver for principal curves that uses neural networks.

    Attributes:
        None
    '''
    def __init__(self):
        self.fit_points = None
        self.model = None
        self.intermediate_layer_model = None

    def fit(self, data, epochs = 500, verbose = 0):
        '''This method creates a model and will fit it to the given m x n 
        dimensional data.

        Args:
            data (np array): A numpy array of shape (m,n), where m is the 
                number of points and n is the number of dimensions. 
            epochs (int): Number of epochs to train neural network, defaults
                to 500.
            verbose (0 or 1): Verbose = 0 mutes the training text from Keras.
                Defaults to 0.

        Returns:   
            None
        '''
        num_dim = data.shape[1] # get number of dimensions for pts

        # create models, base and intermediate
        model = self.create_model(num_dim)
        bname = model.layers[2].name        # bottle-neck layer name

        # The itermediate model gets the output of the bottleneck layer, 
        # which acts as the projection layer.
        self.intermediate_layer_model = Model(inputs=model.input,
                                        outputs=model.get_layer(bname).output)

        # Fit the model and set the instances self.model to model
        model.fit(data, data, epochs = epochs, verbose = verbose)
        self.model = model

        return

    def project(self, data):
        '''The project function will project the points to the curve generated
        by the fit function. Given back is the projection index of the original
        data and a sorted version of the original data. 

        Args:
            data (np array): m x n array to project to the curve

        Returns:
            proj (array): A one-dimension array that contains the projection
                index for each point in data.
            all_sorted (array): A m x n+1 array that contains data sorted by 
                its projection index, along with the index. 
        '''
        pts = self.model.predict(data)
        proj = self.intermediate_layer_model.predict(data)

        self.fit_points = pts
        
        all = np.concatenate([pts, proj], axis = 1)
        all_sorted = all[all[:,2].argsort()]
        
        return proj, all_sorted

    def create_model(self, num_dim):
        '''Creates a tf model.

        Args:
            num_dim (int): How many dimensions the input space is

        Returns: 
            model (object): Keras Model
        '''
        # Create layers:
        # Function G
        input = Input(shape = (num_dim,)) #input layer
        mapping = Dense(20, activation = 'sigmoid')(input)   #mapping layer
        bottle = Dense(1, activation = 'sigmoid')(mapping) #bottle-neck layer

        # Function H
        demapping = Dense(20, activation = 'sigmoid')(bottle)   #mapping layer
        output = Dense(num_dim)(demapping)   #output layer

        # Connect and compile model:
        model = Model(inputs = input, outputs = output)
        gradient_descent = optimizers.adam(learning_rate=.01)
        model.compile(loss = orth_dist, optimizer = gradient_descent)

        return model

    def preprocess(self, data):
        '''Converts individual arrays into a singular m x n array, where
        m is the number of observations and n is the number of dimensions.
        Also scales the data between 0 and 1 as to not break the neural net. 

        Args:
            data (list): List of arrays of points. For example, if you have
                data for x, y, and z stored in arrays x_, y_, and z_, pass
                in [x_, y_, z_]
        
        Returns:
            data_comb (array): A single m x n, where each column 
                is MinMaxScaled.
        '''
        data_lists = []
        s = MinMaxScaler()
        for arr in data:   
            data_lists.append(s.fit_transform(arr.reshape(-1,1)))
        
        return np.concatenate(data_lists, axis = 1)