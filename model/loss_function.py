import numpy as np

class LossFunction:
    """
    LossFunction class

    The model is a 1-layer neural network with sigmoid activation function.
    The loss function is the cross entropy loss.

    """

    def __init__(self, X, y, num_classes=10, num_perceptrons=1e4):
        """
        Initialize the model with data

        Parameters
        ----------
        X : numpy.ndarray
            The input data
        y : numpy.ndarray
            The labels
        num_classes : int
            The number of classes of the data
        num_perceptrons : int
            The number of perceptrons in the hidden layer
        """

        self.X = X
        self.y = y
        self.num_classes = num_classes
        self.num_perceptrons = num_perceptrons
    
    def sigmoid(self, x):
        """
        Sigmoid activation function

        Parameters
        ----------
        x : numpy.ndarray
            The input

        Returns
        -------
        numpy.ndarray
            The output
        """

        return 1 / (1 + np.exp(-x))
    
    def loss(self, W):
        """
        The loss function

        Parameters
        ----------
        W : numpy.ndarray
            The weights

        Returns
        -------
        float
            The loss
        """
