
import numpy as np

from util.loss_functions import *
from model.logistic_layer import LogisticLayer
from model.classifier import Classifier

from sklearn.metrics import accuracy_score

import sys

class MultilayerPerceptron(Classifier):
    """
    A multilayer perceptron used for classification
    """

    def __init__(self, train, valid, test, layers=None, inputWeights=None,
                 outputTask='classification', outputActivation='softmax',
                 loss='bce', learningRate=0.01, l2_rate=0, epochs=50):

        """
        A MNIST recognizer based on multi-layer perceptron algorithm

        Parameters
        ----------
        train : list
        valid : list
        test : list
        learningRate : float
        epochs : positive int

        Attributes
        ----------
        trainingSet : list
        validationSet : list
        testSet : list
        learningRate : float
        epochs : positive int
        performances: array of floats
        """

        self.learningRate = learningRate
        self.l2_rate = l2_rate
        self.epochs = epochs
        self.outputTask = outputTask  # Either classification or regression
        self.outputActivation = outputActivation
        self.cost = []

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test
        
        if loss == 'bce':
            self.loss = BinaryCrossEntropyError()
        elif loss == 'sse':
            self.loss = SumSquaredError()
        elif loss == 'mse':
            self.loss = MeanSquaredError()
        elif loss == 'different':
            self.loss = DifferentError()
        elif loss == 'absolute':
            self.loss = AbsoluteError()
        else:
            raise ValueError('There is no predefined loss function ' +
                             'named ' + str)

        # Record the performance of each epoch for later usages
        # e.g. plotting, reporting..
        self.performances = []

        self.layers = layers

        # Build up the network from specific layers
        self.layers = []

        # Input layer
        inputActivation = "sigmoid"
        self.layers.append(LogisticLayer(train.input.shape[1], 128, 
                           None, inputActivation, False))

        # Output layer
        outputActivation = "softmax"
        self.layers.append(LogisticLayer(128, 10, 
                           None, outputActivation, True))

        self.inputWeights = inputWeights

        # add bias values ("1"s) at the beginning of all data sets
        self.trainingSet.input = np.insert(self.trainingSet.input, 0, 1,
                                            axis=1)
        self.validationSet.input = np.insert(self.validationSet.input, 0, 1,
                                              axis=1)
        self.testSet.input = np.insert(self.testSet.input, 0, 1, axis=1)


    def _get_layer(self, layer_index):
        return self.layers[layer_index]

    def _get_input_layer(self):
        return self._get_layer(0)

    def _get_output_layer(self):
        return self._get_layer(-1)

    def _feed_forward(self, inp):
        """
        Do feed forward through the layers of the network

        Parameters
        ----------
        inp : ndarray
            a numpy array containing the input of the layer

        # Here you have to propagate forward through the layers
        # And remember the activation values of each layer
        """
        
        self.layers[0].inp = inp 
        outp_first_layer = self.layers[0].forward(inp)
        outp_first_layer = np.insert(outp_first_layer, 0, 1)
        self.layers[1].inp = outp_first_layer
        outp_second_layer = self.layers[1].forward(outp_first_layer)

        return outp_second_layer

    def _compute_error(self, target, l2_rate):
        """
        Compute the total error of the network (error terms from the output layer)

        Returns
        -------
        ndarray :
            a numpy array (1,nOut) containing the output of the layer
        """
        
        out_label = [0] * 10
        out_label[target] = 1
        out_label = np.array(out_label)

        #calculate error without l2-regularization
        outp_second_layer = self.layers[1].outp
        error = self.loss.calculateError(out_label, outp_second_layer)
        #calculate l2_error
        l2_error = []
        for i, layer in enumerate(self.layers):
            l2_error.append(np.sum(layer.weights * layer.weights))
        l2_error = np.sum(l2_error)
        #update error 
        upd_error = l2_rate * l2_error + error
        self.cost.append(upd_error)
        
        
        loss_grad = self.loss.calculateDerivative(out_label, self.layers[1].outp)

        for layer in reversed(self.layers): 
            if layer.isClassifierLayer :
                next_derivatives = layer.computeOutputLayerDerivative(loss_grad, 1)
                next_weights = layer.weights
            else:
                next_derivatives = layer.computeDerivative(next_derivatives, next_weights)
                next_weights = layer.weights
       
        
    def _update_weights(self, learningRate, l2_rate):
        """
        Update the weights of the layers by propagating back the error

        """
        ## apply L2-regularization
        for layer in  range(len(self.layers)):
            for neuron in range(0, self.layers[layer].nOut):

                self.layers[layer].weights[:, neuron] -= (learningRate * 
                                                        ((self.layers[layer].deltas[neuron] * 
                                                        self.layers[layer].inp) + 
                                                        (l2_rate * self.layers[layer].weights[:, neuron]) ))
        
    def train(self, verbose=True):
        """Train the Multi-layer Perceptrons

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """
        for epoch in range(self.epochs):
            if verbose:
                print("Training epoch {0}/{1}..".format(epoch+1, self.epochs))

            self._train_one_epoch()

            if verbose:
                
                accuracy = accuracy_score(self.validationSet.label, 
                                        self.evaluate(self.validationSet))

                self.performances.append(accuracy)
                print("Accuracy on validation: {0:.2f}%".format(accuracy*100))
                print("---------------------------------")
            


    def _train_one_epoch(self):

        for img, label in zip(self.trainingSet.input, self.trainingSet.label):


            self._feed_forward(img)
            ## print("outpout: ", outp_second_layer)

            self. _compute_error(label, self.l2_rate)
            
            self._update_weights(self.learningRate, self.l2_rate)
           


    def classify(self, test_instance):
        # Classify an instance given the model of the classifier
        # You need to implement something here

        outp = self._feed_forward(test_instance)
        ## print('outp: ', outp)
        label = np.argmax(outp)
        ## print('label:', label)
        
        return label

        

    def evaluate(self, test=None):
        """Evaluate a whole dataset.

        Parameters
        ----------
        test : the dataset to be classified
        if no test data, the test set associated to the classifier will be used

        Returns
        -------
        List:
            List of classified decisions for the dataset's entries.
        """
        if test is None:
            test = self.testSet.input
        # Once you can classify an instance, just use map for all of the test
        # set.
        return list(map(self.classify, test))

    def __del__(self):
        # Remove the bias from input data
        self.trainingSet.input = np.delete(self.trainingSet.input, 0, axis=1)
        self.validationSet.input = np.delete(self.validationSet.input, 0,
                                              axis=1)
        self.testSet.input = np.delete(self.testSet.input, 0, axis=1)
