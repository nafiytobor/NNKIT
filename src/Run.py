#!/usr/bin/env python
# -*- coding: utf-8 -*-

from data.mnist_seven import MNISTSeven
from model.stupid_recognizer import StupidRecognizer
from model.perceptron import Perceptron
from model.logistic_regression import LogisticRegression
from model.mlp import MultilayerPerceptron
from sklearn.neural_network import MLPClassifier
import numpy as np
import time

from report.evaluator import Evaluator
from report.performance_plot import PerformancePlot


def main():
    data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000,
                                                    oneHot=False)
   
    myMLPClassifier = MultilayerPerceptron(data.trainingSet,
                                        data.validationSet, 
                                        data.testSet,
                                        outputTask='classification',
                                        outputActivation='softmax',
                                        loss='bce',
                                        learningRate=0.01,
                                        epochs=20, l2_rate=0.0)
    
    X = data.trainingSet.input
    y = data.trainingSet.label

    print("=========================")
    print("Training sklearn MLPClassifier..")
             

    t = time.time()
    clf = MLPClassifier(solver='sgd', learning_rate_init=0.001, alpha=9e-5, batch_size=10, 
            hidden_layer_sizes=(128,10), random_state=1, activation='logistic', max_iter=40)
    clf.fit(X,y)
    t2= time.time()
    print(round(t2-t), 'Seconds to train clf.....')
    X_test = data.validationSet.input
    y_test = data.validationSet.label
    print('Test Accuracy of CLF= ', round(clf.score(X_test, y_test), 4))

    
    print("=========================")
    print("Training MLP..")


    # Train the classifiers
    print("\nMLP has been trainning..")
    myMLPClassifier.train()
    print("Done..")
    
                                          
    # Do the recognizer
    # Explicitly specify the test set to be evaluated
    mlpPred = myMLPClassifier.evaluate()
    
    # Report the result
    print("=========================")
    evaluator = Evaluator()
   
    print("\nResult of the MLP classifiers:")
    evaluator.printAccuracy(data.testSet, mlpPred)
    
    # Draw
    
    plot1 = PerformancePlot("MLP error")
    plot1.draw_error_epoch(myMLPClassifier.cost[2999::3000],
                                myMLPClassifier.epochs)

    plot2 = PerformancePlot("MLP validation")
    plot2.draw_performance_epoch(myMLPClassifier.performances,
                                myMLPClassifier.epochs)
    
    
if __name__ == '__main__':
    main()
