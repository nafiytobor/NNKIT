import matplotlib.pyplot as plt


class PerformancePlot(object):
    '''
    Class to plot the performances
    Very simple and badly-styled
    '''

    def __init__(self, name):
        '''
        Constructor
        '''
        self.name = name

    def draw_performance_epoch(self, performances, epochs):
        plt.plot(range(epochs), performances, 'k',
                 range(epochs), performances, 'ro')
        plt.title("Performance of " + self.name + " over the epochs")
        plt.ylim(ymax=1)
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.show()

    def draw_error_epoch(self, errors, epochs):
        plt.plot(range(epochs), errors, 'k',
                 range(epochs), errors, 'ro')
        plt.title("Errors of " + self.name + " over the epochs")
        plt.ylabel("Error")
        plt.xlabel("Epoch")
        plt.show()
