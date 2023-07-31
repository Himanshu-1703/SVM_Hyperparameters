import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC


class DecisionBoundary:

    def __init__(self, X: np.ndarray, y: np.ndarray, kernel: str, clf: SVC):
        self.X = X
        self.y = y
        self.kernel = kernel
        self.clf = clf
        if self.kernel == 'linear':
            self.linear()

        elif self.kernel == 'poly' or self.kernel == 'rbf':
            self.non_linear()

    def linear(self):
        # generate the x and y arrays
        x = np.arange(self.X[:, 0].min() - 1, self.X[:, 0].max() + 1, 0.01)
        y = np.arange(self.X[:, 1].min() - 1, self.X[:, 1].max() + 1, 0.01)

        # form a meshgrid
        XX, YY = np.meshgrid(x, y)

        # form a prediction array
        pred_arr = np.array([XX.ravel(), YY.ravel()]).T

        # predict on the prediction array
        z = self.clf.decision_function(pred_arr).reshape(XX.shape)

        # plot the graph
        fig = plt.figure(figsize=(12, 7))
        # plot the scatter plot
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y)
        # plot the classification line and margins
        plt.contour(XX,YY,z,
                    levels=[-1, 0, 1],
                    colors=['red', 'black', 'green'],
                    linestyles=['dashed', 'solid', 'dashed'])

        # plot the support vectors:
        supp_vectors = self.clf.support_vectors_
        x_vec = supp_vectors[:,0].ravel()
        y_vec = supp_vectors[:,1].ravel()

        # plot the support vectors scatter plot
        plt.scatter(x_vec,y_vec,s=70,linewidths=2,edgecolors='k')

        return fig

    def non_linear(self):
        # generate the x and y arrays
        x = np.arange(self.X[:, 0].min() - 1, self.X[:, 0].max() + 1, 0.01)
        y = np.arange(self.X[:, 1].min() - 1, self.X[:, 1].max() + 1, 0.01)

        # form a meshgrid
        XX, YY = np.meshgrid(x, y)

        # form a prediction array
        pred_arr = np.array([XX.ravel(), YY.ravel()]).T

        # predict on the prediction array
        z = self.clf.predict(pred_arr).reshape(XX.shape)

        # plot the graph
        fig = plt.figure(figsize=(12, 7))
        # plot the scatter plot
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y)

        # plot the contour function
        plt.contourf(XX,YY,z,alpha=0.4)

        return fig