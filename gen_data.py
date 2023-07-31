import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.datasets import make_circles
from sklearn.datasets import make_moons


class DataGen:

    def __init__(self, dtype):
        self.dtype = dtype
        self.X = None
        self.y = None

    @property
    def generate_data(self):
        if self.dtype == 'blobs':
            self.X, self.y = make_blobs(n_samples=500,
                                        n_features=2,
                                        centers=2,
                                        random_state=42,
                                        cluster_std=1.4)

        elif self.dtype == 'circles':
            self.X, self.y = make_circles(n_samples=500,
                                          noise=0.1,
                                          factor=0.2,
                                          random_state=42)

        elif self.dtype == 'moons':
            self.X, self.y = make_moons(n_samples=100,
                                        noise=0.1,
                                        random_state=42)

        return self.X, self.y

    def plot_data(self):
        fig = plt.figure(figsize=(12,7))
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y)
        plt.title(f'Scatter plot for data type "{str.title(self.dtype)}"')

        return fig


