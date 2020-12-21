import numpy as np

class KNearestNeighbours():
    def __init__(self):
        """
        Constructor
        """
        pass

    def train(self, X, y):
        """
        Train: Memorize the data
        X, y: numpy array
        """
        self.X_train = X
        self.y_train = y

    def dist_calc(self, X, k=1):
        """
        Predict: Given the data, predict class
        X: numpy array
        Use L2 norm metric for classification
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))

        # rows = self.X_train.shape[0]
        # cols = self.X_train.shape[1]
        # dims = rows * cols
        # dists = np.sqrt(np.square(self.X_train.reshape(1, dims)) + np.square(X) - 2 * np.matmul(X, self.X_train.T))
        for i in range(num_test):
            dists[i, :] = np.sqrt(np.sum(np.square(X[i, :] - self.X_train), axis=1))


        return dists

    def predict_labels(self, dists, k=1):
            num_test = dists.shape[0]
            y_pred = list()

            for i in range(num_test):
                closest_y = []
                indices = np.argsort(dists[i, :])
                we_want = indices[:k]
                k_values = dists[i, :][indices][:k]
                labels = self.y_train[we_want]
                closest_y = list(labels)

                count = 0
                num = closest_y[0]
                for elem in closest_y:
                  curr_count = closest_y.count(elem)
                  if curr_count > count:
                    count = curr_count
                    num = elem
                #y_pred[i] = count
                y_pred.append(num[0])

            return y_pred[0]
