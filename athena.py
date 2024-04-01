from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px


class LogisticRegression(object):

    def __init__(self):
        self.train_loss_list = []
        self.train_acc_list = []
        self.val_loss_list = []
        self.val_acc_list = []
        self.epoch_list = []

    def sigmoid(self, s: np.ndarray) ->np.ndarray:
        """		
		Sigmoid function 1 / (1 + e^{-s}).
		Args:
		    s: (N, D) numpy array
		Return:
		    (N, D) numpy array, whose values are transformed by sigmoid function to the range (0, 1)
		"""
        return 1 / (1 + np.exp(-s))

    def bias_augment(self, x: np.ndarray) ->np.ndarray:
        """		
		Prepend a column of 1's to the x matrix
		
		Args:
		    x (np.ndarray): (N, D) numpy array, N data points each with D features
		
		Returns:
		    x_aug: (np.ndarray): (N, D + 1) numpy array, N data points each with a column of 1s and D features
		"""
        N, D = x.shape
        
        ones_column = np.ones((N, 1))
        
        x_aug = np.concatenate((ones_column, x), axis=1)
        
        return x_aug

    def predict_probs(self, x_aug: np.ndarray, theta: np.ndarray) ->np.ndarray:
        """		
		Given model weights theta and input data points x, calculate the logistic regression model's
		predicted probabilities for each point
		
		Args:
		    x_aug (np.ndarray): (N, D + 1) numpy array, N data points each with a column of 1s and D features
		    theta (np.ndarray): (D + 1, 1) numpy array, the parameters of the logistic regression model
		
		Returns:
		    h_x (np.ndarray): (N, 1) numpy array, the predicted probabilities of each data point being the positive label
		        this result is h(x) = P(y = 1 | x)
		"""
        return self.sigmoid(np.dot(x_aug, theta))

    def predict_labels(self, h_x: np.ndarray) ->np.ndarray:
        """		
		Given model weights theta and input data points x, calculate the logistic regression model's
		predicted label for each point
		
		Args:
		    h_x (np.ndarray): (N, 1) numpy array, the predicted probabilities of each data point being the positive label
		
		Returns:
		    y_hat (np.ndarray): (N, 1) numpy array, the predicted labels of each data point
		        0 for negative label, 1 for positive label
		"""
        y_hat = np.round(h_x).reshape(-1, 1)
        return y_hat

    def loss(self, y: np.ndarray, h_x: np.ndarray) ->float:
        """		
		Given the true labels y and predicted probabilities h_x, calculate the
		binary cross-entropy loss
		
		Args:
		    y (np.ndarray): (N, 1) numpy array, the true labels for each of the N points
		    h_x (np.ndarray): (N, 1) numpy array, the predicted probabilities of being positive
		Return:
		    loss (float)
		"""
        N = y.shape[0]
        cost = -np.sum(np.log(h_x[y == 1])) - np.sum(np.log(1 - h_x[y == 0]))
        return cost / N

    def gradient(self, x_aug: np.ndarray, y: np.ndarray, h_x: np.ndarray
        ) ->np.ndarray:
        """		
		Calculate the gradient of the loss function with respect to the parameters theta.
		
		Args:
		    x_aug (np.ndarray): (N, D + 1) numpy array, N data points each with a column of 1s and D features
		    y (np.ndarray): (N, 1) numpy array, the true labels for each of the N points
		    h_x: (N, 1) numpy array, the predicted probabilities of being positive
		            it is calculated as sigmoid(x multiply theta)
		
		Return:
		    grad (np.ndarray): (D + 1, 1) numpy array,
		        the gradient of the loss function with respect to the parameters theta.
		
		"""
        N, D = x_aug.shape[0], x_aug.shape[1]
        gradient = np.zeros((D, 1))
        for i in range(D):
            gradient_i = 0
            for j in range(N):
                gradient_i += x_aug.T[i, j] * sum(h_x[j] - y[j])
                
            gradient[i] = gradient_i / N

        return gradient
    


    def accuracy(self, y: np.ndarray, y_hat: np.ndarray) ->float:
        """		
		Calculate the accuracy of the predicted labels y_hat
		
		Args:
		    y (np.ndarray): (N, 1) numpy array, true labels
		    y_hat (np.ndarray): (N, 1) numpy array, predicted labels
		
		Return:
		    accuracy of the given parameters theta on data x, y
		"""

        return np.sum(y == y_hat)/y.shape[0]

    def evaluate(self, x: np.ndarray, y: np.ndarray, theta: np.ndarray
        ) ->Tuple[float, float]:
        """		
		Given data points x, labels y, and weights theta
		Calculate the loss and accuracy
		
		Don't forget to add the bias term to the input data x.
		
		Args:
		    x (np.ndarray): (N, D) numpy array, N data points each with D features
		    y (np.ndarray): (N, 1) numpy array, true labels
		    theta (np.ndarray): (D + 1, 1) numpy array, the parameters of the logistic regression model
		
		Returns:
		    Tuple[float, float]: loss, accuracy
		"""
        x_aug = self.bias_augment(x)
        hx = self.predict_probs(x_aug, theta)
        yhat = self.predict_labels(hx)

        loss = self.loss(y, hx)
        acc = self.accuracy(y, yhat)
        return loss, acc

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.
        ndarray, y_val: np.ndarray, lr: float, epochs: int) ->Tuple[np.
        ndarray, List[float], List[float], List[float], List[float], List[int]
        ]:
        """		
		Use gradient descent to fit a logistic regression model
		
		Pseudocode:
		1) Initialize weights and bias `theta` with zeros
		2) Augment the training data for simplified multication with the `theta`
		3) For every epoch
		    a) For each point in the training data, predict the probability h(x) = P(y = 1 | x)
		    b) Calculate the gradient of the loss using predicted probabilities h(x)
		    c) Update `theta` by "stepping" in the direction of the negative gradient, scaled by the learning rate.
		    d) If the epoch = 0, 100, 200, ..., call the self.update_evaluation_lists function
		4) Return the trained `theta`
		
		Args:
		    x_train (np.ndarray): (N, D) numpy array, N training data points each with D features
		    y_train (np.ndarray): (N, 1) numpy array, the true labels for each of the N training data points
		    x_val (np.ndarray): (N, D) numpy array, N validation data points each with D features
		    y_val (np.ndarray): (N, 1) numpy array, the true labels for each of the N validation data points
		    lr (float): Learning Rate
		    epochs (int): Number of epochs (e.g. training loop iterations)
		Return:
		    theta: (D + 1, 1) numpy array, the parameters of the fitted/trained model
		"""
        D = x_train.shape[1]
        theta = np.zeros((D + 1, 1))
        x_aug = self.bias_augment(x_train)
        
        for i in range(epochs):
            hx = self.predict_probs(x_aug, theta)
            gradient = self.gradient(x_aug, y_train, hx)
            theta = theta - lr * gradient
                
            if i % 100 == 0: 
                self.update_evaluation_lists(x_train, y_train, x_val, y_val, theta, i)
        
        return theta

    def update_evaluation_lists(self, x_train: np.ndarray, y_train: np.
        ndarray, x_val: np.ndarray, y_val: np.ndarray, theta: np.ndarray,
        epoch: int):
        """
        Updates lists of training loss, training accuracy, validation loss, and validation accuracy

        Args:
            x_train (np.ndarray): (N, D) numpy array, N training data points each with D features
            y_train (np.ndarray): (N, 1) numpy array, the true labels for each of the N training data points
            x_val (np.ndarray): (N, D) numpy array, N validation data points each with D features
            y_val (np.ndarray): (N, 1) numpy array, the true labels for each of the N validation data points
            theta: (D + 1, 1) numpy array, the current parameters of the model
            epoch (int): the current epoch number
        """
        train_loss, train_acc = self.evaluate(x_train, y_train, theta)
        val_loss, val_acc = self.evaluate(x_val, y_val, theta)
        self.epoch_list.append(epoch)
        self.train_loss_list.append(train_loss)
        self.train_acc_list.append(train_acc)
        self.val_loss_list.append(val_loss)
        self.val_acc_list.append(val_acc)
        if epoch % 1000 == 0:
            print(
                f"""Epoch {epoch}:
	train loss: {round(train_loss, 3)}	train acc: {round(train_acc, 3)}
	val loss:   {round(val_loss, 3)}	val acc:   {round(val_acc, 3)}"""
                )

    def plot_loss(self, train_loss_list: List[float]=None, val_loss_list:
        List[float]=None, epoch_list: List[int]=None) ->None:
        """
        Plot the loss of the train data and the loss of the test data.

        Args:
            train_loss_list: list of training losses from fit() function
            val_loss_list: list of validation losses from fit() function
            epoch_list: list of epochs at which the training and validation losses were evaluated

        Return:
            Do not return anything.
        """
        if train_loss_list is None:
            assert hasattr(self, 'train_loss_list')
            assert hasattr(self, 'val_loss_list')
            assert hasattr(self, 'epoch_list')
            train_loss_list = self.train_loss_list
            val_loss_list = self.val_loss_list
            epoch_list = self.epoch_list
        fig = px.line(x=epoch_list, y=[train_loss_list, val_loss_list],
            labels={'x': 'Epoch', 'y': 'Loss'}, title='Loss')
        labels = ['Train Loss', 'Validation Loss']
        for idx, trace in enumerate(fig['data']):
            trace['name'] = labels[idx]
        fig.update_layout(legend_title_text='Loss Type')
        fig.show()

    def plot_accuracy(self, train_acc_list: List[float]=None, val_acc_list:
        List[float]=None, epoch_list: List[int]=None) ->None:
        """
        Plot the accuracy of the train data and the accuracy of the test data.

        Args:
            train_acc_list: list of training accuracies from fit() function
            val_acc_list: list of validation accuracies from fit() function
            epoch_list: list of epochs at which the training and validation losses were evaluated

        Return:
            Do not return anything.
        """
        if train_acc_list is None:
            assert hasattr(self, 'train_acc_list')
            assert hasattr(self, 'val_acc_list')
            assert hasattr(self, 'epoch_list')
            train_acc_list = self.train_acc_list
            val_acc_list = self.val_acc_list
            epoch_list = self.epoch_list
        fig = px.line(x=epoch_list, y=[train_acc_list, val_acc_list],
            labels={'x': 'Epoch', 'y': 'Accuracy'}, title='Accuracy')
        labels = ['Train Accuracy', 'Validation Accuracy']
        for idx, trace in enumerate(fig['data']):
            trace['name'] = labels[idx]
        fig.update_layout(legend_title_text='Accuracy Type')
        fig.show()


if __name__ == '__main__':
    sigma1 = np.array([[1, .25],
                       [.25, 4]])
    sigma2 = np.array([[2, .7],
                       [.7,4]])
    
    mu1 = [4,4]
    mu2 = [1,1]

    N = 1000

    rng = np.random.default_rng(1)

    data1 = rng.multivariate_normal(mu1, sigma1, N)
    data2 = rng.multivariate_normal(mu2, sigma2, N)

    data1_label = np.ones((N, 1))
    data2_label = np.zeros((N, 1))

    X = np.concatenate([data1, data2])
    y = np.concatenate([data1_label, data2_label])

    indices = np.arange(X.shape[0])
    rng.shuffle(indices)


    split_idx = int(0.6 * len(indices))


    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    LogReg = LogisticRegression()
    theta = LogReg.fit(X_train, y_train, X_test, y_test, .05, 1000)
    test_loss, test_acc = LogReg.evaluate(X_test, y_test, theta)

    print(f"Test Dataset Accuracy: {round(test_acc, 3)}")

    LogReg.plot_loss()
    LogReg.plot_accuracy()
