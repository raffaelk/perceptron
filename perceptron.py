#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 17:04:05 2020

@author: raffael
"""

import random

import numpy as np
import matplotlib.pyplot as plt


class PerceptronClass:

    def __init__(self, X_data, y_data, X_test=None, y_test=None, bias=False):
        self.n_points = X_data.shape[0]
        self.n_features = X_data.shape[1]
        self.y_data = y_data
        self.train_acc_hist = 0
        self.test_acc_hist = 0
        self.theta0 = np.zeros((self.n_features+1))
        self.theta = self.theta0
        self.theta_hist = self.theta0
        self.X_data = self._append_bias(X_data, bias)
        self.bias = bias

        # check wether test data is provided
        if (X_test is not None) & (y_test is not None):
            self.test_mode = True
        else:
            self.test_mode = False

        if self.test_mode:
            self.X_test = self._append_bias(X_test, bias)
            self.y_test = y_test
            self.n_points_test = X_test.shape[0]

    def _append_bias(self, X, bias):
        n = X.shape[0]
        if bias:
            X_return = np.concatenate((X, np.ones((n, 1))), axis=1)
        else:
            X_return = np.concatenate((X, np.zeros((n, 1))), axis=1)
        return X_return

    def _percep_sign(self, value):
        if value >= 0:
            return 1
        else:
            return -1

    def plot_blobs_2D(self, theta_plot=None, show_test=True, title=''):
        
        if self.n_features != 2:
            raise ValueError("data is not 2D - X_train is of shape ("
                             + str(self.n_points) + ", " + str(self.n_features)
                             + ")")
        
        if theta_plot is None:
            theta_plot = self.theta
        x = self.X_data[:, 0]
        y = self.X_data[:, 1]

        fig, ax = plt.subplots()

        for cluster in np.unique(self.y_data):

            mask = np.array(self.y_data == cluster)

            maskedx = [element for c, element in enumerate(x) if mask[c]]
            maskedy = [element for c, element in enumerate(y) if mask[c]]

            ax.plot(maskedx, maskedy, 'o', label=str(cluster) + ' train')

        if self.test_mode & show_test:
            x_test = self.X_test[:, 0]
            y_test = self.X_test[:, 1]

            for cluster in np.unique(self.y_test):

                mask = np.array(self.y_test == cluster)

                maskedx = [element for c, element in enumerate(x_test) if mask[c]]
                maskedy = [element for c, element in enumerate(y_test) if mask[c]]

                ax.plot(maskedx, maskedy, 'o', label=str(cluster) + ' test')

        x_theta = np.array([np.min(x), np.max(x)])
        y_theta = -(x_theta * theta_plot[0] + theta_plot[2]) / theta_plot[1]

        ax.plot(x_theta, y_theta, '-', label='decision boundary')
        ax.set(title='2D Perceptron' + title,
               xlabel='x',
               ylabel='y'
               )
        ax.legend()
        plt.show()

    def update_theta(self, point_id='random'):

        if point_id == 'random':
            point_id = random.randint(0, self.X_data.shape[0]-1)
        x = self.X_data[point_id, :]
        y = self.y_data[point_id]

        if y * self._percep_sign(np.dot(x, self.theta)) >= 0:
            delta_theta = np.zeros(self.n_features+1)
        else:
            delta_theta = np.dot(y, x)

        self.theta = np.add(self.theta, delta_theta)

    def train(self, n_iter):
        self.theta_hist = np.zeros((n_iter+1, self.n_features+1))
        self.theta = self.theta0
        self.theta_hist[0, :] = self.theta0
        self.train_acc_hist = np.zeros(n_iter+1)
        self.train_acc_hist[0] = self.train_accuracy()

        if self.test_mode:
            self.test_acc_hist = np.zeros(n_iter+1)
            self.test_acc_hist[0] = self.test_accuracy()

        random.seed(41)
        for i in range(n_iter):
            self.update_theta()
            self.train_acc_hist[i+1] = self.train_accuracy()
            self.theta_hist[i+1, :] = self.theta

            if self.test_mode:
                self.test_acc_hist[i+1] = self.test_accuracy()

        return self.train_acc_hist, self.theta_hist

    def predict(self, X):
        lenX = X.shape[0]
        if X.shape[1] != self.n_features+1:
            X = self._append_bias(X, self.bias)
        y = np.zeros(lenX)
        for i in range(lenX):
            y[i] = self._percep_sign(np.dot(X[i, :], self.theta))
        return y

    def accuracy(self, y_pred, y_true):
        acc = np.sum(y_pred == y_true) / len(y_pred)
        return acc

    def train_accuracy(self):
        y_pred = self.predict(self.X_data)
        return self.accuracy(y_pred, self.y_data)

    def test_accuracy(self):
        y_pred = self.predict(self.X_test)
        return self.accuracy(y_pred, self.y_test)




