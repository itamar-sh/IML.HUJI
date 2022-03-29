from __future__ import annotations

import math

import numpy as np
from numpy.linalg import inv, det, slogdet
# from utils import *

class UnivariateGaussian:
    """
    Class for univariate Gaussian Distribution Estimator
    """
    def __init__(self, biased_var: bool = False) -> UnivariateGaussian:
        """
        Estimator for univariate Gaussian mean and variance parameters

        Parameters
        ----------
        biased_var : bool, default=True
            Should fitted estimator of variance be a biased or unbiased estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `UnivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in `UnivariateGaussian.fit`
            function.

        var_: float
            Estimated variance initialized as None. To be set in `UnivariateGaussian.fit`
            function.
        """
        self.biased_ = biased_var
        self.fitted_, self.mu_, self.var_ = False, None, None

    def fit(self, X: np.ndarray) -> UnivariateGaussian:
        """
        Estimate Gaussian expectation and variance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.var_` attributes according to calculated estimation (where
        estimator is either biased or unbiased). Then sets `self.fitted_` attribute to `True`
        """
        m = len(X)
        temp_sum = 0.0
        for i in range(m):
            temp_sum += X[i]
        self.mu_ = temp_sum / m
        temp_sum = 0
        for i in range(m):
            temp_sum += (X[i]-self.mu_)*(X[i]-self.mu_)

        if self.biased_:
            self.mu_ = temp_sum / (m)
        else:
            self.var_ = temp_sum / (m-1)

        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, var_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")
        lenX_tup = X.shape
        lenX = lenX_tup[0]
        pdfs = np.zeros(shape=(1, lenX))
        for i in range(lenX):
            down = (2 * math.pi * self.var_) ** .5
            up = math.exp(-(((X[i] - self.mu_) ** 2) / (2 * self.var_)))
            pdfs[0][i] = up / down
        return pdfs

    @staticmethod
    def log_likelihood(mu: float, sigma: float, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        sigma : float
            Variance of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """
        lenX_tup = X.shape
        m = lenX_tup[0]
        pdfs = np.zeros(shape=(1, m))
        var = sigma * sigma
        sum = 0
        for i in range(m):
            sum += (X[i]-mu)**2
        sum = -(sum / (2*var))
        result = -(m/2)*np.log(math.pi*2)-(m/2)*np.log(var)+sum
        return result



class MultivariateGaussian:
    """
    Class for multivariate Gaussian Distribution Estimator
    """
    def __init__(self):
        """
        Initialize an instance of multivariate Gaussian estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `MultivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in `MultivariateGaussian.ft`
            function.

        cov_: float
            Estimated covariance initialized as None. To be set in `MultivariateGaussian.ft`
            function.
        """
        self.mu_, self.cov_ = None, None
        self.fitted_ = False

    def fit(self, X: np.ndarray) -> MultivariateGaussian:
        """
        Estimate Gaussian expectation and covariance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.cov_` attributes according to calculated estimation.
        Then sets `self.fitted_` attribute to `True`
        """
        self.fitted_ = True
        len_m = len(X)
        len_of_each_x = len(X[0])

        mu = np.zeros(shape=(1, len_of_each_x))
        for i in range(len_of_each_x):
            temp_sum = 0.0
            for j in range(len_m):
                temp_sum += X[j][i]
            mu[0][i] = temp_sum / len_m
        self.mu_ = mu

        self.cov_ = np.cov(X, rowvar=False)
        var = np.zeros((len_of_each_x, len_of_each_x))
        for i in range(len_of_each_x):
            for j in range(len_of_each_x):
                temp_sum = 0.0
                for k in range(len_m):
                    temp_sum += (X[k][i] - mu[0][i])*(X[k][j] - mu[0][j])
                var[i][j] = temp_sum/(len_m - 1)
        return self

    def pdf(self, X: np.ndarray):
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, cov_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")
        lenX_tup = X.shape
        lenX = lenX_tup[0]
        pdfs = np.zeros(shape=(1, lenX))
        inv_cov = np.linalg.inv(self.cov_)
        for i in range(lenX):
            down = (2 * math.pi * np.linalg.det(self.var_)) ** .5
            diff = (X[i] - self.mu_)
            up = math.exp(-0.5*((np.dot(np.dot(diff, inv_cov), diff))))
            pdfs[0][i] = up / down
        return pdfs

    @staticmethod
    def log_likelihood(mu: np.ndarray, cov: np.ndarray, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        cov : float
            covariance matrix of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """
        lenX_tup = X.shape
        m = lenX_tup[0]
        d = mu.shape[0]
        cur_sum = 0.0
        inv_cov = np.linalg.inv(cov)
        for i in range(m):
            diff = X[i]-mu
            cur_sum += np.dot(np.dot(diff, inv_cov), diff)
        cur_sum = cur_sum * -0.5
        log = -(d*m)*np.log(2*math.pi) -m/2*np.log(np.linalg.det(cov)) + cur_sum
        return log

