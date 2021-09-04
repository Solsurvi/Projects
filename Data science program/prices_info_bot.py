import math
import pandas as pd
import numpy as np


class risk_analysis_bot(object):
    '''
    :parameter
    this class is for the gengeral statistical analysis of the time series prices
    name: the name of the portfolio or portfolio manager
    data: the time series data of the portfolio

    Author: Ming NIE
    All Rights Reserved
    '''
    def __init__(self, prices, name):
        self.name = name
        self.prices = prices
        if None in prices:
            print("the price list has None data")

    def info(prices):
        # the general statistical analysis of the prices

        # log return
        returns = []
        for i in range(0, len(prices) - 1):
            temp = math.log(prices[i + 1] / prices[i])
            returns.append(temp)

        # mean mini max
        mean = sum(returns) / len(returns)
        min_value = min(returns)
        max_value = max(returns)

        # root mean
        sum2 = 0
        for i in returns:
            sum2 += i ** 2
        root_mean = (sum2 / len(returns)) ** 0.5

        # sd
        sum_sd = 0
        for i in returns:
            sum_sd += (i - mean) ** 2
        sd = (sum_sd / (len(returns) - 1)) ** 0.5

        # quantiles
        quant_1 = np.quantile(returns, 0.01)
        quant_5 = np.quantile(returns, 0.05)
        quant_95 = np.quantile(returns, 0.95)
        quant_99 = np.quantile(returns, 0.99)

        # skewness
        sum3 = 0
        for i in returns:
            sum3 += (i - mean) ** 3
        skewness = len(returns) ** 0.5 * sum3 / sum_sd ** 1.5


        # kurtosis
        sum4 = 0
        for i in returns:
            sum4 += (i - mean) ** 4
        kurt = len(returns) * sum4 / sum_sd ** 2

        print("The number of daily log return observation is {}".format(len(returns)))
        print(" Root mean square of daily logarithmic returns : {}".format(root_mean))
        print(" Standard deviation of daily logarithmic returns: {}".format(sd))
        print(" Minimum daily logarithmic return: {}".format(min_value))
        print(" Maximum daily logarithmic return: {}".format(max_value))
        print(" 0.01 quantile of daily logarithmic returns: {}".format(quant_1))
        print(" 0.05 quantile of daily logarithmic returns: {}".format(quant_5))
        print(" 0.95 quantile of daily logarithmic returns: {}".format(quant_95))
        print(" 0.99 quantile of daily logarithmic returns: {}".format(quant_99))
        print(" Coefficient of skewness of daily logarithmic returns: {} ".format(skewness,))
        print(" Coefficient of kurtosis (not the excess kurtosis) of daily logarithmic returns: {}".format(kurt))

















