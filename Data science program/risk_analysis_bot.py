import numpy as np
import pandas as pd

class risk_analysis_bot(object):
    '''
    :parameter
    this class is for historical risk analysis of a portfolio
    name: the name of the portfolio or portfolio manager
    prices: the time series prices of the portfolio

    Author: Ming NIE
    All Rights Reserved
    '''
    def __init__(self, name, prices):
        self.name = name
        self.prices = prices

    def run(prices, pos, a):
        '''

        :param pos: position in dollar
        :param a: 1 - confidence level
        :return:
        '''
        # log return calculation
        returns = []
        for i in range(0, len(prices) - 1):
            temp = math.log(prices[i + 1] / prices[i])
            returns.append(temp)

        # sort the return ascending
        returns.sort()
        m = len(returns)
        rank = math.ceil(a * m)
        shortfall_range = returns[0:rank - 1]

        shortfalls = []
        for i in shortfall_range:
            shortfalls.append(pos * (math.exp(i) - 1))

        fall = abs(sum(shortfalls) / len(shortfalls))
        var = abs(pos * (math.exp(returns[rank - 1]) - 1))
        ratio = fall / var
        print("-----------------------------------------------")
        print("Confidence Interval: {:.4%}".format(1 - a) + " Rank in returns : " +
              str(rank) + " Historical Shortfall: " + str(fall) + " Historical VAR: " + str(var) +
              " Ratio of historical shortfall to VAR: " + str(ratio))