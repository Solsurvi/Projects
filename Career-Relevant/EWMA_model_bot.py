import math
import pandas as pd

class ewma_calculator():
    '''
    :parameter
    this class is an implementation of EWMA model for volatility calculation 
    prices: the (adj close) prices in list
    decay: the decay term for EWMA model 

    Author: Ming NIE
    All Rights Reserved
    '''

    def __init__(self, prices, decay):
        self.returns = []
        self.daily_volatility = None
        self.prices = prices
        self.decay = decay

    def run(self):

        #computation of all log returns 
        for i in range(0, len(self.prices) - 1):
            temp = math.log(self.prices[i + 1] / self.prices[i])
            self.returns.append(temp)

        #calculation of daily volatility
        index = 1
        equation = []
        for i in self.returns:
            # apply the decay factor to all returns
            temp = (1 - self.decay) * pow(self.decay, 100 - index) * i * i / (1 - pow(self.decay, 100))
            equation.append(temp)
            index += 1
        res = math.sqrt(sum(equation))
        print(str(res))
        self.daily_volatility = res


    def vol_in_days(self, day_num):
        # for longer period, multiplied by the sqrt of time
        # one week 5 trading days
        # one year 252 trading days
        temp = self.daily_volatility * math.sqrt(day_num)
        print("Volatility in {} is {}".format(day_num,temp))
        return temp


