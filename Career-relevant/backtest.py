import numpy as np
import pandas as pd

class back_test_bot(object):
    '''
    :parameter
    instrument: the financial instrument you are trading, string
    data: the time series of the equity

    Author: Ming NIE
    All Rights Reserved
    '''
    def __init__(self, instrument, data):
        self.instrument = instrument
        # eliminate all na data
        self.data = data.dropna()

    def strategy_init_(self, SMA1, SMA2, choice, smoothing = 2):
        '''
        :parameter
        SMA1: shorter range day number
        SMA2: longer range day number
        Choice: 1 for SMA, you can develop more strategies in this class
        '''
        self.SMA1 = SMA1
        self.SMA2 = SMA2
        self.strategy = choice

        if self.strategy == 1:
            print("---user choice : moving EMA strategy, shorter moving range " + str(SMA1) + " longer moving range: " + str(SMA2))

            ema1 = [None]*(SMA1-1)
            ema2 = [None]*(SMA2-1)

            ema1.append(sum(self.data['Adj Close'][:SMA1]) / SMA1)
            for price in self.data['Adj Close'][SMA1:]:
                ema1.append((price * (smoothing / (1 + SMA1))) + ema1[-1] * (1 - (smoothing / (1 + SMA1))))
            self.data["EMA1"]= ema1


            ema2.append(sum(self.data['Adj Close'][:SMA2]) / SMA2)
            for price in self.data['Adj Close'][SMA2:]:
                ema2.append((price * (smoothing / (1 + SMA2))) + ema2[-1] * (1 - (smoothing / (1 + SMA2))))
            self.data["EMA2"] = ema2

            self.data['Position'] = np.where(self.data['EMA1'] > self.data['EMA2'], 1, 0)

            print("---portfolio position established ...")

            self.data['Returns'] = self.data['Adj Close'] - self.data['Adj Close'].shift(1)
            print("---market returns calculated ...")
            self.data['Strategy'] = self.data['Position'].shift(1) * self.data['Returns']
            print("---strategy returns calculate ...")

            self.data.dropna(inplace=True)

            self.sum = np.exp(self.data[['Returns', 'Strategy']].sum())

            # using 252 trading days
            self.std = self.data[['Returns', 'Strategy']].std() * 252 ** 0.5

            print("---Congrats!!! all result generated, find daily data in results, find stats data in sum and std")


        if self.strategy == 0:
            print("---user choice : moving ave strategy, shorter moving range " + str(SMA1) + " longer moving range: " + str(SMA2))
            self.data["SMA1"] = self.data['Adj Close'].rolling(SMA1).mean()
            self.data["SMA2"] = self.data['Adj Close'].rolling(SMA2).mean()

            self.data['Position'] = np.where(self.data['SMA1'] > self.data['SMA2'], 1, 0)
            print("---portfolio position established ...")

            self.data['Returns'] = self.data['Adj Close'] - self.data['Adj Close'].shift(1)
            print("---market returns calculated ...")
            self.data['Strategy'] = self.data['Position'].shift(1) * self.data['Returns']
            print("---strategy returns calculate ...")

            self.data.dropna(inplace=True)

            self.sum = np.exp(self.data[['Returns', 'Strategy']].sum())

            # using 252 trading days
            self.std = self.data[['Returns', 'Strategy']].std() * 252 ** 0.5

            print("---Congrats!!! all result generated, find daily data in results, find stats data in sum and std")



    def plot1(self):
        print("---plot1: SMA moving average and price lines")
        self.data[["Adj Close","SMA1","SMA2"]].plot(figsize=(10, 6))

    def plot2(self):
        print("---plot2: SMA with positions")
        ax = self.data[["Adj Close", "SMA1", "SMA2", "Position"]].plot(secondary_y='Position', figsize=(10, 6))
        ax.get_legend().set_bbox_to_anchor((0.25, 0.85))

    def plot3(self):
        print("---plot3: SMA strategy returns compared to the market")
        ax = self.data[['Returns', 'Strategy']].cumsum().apply(np.exp).plot(figsize=(10, 6))
        self.data['Position'].plot(ax=ax, secondary_y='Position', style='--')
        ax.get_legend().set_bbox_to_anchor((0.25, 0.85))

    def summary(self, year = 4):
        print("---- SMA: Sums up the returns for the strategy and the market")
        print(self.sum/year)
        print("---The annualized volatility for the strategy and the market")
        print(self.std)

    def ema_plot1(self):
        print("---plot1: EMA moving average and price lines")
        self.data[["Adj Close","EMA1","EMA2"]].plot(figsize=(10, 6))

    def ema_plot2(self):
        print("---plot2: EMA with positions")
        ax = self.data[["Adj Close", "EMA1", "EMA2", "Position"]].plot(secondary_y='Position', figsize=(10, 6))
        ax.get_legend().set_bbox_to_anchor((0.25, 0.85))

    def ema_plot3(self):
        print("---plot3: EMA strategy returns compared to the market")
        ax = self.data[['Returns', 'Strategy']].cumsum().apply(np.exp).plot(figsize=(10, 6))
        self.data['Position'].plot(ax=ax, secondary_y='Position', style='--')
        ax.get_legend().set_bbox_to_anchor((0.25, 0.85))

    def ema_summary(self, year = 4):
        print("---- EMA: Sums up the returns for the strategy and the market")
        print(self.sum/year)
        print("---The annualized volatility for the strategy and the market")
        print(self.std)











