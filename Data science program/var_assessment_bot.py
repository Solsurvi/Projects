

class var_assessment_bot(object):
    '''

    '''
    def __init__(self,data,days):
        '''
        :param data: data frame including prices, volatility
        :param days: how many days you want to test
        '''
        self.data = data
        self.days = days
        self.prices = df["Closing price"].tail(days).tolist()
        self.vols = df["Volatility"].tail(days).tolist()

    def acceptance_range(T, a, quantile = 3.8415):
        '''

        :param T: observation number
        :param a: 1 - confidence level
        :param quantile: quantile for test , default 3.8415, 95% significance level
        :return: range of acceptance
        '''

        T = slef.days - 1
        stats = []
        negative = True
        res = []

        for i in range(0, 50):
            temp = 2 * (math.log((i / T) ** i * (1 - i / T) ** (T - i)) - math.log((1 - a) ** i * a ** (T - i)))
            stats.append(temp)

            if negative:
                if stats[-1] - quantile < 0:
                    negative = False
                    res.append(i)
                continue
            else:
                if stats[-1] - quantile > 0:
                    res.append(i - 1)
                    return res

    def backtest_info(cl, T, quant = 3.8415):
        '''

        :param T: observation number
        :param quant: quantile for test , default 3.8415, 95% significance level
        :return:
        '''
        T = self.days -1
        count = 0
        exceedances = 0


        for i in range(T):
            count += 1
            var = var_return(self.vols[i], cl, 1)
            # compare with tommorrow's return
            tom_return = self.prices[i + 1] - self.prices[i]
            if tom_return < var * self.prices[i]:
                exceedances += 1

        ratio = exceedances / count

        print("Number of comparisons is {}".format(count))
        print("Confidence level of VAR: {}".format(cl))
        print("Number of exceedances is {}".format(exceedances))
        print("Ratio of the number of exceedances to the number of observations in the backtest is {}".format(ratio))

        i = exceedances

        a = cl
        test = 2 * (math.log((i / T) ** i * (1 - i / T) ** (T - i)) - math.log((1 - a) ** i * a ** (T - i)))
        print("-----------------------------------------------------")
        print("the test statistics at {} significance level is {}".format(a, test))
        if test < quant:
            print("we do not reject null hypothesis at 95 percent significance level")
        if test > quant:
            print("we reject null hypothesis at 95 percent significance level ")

        print("_______________end of info_________________")