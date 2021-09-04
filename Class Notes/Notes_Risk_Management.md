---
typora-root-url: pic
---

# Risk Management CU 21S Allan Malz

## 4. Volatility

Facts we know

- time-varying 

- Persistence: Volatility clustering, Autocorrelation 

- Changes: Shifts from low to high volatility are more abrupt, while shifts from high to low are more gradual

- Long-term mean reversion: volatility of an asset’s return tends to gravitate to a long-term level 

- Solution : use conditional volatility, which is to estimate weighted toward more recent information

- The correlation across assets also is time-varying, and they tend to rise during stress periods 

  Assumptions 

- Zero Mean (Return) Assumption (estimation of mean can introduce additional statistical errors, small impact on mean over short intervals, over longer periods means has larger impact than volatility)

- GBM model, variance of price change proportional to time, then volatility proportional to sqrt(time), 1 year =  daily volatility *16

- when computing, size of the return does not matter 

### Traditional Models

A first approach is to use the sample standard deviation, the average square root of daily return deviations from the mean over some past time period:

![](https://i.loli.net/2021/02/22/VHsBvkJhUGRX2QN.jpg)

where T is the number of return observations in the sample, and r¯, the sample mean, is an estimator of μ. Since we’ve decided that, for short periods at least, we can safely ignore the mean, we can use the root mean square of rt:

![](https://i.loli.net/2021/02/22/QL7vAyXxd1intCD.jpg)

The second sample moment and the maximum likelihood estimator of the standard deviation of a normal, zero-mean random variable. 

This estimator adapts more or less slowly to changes in return variability, depending on how long an observation interval T is chosen. 

### EWMA Model 

Exponential Weighted Moving Average model, use a decay factor (lambda, 0.9 to 1 ) close to one to apply more weight in recent observations. 

low decay: reflect the recent, rapid changes,  λ = 0.94 for short-term (e.g. one-day) forecasts  λ = 0.97 for medium-term (e.g. one-month)

![](https://i.loli.net/2021/02/22/N39zgTZSGunFbtJ.jpg)

The sum of weights is unity

```python
import math
import pandas as pd

class ewma_calculator():

    def __init__(self, list_of_prices, decay):
        self.returns = []
        self.daily_volatility = None
        self.prices = list_of_prices
        self.decay = decay

    def returns_cal(self):
        #computation of all log returns 
        for i in range(0, len(self.prices) - 1):
            temp = math.log(self.prices[i + 1] / self.prices[i])
            self.returns.append(temp)

    def daily_vol_cal(self):
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
        return self.daily_volatility * math.sqrt(day_num)


df_excel = pd.read_excel(r'C:\Users\14249\OneDrive\Columbia\$RM\set1\Brent.xlsx', sheet_name='Sheet1')
prices = df_excel["Spot price"].tail(101).tolist()

ewma1 = ewma_calculator(prices, 0.94)
ewma1.returns_cal()
ewma1.daily_vol_cal()
print(ewma1.vol_in_days(5))
```

### GARCH Model 

Generalization of EWMA, the two are very close

The main difference between the two is that, when conditional volatility abruptly spikes, the GARCH(1,1) estimate is somewhat more reactive, updating more drastically

Estimate of the current level of volatility is updated with new return information

Model: Recent volatility + Recent returns (today) + Long-term “point of rest” of volatility or “forever volatility" 

![](https://i.loli.net/2021/02/22/AfRmQgNSu8yU63b.jpg)

large alpha : responsive to daily return, large beta -> less variation

## 5. VAR

Generally, controversial idea, risk limit is more frequently used by practitioners.

Quantiles are points in a distribution that relate to the rank order of values in that distribution.

​	p-quantile X ◦ is smallest value of X s.t. the cumulative probability of X ◦ is at least p

VAR: quantile of the portfolio **loss distribution,** one tail, a loss level that will only be exceeded with a given probability in a given time frame, based on **distribution hypo** (normal or others) on log returns

Assets' market risks more accurately modeled as functions of factor rather than own-price risk (fama-french)

VAR is less accurate at higher confidence interval and longer time horizons

VAR is a poor guide to extreme losses

VAR estimates recurrent losses, the max loss if extreme event does NOT occur 

In fact, short-term volatility forecasts are relatively accurate, compared to forecasts of returns, or to longer-term volatility forecasts

### Confidence Level

VAR at confidence interval alpha = alpha-quantile of the <u>LOSS distribution</u>

VAR at 95 percent, happens every 20 trading days, 99 happens twice a year

The confidence interval: 

​	High ->   provides an estimate of the extremely high, unusual and infrequent losses that the investor seeks to avoid.

​	Low -> an estimate of “main body” risks, the smaller, more regularly occurring losses that, while interesting, are not as crucial an element of risk management. 98 percent, once a month 

### Parametric Method

Assumptions 

- zero mean , log return  r(t, t+T) ~ N(0, vol^2 * T), for short time horizons, drift makes little difference
- constant volatility 
- log return = Z(1-a)* vol  * sqrt(t)

![](https://i.loli.net/2021/02/22/j9Pab3wRl8hZnri.jpg)

we use log return to approximate arithmetic return if numbers are small

the transfer of general normal distribution to standard one, minus mean then divided by the standard deviation.

### Monte Carlo Simulation

1. Independent Simulation
2. Random realization of log return r = vol * normal generated number
3. Order Stats of return / P & L 
4. Find the scenario of our VAR

method: can interpolate scenarios near VAR, average or ... 

use simulation when we are not sure about the distribution

### Historical Simulation

May differs from the two previous method, fatter or thinner tails 

Shorter observation intervals may miss tail events; longer observation intervals may produce results deviating from current return distribution

IF there is strong mean reversion in return volatility, VaR computed via historical simulation may be smaller at a longer than at a shorter horizon

1. set a look_back period ; Use historical risk factor returns but current portfolio position sizes or weights
2. sort m historical realizations RETURNs in ascending order
3. use return to get m ordered simulations of P&L:
4. VAR of long position at confidence level, use quantiles

### Short Position VAR

use α- rather than (1 − α)-quantile
$$
VAR_{short(a,t)} = pos *  ( e^ {z(a) * vol * sqrt(t) }  -1)
$$
VAR(a,t) = pos *  ( exp (z(a) * vol * sqrt(t) )  -1)

VaR slightly larger than for long position (unlimited risk of short position)

## 6. Credit and Counterparty Risk 

Credit is an economic obligation to an “outsider,” an entity that doesn’t own equity in the firm

### Economic Balance Sheet

MKT value of the firm (Assets) = Equity + Debt Obligations

​	Equity owns the company,  takes the hit before debt, limited liability

​	Debt has seniority senior or subordinated 

Leverage: think of equity as down payment to a house or margin of a future 

Many Debt Securities are combinations of debt and equity, for instance, preferred stock, convertible bonds, PIK

### Mechanisms 

Security: unsecured general claim, secured has specific claim on collateral, in bankruptcy 

Secured Credit Transactions: collateral, (e.g. the sales of a auto model as collateral)

A claim on collateral is called a lien on the property

### Bankruptcy 

Securitized first, senior first, short-term first, 

end in reorganization (debt -> equity of the new firm) or liquidation 

the conflict of interest: senior and secured wants liquidation, however, junior and equity want reorganization, want full value back 

### Transaction cost of the credit contracts 

- The main cause is the information acquiring of the borrower's condition 
- Borrower knows more than lender 
- Risk shifting : The classic example is the conflict between equity investors and lenders. Increasing risk to the firm’s assets can benefit the equity investor, since their potential loss is limited to their equity investment, while their potential return is unlimited. Debt holders have no benefit from increased risk, since their return is fixed, only the increased risk of loss. Increasing risk therefore shifts risk from equity to bondholders.
- Moral Hazard/ adverse selection....


### Credit Rating: 

​	higher one,  investment grade;  lower one, speculative grade.

​	rating migration: a matrix of probability 

### Counterparty Risk: 

It arises whenever two conditions both need to be fulfilled (forward) in order for a market participant to profit from an investment. 2 way, not clear who is the borrower who is the lender 

## 7. Default Analytics 

### Exposure at default 

Default (Event):  Caused by insolvency(state), failure to fulfill the terms of a debt contract, cross-default 

Exposure at default : the amount of money the lender can potentially lose in a default.
$$
Exposure  At Default =  Recovery + Loss Given Default (LGD)
$$

$$
Recovery Rate = Recovery / Exposure
$$

 for senior secured bank debt is about 75%, for junior unsecured, it is close to 0. 

Default cluster: high correlation of debt 

Probability of default: π

Exposure at default: for debt, par value + accrued interest

recovery: R

Loss give default: 1- R

Expected Loss: π * ( 1 - R )

Credit VAR : quantile - EL    ( here we do not have zero mean assumption, since we look at longer term)

### Default Intensity Models

Constant hazard rate Assumption, might not be a great assumption, in practice, it is time-varying , 

Long Horizon : default probability converge to 1, every frim will eventually default 

**<u>CDS model</u>** used in practice

term structure, downwards hazard rate, if survive it will continue to survive for long time (2008 Morgan Stanley)

Unconditional one-year default probability lower for remote years

Default probability is memoryless

![](https://i.loli.net/2021/02/22/NK3xGtcgAbMZ9uv.jpg)

#### Hazard Rates

how we get the model, we want a constant default probability in every dt

![](https://i.loli.net/2021/02/22/qOSKeHsoTRB2fZ8.jpg)

### Merton Model

GBM assumption 
$$
dA~(t) = μA(t)dt + σ(A)A(t)dW
$$
Set a threshold for assets, when equity value goes close to zero or assets value close to the par value of the debt 

### Single-obligor credit risk model 

#### Idea: Equity and debt as options 

​	Equity as a long call on firm's assets, with a strike price equal to the par value of the debt, gets extra value of (assets - debt)

​	Debt as a portfolio: riskless bond with the same par value as the debt + an implicit short put on the firm's assets with a strike price equal to the par value of the debt 

​	Debt owners sell the put option to the equity owners

​	use this setup we can use BS formula to derive default probability 

IF assets go below the threshold at the ***end of the time horizon (debt due),** it will default. (if it bounces back, it will not default)

### Single Factor Model 

Market Risk factor +  Idiosyncratic risk factor, build a N (0,1) distribution for the return.

Returns and shocks are measured as deviations from expectations or from a “neutral state"

Default triggered by combination of market and idiosyncratic risk, beta is high, economy has a large impact on the firm's return, firms with high beta tend to have a high default correlation.

![](https://i.loli.net/2021/02/22/RUOT3d7txjfbNzB.jpg)

Used default probability as input π and beta is known; translate the π into default threshold (where the assets may fall below), k =  standard normal quantile of  π

#### Default Risk Decomposition 

Systematic and idiosyncratic risk: fraction of asset return variance explained by variances of  

Market risk factor: β^2 

Idiosyncratic risk factors: 1 - β^2  

#### Conditional Default Distribution of a single obligor

If u know the state of the economy in the future, what is the default probability of the firm. 

Then the mean of the return changes, default distance and probability also changes.

![2021-02-22 (5)](https://i.loli.net/2021/02/23/yNhJUovmxwjkMLS.png)

​                                                                                                                                    Ki is negative number 

![](https://i.loli.net/2021/02/22/Ll274XuD3JMd5WN.jpg)

In the picture, the market negative shock led the default to the left, the variance is smaller in the first picture 

Conditional on this situation, the default situation of each firm is independent.

![](https://i.loli.net/2021/02/22/MfYtcbyqEOBxkJm.jpg)

Higher beta, not much left for the idiosyncratic risk part. We have a tighter pdf, which can also lead to default cluster.

## 8. Credit Portfolios

skewness of credit risk 

![SharedScreenshot](https://i.loli.net/2021/02/23/xlIpqZ9U5maeGV1.jpg)

### Credit VAR 

$$
CreditVAR = \phi(1-\alpha) - ExpectedLoss
$$

Default correlation

![2021-02-23](https://i.loli.net/2021/02/23/rIcYaLnMtws5N1o.png)

**<u>3 things we care about, default prob, default corr, and granularity (n identical loans)</u>**

### Default Correlation

Bernoulli distribution; usually it is relatively small numbers 

2 high default corr -> we tend to have correlated defaults 

### Uncorrelated portfolio 

Simple credit portfolios with uncorrelated defaults; Portfolio of n identical loans or bonds; All pairwise default correlations zero  All default probabilities equal to  π  -> binomial distribution , expected number of default  π  * n 

-> can determine distribution of credit loss (default numbers) 

-> Credit VAR = quantile * loan size- EL( portfolio total par value * π)

### Granularity

​	Many small debt obligations relative to total portfolio; often measured via Herfindahl index; can turn default loss to a kind of cost; granularity / diversification can be offset by high default correlation 

-  Portfolio with low granularity or very low default probability, has binary “0-1” risk, behaves like a coin toss  High probability of no loss, small but material probability π of complete loss  
- Portfolio with high granularity  Has a very narrow range of likely loss rates close to  Loss rate very likely to be close to expected loss/default probability π  
- Portfolio with moderate granularity  Has a wider range of likely outcomes  And thus greater probability of material unexpected loss  Has low probability of complete or near-complete loss  Impact of granularity greater for higher default  probability π  



## 9. Portfolio Credit Risk Models  

### Assumptions

- Idiosyncratic risk is independent - > obligors are identical 
- Conditional independent 
- Obligors are identical 
- Granularity, law of large numbers -> idiosyncratic risk disappears 

### 2 Obligors

product of two betas are the return correlation of two obligors, linked by the market

#### Return and default correlation 

![2021-02-22 (6)](https://i.loli.net/2021/02/23/PcMQCU8asWJvbuH.png)

Joint default probability : their both asset returns are below their own default thresholds

### From conditional default probability to portfolio loss

#### Add Assumptions

- identical obligors with same beta, their pairwise correlation beta^2, default probability  **π** = Φ(k)
- Granularity: homogeneous and completely diversified portfolio
- Zero recovery

-> we have conditional **<u>default probability common to all obligors</u>**, Because of law of large numbers - > <u>**idiosyncratic risk disappears**</u>

-> portfolio loss a function only of market shock 

-> <u>**Fraction X of loans defaulting—portfolio loss rate—equals single-firm default probability p(m), conditional on market shock.**</u>

![2021-02-22 (7)](https://i.loli.net/2021/02/23/cC54kRKhu8ePv62.png)

​             *based on single obligor model, transformed into Z quantile, based on single factor model*

Now we have the distribution of m and its association with the portfolio loss rate, input market shock, get the number of default 

### Distribution of loss rate x

We’ve posited a standard normal distribution for m, from which we can derive distribution of x 

1. Find market shock m that leads to a given loss rate x 

![2021-02-23 (1)](https://i.loli.net/2021/02/23/byveMWziC83mPZ4.png)

2. Probability of loss rate x equals probability of market shock m that leads to it

![2021-02-23 (2)](https://i.loli.net/2021/02/23/FdNhQ3SZAEcLowu.png)

 Random loss rate ˜x below level x ⇔ realized value of market factor m higher ˜ than associated level m

### Impact of Default Probability

- For realistic default probabilities below 50 percent, median portfolio loss rate is below the loan default rate, 0.5 CDF horizontal line in the graph below 
- Low default probability: For moderate correlation, low default probability induces Bernoulli-like, “binary” loss behavior in the portfolio  Loss density very skewed to low loss levels  High likelihood that portfolio losses low  
- High default probability:  Higher likelihood of higher portfolio losses  Loss density more spread out over range of loss levels

![2021-02-23 (3)](https://i.loli.net/2021/02/23/LqwteNKTMSap3WP.png)

### MKT factor and portfolio loss distribution 

default probability is commonly below 50%

![2021-02-23 (5)](https://i.loli.net/2021/02/23/3eAUjqmoaGuwLvT.png)

### Correlation

- Correlation near 1: portfolio behaves as if single loan/obligor; Loss distribution close to binary; P [˜x ≤ ε] (nearly no loss) near 1 − π; P [˜x ≤ 1 − ε] (near-complete loss) near π with ε a tiny positive number ; Low probabilities of intermediate outcomes; Intuition: With high correlation, default clusters very likely 
- Correlation near 0; High probability of portfolio loss rate very close to typical firm’s default probability ; Intuition: With low default rates and low correlation, default clusters close to impossible
- Correlation “in the middle” ; Intuition: with low default rates and intermediate correlation, default clusters are unusual

![2021-02-23 (6)](https://i.loli.net/2021/02/23/c8ih25aO3nZEKet.png)

median: horizontal line; 0 corr -> median close to 5 percent; correlation in the middle

#### Credit VAR

Higher correlation leads to higher Credit VaR ; By increasing likelihood of default clusters

Credit VAR(vertical line) = quantile ( the point) - EL( same for all lines)

![2021-02-23 (8)](https://i.loli.net/2021/02/23/9A3mYSdGsuobewL.png)



## 10. Leverage Risk 

Based on Economic Balance Sheet

Repurchase agreement: security you borrows in order to lend, result in large leverage number 

Leverage = Asset / Equity 

- avoid confusion, alternative definition: debt / asset, regularity leverage ratio; operating leverage.

### Modigliani-Miller Irrelevancy 

Assumptions:

- complete arbitrage
- no taxes
- no bankruptcy costs apart from loss give default
- but debt may be risky

Content: 

- Firm market value is independent of capital structure 
- Only firm asset choices matter for firm value

prop 1 : independence of WACC and leverage

prop 2 : higher leverage increases required equity return 

what matters for return is not financing policy ( it just decide how the return is going to be distributed) , is the value of asset, 

WACC = expected asset return 

intuition : investor can borrow or lend at prevailing rates to undo corporate financing policy for 

![2021-02-24](https://i.loli.net/2021/02/24/MTjgH1ZCvtYlN34.png)

the leverage for the existence of the gap between return on assets and blended cost of debt financing (part of the asset not financed by debt, by equity) .

if the return on assets being lower than the return on debt, no lower limit for losses. so leverage increases risk 

use leverage to amplify the P&L

![2021-02-24 (1)](https://i.loli.net/2021/02/24/wPqCTVO8WdJixIK.png)

### Carry Trade 

by many hedge fund 

search for yield: promised return for clients 

for example, pension fund required return need to meet the obligations for future pension benefits, discounted rate 

low interest rate world, a lot of institutions have to use leverage 

![2021-02-24 (2)](https://i.loli.net/2021/02/24/rg3J9HsypoLj7UN.png)

#### Scenario Analysis 

1. baseline assumptions
2. scenario that assumptions are not met, 
3. calculate the asset price change 

higher leverage, smaller asset price decline is sufficient to wipe out the equity 

#### Fixed income carry trade 

<u>page 14 not required</u> 

**<u>A repurchase agreement (repo)</u>** is a form of short-term borrowing for dealers in government securities. In the case of a repo, a dealer sells government securities to investors usually on an overnight basis, and buys them back the following day at a slightly higher price. That small difference in price is the implicit overnight interest rate. Repos are typically used to raise short-term capital. They are also a common tool of central bank open market operations.

![2021-02-24 (3)](https://i.loli.net/2021/02/24/VZCoYUgkxI8zDjl.png)

*<u>rollover a loan means extend the due day</u>* 

#### Foreign exchange carry 

exchange changes -> value of money market rate you get in the funding currency will also change 

get the carry of high-yield currency, 

Japanese household invest in foreign-currency dominated assets 

![2021-02-24 (5)](https://i.loli.net/2021/02/24/FWDbhwciZ7Sd9kV.png)

### Incentive alignment and capital structure 

### Risk Shifting 

liability -> risk sharing between debt (risk is default, benefit is that fixed rate of return) and equity (unlimited good upside but limited downside)

Limited liability generates incentive for equity owners to take on greater risk at expense of debtholders, additional benefits go to the equity 

Debt holders prefer less risk, baseline is no default

### Debt Overhang

Recovery R,  Default Probability π, Weighted average approach



![2021-03-19 (3)](https://i.loli.net/2021/03/20/1BKyIl3CZLxE79S.png)
$$
Firm Value = \pi * R + (1-\pi)*Value Without Default  
$$

$$
Debt value = \pi * R + (1-\pi)*ParValue
$$

![2021-02-25](https://i.loli.net/2021/02/26/3pSJcYGQBUK2xuR.png)

High default probability, firm value close to debt value, weight on the first half, additional benefits go to the recovery 

Low default, debt value close to the bar, additional benefits go to the equity

## 11. Forms of Leverage

### Background

- funding disintermediation 

- shadow banking security or equity market to find funding rather than banks 
- increased abs, corporate bonds, FDI....
- securities, a new form of risk sharing

Leverage (A/E) since the crisis, hard to measure in the economy, book value is not a good measure of the market value, market value hard to aggregate, debt-to-GDP to measure the leverage of the economy

Leverage of the economy comes mostly from the finance sector, down since the crisis, but still high, collateralized security may be one major source 



### Collateralized security loan 

Usually short term, loan to buy the asset and asset as collateral

SFTs in regulator side means collateral security

Idea is overcollateralization, has many forms, a rising of collateralization requirements 

Rehypothecation is a practice whereby banks and brokers use, for their own purposes, assets that have been posted as collateral by their clients. Clients who permit rehypothecation of their collateral may be compensated either through a lower cost of borrowing or a rebate on fees. In a typical example of rehypothecation, securities that have been posted with a prime brokerage as collateral by a hedge fund are used by the brokerage to back its own transactions and trades.

#### Types

- Margin loans: loans financing security transactions in which the loan is collateralized by the security, purchase asset use loan, the collateral of the loan will be the asset 

- REPO, Repurchase agreements repo, spot sale + forward, in any repo, borrower repo out the bond, lender repo in the bond or engage in reverse repo

- Securities lending in exchange for a fee




Haircuts = margin

Variation margin (the maintenance margin, can due to decline or other events)

### Participants of Repo

For those who temporarily need some assets, one example, short selling, delivery

Mainly through broker dealer 

*Dealer matched-book repo: simultaneously enter repo and reverse-repo with different counterparties, earn spread*

### Types of Repo

mostly treasury

Tri-party repo: third-party custodian values securities, settles trades via transfers of cash and securities between participants’ accounts

(A custodian bank, or simply **custodian**, is a specialized [financial institution](https://en.wikipedia.org/wiki/Financial_institution) responsible for safeguarding a [brokerage firm](https://en.wikipedia.org/wiki/Brokerage_firm)'s or individual's financial assets and is not engaged in "traditional" commercial or consumer/retail banking) 

the biggest repo type  

bank of New York will carry out the clearing 

## 12. Liquidity Risk

### 2 forms

own a asset 2 ways to raise fund, 

Funding liquidity :   

​	Ability to maintain debt-financed positions

Market liquidity :

​	Ability to buy or sell without pushing prices against you

### 2 ways to create assets like cash: 

Maturity transformation : change longer to shorter term debt, borrowing short lending long,

​	Motivations of short-term borrowers and lenders: Borrower pays lower interest rate than he earns on longer-term assets.  Lender has a short-term asset→liquidity transform 

Liquidity transformation: making an asset more readily transformable into goods or other as assets

​	information insensitive debt low monitor cost

​	*most assets used in repo are us treasury bonds (inflation risk, market risk), but they are considered info insensitive, no need to do analysis on us treasury, basically little info comes in that can change the situation.*

Liquidity of banks: Depositor in the bank, fixed deposit money / liquid assets may be asked back by the depositors. Long term funding is essential to commercial banks, for lending. Banks would meet the need of depositor cash back, a bank may be forced to liquidate one of its long-term loan. Banks have stock in liquidity, continuous cash flow, but the amount is smaller than the total amount of depositors , causing liquidity risk. Deposit rate lower-> because loan is used as liquid asset jus like money 

### Funding Liquidity Risk 

- rollover risk : change of terms 

- leman bros short term debt check, called fire sales,

- stock measure : mismatch of inflow and outflow of cash

- large corps with high cash, sensitive to the change of interest in each place (e.g. Apple)


### Market Liquidity Risk

- Determined by search, trade processing costs, and information asymmetries
- Find a counterparty

- cost of execution / clearing 

- info asymmetries   market microstructure 


quote-driven, dealer-based trading system  another one is order-driven like an auction

#### Characteristics of MKT Liquidity

- Tightness: cost of a round trip, bid-ask spread 

- Depth: how large an order can move the market

- Resiliency: how long it takes for market to bounce from order change the price , purely for liquidity reason 


Slippage: change in market price induced by time it takes to get a trade done in a moving market (an insurance want to sell 1000000 IBM shares through JP, JP takes the whole afternoon to get this done, but the price may change during the transaction)

### Liquidity-adjusted VAR

![2021-02-25 (1)](https://i.loli.net/2021/02/26/KPqzgxsMlZi3G9o.png)

A mount of time u need to put on or get out of the position 

*a week to liquidate a position without pushing the price down, on the first day, equal amount on each day, 1/5, adjusted term, larger VAR, if we can liquidate the position instantly, the liquid adjusted VAR will be equal to the VAR we have* 

Private liquidity creation: focus on commercial banks, but insurance companies, and large intermediary can all have this problem

Net interest margin = net interest earning / interest earning assets; difference in maturity, so NIM is positive for commercial banks 1.5-4.5%

### Liquidity Risk and Run

Banks keep reserves of liquid asset, but a small share of the total deposit

bank run : many depositors to redeem their deposit, they are afraid others are also doing

sequential service constraint : first deposit show up get the money back first 

par redemption: pay in cash without delay, which should be provided by banks

demand deposit only works in the no run equilibrium 

![2021-02-25 (2)](https://i.loli.net/2021/02/26/hryGzSmEWYApQTR.png)

No triple A banks , AA is trustworthy 

commercial paper short term not more than a year 

the change in 2008 financial crisis market resilience not willing to lend the banks more than 10 days

## 13.  Extreme Market Risk Events

### Limitations of the standard models

#### Model

distribution of returns are log normal 

parameter of the GBM mu and sigma are constant, 

cannot predict the future using the model

autocorrelation of success return 

#### Reality

Tail risk 

params are time-varying, distribution is also changing 

historical approach is a changing approach 

autocorrelations of daily and lower-frequency returns in fact close to zero

cross-asset correlations vary over time, and with other distributional characteristics

### NON-normality 

Skewness: one sign more

​	normal = 0

​	compare mean to median

​	Conditional skewness: on current vol

Kurtosis: large returns more frequent, "fat tails", 

​	normal = 3, excess kurtosis

### Volatility Asymmetry 

the observation that **volatility** is higher in declining markets than in rising markets.

![2021-03-19 (4)](https://i.loli.net/2021/03/20/Yx9oeQAR3PTzMjV.png)

​                                                      *spike in vol and sharp dip in returns*

causes:

1. Often called leverage effect: hypothesis that negative returns increase leverage, thus raising volatility  Alternatively
2.  volatility feedback: hypothesis that higher volatility requires higher expected return, hence decline in price

Leverage of the firm goes up, vol increase, asset vol may stay the same, equity vol increase, so we may have a decline of value in the equity value.

vol goes up, market value must go down, because of the higher risk exposed, which means a lower price now, if the cash flows have not changed. in order to achieve higher expected returns, 

![2021-03-19 (6)](https://i.loli.net/2021/03/20/pZ7Fqw4UJaWLXg6.png)



### Examining data

history cannot give you enough information, the situation is always evolving,

no observations of negative HPA, 

conditional volatility 



### quantile-quantile plot to compare two distributions

![2021-03-19 (7)](https://i.loli.net/2021/03/20/iFAyHBTag1DhcS7.png)

dashed should be normal, larger negative returns

### Kernel Density Estimate $

alternative distribution hypo

no agreed winner

Jump diffusion an



## 14. Assessing VAR

backtesting of VAR, model validation

required by regulators

### Testable Dimensions

- Unconditional coverage: proportion of exceedance in the entire sample.

- Independence: absence of cluster 

- Magnitude of Exceedances

### Statistical Hypothesis 

H_0: in acceptance region, we accept this

H_1: in critical region, we accept this

likelihood function is the likelihood of what you see under H_0

confidence level alpha, 95%, happen every 20 trading days 

basket is a sequence of comparisons of VAR estimate with P&L realized in the VAR forecast horizon

under H_0, comparisons are Bernoulli trials :

- 1-a, result is 1, VAR exceeded 
- a, VAR not exceeded 

In theory, i.i.d. , in practice, we have exceedances clustered 

we are not assuming lognormal returns

### log likelihood ratio

![2021-03-20](https://i.loli.net/2021/03/20/BqpySeit4VaOglN.png)

the gap between the maintain the hypo, the value of the likelihood you 

in order to not reject, the hypo, you want the log likelihood ratio to be small.

### Chi-square Test

Test statistic measures distance between data and model prediction 

Follows a χ 2 distribution (for large enough T) if H0 is true 

- With one degree of freedom (df), for the one parameter α 
- χ 2 test a standard approach to assessing goodness of fit of a distributional hypothesis 
- In this case, exceedances i.i.d. Bernoulli trials with parameter α 

 p-value: probability, if H0 true, of a test statistic greater than or equal to that actually obtained in the sample  I.e., the CDF of a χ 2 [1] variate at a value equal to the test statistic

#### Process

- significance level -> critical value of CDF of chi-square 

- acceptance range in no. of outliers: test statistic < critical value , wider at a low significance level 

- compute VAR and no. of outliers (for daily VAR, each day's VAR compared with tomorrow's P&L

  weekly: you have to use today and next week, cannot have overlapped time horizon)

be aware 

- Confidence level of VaR enters into test statistic (together with number of observations, number of exceedances) 
- Significance level of backtest determines χ 2 quantile to compare (together with number of degrees of freedom)
- Confidence level to construct the VAR
- Significance level is high, loose criteria to accept the VAR
- too high or too low, we reject H_0
- too few exceedance, too loose criteria

#### Example:

in long and short position basketing

![2021-03-20 (1)](https://i.loli.net/2021/03/20/3oNIZnvGwJ1ERq5.png)



![2021-03-20 (2)](https://i.loli.net/2021/03/20/gpL5qnbaQkxBRsi.png)

reject the H_0 to sp position but not the aud pos

### Critique of VAR

but the magnitude does not go in to play

also we want to assess the clustering 



## MKT risk management in practice

### Nonlinearity

The derivative of the security price to the risk factor has second and higher terms, also called convexity in options and bond markets

#### VAR techniques for Nonlinearity 

1. use simulation based on pricing model, may not be practical, but accurate
2. Delta-gamma, use the first and second derivatives to approximate 

#### The Greeks 

Delta  for EU (-1,0), (0,1)

Gamma is positive for long option position, negative for short

![2021-03-23 (1)](https://i.loli.net/2021/03/23/JnZHymX39ocUpIC.png)

- Gamma dampens P&L for long option positions and amplifies P&L for short option positions 
- High γt reduces VaR for a long option position and increases VaR for a short option position 
- Difference between P&L results of large and very large underlying price changes is also greater for short positions

Analysis based on the 1/2 second order term, draw a Tangent Line

Delta-gamma approximation is more convex than whole bs model



## Extreme Test

### Stress Test 

- Determine appropriate scenarios 

- Calculate shocks to risk factors in each scenario 
- value the portfolio 

Objectives

- address the tail risk
- reduce model risk by reducing reliance on models
- know the vulnerabilities



### Expected Shortfall

or called conditional/tail VAR

average value of the truncated loss distribution
$$
E[x|x<VAR]
$$
can use the same simulation the same as the VAR

#### Historical simulation

expected shortfall : take the average of the all **<u>top worst scenario</u>** 

 

ratio of expected shortfall to VAR is higher at lower confidence levels,

have more distribution involved

Expected shortfall for tail risk, work only we measure the tail risk well



## Securitization

types

- Securitization: cash securities backed by mortgages, customer debt or leases
- Structured credit/finance: securities backed by bank debt or bonds or securitization in synthetic credit derivatives form
- Collateralized Debt Obligations: on asset pool of bank loans 

why 

- diversification

- risk transfer: separation of loan origination from balance-sheet investment/use of capital

- risk distribution : different risk-reward


risk types in collateral pool

- prepayment risk: when interest goes down, so they can refinance 

- credit risk


### Securitization Structure 

Tranches: Equity note (only receive money if the seniors are paid in full), Mezzanine Note, Senior Note.

The overcollateralization of senior note is Mezzanine Note and Equity Note.

Waterfall: losses from bottom up, equity takes the hit first 

attachment point: start to losing money

detachment point: wiped out 

if thin, binary risk, default correlation, can be completely wiped out

overcollateralization

example p17

![2021-04-04](https://i.loli.net/2021/04/05/eikxYUd2PApJnQ4.png)

credit enhancement of the Senior bond is mezzanine and equity 

Mezzanine bond has equity

equity is not promised by anything, no legal results

### Stipulated Cash Flows 

P & I : principle and interest 

a:  attachment point

![2021-04-04 (1)](https://i.loli.net/2021/04/05/xcTaKojykR6gU8M.png)



## Structured Credit Risk 

default rate x

![2021-04-06](https://i.loli.net/2021/04/06/lTO1djg635Jpzkw.png)
$$
SeniorReturn(x)= SeniorCashFlow(x)/(1-a_s)-1
$$

$$
MezzanineReturn(x)= MezzanineCashFlow(x)/(a_s-a_s)-1
$$

$$
EquityReturn(x)= EquityCashFlow(x)/(a_m)-1
$$



**compare the default and stipulated** 

Even in stress case  senior can still get returns 

equity however easily got wiped out in the stress case

but 122.5% return as max 

### Option-like tranche

Strike levels: attachment/detachment points

- Senior tranche behaves like a “short call” on loan pool proceeds 
- Mezzanine tranche behaves like a “collar” on loan pool proceeds 
- Equity tranche behaves like a “long put” on loan pool proceeds



![2021-04-06 (1)](https://i.loli.net/2021/04/06/ElbJfX6K7z5WNpo.png)

### Embedded leverage 

option -> leverage

equity is leveraged by mezzanine and senior 

default of a tranche

the equity tranche cannot default legally

### Risk Modelling

p14 a case of auto loan single factor model(sfm)

1. probability of loss and correlation of SFM
2. use waterfall for each tranche 
3. calculate the one to one return for each of the tranche -> CDF

p19

2 tails in p20 in CDF

zero recovery 

P20 senior tranches hit by a lighting 

P21 with more corr, the loss rate goes down, things won't happen at all or it just happen

p18 the x parameter is from p10, the loss level for the tranches -> for the default of the tranche, which is the start 

understanding of the mezzanine between senior and equity 



## Financial Crisis 

### Facts

- high external equity debt, turkey external debt equity is more than government 

- in eastern EU, people borrow in Suisse banks for mortgage and live in Euros, exchange rate and interest rate 

- within EU, government borrow money easily for law reasons

- when government credit is highly associated with the banks, sovereigns                                                           

- in a recession, reluctant to lend money, even the project is profitable, the profit will just go to debt holder, not the shareholders 


### Causes of the crisis 

- macro-eco

- financial imbalances

- reaching for yield
- endogeneity

Monetary Policy: interest low, increased leverage, volatility high means the risk is realized by the investors, low volatility encourages risk-taking, so we have more and more risk in the market, financial imbalances building happens

How things become a market crisis

variance risk premium: difference between implied volatility and expected volatility 



### Reaching for yield 

agent money manager , they have a yield target, they under pressure 

p25 property insurance, ratings go down

Interest rate goes down, so the insurance companies were forced to make riskier investment to keep profit. As shown in the picture in p25, it leads to more lower ratings.



### Crisis

triggers and propagation, usually the combination of the two 

if the financial system is stable, then the trigger is the shock

if small trigger , contagion to explain why becomes a crisis



#### Common shocks 

bad underwriting, policy error ( not save Lehman)

#### Endogeneity

the shock can spread and magnify, loops( participants), internalities, procyclicality (shock reaching for breaking point)

#### Contagion

inherited contagion of the financial system 

interconnectedness: derivatives like swap involves many parties, so we have clearing mandate,(one contract becomes two)



## Financial Regulation 

EU is more advanced in this regulation 

us and uk systems are old, sometimes out of dated 

the system is complex 

institution is important concept 



2 things get regulated : firm and market 

fed and local system 

### Banks

dual banking system: most banks are state banks and we also have national banks 

fed is regulator of banks and holdings ( hold bank), chartered state banks, us foreign banking(foreign banks set subsidiary , or open a branch ). 

FDIC regulates national and state-chartered banks with (→)insured deposits

each state has regulatory agent



### non bank intermediary 

SEC regulates broker dealer 

also stock market participants (short selling institution)

insurance has no national regulator, only local 



DEBATE: regulation should not involve central bank 

central banks should provide collateralized liquidity 

insolvency whether to bail out the bank should be up to tax payers, which is a government issue.

government change when reelection, but not for central banks

contradiction in regulatory and monetary policies  



### International 

they set standards, national agents adapt 

jurisdiction set the rules, political process required 

G10 

fed is not comfortable with leverage loans

A leveraged loan is a type of loan that is extended to companies or individuals that already have considerable amounts of debt or poor credit history.



### Methods of regulation

- chartering and scope restriction **bank charter** - a **charter** authorizing the operation of a **bank**. **charter** - a document incorporating an institution and specifying its rights; includes the articles of incorporation and the certificate of incorporation.
- resolution mechanism 
- acct standards
- on-site supervision, fed has staff inside the banks, jp morgan.. bank of am
- deposit insurance, EU deposit not welcomed in some countries 
- macroprudential 

In EU, investment banks, commercial banks, brokers, dealers, can be one

but in us, not ok, loosen up though recently

narrow banking, equity-financed lending only, separation of bank's payment and lending systems

#### Rating 

camel rating

RFI rating 

#### Deposit insurance 

prevent run, protection of depositors 



### Post-crisis Regulatory Reform

single supervisory mechanism, 

Volcker rule bans proprietary trading by banks 

FSOC 

FSB : global systemically important banks



Economic arguments

principle-agent problems in financial system : regulation of compensation 



## Regulatory Capital Standards

Bailout or not 

taxpayer pay or run of the institutions

risk weighted assets 

leverage-based capital 



### Loss absorbency hierarchy 

Tier 1: first loss components, common equity, retained earning, going concern, should be able to survive even suffered large loss

Common Equity Tier 1 (CET1) is a component of Tier 1 capital that is mostly common stock held by a bank or other financial institution. It is a [capital ](https://www.investopedia.com/terms/c/capital.asp)measure introduced in 2014 as a precautionary means to protect the economy from a financial crisis. It is expected that all banks should meet the minimum required CET1 ratio of 4.5% by 2019.

Tier 2: supplementary capital subordinated debt, preferred stock and loan loss reserves, protect taxpayer and most senior liabilities 

Bail-in-able liabilities: other forms of longer-term unsecured subordinated debt



### Scope of risk-based capital 

regulatory arbitrage: higher lower 



### Key components of capital standards

banking book: held to maturity 

trading books: positions for liquidity, market making and proprietary trading 



### credit risk weighted assets

standard approach: assign larger weight to riskier assets, use rating 

IRB approach: internal model, bank computes risk weights



### Market risk 

general mkt risk: shocks 

default risk: idiosyncratic risk 

residual risk

Standardized approach

internal models approach: VAR -> FRTB, folder 

stressed VAR

Risk-based capital: buffers 

Leverage-based capital: exposure measure and adjusted assets 

Total loss absorbing capacity 



![2021-04-07](https://i.loli.net/2021/04/07/dSCJjU5IMlqB8Dk.png)

​                                                           table summary of regulatory requirements

**Pillar 3** requires firms to publicly disclose information relating to their risks, capital adequacy, and policies for managing risk with the aim of promoting market discipline. called pillar 3 disclosure of BASEL framework  

![2021-04-07 (1)](https://i.loli.net/2021/04/07/DyEoe1NF2gc4OZu.png)

operational risk one third

credit risk for BAC DB, commercial banks 

MS broker dealer business 

DB tier 1 leverage is close to the standard 

RWA density : risk-weighted assets is small done by internal computation.



## Liquidity Regulation 

### Basel liquidity standards 

Liquidity Coverage Ratio: 30 days stress cash outflows 

Net Stable funding ratio NSFR:  liability 

Liquidity Coverage Ratio 



### Liquidity Coverage Ratio 

the key stone, more important than NSFR

30 days stress test on asset

high quality liquid asset  = HQLA
$$
HQLA/NetOutflow_{30days} = LCR > 1
$$

$$
Total Net Outflows = Outflows - min(inflows, 0.75outflows )
$$

Operating deposits: used by depositors for day-to-day business and to support transactions, retail deposit compliance with LCR, does not immunize bank from funs

Nonoperating deposits: used primarily as investments or liquidity reserve rather than to support transactions

p10

p11 



level 2 cannot be higher than a proportion, banks cannot solely rely on level 2

the technical requirement of the regulation, level 2 

In finance, a haircut is the difference between the current market value of an asset and the value ascribed to that asset for purposes of calculating regulatory capital or loan collateral.




### Net Stable Funding Ratio

Asset and liability 

$$
NSFR = Available Stable Funding /Required Stable Funding>=1
$$
p12

matched book : chain links in the market of funding and investing 



## Financial market impact of crises and policy responses



dollar appreciation during the crisis 

swap plot can also show 

### solvency and liquidity 

1. cash flow solvency: meet the liability as they fall due
2. balance sheet solvency: asset is bigger than liability 

meet one does not mean meet the other 

liquidity hoarding 

nowadays bank failures are uncommon 



























