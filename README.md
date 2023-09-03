# trading strategy analysis with Python
tests the hypothesis that it is possible to "beat the market" using a swing trading strategy that is based on technical analysis.

Introduction
Amid debates on trading strategies among students, many are venturing into algorithmic trading societies or individual trading. Instead of adopting my coursemates' strategies blindly, I sought to evaluate their long-term profitability potential in the stock market.

Methodology
Stock Selection and Data Collection
I selected prominent stocks from portfolios of renowned institutions like Bridgewater Associates and Berkshire Hathaway. Using a third-party API, I retrieved six years' worth of stock price data for these selections.
Swing Trading Logic Evaluation
I simulated trading for selected stocks from 2021 to 2023 using various trading logics. The rules were crafted to mirror real-world trading scenarios. For instance, a new stock could only be bought after the previous one was sold, ensuring account balance longevity. The best-performing logic combinations were identified based on their ability to spot trend reversals and consistently realize profits. The trading logics that looked the most promising were later combined together to supplement each other's weaknesses.
Technical Indicator Parameter Optimization
Despite the optimized trading logic, other factors like stock volatility and market cap influenced trading performance when trading multiple different stocks. To address this, I introduced the SWEEP analysis. It iterated approximately 20,000 times over the 2021-2023 period, with varying technical indicator combinations in each iteration. The top-performing combinations in terms of profit, efficiency, and optimal efficiency were recorded.
Real-world Scenario Testing
To address overfitting concerns from SWEEP, I conducted another simulation using data from 2017-2021. This simulation showed me how well the strategy would perform if it was given a new set of data that was not used in the technical indicator optimisation process. As anticipated, returns were lower due to the absence of fitting, yet some strategies still outperformed the market.

Findings
About 60% of the tested stocks yielded higher-than-market gains using the swing trading strategy. However, these strategies require periodic re-evaluation due to stock movement's ever-changing nature and were not tested against global recessions like the 2008 crisis. Additionally, the assumed transactional ease during simulations may not reflect real-world trading conditions, particularly with high trading volumes.

Future Directions
While the results are promising, there's room for enhancement. Machine learning could validate transaction signals further. Expanding the stock list, incorporating company earnings dates, and tailoring unique trading logics for each stock could also optimize outcomes. In conclusion, with a curated stock list and finely-tuned strategies, it's plausible to outperform the market, albeit with periodic re-evaluations and low trading volumes.



