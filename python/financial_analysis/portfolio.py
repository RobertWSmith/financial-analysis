import dataclasses
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclasses.dataclass
class Portfolio:
    # dataframe indexed by date, column names of the portfolio components
    portfolio: pd.DataFrame
    # series indexed by date, named by market returns
    market_data: pd.Series
    # risk free rate for one month
    risk_free_rate: float

    @property
    def annual_risk_free_rate(self) -> float:
        return self.risk_free_rate * 12.

    @property
    def portfolio_returns(self) -> pd.DataFrame:
        return self.portfolio.pct_change().dropna()

    @property
    def market_data_returns(self) -> pd.Series:
        return self.market_data.pct_change().dropna()

    def covariance(self) -> pd.DataFrame:
        return pd.DataFrame(
            np.cov(np.transpose(self.portfolio_returns)),
            columns=self.portfolio.columns,
            index=self.portfolio.columns
        )

    def portfolio_statistics(self) -> pd.DataFrame:
        return self.portfolio_returns.describe(include=[np.number])

    def portfolio_market_fits(self) -> Dict[str, Any]:
        output = dict()
        exog = sm.add_constant(self.market_data_returns)
        for field in list(self.portfolio_returns):
            mdl = sm.OLS(self.portfolio_returns[field], exog, hasconst=True)
            fit = mdl.fit(use_t=True)
            output[field] = fit
        return output

    def portfolio_to_market(self) -> pd.DataFrame:
        tickers = list()
        for key, result in self.portfolio_market_fits().items():
            summary = dict(ticker=key)
            summary.update(dict(zip(("alpha", "beta"), result.params.values)))
            summary.update(dict(zip(("alpha_t", "beta_t"), result.tvalues.values)))
            tickers.append(summary)
        return pd.DataFrame(tickers)

    def global_minimum_variance_portfolio(self) -> Tuple[pd.Series, pd.Series]:
        col_ones = np.ones((len(self.portfolio.columns), 1))

        cov = self.covariance()
        cov_inv = np.linalg.inv(cov)

        numerator = np.matmul(cov_inv, col_ones)
        denominator = np.sum(numerator)

        mvp = numerator / denominator
        mvp = pd.Series([x[0] for x in mvp.tolist()], index=list(self.portfolio), dtype=np.float64)

        portfolio_means = self.portfolio.mean(axis=0)

        stats = pd.Series(dtype=np.float64)
        stats["mean"] = np.dot(portfolio_means, mvp)

        return mvp, stats

    # alias
    gmvp = global_minimum_variance_portfolio

    # def efficient_portfolio_with_short_sales(self):
