import dataclasses
import enum
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm


class DataFrequency(enum.IntEnum):
    daily = 253
    weekly = 52
    monthly = 12


@dataclasses.dataclass
class Portfolio:
    # dataframe indexed by date, column names of the portfolio components
    portfolio: pd.DataFrame

    # series indexed by date, named by market returns
    market_data: pd.Series

    # risk free rate for one month
    risk_free_rate: float

    data_frequency: DataFrequency = DataFrequency.monthly

    @property
    def annual_risk_free_rate(self) -> float:
        return self.risk_free_rate * float(self.data_frequency.value)

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

    @staticmethod
    def portfolio_statistics(
            weights: np.ndarray,
            mean_returns: np.ndarray,
            covariance_matrix: np.ndarray
    ) -> Dict[str, Any]:
        variance = np.matmul(np.matmul(np.transpose(x), covariance_matrix), x)

        return {
            "weights":           weights,
            "mean_returns":      mean_returns,
            "covariance_matrix": covariance_matrix,
            "mean":              np.dot(mean_returns, weights),
            "variance":          variance,
            "sigma":             np.sqrt(variance)
        }

    @staticmethod
    def portfolio_comparison(
            portfolio_one: Dict[str, float],
            portfolio_two: Dict[str, float]
    ) -> Dict[str, Any]:
        covariance = np.matnul(
            np.matmul(
                np.transpose(portfolio_one["weights"]),
                portfolio_one["covariance"]
            ),
            portfolio_two["weights"]
        )
        correlation = covariance / (portfolio_one["sigma"] * portfolio_two["sigma"])
        return {
            "portfolio_one": portfolio_one,
            "portfolio_two": portfolio_two,
            "covariance":    covariance,
            "correlation":   correlation,
        }

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

        z = np.matmul(cov_inv, col_ones)
        denominator = np.sum(z)

        mvp = z / denominator
        mvp = pd.Series([x[0] for x in mvp.tolist()], index=list(self.portfolio), dtype=np.float64)

        portfolio_means = self.portfolio.mean(axis=0)

        stats = pd.Series(dtype=np.float64)
        stats["mean"] = np.dot(portfolio_means, mvp)
        stats["variance"] = np.matmul(np.matmul(np.transpose(x), covariance_matrix), x)
        stats["sigma"] = np.sqrt(stats["variance"])

        return mvp, stats

    def tangent_portfolio(self) -> Tuple[pd.Series, pd.Series]:
        mean_rets = self.portfolio_returns.mean() - self.risk_free_rate
        market_rets = self.market_data_returns.mean() - self.risk_free_rate

        covariance_matrix = self.covariance()
        z = np.matmul(np.linalg.inv(covariance_matrix), mean_rets - market_rets)
        denominator = np.sum(z)
        x = z / denominator
        x = pd.Series([y[0] for y in x.tolist()], index=list(self.portfolio), dtype=np.float64)

        portfolio_means = self.portfolio.mean(axis=0)

        stats = pd.Series(dtype=np.float64)
        stats["mean"] = np.dot(portfolio_means, x)
        stats["variance"] = np.matmul(np.matmul(np.transpose(x), covariance_matrix), x)
        stats["sigma"] = np.sqrt(stats["variance"])

        return x, stats

    def efficient_frontier(self):
        gmvp = self.global_minimum_variance_portfolio()
        tp = self.tangent_portfolio()
