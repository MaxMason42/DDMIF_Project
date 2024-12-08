import numpy as np
import pandas as pd

from typing import Dict, List, Tuple

from empyrical import (
    sharpe_ratio,
    calmar_ratio,
    sortino_ratio,
    max_drawdown,
    downside_risk,
    annual_return,
    annual_volatility
)


def calc_performance_metrics(data: pd.DataFrame, metric_suffix="", num_identifiers = None):
    if not num_identifiers:
        num_identifiers = len(data.dropna()["ticker"].unique()) #Might just make ticker or permno
    
    srs = data.dropna().groupby(level=0)["captured_returns"].sum() / num_identifiers
    
    return {
        f"annual_return{metric_suffix}": annual_return(srs),
        f"annual_volatility{metric_suffix}": annual_volatility(srs),
        f"sharpe_ratio{metric_suffix}": sharpe_ratio(srs),
        f"downside_risk{metric_suffix}": downside_risk(srs),
        f"max_drawdown{metric_suffix}": -max_drawdown(srs),
        f"perc_pos_return{metric_suffix}": len(srs[srs > 0.0]) / len(srs),
        f"profit_loss_ratio{metric_suffix}": np.mean(srs[srs > 0.0])/ np.mean(np.abs(srs[srs < 0.0]))
    }


def calc_performance_metrics_subset(srs: pd.Series, metric_suffix=""):
     return {
        f"annual_return{metric_suffix}": annual_return(srs),
        f"annual_volatility{metric_suffix}": annual_volatility(srs),
        f"downside_risk{metric_suffix}": downside_risk(srs),
        f"max_drawdown{metric_suffix}": -max_drawdown(srs)
    }


def calc_net_returns(data: pd.DataFrame, list_basis_points: List[float], identifiers = None):
    if not identifiers:
        identifiers = data["identifier"].unique().tolist()
    cost = np.atleast_2d(list_basis_points) * 1e-4

    dfs = []
    for i in identifiers:
        data_slice = data[data["identifier"] == i].reset_index(drop=True)
        annualised_vol = data_slice["daily_vol"]*np.sqrt(252)
        scaled_position = data_slice["position"] #Was scaled by volatility but don't really care
        transaction_costs =  scaled_position.diff().abs().fillna(0.0).to_frame().to_numpy()* 0 #cost for transaction costs
        net_captured_returns = data_slice[["captured_returns"]].to_numpy() - transaction_costs
        columns = list(map(lambda c: "captured_returns_" + str(c).replace(".", "_") +"_bps", list_basis_points))
        dfs.append(pd.concat([data_slice, pd.DataFrame(net_captured_returns, columns=columns)], axis=1))
    return pd.concat(dfs).reset_index(drop=True)


def calc_sharpe_by_year(data: pd.DataFrame, suffix=""):
    data = data.copy()
    data["year"] = data.index.year

    sharpes = (
        data.dropna()[["year", "captured_returns"]]
        .groupby(level=0)
        .mean()
        .groupby("year")
        .apply(lambda y: sharpe_ratio(y["captured_returns"]))
    )

    sharpes.index = "sharpe_ratio_" + sharpes.index.map(int).map(str) + suffix

    return sharpes.to_dict()


def calc_returns(srs: pd.Series, day_offset: int = 1):
    returns = srs / srs.shift(day_offset) - 1
    return returns


def calc_daily_vol(daily_returns):
    return (
        daily_returns.ewm(span = 60, min_periods = 60).std().fillna(method="bfill")
    )

