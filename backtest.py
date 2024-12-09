# Based on the papers:
# 
# @article{wood2021trading,
#   title={Trading with the Momentum Transformer: An Intelligent and Interpretable Architecture},
#   author={Wood, Kieran and Giegerich, Sven and Roberts, Stephen and Zohren, Stefan},
#   journal={arXiv preprint arXiv:2112.08534},
#   year={2021}
# }
# 
# @article {Wood111,
# 	author = {Wood, Kieran and Roberts, Stephen and Zohren, Stefan},
# 	title = {Slow Momentum with Fast Reversion: A Trading Strategy Using Deep Learning and Changepoint Detection},
# 	volume = {4},
# 	number = {1},
# 	pages = {111--129},
# 	year = {2022},
# 	doi = {10.3905/jfds.2021.1.081},
# 	publisher = {Institutional Investor Journals Umbrella},
# 	issn = {2640-3943},
# 	URL = {https://jfds.pm-research.com/content/4/1/111},
# 	eprint = {https://jfds.pm-research.com/content/4/1/111.full.pdf},
# 	journal = {The Journal of Financial Data Science}
# }

import os
from typing import Tuple, List, Dict
import tensorflow as tf
import pandas as pd
import datetime as dt
import numpy as np
import shutil
import gc
import copy

import json

from model_inputs import ModelFeatures
from deep_momentum_network import LstmDeepMomentumNetworkModel
from momentum_transformer import TftDeepMomentumNetworkModel
from classical_strategies import (
    calc_performance_metrics,
    calc_performance_metrics_subset,
    calc_sharpe_by_year,
    calc_net_returns,
    annual_volatility
)

os.environ['CUDA_HOME'] = r"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2" 
os.environ['PATH'] += r";C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin" 
os.environ['PATH'] += r";C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/extras/CUPTI/libx64" 
os.environ['PATH'] += r";C:/tools/cuda/bin"


physical_devices = tf.config.list_physical_devices("GPU")
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


def _get_directory_name(experiment_name: str, train_interval: Tuple[int, int, int] = None):
    if train_interval:
        return os.path.join(
            "results", experiment_name, f"{train_interval[1]}-{train_interval[2]}"
        )
    else:
        return os.path.join(
            "results",
            experiment_name,
        )


def _basis_point_suffix(basis_points: float = None):
    if not basis_points:
        return ""
    return "_" + str(basis_points).replace(".", "_") + "_bps"


def _interval_suffix(train_interval: Tuple[int, int, int], basis_points: float = None):

    return f"_{train_interval[1]}_{train_interval[2]}" + _basis_point_suffix(
        basis_points
    )


def _results_from_all_windows(experiment_name: str, train_intervals: List[Tuple[int, int, int]]):
    return pd.concat(
        [
            pd.read_json(
                os.path.join(
                    _get_directory_name(experiment_name, interval), "results.json"
                ),
                # typ="series",
            )
            for interval in train_intervals
        ]
    )



def _get_asset_classes(asset_class_dictionary: Dict[str, str]):
    return np.unique(list(asset_class_dictionary.values())).tolist()


def _captured_returns_from_all_windows(
    experiment_name: str,
    train_intervals: List[Tuple[int, int, int]],
    volatility_rescaling: bool = False,
    only_standard_windows: bool = True,
    volatilites_known: List[float] = None,
    filter_identifiers: List[str] = None,
    captured_returns_col: str = "captured_returns",
    standard_window_size: int = 1):


    srs_list = []
    volatilites = volatilites_known if volatilites_known else []
    for interval in train_intervals:
        if only_standard_windows and (
            interval[2] - interval[1] == standard_window_size
        ):
            df = pd.read_csv(
                os.path.join(
                    _get_directory_name(experiment_name, interval),
                    "captured_returns_sw.csv",
                ),
            )

            if filter_identifiers:
                filter = pd.DataFrame({"identifier": filter_identifiers})
                df = df.merge(filter, on="identifier")
            num_identifiers = len(df["identifier"].unique())
            srs = df.groupby("time")[captured_returns_col].sum() / num_identifiers
            srs_list.append(srs)
            if volatility_rescaling and not volatilites_known:
                volatilites.append(annual_volatility(srs))
    if volatility_rescaling:
        return pd.concat(srs_list) * 0.15 / np.mean(volatilites)
    else:
        return pd.concat(srs_list)



def save_results(results_sw: pd.DataFrame, output_directory: str, train_interval: Tuple[int, int, int],
                num_identifiers: int, asset_class_dictionary: Dict[str, str], extra_metrics: dict = {}):

    asset_classes = ["ALL"]
    results_asset_class = [results_sw]
    if asset_class_dictionary:
        results_sw["asset_class"] = results_sw["identifier"].map(
            lambda i: asset_class_dictionary[i]
        )
        classes = _get_asset_classes(asset_class_dictionary)
        for ac in classes:
            results_asset_class += [results_sw[results_sw["asset_class"] == ac]]
        asset_classes += classes

    metrics = {}
    for ac, results_ac in zip(asset_classes, results_asset_class):
        suffix = _interval_suffix(train_interval)
        if ac == "ALL" and extra_metrics:
            ac_metrics = extra_metrics.copy()
        else:
            ac_metrics = {}
        for basis_points in [None, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
            suffix = _interval_suffix(train_interval, basis_points)
            if basis_points:
                results_ac_bps = results_ac.drop(columns="captured_returns").rename(
                    columns={
                        "captured_returns"
                        + _basis_point_suffix(basis_points): "captured_returns"
                    }
                )
            else:
                results_ac_bps = results_ac

            ac_metrics = {
                **ac_metrics,
                **calc_performance_metrics(
                    results_ac_bps.set_index("time"), suffix, num_identifiers
                ),
                **calc_sharpe_by_year(
                    results_ac_bps.set_index("time"), _basis_point_suffix(basis_points)
                ),
            }
        metrics = {**metrics, ac: ac_metrics}

    with open(os.path.join(output_directory, "results.json"), "w") as file:
        file.write(json.dumps(metrics, indent=4))



def aggregate_and_save_all_windows(experiment_name: str, train_intervals: List[Tuple[int, int, int]],
                                    asset_class_dictionary: Dict[str, str], standard_window_size: int):

    directory = _get_directory_name(experiment_name)
    all_results = _results_from_all_windows(experiment_name, train_intervals)

    _metrics = [
        "annual_return",
        "annual_volatility",
        "sharpe_ratio",
        "downside_risk",
        "sortino_ratio",
        "max_drawdown",
        "calmar_ratio",
        "perc_pos_return",
        "profit_loss_ratio",
    ]
    _rescaled_metrics = [
        "annual_return_rescaled",
        "annual_volatility_rescaled",
        "downside_risk_rescaled",
        "max_drawdown_rescaled",
    ]

    metrics = []
    rescaled_metrics = []
    for bp in [None, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        suffix = _basis_point_suffix(bp)
        metrics += list(map(lambda m: m + suffix, _metrics))
        rescaled_metrics += list(map(lambda m: m + suffix, _rescaled_metrics))

    if asset_class_dictionary:
        asset_classes = ["ALL"] + _get_asset_classes(asset_class_dictionary)
    else:
        asset_classes = ["ALL"]

    average_metrics = {}
    list_metrics = {}

    asset_class_tickers = (
        pd.DataFrame.from_dict(asset_class_dictionary, orient="index")
        .reset_index()
        .set_index(0)
    )

    for asset_class in asset_classes:
        average_results = dict(
            zip(
                metrics + rescaled_metrics,
                [[] for _ in range(len(metrics + rescaled_metrics))],
            )
        )
        asset_results = all_results[asset_class]

        for bp in [None, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
            suffix = _basis_point_suffix(bp)
            average_results[f"sharpe_ratio_years{suffix}"] = []
        # average_results["sharpe_ratio_years_std"] = 0.0

        for interval in train_intervals:
            # only want full windows here
            if interval[2] - interval[1] == standard_window_size:
                for m in _metrics:
                    for bp in [None, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
                        suffix = _interval_suffix(interval, bp)
                        average_results[m + _basis_point_suffix(bp)].append(
                            asset_results[m + suffix]
                        )

            for bp in [None, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
                suffix = _basis_point_suffix(bp)
                for year in range(interval[1], interval[2]):
                    average_results["sharpe_ratio_years" + suffix].append(
                        asset_results[f"sharpe_ratio_{int(year)}{suffix}"]
                    )
        for bp in [None, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
            suffix = _basis_point_suffix(bp)
            all_captured_returns = _captured_returns_from_all_windows(
                experiment_name,
                train_intervals,
                volatility_rescaling=True,
                only_standard_windows=True,
                volatilites_known=average_results["annual_volatility" + suffix],
                filter_identifiers=(
                    None
                    if asset_class == "ALL"
                    else asset_class_tickers.loc[
                        asset_class, asset_class_tickers.columns[0]
                    ].tolist()
                ),
                captured_returns_col=f"captured_returns{suffix}",
            )
            yrs = pd.to_datetime(all_captured_returns.index).year
            for interval in train_intervals:
                if interval[2] - interval[1] == standard_window_size:
                    srs = all_captured_returns[
                        (yrs >= interval[1]) & (yrs < interval[2])
                    ]
                    rescaled_dict = calc_performance_metrics_subset(
                        srs, f"_rescaled{suffix}"
                    )
                    for m in _rescaled_metrics:
                        average_results[m + suffix].append(rescaled_dict[m + suffix])

        window_history = copy.deepcopy(average_results)
        for key in average_results:
            average_results[key] = np.mean(average_results[key])

        for bp in [None, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
            suffix = _basis_point_suffix(bp)
            average_results[f"sharpe_ratio_years_std{suffix}"] = np.std(
                window_history[f"sharpe_ratio_years{suffix}"]
            )

        average_metrics = {**average_metrics, asset_class: average_results}
        list_metrics = {**list_metrics, asset_class: window_history}

    with open(os.path.join(directory, "average_results.json"), "w") as file:
        file.write(json.dumps(average_metrics, indent=4))
    with open(os.path.join(directory, "list_results.json"), "w") as file:
        file.write(json.dumps(list_metrics, indent=4))



def run_single_window(
    experiment_name: str,
    features_file_path: str,
    train_interval: Tuple[int, int, int],
    params: dict,
    changepoint_lbws: List[int],
    skip_if_completed: bool = True,
    asset_class_dictionary: Dict[str, str] = None,
    hp_minibatch_size: List[int] = [64, 128, 256]
):
    
    directory = _get_directory_name(experiment_name, train_interval)

    if skip_if_completed and os.path.exists(os.path.join(directory, "results.json")):
        print(
            f"Skipping {train_interval[1]}-{train_interval[2]} because already completed."
        )
        return

    raw_data = pd.read_csv(features_file_path, index_col=0, parse_dates=True)
    raw_data.rename(columns={'date.1': 'date'}, inplace=True)
    raw_data["date"] = raw_data["date"].astype("datetime64[ns]")

    model_features = ModelFeatures(
        raw_data,
        params["total_time_steps"],
        start_boundary=train_interval[0],
        test_boundary=train_interval[1],
        test_end=train_interval[2],
        changepoint_lbws=changepoint_lbws,
        split_tickers_individually=params["split_tickers_individually"],
        train_valid_ratio=params["train_valid_ratio"],
        add_ticker_as_static=(params["architecture"] == "TFT"),
        time_features=params["time_features"],
        lags=params["force_output_sharpe_length"],
        asset_class_dictionary=asset_class_dictionary,
    )

    hp_directory = os.path.join(directory, "hp")

    if params["architecture"] == "LSTM":
        dmn = LstmDeepMomentumNetworkModel(
            experiment_name,
            hp_directory,
            hp_minibatch_size,
            **params,
            **model_features.input_params,
        )
    elif params["architecture"] == "TFT":
        dmn = TftDeepMomentumNetworkModel(
            experiment_name,
            hp_directory,
            hp_minibatch_size,
            **params,
            **model_features.input_params,
            **{
                "column_definition": model_features.get_column_definition(),
                "num_encoder_steps": 0, 
                "stack_size": 1,
                "num_heads": 4
            },
        )
    else:
        dmn = None
        raise Exception(f"{params['architecture']} is not a valid architecture.")

    best_hp, best_model = dmn.hyperparameter_search(
        model_features.train, model_features.valid
    )
    val_loss = dmn.evaluate(model_features.valid, best_model)

    print(f"Best validation loss = {val_loss}")
    print(f"Best params:")
    for k in best_hp:
        print(f"{k} = {best_hp[k]}")

    with open(os.path.join(directory, "best_hyperparameters.json"), "w") as file:
        file.write(json.dumps(best_hp))

    # if predict_on_test_set:
    print("Predicting on test set...")

    results_sw, performance_sw = dmn.get_positions(
        model_features.test_sliding,
        best_model,
        sliding_window=True,
        years_geq=train_interval[1],
        years_lt=train_interval[2],
    )
    print(f"performance (sliding window) = {performance_sw}")

    results_sw = results_sw.merge(
        raw_data.reset_index(drop=True)[["ticker", "date", "daily_vol"]].rename(
            columns={"ticker": "identifier", "date": "time"}
        ),
        on=["identifier", "time"],
    )
    results_sw = calc_net_returns(
        results_sw, [None, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0][1:], model_features.tickers
    )
    results_sw.to_csv(os.path.join(directory, "captured_returns_sw.csv"))

    # keep fixed window just in case
    results_fw, performance_fw = dmn.get_positions(
        model_features.test_fixed,
        best_model,
        sliding_window=False,
        years_geq=train_interval[1],
        years_lt=train_interval[2],
    )
    print(f"performance (fixed window) = {performance_fw}")
    results_fw = results_fw.merge(
        raw_data.reset_index()[["ticker", "date", "daily_vol"]].rename(
            columns={"ticker": "identifier", "date": "time"}
        ),
        on=["identifier", "time"],
    )
    results_fw = calc_net_returns(
        results_fw, [None, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0][1:], model_features.tickers
    )
    results_fw.to_csv(os.path.join(directory, "captured_returns_fw.csv"))

    with open(os.path.join(directory, "fixed_params.json"), "w") as file:
        file.write(
            json.dumps(
                dict(
                    **params,
                    **model_features.input_params,
                    **{
                        "changepoint_lbws": changepoint_lbws
                        if changepoint_lbws
                        else [],
                        "features_file_path": features_file_path,
                    },
                ),
                indent=4,
            )
        )

    # save model and get rid of the hp dir
    best_directory = os.path.join(directory, "best")
    best_model.save_weights(os.path.join(best_directory, "checkpoints", "checkpoint"))
    with open(os.path.join(best_directory, "hyperparameters.json"), "w") as file:
        file.write(json.dumps(best_hp, indent=4))
    shutil.rmtree(hp_directory)

    save_results(
        results_sw,
        directory,
        train_interval,
        model_features.num_tickers,
        asset_class_dictionary,
        {
            "performance_sw": performance_sw,
            "performance_fw": performance_fw,
            "val_loss": val_loss,
        },
    )


    del best_model
    gc.collect()
    tf.keras.backend.clear_session()
    physical_devices = tf.config.list_physical_devices("GPU")
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)



def run_all_windows(
    experiment_name: str,
    features_file_path: str,
    train_intervals: List[Tuple[int, int, int]],
    params: dict,
    changepoint_lbws: List[int],
    asset_class_dictionary=Dict[str, str],
    hp_minibatch_size: List[int]=[64, 128, 256],
    standard_window_size=1
):
    # run the expanding window
    for interval in train_intervals:
        run_single_window(
            experiment_name,
            features_file_path,
            interval,
            params,
            changepoint_lbws,
            asset_class_dictionary=asset_class_dictionary,
            hp_minibatch_size=hp_minibatch_size,
        )

    aggregate_and_save_all_windows(
        experiment_name, train_intervals, asset_class_dictionary, standard_window_size
    )







#Basic long only momentum strategy

def intermediate_momentum_position(w: float, returns_data: pd.DataFrame) -> pd.Series:
    return w * np.sign(returns_data["norm_monthly_return"]) + (1 - w) * np.sign(
        returns_data["norm_annual_return"]
    )


def run_classical_methods(
    features_file_path,
    train_intervals,
    reference_experiment,
    long_only_experiment_name="long_only",
    tsmom_experiment_name="tsmom",
):
    """Run classical TSMOM method and Long Only as defined in https://arxiv.org/pdf/2105.13727.pdf."""

    directory = _get_directory_name(long_only_experiment_name)
    if not os.path.exists(directory):
        os.mkdir(directory)

    directory = _get_directory_name(tsmom_experiment_name)
    if not os.path.exists(directory):
        os.mkdir(directory)

    for train_interval in train_intervals:
        directory = _get_directory_name(tsmom_experiment_name, train_interval)
        if not os.path.exists(directory):
            os.mkdir(directory)
        raw_data = pd.read_csv(features_file_path, parse_dates=True)
        reference = pd.read_csv(
            f"results/{reference_experiment}/{train_interval[1]}-{train_interval[2]}/captured_returns_sw.csv",
            parse_dates=True,
        )
        returns_data = raw_data.merge(
            reference[["time", "identifier", "returns"]],
            left_on=["date", "ticker"],
            right_on=["time", "identifier"],
        )
        returns_data["position"] = intermediate_momentum_position(0, returns_data)
        # returns_data["returns"] = returns_data["scaled_return_target"]
        returns_data["captured_returns"] = (
            returns_data["position"] * returns_data["returns"]
        )
        returns_data = returns_data.reset_index()[
            ["identifier", "time", "returns", "position", "captured_returns"]
        ]
        returns_data.to_csv(f"{directory}/captured_returns_sw.csv")

        directory = _get_directory_name(long_only_experiment_name, train_interval)
        if not os.path.exists(directory):
            os.mkdir(directory)
        returns_data["position"] = 1.0
        returns_data["captured_returns"] = (
            returns_data["position"] * returns_data["returns"]
        )
        returns_data.to_csv(f"{directory}/captured_returns_sw.csv")

