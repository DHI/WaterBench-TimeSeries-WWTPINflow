import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures

from typing import Optional, Callable, Tuple, List

from darts import TimeSeries
from darts.utils.missing_values import extract_subseries
from darts.models import NaiveMovingAverage
from darts.models.forecasting.forecasting_model import ForecastingModel


def holt_smoother(x: np.array, alpha: float) -> np.array:
    # https://empslocal.ex.ac.uk/people/staff/dbs202/cag/courses/MT37C/course/node102.html
    y = np.zeros_like(x)
    y[0] = x[0]

    for t in range(1, len(x)):
        y[t] = alpha * x[t] + (1 - alpha) * y[t - 1]

    return y


def generate_onehot_dayofweek_ts(ts: TimeSeries) -> TimeSeries:
    """Generates time series of date time features using one hot encoding"""
    dayofweek = (
        pd.get_dummies(ts.time_index.day_of_week)
        .rename(columns=lambda x: f"dow_{x}")
        .set_index(ts.time_index)
    )

    return TimeSeries.from_dataframe(dayofweek)


def generate_poly_ts(ts: TimeSeries, degree: int) -> TimeSeries:
    poly_values = PolynomialFeatures(degree, include_bias=False).fit_transform(
        ts.values()
    )
    poly_features = pd.DataFrame(
        poly_values,
        columns=[f"polyfeature_{i}" for i in range(poly_values.shape[1])],
        index=ts.time_index,
    )

    return TimeSeries.from_dataframe(
        poly_features,
        freq="h",
    )


def add_smoothed_precip(ts: TimeSeries, alpha: float) -> TimeSeries:
    lp_precip = holt_smoother(ts["acc_precip"].values().ravel(), alpha)
    ts_lp = TimeSeries.from_series(
        pd.Series(lp_precip, index=ts.time_index, name="smooth_precip")
    )
    return ts.concatenate(ts_lp, axis=1)


def compute_errors(
    ts: TimeSeries,
    model: ForecastingModel,
    horizon: int,
    *,
    past_covariates: Optional[TimeSeries] = None,
    future_covariates: Optional[TimeSeries] = None,
    output_transform: Optional[Callable] = None,
    is_naive: bool = False,
) -> pd.DataFrame:
    """Computes forecasting errors for passed time series data. The results are sorted by lead time."""
    lags = [-l for l in model.extreme_lags if l is not None and l < 0]  # noqa: E741
    n_lags = max(lags)

    n_steps = len(ts) - n_lags - horizon

    errors = []
    for t in range(n_steps):
        ts_in, ts_out = ts.split_before(t + n_lags)

        kwargs = {}
        if model.supports_future_covariates:
            kwargs["future_covariates"] = future_covariates
        if model.supports_past_covariates:
            kwargs["past_covariates"] = past_covariates
        if model.supports_transferable_series_prediction:
            kwargs["series"] = ts_in
        if is_naive:
            # Naive models can only _continue_ the series they have been trained on
            model.fit(ts_in)

        pred = model.predict(
            horizon,
            **kwargs,
        )

        actual_values = ts_out.slice_intersect(pred).values()
        pred_values = pred.values()
        if output_transform is not None:
            actual_values = output_transform(actual_values)
            pred_values = output_transform(pred_values)

        error = actual_values - pred_values
        error_df = pd.DataFrame(
            {
                "error": error.ravel(),
                "actual": actual_values.ravel(),
                "lead_time": list(range(1, horizon + 1)),
                "observation_time": pred.time_index,
            },
        )
        errors.append(error_df)

    return pd.concat([e for e in errors], ignore_index=True)


def get_mape(errors: pd.DataFrame):
    return (
        errors.apply(lambda x: 100 * abs(x.error / x.actual), axis=1)
        .groupby(errors["lead_time"])
        .mean()
    )


def get_rmse(errors: pd.DataFrame):
    return (
        errors.apply(lambda x: x.error**2, axis=1)
        .groupby(errors["lead_time"])
        .mean()
        .apply(np.sqrt)
    )


def generate_baseline_df(
    ts: TimeSeries, n_lags: int, output_chunk_length: int
) -> pd.DataFrame:
    naive = NaiveMovingAverage(input_chunk_length=n_lags)
    naive_errors = compute_errors(ts, naive, output_chunk_length, is_naive=True)

    return pd.DataFrame(
        {
            "RMSE": get_rmse(naive_errors),
            "MAPE": get_mape(naive_errors),
            "model": "baseline",
        }
    )


def assemble_comparison(candidates: List[pd.DataFrame]) -> pd.DataFrame:
    """Returns a dataframe comparing multiple models by lead time and metric"""
    return (
        pd.concat(candidates, axis=0)
        .reset_index()
        .melt(id_vars=["lead_time", "model"], var_name="metric")
    )


#################
# EXAMPLE UTILS #
#################


def example_train_test_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Get data to make a modelling example. The selected data period is based
    on prior data analysis and is fixed to data from 01/03/2024 to 17/04/2024
    since we want to test our models with a long-enough continuous period
    that experienced significant rainfall.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Train and test split for test data
    """
    data_path = "../processed/data.csv"

    start_test_period = pd.Timestamp(day=1, month=3, year=2024, hour=7)
    end_test_period = pd.Timestamp(day=17, month=4, year=2024, hour=11)

    data = pd.read_csv(data_path, index_col=0, parse_dates=True)

    test_mask = data.index.to_series().between(
        start_test_period, end_test_period, inclusive="both"
    )
    train_mask = ~test_mask

    train_data = data[train_mask].copy()
    test_data = data[test_mask].copy()

    # The temperature at grass level and 30 cm below the surface have
    # a big gap at the end. Since we have other temperature measurements,
    # e.g. `mean_temp` and `temp_soil_10`, we discard the following to
    # preserve as many samples as possible.
    faulty_variables = ["temp_grass", "temp_soil_30"]
    train_data.drop(columns=faulty_variables, inplace=True)
    test_data.drop(columns=faulty_variables, inplace=True)

    return train_data, test_data


def example_train_subseries() -> List[TimeSeries]:
    min_ts_length = 24 * 7
    train_data, _ = example_train_test_data()
    train_ts = TimeSeries.from_dataframe(train_data, freq="h")
    subseries = extract_subseries(train_ts, mode="any")

    return [s for s in subseries if len(s) >= min_ts_length]


def example_test_series() -> TimeSeries:
    _, test_data = example_train_test_data()
    return TimeSeries.from_dataframe(test_data)


def example_pipeline(ts: TimeSeries) -> TimeSeries:
    """We transform the dataset by:
        - Including a smoothed rain
        - The value of the parameter $\alpha$ is based on the correlation study and simple trial and error to minimize forecasting error
        - Including polynomial functions of the weather features
        - Including datetime features, using one hot encoding

    Returns
    -------
    TimeSeries
        Expanded timeseries after adding features
    """
    alpha = 0.2  # smoothing coefficient
    deg = 3  # polynomial degree

    # We use our future precipitation observations as a "perfect forecast"
    future_components = [
        "acc_precip",
        "smooth_precip",
    ]

    ts = add_smoothed_precip(ts, alpha)
    ts1 = generate_poly_ts(ts[future_components], deg)
    ts2 = generate_onehot_dayofweek_ts(ts)

    return ts.drop_columns(future_components).concatenate(
        ts1.concatenate(ts2, axis=1), axis=1
    )


def example_target_and_features(
    ts: TimeSeries | List[TimeSeries], target_var: str
) -> Tuple[TimeSeries, TimeSeries] | Tuple[List[TimeSeries], List[TimeSeries]]:
    if isinstance(ts, TimeSeries):
        target_ts = ts[target_var]
        features = ts.drop_columns(target_var)
        expanded_features = example_pipeline(features)
    else:
        target_ts = [s[target_var] for s in ts]
        subts = [s.drop_columns(target_var) for s in ts]
        expanded_features = []
        for s in subts:
            expanded_features.append(example_pipeline(s))

    return target_ts, expanded_features


def visualize_example_measurements():
    subseries = example_train_subseries()
    test_ts = example_test_series()

    fig, axes = plt.subplots(2, 1, figsize=(15, 5), sharex=True)
    for s in subseries:
        s["flow"].plot(ax=axes[0], linewidth=0.8)
        s["acc_precip"].plot(ax=axes[1], linewidth=0.8)

    test_ts["flow"].plot(linewidth=0.8, ax=axes[0], color="lime")
    test_ts["acc_precip"].plot(linewidth=0.8, ax=axes[1], color="lime")

    start_test = test_ts.time_index[0]
    end_test = test_ts.time_index[-1]

    for j in [0, 1]:
        axes[j].axvspan(start_test, end_test, color="grey", alpha=0.3)
        axes[j].text(
            start_test + (end_test - start_test) / 2,
            2.5,
            "TEST",
            ha="center",
            va="center",
            fontsize=12,
        )

    axes[0].legend().set_visible(False), axes[1].legend().set_visible(False)
    axes[0].set_ylabel("Flow [m^3/h]"), axes[1].set_ylabel("Acc. Precip. [mm]")
