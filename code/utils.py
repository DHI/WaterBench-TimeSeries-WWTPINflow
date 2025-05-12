import numpy as np
import pandas as pd

from typing import Optional, Callable, Tuple

from darts import TimeSeries
from darts.models.forecasting.forecasting_model import ForecastingModel


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

    return train_data, test_data


def holt_smoother(x: np.array, alpha: float) -> np.array:
    # https://empslocal.ex.ac.uk/people/staff/dbs202/cag/courses/MT37C/course/node102.html
    y = np.zeros_like(x)
    y[0] = x[0]

    for t in range(1, len(x)):
        y[t] = alpha * x[t] + (1 - alpha) * y[t - 1]

    return y


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
