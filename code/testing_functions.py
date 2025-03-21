import numpy as np
import pandas as pd

from typing import Optional, Callable

from darts import TimeSeries
from darts.models.forecasting.forecasting_model import ForecastingModel


def compute_errors(
    ts: TimeSeries,
    model: ForecastingModel,
    horizon: int,
    *,
    future_covariates: Optional[TimeSeries] = None,
    output_transform: Optional[Callable] = None,
    is_naive: bool = False,
) -> pd.DataFrame:
    lags = [-l for l in model.extreme_lags if l is not None and l < 0]  # noqa: E741
    n_lags = max(lags)

    n_steps = len(ts) - n_lags - horizon

    errors = []
    for t in range(n_steps):
        ts_in, ts_out = ts.split_before(t + n_lags)

        kwargs = {}
        if model.supports_future_covariates:
            kwargs["future_covariates"] = future_covariates
        if model.supports_transferrable_series_prediction:
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
