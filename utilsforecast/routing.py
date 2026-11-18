"""Budget-constrained model routing.

Given several candidate forecast models, a per-series nonnegative integer
business loss and weight for every available model, a one-time enablement
cost and a per-assignment cost for each model, and a total budget, this
module:

1. solves the exact integer routing problem (each series assigned to exactly
   one model, weighted loss minimized, fixed charges paid once per used
   model) and fails explicitly when the budget cannot be met;
2. drives the forecasts from that assignment: every used model is called
   exactly once and receives only the history and future frames of the
   series assigned to it, while unused models are never called;
3. validates every returned frame against the exact future keys it should
   cover, so no partial result is ever delivered;
4. assembles the forecasts in the original pandas/polars type for native
   evaluation and plotting, and reports the routing in a separate table that
   never pollutes the forecast columns.

The existing library interface is unchanged.
"""

__all__ = [
    "BudgetInfeasibleError",
    "RoutingResult",
    "route_forecasts",
]


from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

from .compat import DataFrame, pl, pl_DataFrame
from .processing import ensure_sorted
from .routing_solver import (
    BudgetInfeasibleError,
    RoutingOption,
    RoutingProblem,
    solve_routing,
)
from .validation import validate_format


@dataclass
class RoutingResult:
    """Output of :func:`route_forecasts`.

    Attributes:
        forecasts_df: Forecasts in the same pandas/polars type as the inputs,
            with the id and time columns plus one column per used model
            (named after the model). Ready for ``evaluate`` and
            ``plot_series``.
        assignments_df: Separate table with one row per series recording the
            assigned model, its business loss, weight, assignment cost and
            whether the model's enablement fee is charged on that row. It is
            kept separate from ``forecasts_df``.
        weighted_loss: Exact total weighted loss of the chosen assignment.
        total_cost: Exact total cost, including every enablement fee paid.
        used_models: Models that were actually called.
    """

    forecasts_df: DataFrame
    assignments_df: DataFrame
    weighted_loss: int
    total_cost: int
    used_models: List[str]


ModelCallback = Callable[[DataFrame, DataFrame], DataFrame]


def _is_pandas(df: DataFrame) -> bool:
    return isinstance(df, pd.DataFrame)


def _require_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"`{name}` must be a nonnegative integer, got {value!r}.")
    if value < 0:
        raise ValueError(f"`{name}` must be nonnegative, got {value}.")
    return value


def _column_values(df: DataFrame, col: str) -> np.ndarray:
    return df[col].to_numpy()


def _unique_ids(df: DataFrame, id_col: str) -> List[Any]:
    """Stable, first-occurrence ordering of the ids."""
    if _is_pandas(df):
        return df[id_col].drop_duplicates().tolist()
    # polars' plain unique() is unordered; maintain_order reproduces the
    # first-occurrence ordering of the current frame
    return df[id_col].unique(maintain_order=True).to_list()


def _filter_ids(df: DataFrame, id_col: str, ids: List[Any]) -> DataFrame:
    if _is_pandas(df):
        out = df[df[id_col].isin(ids)].reset_index(drop=True)
    else:
        out = df.filter(df[id_col].is_in(ids))
    return out


def _frame_keys(
    df: DataFrame, id_col: str, time_col: str
) -> set:
    ids = _column_values(df, id_col)
    times = _column_values(df, time_col)
    return set(zip(ids.tolist(), times.tolist()))


def _normalize_score_table(
    table: Mapping[str, Mapping[str, int]],
    name: str,
    series_ids: Sequence[Any],
    model_names: Sequence[str],
    require_full: bool,
) -> Dict[Any, Dict[str, int]]:
    """Validate a {series: {model: nonnegative int}} mapping."""
    if not isinstance(table, Mapping):
        raise TypeError(f"`{name}` must be a mapping, got {type(table)!r}.")
    series_set = set(series_ids)
    model_set = set(model_names)
    normalized: Dict[Any, Dict[str, int]] = {}
    for series, row in table.items():
        if series not in series_set:
            raise ValueError(
                f"`{name}` contains unknown series id {series!r}; "
                f"valid ids are {sorted(map(str, series_set))[:5]}..."
            )
        if not isinstance(row, Mapping):
            raise TypeError(
                f"`{name}[{series!r}] must be a mapping of model to integer."
            )
        clean: Dict[str, int] = {}
        for model, value in row.items():
            if model not in model_set:
                raise ValueError(
                    f"`{name}` references unknown model {model!r}."
                )
            clean[model] = _require_int(value, f"{name}[{series!r}][{model!r}]")
        normalized[series] = clean
    if require_full:
        missing_series = series_set - set(normalized)
        if missing_series:
            raise ValueError(
                f"`{name}` is missing series: {sorted(map(str, missing_series))}."
            )
    return normalized


def route_forecasts(
    history_df: DataFrame,
    future_df: DataFrame,
    models: Mapping[str, ModelCallback],
    losses: Mapping[Any, Mapping[str, int]],
    weights: Mapping[Any, Mapping[str, int]],
    enable_costs: Mapping[str, int],
    assignment_costs: Mapping[Any, Mapping[str, int]],
    budget: int,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
) -> RoutingResult:
    """Route series to models under a budget and produce the forecasts.

    The budget and routing are resolved *before* any model is called, so an
    infeasible budget always raises without invoking a single callback. Each
    used model is then called exactly once with only the history and future
    rows of the series assigned to it.

    Args:
        history_df: Historical data with columns ``[id_col, time_col,
            target_col, ...]``.
        future_df: Future timestamps to forecast, with ``[id_col, time_col,
            ...]``.
        models: Mapping of model name to a callback
            ``fn(history_subset, future_subset) -> forecast_df``. The returned
            frame must contain ``id_col``, ``time_col`` and ``target_col`` and
            cover exactly the future keys of the series handed to the model.
        losses: Per-series, per-model nonnegative integer business losses.
        weights: Per-series, per-model nonnegative integer weights applied to
            the losses.
        enable_costs: Nonnegative integer one-time enablement cost per model.
        assignment_costs: Nonnegative integer cost charged for every series
            assigned to a model.
        budget: Inclusive integer total-cost budget.
        id_col: Name of the id column.
        time_col: Name of the time column.
        target_col: Name of the forecast/target column.

    Returns:
        RoutingResult: The assembled forecasts, the separate assignment table
            and the exact objective and cost.

    Raises:
        BudgetInfeasibleError: If no assignment fits within the budget. No
            model callback is invoked in that case.
        ValueError: If the inputs are inconsistent or a callback returns an
            invalid frame (missing, duplicated or extra keys, wrong type).
    """
    # ---- structural validation -------------------------------------------------
    if not isinstance(models, Mapping) or not models:
        raise ValueError("`models` must be a non-empty mapping of name to callback.")
    model_names = list(models.keys())
    if len(set(model_names)) != len(model_names):
        raise ValueError("Model names must be unique.")
    for name, cb in models.items():
        if not callable(cb):
            raise TypeError(f"Model {name!r} must be callable.")

    is_pandas = _is_pandas(history_df)
    if is_pandas != _is_pandas(future_df):
        raise ValueError(
            "`history_df` and `future_df` must be the same frame type "
            "(both pandas or both polars)."
        )
    if not is_pandas and not isinstance(history_df, pl_DataFrame):
        raise ValueError(
            "Inputs must be pandas or polars DataFrames, got "
            f"{type(history_df)!r}."
        )
    validate_format(history_df, id_col, time_col, target_col)
    validate_format(future_df, id_col, time_col, None)

    budget = _require_int(budget, "budget")
    for model, cost in enable_costs.items():
        if model not in models:
            raise ValueError(f"`enable_costs` references unknown model {model!r}.")
        _require_int(cost, f"enable_costs[{model!r}]")
    missing_fees = set(model_names) - set(enable_costs)
    if missing_fees:
        raise ValueError(
            "Every model needs an enablement cost; missing: "
            f"{sorted(missing_fees)}."
        )

    history_df = ensure_sorted(history_df, id_col, time_col)
    future_df = ensure_sorted(future_df, id_col, time_col)

    series_ids = _unique_ids(future_df, id_col)
    if not series_ids:
        raise ValueError("`future_df` contains no series to route.")
    history_ids = set(_unique_ids(history_df, id_col))
    unknown = [s for s in series_ids if s not in history_ids]
    if unknown:
        raise ValueError(
            "Every future series must have history; missing: "
            f"{unknown[:5]}{'...' if len(unknown) > 5 else ''}."
        )

    loss_table = _normalize_score_table(
        losses, "losses", series_ids, model_names, require_full=True
    )
    weight_table = _normalize_score_table(
        weights, "weights", series_ids, model_names, require_full=True
    )
    cost_table = _normalize_score_table(
        assignment_costs,
        "assignment_costs",
        series_ids,
        model_names,
        require_full=True,
    )

    # ---- build the exact integer problem --------------------------------------
    model_index = {name: k for k, name in enumerate(model_names)}
    options: List[List[RoutingOption]] = []
    # availability is defined by the loss table; weights/costs must match it
    for i, series in enumerate(series_ids):
        available = list(loss_table[series].keys())
        if not available:
            raise BudgetInfeasibleError(
                f"Series {series!r} has no available model, so the budget "
                "cannot be satisfied."
            )
        row: List[RoutingOption] = []
        for name in available:
            if name not in weight_table[series]:
                raise ValueError(
                    f"Missing weight for series {series!r}, model {name!r}."
                )
            if name not in cost_table[series]:
                raise ValueError(
                    f"Missing assignment cost for series {series!r}, "
                    f"model {name!r}."
                )
            row.append(
                RoutingOption(
                    series=i,
                    model=model_index[name],
                    loss=loss_table[series][name],
                    weight=weight_table[series][name],
                    assign_cost=cost_table[series][name],
                )
            )
        options.append(row)

    enable_by_index = {
        model_index[name]: enable_costs[name] for name in model_names
    }
    problem = RoutingProblem(
        n_series=len(series_ids),
        options=options,
        enable_cost=enable_by_index,
        budget=budget,
    )
    choices, weighted_loss, total_cost = solve_routing(problem)

    assignment_by_series: Dict[Any, str] = {
        series: model_names[choices[i]] for i, series in enumerate(series_ids)
    }
    used_models = [
        name for name in model_names if name in set(assignment_by_series.values())
    ]

    # ---- drive the forecasts: one call per used model -------------------------
    expected_keys = _frame_keys(future_df, id_col, time_col)
    model_outputs: Dict[str, DataFrame] = {}
    for name in used_models:
        assigned = [s for s, m in assignment_by_series.items() if m == name]
        hist_subset = _filter_ids(history_df, id_col, assigned)
        future_subset = _filter_ids(future_df, id_col, assigned)
        if hist_subset.shape[0] == 0 or future_subset.shape[0] == 0:
            raise ValueError(
                f"Model {name!r} was assigned series but received an empty frame."
            )
        try:
            forecast = models[name](hist_subset, future_subset)
        except Exception as exc:  # callback failure must not deliver half results
            raise RuntimeError(
                f"Model {name!r} callback failed: {exc}"
            ) from exc

        _validate_model_forecast(
            forecast=forecast,
            expected_frame=future_subset,
            model_name=name,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
            is_pandas=is_pandas,
        )
        model_outputs[name] = forecast

    # final safety net: the union of returned keys equals the future keys,
    # with no duplicates, no missing keys and no extra keys
    returned_keys: set = set()
    for name, forecast in model_outputs.items():
        keys = _frame_keys(forecast, id_col, time_col)
        if returned_keys & keys:
            raise RuntimeError(
                f"Model {name!r} returned keys already covered by another model."
            )
        returned_keys |= keys
    if returned_keys != expected_keys:
        missing = expected_keys - returned_keys
        extra = returned_keys - expected_keys
        raise RuntimeError(
            "Assembled forecasts do not match the future keys. "
            f"Missing {len(missing)}, extra {len(extra)}."
        )

    forecasts_df = _assemble_forecasts(
        model_outputs=model_outputs,
        future_df=future_df,
        used_models=used_models,
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
        is_pandas=is_pandas,
    )
    assignments_df = _build_assignments(
        series_ids=series_ids,
        assignment_by_series=assignment_by_series,
        loss_table=loss_table,
        weight_table=weight_table,
        cost_table=cost_table,
        enable_costs=enable_costs,
        used_models=used_models,
        is_pandas=is_pandas,
    )
    return RoutingResult(
        forecasts_df=forecasts_df,
        assignments_df=assignments_df,
        weighted_loss=weighted_loss,
        total_cost=total_cost,
        used_models=used_models,
    )


def _validate_model_forecast(
    forecast: DataFrame,
    expected_frame: DataFrame,
    model_name: str,
    id_col: str,
    time_col: str,
    target_col: str,
    is_pandas: bool,
) -> None:
    if is_pandas:
        if not isinstance(forecast, pd.DataFrame):
            raise ValueError(
                f"Model {model_name!r} must return a pandas DataFrame, "
                f"got {type(forecast)!r}."
            )
    else:
        if not isinstance(forecast, pl_DataFrame):
            raise ValueError(
                f"Model {model_name!r} must return a polars DataFrame, "
                f"got {type(forecast)!r}."
            )
    for col in (id_col, time_col, target_col):
        if col not in forecast.columns:
            raise ValueError(
                f"Model {model_name!r} output is missing column {col!r}."
            )
    expected = _frame_keys(expected_frame, id_col, time_col)
    actual = _frame_keys(forecast, id_col, time_col)
    if len(actual) != forecast.shape[0]:
        raise ValueError(
            f"Model {model_name!r} returned duplicated (id, time) keys."
        )
    if actual != expected:
        missing = expected - actual
        extra = actual - expected
        raise ValueError(
            f"Model {model_name!r} must cover exactly its assigned future "
            f"keys; missing {len(missing)}, extra {len(extra)}."
        )
    target = forecast[target_col]
    if is_pandas:
        if not pd.api.types.is_numeric_dtype(target.dtype):
            raise ValueError(
                f"Model {model_name!r} target column must be numeric."
            )
    else:
        if not target.dtype.is_numeric():
            raise ValueError(
                f"Model {model_name!r} target column must be numeric."
            )


def _assemble_forecasts(
    model_outputs: Dict[str, DataFrame],
    future_df: DataFrame,
    used_models: List[str],
    id_col: str,
    time_col: str,
    target_col: str,
    is_pandas: bool,
) -> DataFrame:
    """Build one column per used model, preserving the input frame type."""
    if is_pandas:
        pieces = []
        for name in used_models:
            piece = model_outputs[name][[id_col, time_col, target_col]].rename(
                columns={target_col: name}
            )
            pieces.append(piece)
        merged = pieces[0]
        for piece in pieces[1:]:
            merged = merged.merge(piece, on=[id_col, time_col], how="outer")
        # order rows like the future frame for deterministic, native output
        order_key = future_df[[id_col, time_col]].copy()
        order_key["__order"] = np.arange(order_key.shape[0])
        merged = merged.merge(order_key, on=[id_col, time_col], how="left")
        merged = merged.sort_values("__order").drop(columns="__order")
        merged = merged.reset_index(drop=True)
        return merged

    pieces = []
    for name in used_models:
        piece = model_outputs[name].select(
            [id_col, time_col, pl.col(target_col).alias(name)]
        )
        pieces.append(piece)
    merged = pieces[0]
    for piece in pieces[1:]:
        joined = merged.join(piece, on=[id_col, time_col], how="full")
        # polars keeps the join keys from both sides under a full join; coalesce
        # them into a single key with a version-independent expression
        joined = joined.with_columns(
            pl.coalesce([id_col, f"{id_col}_right"]).alias(id_col),
            pl.coalesce([time_col, f"{time_col}_right"]).alias(time_col),
        ).drop(f"{id_col}_right", f"{time_col}_right")
        merged = joined
    # restore the future frame's row order
    order = future_df.select([id_col, time_col]).with_row_index("__order")
    merged = (
        merged.join(order, on=[id_col, time_col], how="left")
        .sort("__order")
        .drop("__order")
    )
    return merged


def _build_assignments(
    series_ids: Sequence[Any],
    assignment_by_series: Mapping[Any, str],
    loss_table: Mapping[Any, Mapping[str, int]],
    weight_table: Mapping[Any, Mapping[str, int]],
    cost_table: Mapping[Any, Mapping[str, int]],
    enable_costs: Mapping[str, int],
    used_models: Sequence[str],
    is_pandas: bool,
) -> DataFrame:
    # the enablement fee is attributed once, to the first series of each model
    fee_charged: Dict[str, bool] = {name: False for name in used_models}
    rows = []
    for series in series_ids:
        model = assignment_by_series[series]
        first_use = not fee_charged[model]
        if first_use:
            fee_charged[model] = True
        rows.append(
            {
                "unique_id": series,
                "model": model,
                "loss": loss_table[series][model],
                "weight": weight_table[series][model],
                "assignment_cost": cost_table[series][model],
                "enable_cost": enable_costs[model] if first_use else 0,
            }
        )
    if is_pandas:
        return pd.DataFrame(rows)
    out = pl_DataFrame(rows)
    # preserve the input series ordering instead of polars' default ordering
    order_df = pl_DataFrame(
        {"unique_id": list(series_ids), "__order": list(range(len(series_ids)))}
    )
    return out.join(order_df, on="unique_id", how="left").sort("__order").drop(
        "__order"
    )
