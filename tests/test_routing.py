import itertools

import numpy as np
import pandas as pd
import pytest

from utilsforecast.compat import POLARS_INSTALLED
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mae
from utilsforecast.routing import RoutingResult, route_forecasts
from utilsforecast.routing_solver import (
    BudgetInfeasibleError,
    RoutingOption,
    RoutingProblem,
    solve_routing,
)

if POLARS_INSTALLED:
    import polars as pl


# --------------------------------------------------------------------------- #
# solver fixtures and helpers
# --------------------------------------------------------------------------- #


def brute_force(problem):
    """Enumerate every assignment; return (loss, cost, choices) or None."""
    best = None
    groups = problem.options
    for combo in itertools.product(*[range(len(g)) for g in groups]):
        used = set()
        cost = 0
        loss = 0
        choices = []
        for i, k in enumerate(combo):
            opt = groups[i][k]
            choices.append(opt.model)
            used.add(opt.model)
            cost += opt.assign_cost
            loss += opt.weighted_loss
        cost += sum(problem.enable_cost[m] for m in used)
        if cost <= problem.budget and (
            best is None
            or loss < best[0]
            or (loss == best[0] and cost < best[1])
        ):
            best = (loss, cost, choices)
    return best


def make_problem(rng, n_series, n_models, missing=0.3):
    models = list(range(n_models))
    options = []
    enable = {m: rng.randint(0, 20) for m in models}
    for i in range(n_series):
        row = []
        for m in models:
            if rng.random() < missing:
                continue
            row.append(
                RoutingOption(
                    series=i,
                    model=m,
                    loss=rng.randint(0, 30),
                    weight=rng.randint(1, 5),
                    assign_cost=rng.randint(0, 15),
                )
            )
        if not row:
            row.append(
                RoutingOption(i, models[0], rng.randint(0, 30), 1, rng.randint(0, 15))
            )
        options.append(row)
    min_budget = sum(min(o.assign_cost for o in g) for g in options)
    budget = rng.randint(min_budget, min_budget + sum(enable.values()) + 20)
    return RoutingProblem(n_series, options, enable, budget)


# --------------------------------------------------------------------------- #
# exact solver
# --------------------------------------------------------------------------- #


def test_solver_matches_brute_force():
    py_rng = __import__("random").Random(0)
    for _ in range(200):
        problem = make_problem(
            py_rng, py_rng.randint(1, 5), py_rng.randint(1, 4)
        )
        expected = brute_force(problem)
        try:
            choices, loss, cost = solve_routing(problem)
        except BudgetInfeasibleError:
            assert expected is None
            continue
        assert expected is not None
        assert loss == expected[0]
        assert cost == expected[1]
        assert cost <= problem.budget
        # every series assigned exactly once to an available model
        assert len(choices) == problem.n_series
        available = set(problem.model_ids)
        assert all(c in available for c in choices)


def test_solver_pure_integer_arithmetic():
    # Values large enough that any float rounding would change the answer.
    # Model 0 is strictly best on loss for both series, so the optimum must
    # carry the 10^40 weighted loss and 10^20 enable fee through exactly.
    big = 10**20
    problem = RoutingProblem(
        n_series=2,
        options=[
            [RoutingOption(0, 0, big, big, 5), RoutingOption(0, 1, big, big, 9)],
            [RoutingOption(1, 0, 7, 1, 5), RoutingOption(1, 1, big, big, 9)],
        ],
        enable_cost={0: big, 1: 3},
        budget=10**40,
    )
    choices, loss, cost = solve_routing(problem)
    assert isinstance(loss, int) and isinstance(cost, int)
    assert choices == [0, 0]
    assert loss == big * big + 7  # exact, no float rounding
    assert cost == big + 10
    # the near-miss using model 1 for the second series costs more in loss
    # despite model 1's cheaper enable fee, so it must not be selected
    assert big * big + big * big > loss


def test_solver_infeasible_budget_raises():
    problem = RoutingProblem(
        n_series=2,
        options=[
            [RoutingOption(0, 0, 1, 1, 10)],
            [RoutingOption(1, 0, 1, 1, 10)],
        ],
        enable_cost={0: 5},
        budget=10,  # need 25
    )
    with pytest.raises(BudgetInfeasibleError):
        solve_routing(problem)


def test_solver_enable_fee_charged_once():
    # both series use model 0: fee paid once, assignment cost twice
    problem = RoutingProblem(
        n_series=2,
        options=[
            [RoutingOption(0, 0, 5, 1, 3), RoutingOption(0, 1, 1, 1, 1)],
            [RoutingOption(1, 0, 5, 1, 3), RoutingOption(1, 1, 1, 1, 1)],
        ],
        enable_cost={0: 7, 1: 100},
        budget=50,
    )
    choices, loss, cost = solve_routing(problem)
    assert choices == [0, 0]
    assert cost == 7 + 3 + 3
    assert loss == 10


def test_solver_tie_prefers_lower_cost():
    # equal weighted loss, model 2 has lower total cost than model 3
    problem = RoutingProblem(
        n_series=1,
        options=[
            [
                RoutingOption(0, 2, 9, 4, 14),
                RoutingOption(0, 3, 9, 4, 7),
            ]
        ],
        enable_cost={2: 4, 3: 15},
        budget=100,
    )
    choices, loss, cost = solve_routing(problem)
    assert choices == [2]
    assert cost == 18  # 4 + 14, cheaper than 15 + 7 = 22


def test_solver_deterministic():
    py_rng = __import__("random").Random(7)
    problem = make_problem(py_rng, 30, 6)
    problem.budget = (
        sum(max(o.assign_cost for o in g) for g in problem.options)
        + sum(problem.enable_cost.values())
    )
    first = solve_routing(problem)
    second = solve_routing(problem)
    assert first == second


def test_problem_rejects_empty_options():
    with pytest.raises(BudgetInfeasibleError):
        RoutingProblem(1, [[]], {0: 1}, 10)


def test_solver_matches_milp_oracle():
    """Cross-check against the HiGHS MILP solver when scipy is available."""
    scipy = pytest.importorskip("scipy")
    from scipy.optimize import Bounds, LinearConstraint, milp

    rng = __import__("random").Random(123)

    def milp_oracle(problem):
        # one binary x per (series, option); one binary y per enabled model
        flat = [
            (i, k)
            for i, group in enumerate(problem.options)
            for k in range(len(group))
        ]
        n_x, n_y = len(flat), len(problem.model_ids)
        model_y = {m: k for k, m in enumerate(problem.model_ids)}
        n_vars = n_x + n_y
        integrality = np.ones(n_vars)
        bounds = Bounds(np.zeros(n_vars), np.ones(n_vars))
        constraints = []
        # each series picks exactly one option
        for i, group in enumerate(problem.options):
            row = np.zeros(n_vars)
            for k in range(len(group)):
                row[flat.index((i, k))] = 1.0
            constraints.append(LinearConstraint(row, 1.0, 1.0))
        # x <= y: an option's model must be enabled
        for idx, (i, k) in enumerate(flat):
            model = problem.options[i][k].model
            row = np.zeros(n_vars)
            row[idx] = 1.0
            row[n_x + model_y[model]] = -1.0
            constraints.append(LinearConstraint(row, -np.inf, 0.0))
        # assignment cost + enable fees <= budget
        row = np.zeros(n_vars)
        for idx, (i, k) in enumerate(flat):
            row[idx] = problem.options[i][k].assign_cost
        for m, j in model_y.items():
            row[n_x + j] = problem.enable_cost[m]
        constraints.append(LinearConstraint(row, -np.inf, float(problem.budget)))
        objective = np.zeros(n_vars)
        for idx, (i, k) in enumerate(flat):
            objective[idx] = problem.options[i][k].weighted_loss
        result = milp(
            c=objective,
            constraints=constraints,
            integrality=integrality,
            bounds=bounds,
            options={"disp": False},
        )
        if result.status == 2:  # infeasible
            return None
        assert result.success, result.message
        chosen = []
        for i, group in enumerate(problem.options):
            for k in range(len(group)):
                if result.x[flat.index((i, k))] > 0.5:
                    chosen.append(group[k].model)
        return int(round(result.fun)), chosen

    for _ in range(40):
        problem = make_problem(rng, rng.randint(8, 14), rng.randint(3, 6))
        oracle = milp_oracle(problem)
        try:
            choices, loss, cost = solve_routing(problem)
        except BudgetInfeasibleError:
            assert oracle is None
            continue
        assert oracle is not None
        oracle_loss, _ = oracle
        assert loss == oracle_loss
        assert cost <= problem.budget
        assert len(choices) == problem.n_series


def test_branch_and_bound_fallback_matches_milp():
    """Force the DP frontier past its safety limit so branch-and-bound runs."""
    pytest.importorskip("scipy")
    from scipy.optimize import Bounds, LinearConstraint, milp

    from utilsforecast.routing_solver import _pareto_dp

    rng = __import__("random").Random(99)
    n_series, n_models = 12, 16
    enable = {m: rng.randint(0, 20) for m in range(n_models)}
    options = [
        [
            RoutingOption(
                i, m, rng.randint(0, 30), rng.randint(1, 4), rng.randint(0, 15)
            )
            for m in range(n_models)
        ]
        for i in range(n_series)
    ]
    # a loose budget keeps every model affordable; the many models overflow
    # the Pareto DP frontier, forcing the branch-and-bound fallback
    budget = (
        sum(max(o.assign_cost for o in g) for g in options) + sum(enable.values())
    )
    assert _pareto_dp(options, enable, budget) is None  # fallback really engaged
    problem = RoutingProblem(n_series, options, enable, budget)
    choices, loss, cost = solve_routing(problem)

    flat = [
        (i, k)
        for i, group in enumerate(options)
        for k in range(len(group))
    ]
    n_x, n_y = len(flat), n_models
    n_vars = n_x + n_y
    constraints = []
    for i, group in enumerate(options):
        row = np.zeros(n_vars)
        for k in range(len(group)):
            row[flat.index((i, k))] = 1.0
        constraints.append(LinearConstraint(row, 1.0, 1.0))
    for idx, (i, k) in enumerate(flat):
        model = options[i][k].model
        row = np.zeros(n_vars)
        row[idx] = 1.0
        row[n_x + model] = -1.0
        constraints.append(LinearConstraint(row, -np.inf, 0.0))
    row = np.zeros(n_vars)
    for idx, (i, k) in enumerate(flat):
        row[idx] = options[i][k].assign_cost
    for m in range(n_models):
        row[n_x + m] = enable[m]
    constraints.append(LinearConstraint(row, -np.inf, float(budget)))
    objective = np.zeros(n_vars)
    for idx, (i, k) in enumerate(flat):
        objective[idx] = options[i][k].weighted_loss
    result = milp(
        c=objective,
        constraints=constraints,
        integrality=np.ones(n_vars),
        bounds=Bounds(np.zeros(n_vars), np.ones(n_vars)),
        options={"disp": False},
    )
    assert result.success
    assert loss == int(round(result.fun))
    assert cost <= budget
    assert len(choices) == n_series


def test_branch_and_bound_infeasible_guard():
    """Overflow the DP frontier with a budget below the cheapest completion."""
    rng = __import__("random").Random(3)
    n_series, n_models = 12, 16
    options = [
        [
            RoutingOption(i, m, rng.randint(0, 10), 1, rng.randint(5, 20))
            for m in range(n_models)
        ]
        for i in range(n_series)
    ]
    enable = {m: rng.randint(5, 20) for m in range(n_models)}
    cheapest = sum(min(o.assign_cost for o in g) for g in options) + min(
        enable.values()
    )
    problem = RoutingProblem(n_series, options, enable, cheapest - 1)
    with pytest.raises(BudgetInfeasibleError):
        solve_routing(problem)


# --------------------------------------------------------------------------- #
# routing end-to-end fixtures
# --------------------------------------------------------------------------- #


class Recorder:
    """Stand-in forecast model that records exactly what it receives."""

    def __init__(self, name, value):
        self.name = name
        self.value = value
        self.calls = 0
        self.received = []

    def __call__(self, hist, future):
        self.calls += 1
        if isinstance(hist, pd.DataFrame):
            ids = sorted(hist["unique_id"].unique().tolist())
            out = future.copy()
            out["y"] = float(self.value)
        else:
            ids = sorted(hist["unique_id"].unique().to_list())
            out = future.with_columns(pl.lit(float(self.value)).alias("y"))
        self.received.append(ids)
        return out


def build_frames(engine, n=4, h=3):
    hist_rows, fut_rows = [], []
    for i in range(n):
        uid = f"s{i}"
        for t in range(10):
            hist_rows.append({"unique_id": uid, "ds": t, "y": float(i + t)})
        for t in range(10, 10 + h):
            fut_rows.append({"unique_id": uid, "ds": t})
    if engine == "pandas":
        return pd.DataFrame(hist_rows), pd.DataFrame(fut_rows)
    return pl.DataFrame(hist_rows), pl.DataFrame(fut_rows)


def score_maps(series, models, loss_fn, cost_fn):
    losses, weights, costs = {}, {}, {}
    for s in series:
        losses[s] = {m: loss_fn(s, m) for m in models}
        weights[s] = {m: 1 for m in models}
        costs[s] = {m: cost_fn(s, m) for m in models}
    return losses, weights, costs


@pytest.fixture(params=["pandas"] + (["polars"] if POLARS_INSTALLED else []))
def engine(request):
    return request.param


def test_route_returns_native_type_and_separate_tables(engine):
    hist, future = build_frames(engine)
    series = [f"s{i}" for i in range(4)]
    models = {"a": Recorder("a", 1.0), "b": Recorder("b", 2.0)}
    losses, weights, costs = score_maps(
        series, ["a", "b"], lambda s, m: 1 if m == "a" else 5, lambda s, m: 2
    )
    result = route_forecasts(
        hist, future, models, losses, weights,
        {"a": 1, "b": 1}, costs, budget=100,
    )
    assert isinstance(result, RoutingResult)
    if engine == "pandas":
        assert isinstance(result.forecasts_df, pd.DataFrame)
        assert isinstance(result.assignments_df, pd.DataFrame)
    else:
        assert isinstance(result.forecasts_df, pl.DataFrame)
        assert isinstance(result.assignments_df, pl.DataFrame)
    # routing table is separate: forecast columns are keys + used models only
    assert set(result.forecasts_df.columns) == {
        "unique_id", "ds", *result.used_models
    }
    assert "model" in result.assignments_df.columns
    assert result.assignments_df.shape[0] == len(series)


def test_each_used_model_called_once_with_only_its_series(engine):
    hist, future = build_frames(engine)
    series = [f"s{i}" for i in range(4)]
    models = {"a": Recorder("a", 1.0), "b": Recorder("b", 2.0)}
    losses, weights, costs = score_maps(
        series, ["a", "b"],
        lambda s, m: {"a": 10, "b": 1}[m], lambda s, m: {"a": 1, "b": 4}[m],
    )
    result = route_forecasts(
        hist, future, models, losses, weights,
        {"a": 2, "b": 5}, costs, budget=18,
    )
    # b-only would cost 5 + 4*4 = 21, so the tight budget forces both models
    assert set(result.used_models) == {"a", "b"}
    assign = result.assignments_df
    if engine == "polars":
        assign = assign.to_pandas()
    for name, model in models.items():
        assert model.calls == 1
        expected = sorted(
            assign.loc[assign["model"] == name, "unique_id"].tolist()
        )
        assert expected  # each model really got at least one series
        assert model.received == [expected]
        # it only saw its own series in the history subset too
        assert set(model.received[0]) == set(expected)


def test_forecast_covers_exactly_future_keys(engine):
    hist, future = build_frames(engine)
    series = [f"s{i}" for i in range(4)]
    models = {"a": Recorder("a", 1.0)}
    losses, weights, costs = score_maps(
        series, ["a"], lambda s, m: 1, lambda s, m: 1
    )
    result = route_forecasts(
        hist, future, models, losses, weights,
        {"a": 0}, costs, budget=100,
    )
    fkeys = set(map(tuple, future[["unique_id", "ds"]].to_numpy().tolist()))
    rkeys = set(
        map(tuple, result.forecasts_df[["unique_id", "ds"]].to_numpy().tolist())
    )
    assert fkeys == rkeys
    assert result.forecasts_df.shape[0] == future.shape[0]
    # no duplicated keys
    keys_df = result.forecasts_df[["unique_id", "ds"]]
    if engine == "pandas":
        assert not keys_df.duplicated().any()
    else:
        assert keys_df.is_duplicated().sum() == 0


def test_infeasible_budget_calls_no_model(engine):
    hist, future = build_frames(engine)
    series = [f"s{i}" for i in range(4)]
    model = Recorder("a", 1.0)
    losses, weights, costs = score_maps(
        series, ["a"], lambda s, m: 1, lambda s, m: 10
    )
    with pytest.raises(BudgetInfeasibleError):
        route_forecasts(
            hist, future, {"a": model}, losses, weights,
            {"a": 5}, costs, budget=2,
        )
    assert model.calls == 0


class FailingRecorder(Recorder):
    def __call__(self, _hist, _future):
        self.calls += 1
        raise RuntimeError("boom")


def test_callback_failure_raises_and_no_partial_delivery(engine):
    hist, future = build_frames(engine)
    series = [f"s{i}" for i in range(4)]
    model = FailingRecorder("a", 1.0)
    losses, weights, costs = score_maps(
        series, ["a"], lambda s, m: 1, lambda s, m: 0
    )
    with pytest.raises(RuntimeError, match="boom"):
        route_forecasts(
            hist, future, {"a": model}, losses, weights,
            {"a": 0}, costs, budget=100,
        )


class BadKeyRecorder(Recorder):
    def __call__(self, hist, future):
        out = super().__call__(hist, future)
        if isinstance(out, pd.DataFrame):
            return out.iloc[:-1]  # drop the last row -> missing key
        return out.head(out.shape[0] - 1)


def test_callback_missing_key_rejected(engine):
    hist, future = build_frames(engine)
    series = [f"s{i}" for i in range(4)]
    model = BadKeyRecorder("a", 1.0)
    losses, weights, costs = score_maps(
        series, ["a"], lambda s, m: 1, lambda s, m: 0
    )
    with pytest.raises(ValueError, match="exactly its assigned future"):
        route_forecasts(
            hist, future, {"a": model}, losses, weights,
            {"a": 0}, costs, budget=100,
        )


class ExtraKeyRecorder(Recorder):
    def __call__(self, hist, future):
        out = super().__call__(hist, future)
        extra = out.head(1)  # same schema, duplicated (id, time) key
        if isinstance(out, pd.DataFrame):
            return pd.concat([out, extra], ignore_index=True)
        return pl.concat([out, extra])


def test_callback_duplicate_key_rejected(engine):
    hist, future = build_frames(engine)
    series = [f"s{i}" for i in range(4)]
    model = ExtraKeyRecorder("a", 1.0)
    losses, weights, costs = score_maps(
        series, ["a"], lambda s, m: 1, lambda s, m: 0
    )
    with pytest.raises(ValueError, match="duplicated"):
        route_forecasts(
            hist, future, {"a": model}, losses, weights,
            {"a": 0}, costs, budget=100,
        )


def test_wrong_return_type_rejected(engine):
    hist, future = build_frames(engine)
    series = [f"s{i}" for i in range(4)]

    def bad_callback(_hist, _future):
        return {"not": "a frame"}

    losses, weights, costs = score_maps(
        series, ["a"], lambda s, m: 1, lambda s, m: 0
    )
    with pytest.raises(ValueError, match="must return"):
        route_forecasts(
            hist, future, {"a": bad_callback}, losses, weights,
            {"a": 0}, costs, budget=100,
        )


def test_forecast_works_with_native_evaluate(engine):
    hist, future = build_frames(engine)
    series = [f"s{i}" for i in range(4)]
    models = {"a": Recorder("a", 1.0), "b": Recorder("b", 2.0)}
    losses, weights, costs = score_maps(
        series, ["a", "b"],
        lambda s, m: {"a": 10, "b": 1}[m], lambda s, m: 1,
    )
    result = route_forecasts(
        hist, future, models, losses, weights,
        {"a": 1, "b": 1}, costs, budget=100,
    )
    # build an actuals frame matching the future keys
    if engine == "pandas":
        actual = future.copy()
        actual["y"] = 1.5
        eval_df = actual.merge(result.forecasts_df, on=["unique_id", "ds"])
    else:
        actual = future.with_columns(pl.lit(1.5).alias("y"))
        eval_df = actual.join(result.forecasts_df, on=["unique_id", "ds"])
    scores = evaluate(eval_df, metrics=[mae], models=result.used_models)
    assert all(m in scores.columns for m in result.used_models)


def test_rejects_non_integer_inputs(engine):
    hist, future = build_frames(engine)
    series = [f"s{i}" for i in range(4)]
    models = {"a": Recorder("a", 1.0)}
    losses, weights, costs = score_maps(
        series, ["a"], lambda s, m: 1, lambda s, m: 1
    )
    losses["s0"]["a"] = 1.5  # float, not a nonnegative integer
    with pytest.raises(TypeError, match="nonnegative integer"):
        route_forecasts(
            hist, future, models, losses, weights,
            {"a": 0}, costs, budget=100,
        )


def test_rejects_negative_values(engine):
    hist, future = build_frames(engine)
    series = [f"s{i}" for i in range(4)]
    models = {"a": Recorder("a", 1.0)}
    losses, weights, costs = score_maps(
        series, ["a"], lambda s, m: 1, lambda s, m: 1
    )
    costs["s1"]["a"] = -3
    with pytest.raises(ValueError, match="nonnegative"):
        route_forecasts(
            hist, future, models, losses, weights,
            {"a": 0}, costs, budget=100,
        )


def test_rejects_missing_enable_cost(engine):
    hist, future = build_frames(engine)
    series = [f"s{i}" for i in range(4)]
    models = {"a": Recorder("a", 1.0), "b": Recorder("b", 2.0)}
    losses, weights, costs = score_maps(
        series, ["a", "b"], lambda s, m: 1, lambda s, m: 1
    )
    with pytest.raises(ValueError, match="enablement cost"):
        route_forecasts(
            hist, future, models, losses, weights,
            {"a": 0}, costs, budget=100,  # missing b
        )


def test_mixed_frame_types_rejected():
    hist_pd, _ = build_frames("pandas")
    _, future_pl = build_frames("polars")
    series = [f"s{i}" for i in range(4)]
    model = Recorder("a", 1.0)
    losses, weights, costs = score_maps(
        series, ["a"], lambda s, m: 1, lambda s, m: 1
    )
    with pytest.raises(ValueError, match="same frame type"):
        route_forecasts(
            hist_pd, future_pl, {"a": model}, losses, weights,
            {"a": 0}, costs, budget=100,
        )


def test_series_without_history_rejected(engine):
    hist, future = build_frames(engine)
    # add a future series absent from history
    extra = {"unique_id": "ghost", "ds": 10}
    if engine == "pandas":
        future = pd.concat([future, pd.DataFrame([extra])], ignore_index=True)
    else:
        future = pl.concat([future, pl.DataFrame([extra])])
    series = [f"s{i}" for i in range(4)] + ["ghost"]
    model = Recorder("a", 1.0)
    losses = {s: {"a": 1} for s in series}
    weights = {s: {"a": 1} for s in series}
    costs = {s: {"a": 1} for s in series}
    with pytest.raises(ValueError, match="history"):
        route_forecasts(
            hist, future, {"a": model}, losses, weights,
            {"a": 0}, costs, budget=100,
        )


def test_assignment_table_costs_sum_to_total(engine):
    hist, future = build_frames(engine)
    series = [f"s{i}" for i in range(4)]
    models = {"a": Recorder("a", 1.0), "b": Recorder("b", 2.0)}
    losses, weights, costs = score_maps(
        series, ["a", "b"],
        lambda s, m: {"a": 10, "b": 1}[m], lambda s, m: {"a": 1, "b": 4}[m],
    )
    result = route_forecasts(
        hist, future, models, losses, weights,
        {"a": 2, "b": 5}, costs, budget=20,
    )
    assert set(result.used_models) == {"a", "b"}
    assign = result.assignments_df
    if engine == "polars":
        assign = assign.to_pandas()
    summed = assign["assignment_cost"].sum() + assign["enable_cost"].sum()
    assert summed == result.total_cost
    # enable fee appears exactly once per used model
    for name in result.used_models:
        assert (assign.loc[assign["model"] == name, "enable_cost"] > 0).sum() == 1
    # weighted loss matches the table exactly
    wl = (assign["loss"] * assign["weight"]).sum()
    assert wl == result.weighted_loss


def test_rejects_empty_models_mapping(engine):
    hist, future = build_frames(engine)
    series = [f"s{i}" for i in range(4)]
    losses, weights, costs = score_maps(
        series, ["a"], lambda s, m: 1, lambda s, m: 1
    )
    with pytest.raises(ValueError, match="non-empty mapping"):
        route_forecasts(
            hist, future, {}, losses, weights,
            {"a": 0}, costs, budget=100,
        )


def test_rejects_non_callable_model(engine):
    hist, future = build_frames(engine)
    series = [f"s{i}" for i in range(4)]
    losses, weights, costs = score_maps(
        series, ["a"], lambda s, m: 1, lambda s, m: 1
    )
    with pytest.raises(TypeError, match="callable"):
        route_forecasts(
            hist, future, {"a": "not a function"}, losses, weights,
            {"a": 0}, costs, budget=100,
        )


def test_rejects_boolean_as_integer(engine):
    hist, future = build_frames(engine)
    series = [f"s{i}" for i in range(4)]
    models = {"a": Recorder("a", 1.0)}
    losses, weights, costs = score_maps(
        series, ["a"], lambda s, m: 1, lambda s, m: 1
    )
    with pytest.raises(TypeError, match="nonnegative integer"):
        route_forecasts(
            hist, future, models, losses, weights,
            {"a": 0}, costs, budget=True,  # bool must not be accepted
        )


def test_rejects_unknown_model_in_score_table(engine):
    hist, future = build_frames(engine)
    series = [f"s{i}" for i in range(4)]
    models = {"a": Recorder("a", 1.0)}
    losses, weights, costs = score_maps(
        series, ["a"], lambda s, m: 1, lambda s, m: 1
    )
    losses["s0"] = {"ghost": 1}
    with pytest.raises(ValueError, match="unknown model"):
        route_forecasts(
            hist, future, models, losses, weights,
            {"a": 0}, costs, budget=100,
        )


def test_rejects_unknown_series_in_score_table(engine):
    hist, future = build_frames(engine)
    series = [f"s{i}" for i in range(4)]
    models = {"a": Recorder("a", 1.0)}
    losses, weights, costs = score_maps(
        series, ["a"], lambda s, m: 1, lambda s, m: 1
    )
    losses["ghost"] = {"a": 1}
    with pytest.raises(ValueError, match="unknown series id"):
        route_forecasts(
            hist, future, models, losses, weights,
            {"a": 0}, costs, budget=100,
        )


def test_rejects_negative_budget(engine):
    hist, future = build_frames(engine)
    series = [f"s{i}" for i in range(4)]
    models = {"a": Recorder("a", 1.0)}
    losses, weights, costs = score_maps(
        series, ["a"], lambda s, m: 1, lambda s, m: 0
    )
    with pytest.raises(ValueError, match="nonnegative"):
        route_forecasts(
            hist, future, models, losses, weights,
            {"a": 0}, costs, budget=-1,
        )


def test_rejects_non_frame_input(engine):
    hist, _ = build_frames(engine)
    series = [f"s{i}" for i in range(4)]
    model = Recorder("a", 1.0)
    losses, weights, costs = score_maps(
        series, ["a"], lambda s, m: 1, lambda s, m: 1
    )
    with pytest.raises((ValueError, TypeError)):
        route_forecasts(
            hist, [("s0", 10)], {"a": model}, losses, weights,
            {"a": 0}, costs, budget=100,
        )
