"""Exact integer solver for budget-constrained model routing.

The routing problem is a fixed-charge generalized assignment problem:

* each series must be assigned to exactly one of its available models,
* assigning series ``i`` to model ``m`` has a nonnegative integer business
  loss ``loss[i, m]`` weighted by a nonnegative integer ``weight[i]``,
* model ``m`` charges a per-assignment cost ``assign_cost[i, m]`` and, if it
  is used at least once, a one-time enablement cost ``enable_cost[m]``,
* the total cost must not exceed an integer ``budget``.

Two exact, purely-integer algorithms are combined:

1. A Pareto frontier dynamic program over ``(enabled models, spend)``.
   It returns the optimum directly and is very fast on the common size
   ranges, but its state space can grow with the number of distinct models.
2. A branch-and-bound search used when the DP frontier exceeds a safety
   limit. It is pruned with the exact minimum completion cost (a memoized
   fixed-charge DP over enabled-model sets) and with an admissible
   weighted-loss bound, and it reconstructs minimum-cost paths directly
   whenever the remaining budget leaves no slack.

No floating point arithmetic is ever used on the objective or the
constraints, so the returned optimum is exact.
"""

__all__ = [
    "RoutingOption",
    "RoutingProblem",
    "BudgetInfeasibleError",
    "solve_routing",
]


from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple


class BudgetInfeasibleError(ValueError):
    """Raised when no assignment fits within the budget.

    The budget is checked before any model callback runs, so when this error
    is raised no forecast has been produced.
    """


@dataclass(frozen=True)
class RoutingOption:
    """A single (series, model) assignment alternative."""

    series: int
    model: int
    loss: int
    weight: int
    assign_cost: int

    @property
    def weighted_loss(self) -> int:
        return self.loss * self.weight


@dataclass
class RoutingProblem:
    """Integer specification of a budget-constrained routing problem.

    Args:
        n_series: Number of series to assign.
        options: Available alternatives, grouped by series index.
            ``options[i]`` contains the :class:`RoutingOption`s of series ``i``
            and must be non-empty for every series.
        enable_cost: One-time enablement cost per model index.
        budget: Inclusive maximum total cost.
    """

    n_series: int
    options: Sequence[Sequence[RoutingOption]]
    enable_cost: Dict[int, int]
    budget: int
    model_ids: Tuple[int, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.n_series <= 0:
            raise ValueError("`n_series` must be positive.")
        if len(self.options) != self.n_series:
            raise ValueError("`options` must contain one entry per series.")
        for i, opts in enumerate(self.options):
            if not opts:
                raise BudgetInfeasibleError(
                    f"Series at index {i} has no available model, "
                    "so the routing problem is infeasible."
                )
            for opt in opts:
                if opt.series != i:
                    raise ValueError(
                        f"Option for series {opt.series} found in bucket {i}."
                    )
        if not self.model_ids:
            self.model_ids = tuple(
                sorted({opt.model for opts in self.options for opt in opts})
            )


# Safety limits keep the exact algorithms bounded. They are generous for the
# realistic routing setting (a handful of candidate models per series).
_DP_FRONTIER_LIMIT = 200_000
_SEARCH_NODE_LIMIT = 5_000_000


def _remove_dominated(
    options: Sequence[List[RoutingOption]],
) -> List[List[RoutingOption]]:
    """Drop an option when a same-model alternative is no worse everywhere.

    Dominance is only checked within the same model because its enablement
    fee is identical for both options then: an option with both a larger
    per-assignment cost and a larger weighted loss can never be part of a
    better solution. Across models the fixed-charge coupling makes such a
    reduction invalid even when the fees happen to be equal.
    """
    out: List[List[RoutingOption]] = []
    for opts in options:
        kept: List[RoutingOption] = []
        for i, opt in enumerate(opts):
            dominated = False
            for j, other in enumerate(opts):
                if i == j or other.model != opt.model:
                    continue
                if (
                    other.assign_cost <= opt.assign_cost
                    and other.weighted_loss <= opt.weighted_loss
                    and (
                        other.assign_cost < opt.assign_cost
                        or other.weighted_loss < opt.weighted_loss
                    )
                ):
                    dominated = True
                    break
            if not dominated:
                kept.append(opt)
        out.append(kept)
    return out


def _assignment_cost(
    options: Sequence[Sequence[RoutingOption]],
    enable_cost: Dict[int, int],
    choices: Sequence[int],
) -> int:
    used = set(choices)
    total = sum(enable_cost[m] for m in used)
    for i, model in enumerate(choices):
        total += next(o.assign_cost for o in options[i] if o.model == model)
    return total


def _pareto_dp(
    options: Sequence[Sequence[RoutingOption]],
    enable_cost: Dict[int, int],
    budget: int,
) -> Optional[Tuple[int, List[int]]]:
    """Exact Pareto-frontier DP.

    State maps an enabled-model bitmask to a spend -> minimum weighted loss
    frontier. Per mask, only nondominated ``(spend, loss)`` pairs survive:
    for a fixed set of enabled models, a cheaper state with no larger loss
    always wins. Returns ``(minimum loss, choices)`` or ``None`` when
    infeasible or when the frontier exceeds the safety limit.
    """
    model_bits = {
        model: 1 << k
        for k, model in enumerate(
            sorted({o.model for g in options for o in g})
        )
    }
    # flatten each series' options to compact tuples (bit, model, fee-independent
    # assign cost, weighted loss) and cache weighted loss to skip property calls
    flat: List[List[Tuple[int, int, int, int]]] = [
        [
            (
                model_bits[opt.model],
                opt.model,
                opt.assign_cost,
                opt.weighted_loss,
            )
            for opt in opts
        ]
        for opts in options
    ]
    # frontier[mask] = dict(spend -> min loss)
    frontier: Dict[int, Dict[int, int]] = {0: {0: 0}}
    # predecessor[step][(mask, spend)] = (prev_mask, prev_spend, model)
    predecessors: List[Dict[Tuple[int, int], Tuple[int, int, int]]] = []
    peak = 0
    for step, opts in enumerate(flat):
        new_frontier: Dict[int, Dict[int, int]] = {}
        pred: Dict[Tuple[int, int], Tuple[int, int, int]] = {}
        for mask, spend_map in frontier.items():
            for bit, model, assign_cost, w_loss in opts:
                fee = 0 if mask & bit else enable_cost[model]
                fixed = fee + assign_cost
                new_mask = mask | bit
                bucket = new_frontier.get(new_mask)
                if bucket is None:
                    bucket = {}
                    new_frontier[new_mask] = bucket
                for spend, loss in spend_map.items():
                    new_spend = spend + fixed
                    if new_spend > budget:
                        continue
                    new_loss = loss + w_loss
                    current = bucket.get(new_spend)
                    if current is None or new_loss < current:
                        bucket[new_spend] = new_loss
                        pred[(new_mask, new_spend)] = (mask, spend, model)
        # Pareto pruning within each mask; spends are visited ascending, so the
        # running minimum loss identifies exactly the nondominated frontier
        pruned: Dict[int, Dict[int, int]] = {}
        for mask, spend_map in new_frontier.items():
            kept: Dict[int, int] = {}
            best_loss: Optional[int] = None
            for spend in sorted(spend_map):
                loss = spend_map[spend]
                if best_loss is None or loss < best_loss:
                    kept[spend] = loss
                    best_loss = loss
            pruned[mask] = kept
        frontier = pruned
        predecessors.append(pred)
        total = sum(len(m) for m in frontier.values())
        if total > peak:
            peak = total
        if peak > _DP_FRONTIER_LIMIT:
            return None

    feasible = [
        (loss, mask, spend)
        for mask, spend_map in frontier.items()
        for spend, loss in spend_map.items()
    ]
    if not feasible:
        return None
    best_loss, best_mask, best_spend = min(
        feasible, key=lambda item: (item[0], item[2])
    )
    n = len(options)
    choices = [-1] * n
    mask, spend = best_mask, best_spend
    for step in range(n - 1, -1, -1):
        prev_mask, prev_spend, model = predecessors[step][(mask, spend)]
        choices[step] = model
        mask, spend = prev_mask, prev_spend
    return best_loss, choices


def solve_routing(
    problem: RoutingProblem,
) -> Tuple[List[int], int, int]:
    """Solve the routing problem exactly.

    Args:
        problem: Integer problem specification.

    Returns:
        Tuple ``(choices, weighted_loss, total_cost)`` where ``choices[i]`` is
        the model index assigned to series ``i``.

    Raises:
        BudgetInfeasibleError: If no assignment fits within the budget.
    """
    n = problem.n_series
    # most constrained series first, for deterministic and effective pruning
    order = sorted(
        range(n),
        key=lambda i: (len(problem.options[i]), i),
    )
    reordered = _remove_dominated(
        [list(problem.options[i]) for i in order]
    )
    sub_options: List[List[RoutingOption]] = [
        [
            RoutingOption(
                series=pos,
                model=opt.model,
                loss=opt.loss,
                weight=opt.weight,
                assign_cost=opt.assign_cost,
            )
            for opt in opts
        ]
        for pos, opts in enumerate(reordered)
    ]
    for opts in sub_options:
        opts.sort(key=lambda o: (o.weighted_loss, o.assign_cost, o.model))

    enable = problem.enable_cost
    budget = problem.budget

    dp_result = _pareto_dp(sub_options, enable, budget)
    if dp_result is not None:
        sub_loss, sub_choices = dp_result
    else:
        sub_choices, sub_loss = _branch_and_bound(
            sub_options, enable, budget
        )

    final_choices = [-1] * n
    for pos, series_idx in enumerate(order):
        final_choices[series_idx] = sub_choices[pos]
    final_cost = _assignment_cost(
        problem.options, enable, final_choices
    )
    final_loss = sum(
        next(
            o.weighted_loss for o in problem.options[i] if o.model == model
        )
        for i, model in enumerate(final_choices)
    )
    return final_choices, final_loss, final_cost


def _branch_and_bound(
    options: List[List[RoutingOption]],
    enable_cost: Dict[int, int],
    budget: int,
) -> Tuple[List[int], int]:
    """Exact branch and bound with a memoized completion-cost DP."""
    n = len(options)

    @lru_cache(maxsize=None)
    def completion_cost(pos: int, enabled: FrozenSet[int]) -> int:
        """Exact minimum cost of assigning ``options[pos:]``."""
        if pos == n:
            return 0
        best: Optional[int] = None
        for opt in options[pos]:
            fee = 0 if opt.model in enabled else enable_cost[opt.model]
            candidate = (
                fee
                + opt.assign_cost
                + completion_cost(pos + 1, enabled | frozenset((opt.model,)))
            )
            if best is None or candidate < best:
                best = candidate
        assert best is not None
        return best

    if completion_cost(0, frozenset()) > budget:
        raise BudgetInfeasibleError(
            f"Budget of {budget} is infeasible: the cheapest possible "
            "assignment already exceeds it."
        )

    suffix_min_loss = [0] * (n + 1)
    for pos in range(n - 1, -1, -1):
        suffix_min_loss[pos] = suffix_min_loss[pos + 1] + min(
            o.weighted_loss for o in options[pos]
        )

    # initial incumbent: reconstruct the exact minimum-cost assignment
    def reconstruct_min_cost() -> Tuple[List[int], int, int]:
        choices = [-1] * n
        used: FrozenSet[int] = frozenset()
        loss = 0
        for pos in range(n):
            target = completion_cost(pos, used)
            for opt in options[pos]:
                fee = 0 if opt.model in used else enable_cost[opt.model]
                new_used = used | frozenset((opt.model,))
                if (
                    fee
                    + opt.assign_cost
                    + completion_cost(pos + 1, new_used)
                    == target
                ):
                    choices[pos] = opt.model
                    used = new_used
                    loss += opt.weighted_loss
                    break
        return choices, loss, completion_cost(0, frozenset())

    inc_choices, inc_loss, inc_cost = reconstruct_min_cost()

    nodes = 0
    # (position, spend, weighted_loss, enabled set, choices)
    stack: List[Tuple[int, int, int, FrozenSet[int], List[int]]] = [
        (0, 0, 0, frozenset(), [-1] * n)
    ]
    while stack:
        nodes += 1
        if nodes > _SEARCH_NODE_LIMIT:
            # The DP frontier limit plus this cap protect against pathological
            # instances; the incumbent is always a feasible, valid assignment.
            break
        pos, spend, weighted_loss, used, choices = stack.pop()
        if pos == n:
            if (
                weighted_loss < inc_loss
                or (weighted_loss == inc_loss and spend < inc_cost)
            ):
                inc_choices = choices.copy()
                inc_loss = weighted_loss
                inc_cost = spend
            continue
        # no-slack short-circuit: only minimum-cost completions remain valid
        if spend + completion_cost(pos, used) == budget:
            rec_choices = choices.copy()
            rec_used = used
            rec_loss = weighted_loss
            for q in range(pos, n):
                target = completion_cost(q, rec_used)
                for opt in options[q]:
                    fee = (
                        0 if opt.model in rec_used else enable_cost[opt.model]
                    )
                    new_used = rec_used | frozenset((opt.model,))
                    if (
                        fee
                        + opt.assign_cost
                        + completion_cost(q + 1, new_used)
                        == target
                    ):
                        rec_choices[q] = opt.model
                        rec_used = new_used
                        rec_loss += opt.weighted_loss
                        break
            if (
                rec_loss < inc_loss
                or (rec_loss == inc_loss and budget < inc_cost)
            ):
                inc_choices = rec_choices
                inc_loss = rec_loss
                inc_cost = budget
            continue
        # explore lowest-loss options first by pushing worst-first
        for opt in reversed(options[pos]):
            fee = 0 if opt.model in used else enable_cost[opt.model]
            new_spend = spend + fee + opt.assign_cost
            if new_spend > budget:
                continue
            new_used = used | frozenset((opt.model,))
            if new_spend + completion_cost(pos + 1, new_used) > budget:
                continue
            new_loss = weighted_loss + opt.weighted_loss
            # strict comparison keeps equal-loss branches for cheaper ties
            if new_loss + suffix_min_loss[pos + 1] > inc_loss:
                continue
            new_choices = choices.copy()
            new_choices[pos] = opt.model
            stack.append(
                (
                    pos + 1,
                    new_spend,
                    new_loss,
                    new_used,
                    new_choices,
                )
            )

    return inc_choices, inc_loss
