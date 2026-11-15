"""wrongway.Arena.match — two-mode match API for the Wrongway game engine.

Modes
-----
**Simulation mode** — both agents are Python callables living in the same
process. Call :func:`run_simulation` and get back a complete
:class:`MatchResult` with every turn recorded.

**Session mode** — an external controller (another script, a web server, a
test harness, …) owns the game loop. Create a :class:`Session`, read the
current observation with :meth:`Session.observe`, push your action with
:meth:`Session.step`, and check :attr:`Session.result` when it is done.

**Instant simulation (rollout)** — from any game state, evaluate a candidate
action by applying it on a *copy* of the board and letting two policy callables
play out the rest of the game automatically.  Use :func:`simulate_action` as a
standalone helper or :meth:`Session.simulate` on a live session.  The original
game state is never touched.

Both modes share the same :class:`TurnRecord` / :class:`MatchResult` schema so
replay, logging, and visualisation helpers work identically.

Quick-start examples
--------------------
Simulation::

    from Arena import run_simulation

    def my_agent(obs):
        moves = obs["legal_moves"]
        return {"type": "move", "to": moves[0]} if moves else {"type": "pass"}

    result = run_simulation(my_agent, my_agent)
    print(result.winner, result.total_turns)
    for t in result.turns:
        print(t.turn, t.player, t.action)

Session (two outside callers)::

    from Arena import Session

    session = Session()
    while not session.done:
        obs    = session.observe()          # dict – give to whichever side is up
        pid    = obs["current_player"]      # 0 or 1
        action = your_agents[pid](obs)      # you decide the action externally
        session.step(action)                # push it in

    result = session.result
    print(result.winner, result.history)    # history == list[TurnRecord]

Instant simulation (rollout) — evaluate a candidate action without committing::

    from Arena import Session, simulate_action

    session = Session()
    # ... play some turns ...
    obs = session.observe()

    # Pick a candidate action and roll out the rest with two policies
    candidate = {"type": "move", "to": obs["legal_moves"][0]}
    rollout = simulate_action(candidate, session.snapshot(), policy0, policy1)

    print(rollout.winner)        # projected winner if this action is taken
    print(rollout.total_turns)   # projected game length
    # session is UNCHANGED — keep playing normally

    # Or use the convenience method directly on the session:
    rollout = session.simulate(candidate, policy0, policy1)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .environment import Environment


# ---------------------------------------------------------------------------
# Public data-classes
# ---------------------------------------------------------------------------

@dataclass
class TurnRecord:
    """Everything that happened during one half-turn."""

    turn: int
    """Turn counter *after* the action was applied (matches ``Environment.turn_count``)."""

    player: int
    """Which player acted (0 or 1)."""

    action: Optional[Dict[str, Any]]
    """The action dict that was applied, or ``None`` for a forfeit / pass."""

    status: str
    """``'ok'``, ``'pass'``, ``'invalid'``, or ``'error'``."""

    reason: Optional[str]
    """Engine reason string (e.g. ``'move'``, ``'illegal_move'``, ``'timeout'``)."""

    elapsed: float
    """Wall-clock seconds the agent spent choosing (always 0.0 in Session mode)."""

    error: Optional[str]
    """Agent-level error message, if any."""

    state: Dict[str, Any]
    """Full board snapshot *after* the action."""


@dataclass
class MatchResult:
    """Returned by :func:`run_simulation` and exposed as :attr:`Session.result`."""

    winner: Optional[int]
    """``0``, ``1``, or ``None`` for a draw or an in-progress game."""

    winner_name: Optional[str]
    """Friendly name passed to the factory, or ``'player0'`` / ``'player1'``."""

    termination_reason: Optional[str]
    """``'goal'``, ``'max_turns'``, or ``None`` if the game is still live."""

    total_turns: int
    """Number of half-turns that have been played so far."""

    turns: List[TurnRecord]
    """All turn records in chronological order."""

    initial_state: Dict[str, Any]
    """Board snapshot before the first action."""

    final_state: Optional[Dict[str, Any]]
    """Board snapshot after the last action (``None`` while the game is live)."""

    config: Dict[str, Any]
    """The parameters the match was created with."""

    @property
    def history(self) -> List[TurnRecord]:
        """Alias for :attr:`turns` — same list, different name."""
        return self.turns

    @property
    def done(self) -> bool:
        """``True`` once the game has ended."""
        return self.termination_reason is not None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _apply(env: Environment, action: Any, elapsed: float = 0.0) -> TurnRecord:
    """Submit *action* to the environment and return a :class:`TurnRecord`."""
    player = env.current_player

    if not isinstance(action, dict):
        env.skip_turn("invalid_action_format")
        return TurnRecord(
            turn=env.turn_count,
            player=player,
            action=None,
            status="invalid",
            reason="invalid_action_format",
            elapsed=elapsed,
            error="Action must be a dict",
            state=env.snapshot(),
        )

    update = env.step(action)
    status = update["status"]
    return TurnRecord(
        turn=env.turn_count,
        player=player,
        action=update.get("action"),
        status=status,
        reason=update.get("reason"),
        elapsed=elapsed,
        error=update.get("reason") if status == "invalid" else None,
        state=env.snapshot(),
    )


def _make_config(
    max_turns: int,
    seed: Optional[int],
    starting_player: int,
    names: List[str],
) -> Dict[str, Any]:
    return {
        "max_turns": max_turns,
        "seed": seed,
        "starting_player": starting_player,
        "player_names": names,
    }


# ---------------------------------------------------------------------------
# Mode 1 — Simulation
# ---------------------------------------------------------------------------

AgentFn = Callable[[Dict[str, Any]], Dict[str, Any]]


def run_simulation(
    agent0: AgentFn,
    agent1: AgentFn,
    *,
    max_turns: int = 500,
    starting_player: int = 0,
    seed: Optional[int] = None,
    names: Optional[List[str]] = None,
    on_turn: Optional[Callable[[TurnRecord], None]] = None,
) -> MatchResult:
    """Run a complete match between two in-process callables.

    Parameters
    ----------
    agent0, agent1:
        Callables with signature ``(observation: dict) -> dict``.  They are
        called synchronously; exceptions are caught, the turn is forfeited, and
        the error message is stored in :attr:`TurnRecord.error`.
    max_turns:
        Maximum number of half-turns before the game ends in a draw.
    starting_player:
        ``0`` (default) or ``1``.
    seed:
        If given, passed to :mod:`random` before each agent call so that
        random-based agents reproduce deterministically.  ``None`` leaves the
        RNG unseeded.
    names:
        ``[name_for_player0, name_for_player1]``. Defaults to
        ``['player0', 'player1']``.
    on_turn:
        Optional callback invoked with the :class:`TurnRecord` after every
        half-turn — useful for live rendering without touching the return value.

    Returns
    -------
    MatchResult
        A fully-populated result including every :class:`TurnRecord`.
    """
    if names is None:
        names = ["player0", "player1"]
    agents: List[AgentFn] = [agent0, agent1]

    env = Environment(max_turns=max_turns, starting_player=starting_player)
    initial = env.snapshot()
    turns: List[TurnRecord] = []

    if seed is not None:
        import random as _random
        _random.seed(seed)

    while not env.done:
        player = env.current_player
        obs = env.get_observation(player)

        t0 = time.monotonic()
        error: Optional[str] = None
        try:
            action = agents[player](obs)
        except Exception as exc:
            action = None
            error = f"{type(exc).__name__}: {exc}"
        elapsed = time.monotonic() - t0

        if error is not None:
            env.skip_turn("agent_error")
            record = TurnRecord(
                turn=env.turn_count,
                player=player,
                action=None,
                status="error",
                reason="agent_error",
                elapsed=elapsed,
                error=error,
                state=env.snapshot(),
            )
        else:
            record = _apply(env, action, elapsed)

        turns.append(record)
        if on_turn is not None:
            on_turn(record)

    winner_name = names[env.winner] if env.winner is not None else None
    return MatchResult(
        winner=env.winner,
        winner_name=winner_name,
        termination_reason=env.termination_reason,
        total_turns=env.turn_count,
        turns=turns,
        initial_state=initial,
        final_state=env.snapshot(),
        config=_make_config(max_turns, seed, starting_player, names),
    )


# ---------------------------------------------------------------------------
# Mode 1b — Instant simulation (rollout / what-if)
# ---------------------------------------------------------------------------

def simulate_action(
    action: Any,
    state_snapshot: Dict[str, Any],
    policy0: AgentFn,
    policy1: AgentFn,
    *,
    names: Optional[List[str]] = None,
    on_turn: Optional[Callable[[TurnRecord], None]] = None,
) -> "MatchResult":
    """Apply *action* on a copy of *state_snapshot* and auto-play to the end.

    This is the **instant simulation** (rollout) primitive.  It lets a training
    agent evaluate any candidate action without touching the live game state.

    How it works
    ------------
    1. A fresh :class:`~Arena.environment.Environment` is reconstructed from
       *state_snapshot* using :meth:`~Arena.environment.Environment.from_snapshot`.
    2. *action* is applied for the **currently active player**.
    3. Both *policy0* and *policy1* then play alternate turns until the game
       ends naturally (win or draw).
    4. The full :class:`MatchResult` — including every turn from step 2 onward
       — is returned.  The original session / environment is **never mutated**.

    Parameters
    ----------
    action:
        The candidate action dict to evaluate (e.g.
        ``{"type": "move", "to": [6, 4]}``).  If the action is illegal the
        turn is forfeited (same as the main arena) and the rollout continues.
    state_snapshot:
        A snapshot dict returned by :meth:`Session.snapshot` or
        :meth:`~Arena.environment.Environment.snapshot`.  The snapshot is not
        mutated.
    policy0, policy1:
        Callables with signature ``(observation: dict) -> dict``.  These are
        the policies used to play out the remainder of the game after *action*
        is applied.  Typically both are set to your current trained policy (or
        a cheap heuristic) to get a quick value estimate.
    names:
        ``[name_for_player0, name_for_player1]``.  Defaults to
        ``['player0', 'player1']``.
    on_turn:
        Optional callback invoked with the :class:`TurnRecord` after every
        half-turn of the rollout.

    Returns
    -------
    MatchResult
        A complete result covering the entire rollout (starting from *action*).
        Key attributes for training:

        * ``result.winner`` — projected winner (``0``, ``1``, or ``None``).
        * ``result.total_turns`` — projected game length.
        * ``result.turns`` — every turn played during the rollout.
        * ``result.initial_state`` — board state *before* the rollout action.
        * ``result.final_state`` — board state at the end of the rollout.

    Example
    -------
    ::

        from Arena import Session, simulate_action

        def my_policy(obs):
            moves = obs["legal_moves"]
            return {"type": "move", "to": moves[0]} if moves else {"type": "pass"}

        session = Session()
        # ... play several real turns ...
        obs = session.observe()

        best_action, best_winner = None, -1
        for candidate in obs["legal_moves"]:
            action = {"type": "move", "to": candidate}
            rollout = simulate_action(action, session.snapshot(), my_policy, my_policy)
            my_pid  = obs["player_id"]
            if rollout.winner == my_pid:
                best_action = action
                break

        # Commit the real turn — session is still live and unmodified
        session.step(best_action or obs["legal_moves"][0])
    """
    if names is None:
        names = ["player0", "player1"]
    policies: List[AgentFn] = [policy0, policy1]

    env = Environment.from_snapshot(state_snapshot)
    initial = env.snapshot()           # snapshot *before* the rollout action
    turns: List[TurnRecord] = []

    if env.done:
        # Nothing to simulate — return a completed-game result immediately.
        winner_name = names[env.winner] if env.winner is not None else None
        return MatchResult(
            winner=env.winner,
            winner_name=winner_name,
            termination_reason=env.termination_reason,
            total_turns=env.turn_count,
            turns=turns,
            initial_state=initial,
            final_state=env.snapshot(),
            config=_make_config(env.max_turns, None, state_snapshot.get("current_player", 0), names),
        )

    # Step 1 — apply the candidate action (may forfeit if illegal).
    first_record = _apply(env, action)
    turns.append(first_record)
    if on_turn is not None:
        on_turn(first_record)

    # Step 2 — auto-play the remainder with the supplied policies.
    while not env.done:
        player = env.current_player
        obs = env.get_observation(player)

        t0 = time.monotonic()
        error: Optional[str] = None
        try:
            act = policies[player](obs)
        except Exception as exc:
            act = None
            error = f"{type(exc).__name__}: {exc}"
        elapsed = time.monotonic() - t0

        if error is not None:
            env.skip_turn("agent_error")
            record = TurnRecord(
                turn=env.turn_count,
                player=player,
                action=None,
                status="error",
                reason="agent_error",
                elapsed=elapsed,
                error=error,
                state=env.snapshot(),
            )
        else:
            record = _apply(env, act, elapsed)

        turns.append(record)
        if on_turn is not None:
            on_turn(record)

    winner_name = names[env.winner] if env.winner is not None else None
    return MatchResult(
        winner=env.winner,
        winner_name=winner_name,
        termination_reason=env.termination_reason,
        total_turns=env.turn_count,
        turns=turns,
        initial_state=initial,
        final_state=env.snapshot(),
        config=_make_config(env.max_turns, None, state_snapshot.get("current_player", 0), names),
    )


# ---------------------------------------------------------------------------
# Mode 2 — Session (external turn-by-turn control)
# ---------------------------------------------------------------------------

class Session:
    """Stateful match where external callers take turns pushing actions.

    The game loop lives entirely outside this object. Typical usage::

        session = Session()

        while not session.done:
            obs    = session.observe()           # current state from active player's POV
            action = your_logic(obs)             # you decide what to play
            session.step(action)                 # commit it

        print(session.result.winner)
        print(session.result.history)            # list[TurnRecord]

    Thread safety
    -------------
    ``Session`` is **not** thread-safe. Use external locking if multiple
    threads need to share one instance.
    """

    def __init__(
        self,
        *,
        max_turns: int = 500,
        starting_player: int = 0,
        names: Optional[List[str]] = None,
    ) -> None:
        """
        Parameters
        ----------
        max_turns:
            Draw limit in half-turns.
        starting_player:
            ``0`` (default) or ``1``.
        names:
            ``[name_for_player0, name_for_player1]``. Defaults to
            ``['player0', 'player1']``.
        """
        if names is None:
            names = ["player0", "player1"]
        self._names = names
        self._env = Environment(max_turns=max_turns, starting_player=starting_player)
        self._turns: List[TurnRecord] = []
        self._initial_state: Dict[str, Any] = self._env.snapshot()
        self._config = _make_config(max_turns, None, starting_player, names)

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def done(self) -> bool:
        """``True`` once the game has ended (win or draw)."""
        return self._env.done

    @property
    def current_player(self) -> Optional[int]:
        """Active player (``0`` or ``1``), or ``None`` when the game is over."""
        return self._env.current_player if not self._env.done else None

    @property
    def winner(self) -> Optional[int]:
        """``0``, ``1``, or ``None`` if the game is not yet decided."""
        return self._env.winner

    @property
    def history(self) -> List[TurnRecord]:
        """All :class:`TurnRecord` objects so far (live list — do not mutate)."""
        return self._turns

    @property
    def result(self) -> MatchResult:
        """A :class:`MatchResult` snapshot valid at any point, including mid-game.

        :attr:`MatchResult.final_state` is ``None`` and
        :attr:`MatchResult.done` is ``False`` while the game is still live.
        """
        winner_name = (
            self._names[self._env.winner] if self._env.winner is not None else None
        )
        return MatchResult(
            winner=self._env.winner,
            winner_name=winner_name,
            termination_reason=self._env.termination_reason,
            total_turns=self._env.turn_count,
            turns=list(self._turns),       # snapshot copy
            initial_state=self._initial_state,
            final_state=self._env.snapshot() if self._env.done else None,
            config=self._config,
        )

    # ------------------------------------------------------------------
    # Active interface
    # ------------------------------------------------------------------

    def observe(self, player: Optional[int] = None) -> Dict[str, Any]:
        """Return the current observation for *player* (default: active player).

        Raises
        ------
        RuntimeError
            If the game is already over.
        """
        if self._env.done:
            raise RuntimeError("The game is already over. Check session.result.")
        return self._env.get_observation(player)

    def step(self, action: Any) -> TurnRecord:
        """Submit an action for the currently active player.

        Any action accepted by :class:`~Arena.environment.Environment`
        (move, wall, pass) works here. Invalid or malformed actions forfeit
        the turn just like the main arena does, so the game always advances.

        Parameters
        ----------
        action:
            A dict such as ``{"type": "move", "to": [row, col]}``.

        Returns
        -------
        TurnRecord
            The record for the turn that was just played.

        Raises
        ------
        RuntimeError
            If the game is already over.
        """
        if self._env.done:
            raise RuntimeError("The game is already over. Check session.result.")
        record = _apply(self._env, action, elapsed=0.0)
        self._turns.append(record)
        return record

    def skip(self, reason: str = "external_pass") -> TurnRecord:
        """Forfeit the current player's turn without submitting an action.

        Useful when the external controller cannot decide in time or wants to
        yield deliberately.
        """
        if self._env.done:
            raise RuntimeError("The game is already over.")
        player = self._env.current_player
        update = self._env.skip_turn(reason)
        record = TurnRecord(
            turn=self._env.turn_count,
            player=player,
            action=update.get("action"),
            status="pass",
            reason=reason,
            elapsed=0.0,
            error=None,
            state=self._env.snapshot(),
        )
        self._turns.append(record)
        return record

    def snapshot(self) -> Dict[str, Any]:
        """Raw board snapshot (same format as :meth:`~Arena.environment.Environment.snapshot`)."""
        return self._env.snapshot()

    def simulate(
        self,
        action: Any,
        policy0: AgentFn,
        policy1: AgentFn,
        *,
        names: Optional[List[str]] = None,
        on_turn: Optional[Callable[[TurnRecord], None]] = None,
    ) -> "MatchResult":
        """Evaluate *action* by rolling out the rest of the game on a copy of this session.

        This is a convenience wrapper around :func:`simulate_action` that
        automatically passes the current session's snapshot as the starting
        state.  The live session is **never modified**.

        Parameters
        ----------
        action:
            The candidate action to evaluate.
        policy0, policy1:
            Callables ``(obs: dict) -> dict`` used to auto-play every turn
            after *action* is committed on the copy.  Pass your current agent
            as both arguments to simulate self-play, or use a cheap heuristic
            for speed.
        names:
            Player names for the rollout result.  Defaults to the session's
            own names.
        on_turn:
            Optional callback called with each :class:`TurnRecord` during the
            rollout.

        Returns
        -------
        MatchResult
            Projected outcome if *action* is taken now.

        Example
        -------
        ::

            session = Session(names=["Trainer", "Opponent"])

            while not session.done:
                obs = session.observe()
                pid = obs["current_player"]

                if pid == 0:
                    # Evaluate all legal moves via instant simulation
                    best, best_score = None, -1
                    for m in obs["legal_moves"]:
                        candidate = {"type": "move", "to": m}
                        rollout = session.simulate(candidate, my_policy, my_policy)
                        score = 1 if rollout.winner == 0 else 0
                        if score > best_score:
                            best, best_score = candidate, score
                    action = best or {"type": "pass"}
                else:
                    action = opponent_policy(obs)

                # Commit the REAL turn — session is unchanged by simulate()
                session.step(action)
        """
        if self._env.done:
            raise RuntimeError("The game is already over. Check session.result.")
        return simulate_action(
            action,
            self._env.snapshot(),
            policy0,
            policy1,
            names=names if names is not None else list(self._names),
            on_turn=on_turn,
        )

    def __repr__(self) -> str:
        if self._env.done:
            return f"<Session done, winner={self._env.winner}>"
        return f"<Session turn {self._env.turn_count}, P{self._env.current_player}, winner=None>"
