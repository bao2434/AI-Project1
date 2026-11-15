"""Arena — Wrongway game engine, importable as a package.

Public API
----------

Simulation mode (both agents are callables in the same process)::

    from Arena import run_simulation, MatchResult, TurnRecord

    def my_agent(obs):
        moves = obs["legal_moves"]
        return {"type": "move", "to": moves[0]} if moves else {"type": "pass"}

    result = run_simulation(my_agent, my_agent, seed=42, names=["Alice", "Bob"])
    print(result.winner, result.winner_name, result.total_turns)
    for t in result.turns:          # list[TurnRecord]
        print(t.turn, t.player, t.action, t.status)

Session mode (you own the game loop, agents live outside)::

    from Arena import Session

    session = Session(names=["X", "Y"])
    while not session.done:
        obs    = session.observe()           # POV of whoever is active
        pid    = obs["current_player"]       # 0 or 1
        action = your_agents[pid](obs)
        session.step(action)

    result = session.result                  # MatchResult
    print(result.winner, result.history)     # history == list[TurnRecord]

CLI (run from the wrongway/ directory)::

    python -m Arena.cli --player1 submissions/greedy.py --player2 submissions/wall_builder.py --render

Lower-level access::

    from Arena.environment import Environment
    from Arena.visualizer  import render_board
    from Arena.agent_runner import StudentAgent   # subprocess-isolated file runner
"""

from .match import (
    run_simulation,
    Session,
    MatchResult,
    TurnRecord,
    AgentFn,
)
from .environment import Environment
from .visualizer import render_board

__all__ = [
    # High-level API
    "run_simulation",
    "Session",
    # Data classes
    "MatchResult",
    "TurnRecord",
    # Type alias
    "AgentFn",
    # Low-level
    "Environment",
    "render_board",
]
