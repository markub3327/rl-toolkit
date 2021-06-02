from typing import Any, NamedTuple


class Transition(NamedTuple):
    observation: Any
    action: Any
    reward: Any
    next_observation: Any
    terminal: Any
