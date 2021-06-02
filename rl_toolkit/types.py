from typing import Any, NamedTuple


class Transition(NamedTuple):
  """Container for a transition."""
  observation: Any
  action: Any
  reward: Any
  next_observation: Any
  terminal: Any
