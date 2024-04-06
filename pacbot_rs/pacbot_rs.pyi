from typing import Callable, Literal, Mapping, Optional

import numpy as np
from numpy.typing import NDArray

class GhostAgent:
    def clear_start_path(self) -> None: ...
    @property
    def pos(self) -> Mapping[Literal["current", "next"], tuple[int, int]]: ...

class PacBot:
    def update(self, position: tuple[int, int]) -> None: ...
    @property
    def direction(self) -> int: ...
    @property
    def pos(self) -> tuple[int, int]: ...

class GameState:
    def __init__(self) -> None: ...
    def is_frightened(self) -> bool: ...
    def next_step(self) -> None: ...
    def pause(self) -> None: ...
    def unpause(self) -> None: ...
    def restart(self) -> None: ...
    def print_ghost_pos(self) -> None: ...
    def frightened_counter(self) -> int: ...
    def state(self) -> int: ...
    @property
    def pacbot(self) -> PacBot: ...
    @property
    def red(self) -> GhostAgent: ...
    @property
    def pink(self) -> GhostAgent: ...
    @property
    def orange(self) -> GhostAgent: ...
    @property
    def blue(self) -> GhostAgent: ...
    @property
    def pellets(self) -> int: ...
    @property
    def power_pellets(self) -> int: ...
    @property
    def cherry(self) -> bool: ...
    @property
    def score(self) -> int: ...
    @property
    def play(self) -> bool: ...
    @property
    def lives(self) -> int: ...

class PacmanGym:
    def __init__(self, random_start: bool) -> None: ...
    def reset(self) -> None: ...
    def step(self, action: int) -> tuple[int, bool]: ...
    def score(self) -> int: ...
    def lives(self) -> int: ...
    def is_done(self) -> bool: ...
    def action_mask(self) -> list[bool]: ...
    def obs_numpy(self) -> np.ndarray: ...
    def print_game_state(self) -> None: ...
    @property
    def game_state(self) -> GameState: ...
    @property
    def random_start(self) -> bool: ...

EvaluatorFunc = Callable[
    [NDArray[np.float32], NDArray[np.bool_]],
    tuple[NDArray[np.float32], NDArray[np.float32]],
]

class MCTSContext:
    def __init__(self, env: PacmanGym, evaluator: EvaluatorFunc) -> None: ...
    def reset(self) -> None: ...
    def take_action(self, action: int) -> tuple[int, bool]: ...
    def best_action(self) -> int: ...
    def action_distribution(self) -> list[float]: ...
    def policy_prior(self) -> list[float]: ...
    def action_values(self) -> list[float]: ...
    def value(self) -> float: ...
    def ponder_and_choose(self, max_tree_size: int) -> int: ...
    def max_depth(self) -> int: ...
    def node_count(self) -> int: ...
    def root_obs_numpy(self) -> np.ndarray: ...
    @property
    def env(self) -> PacmanGym: ...
    evaluator: EvaluatorFunc

class AlphaZeroConfig:
    def __init__(
        self,
        tree_size: int,
        max_episode_length: int,
        discount_factor: float,
        num_parallel_envs: int,
    ) -> None: ...

    tree_size: int
    max_episode_length: int
    discount_factor: float
    num_parallel_envs: int

class ExperienceItem:
    @property
    def obs(self) -> np.ndarray: ...
    @property
    def action_mask(self) -> list[bool]: ...
    @property
    def value(self) -> float: ...
    @property
    def action_distribution(self) -> list[float]: ...

class ExperienceCollector:
    def __init__(self, evaluator: EvaluatorFunc, config: AlphaZeroConfig) -> None: ...
    def generate_experience(self) -> list[ExperienceItem]: ...
    @property
    def config(self) -> AlphaZeroConfig: ...

def create_obs_semantic(game_state: GameState) -> np.ndarray: ...
def get_heuristic_value(
    game_state: GameState, pos: tuple[int, int]
) -> Optional[float]: ...
def get_action_heuristic_values(
    game_state: GameState,
) -> tuple[list[Optional[float]], int]: ...
