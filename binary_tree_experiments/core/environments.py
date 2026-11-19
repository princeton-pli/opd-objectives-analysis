from __future__ import annotations

from dataclasses import dataclass
from typing import List
import numpy as np

State = int


@dataclass(frozen=True)
class NaryTreeNavigationConfig:
    depth: int
    num_children: int


def build_environment(config: NaryTreeNavigationConfig) -> NaryTreeNavigationEnvironment:
    return NaryTreeNavigationEnvironment(
        depth=config.depth,
        num_children=config.num_children,
    )


@dataclass(frozen=True)
class NaryTreeNavigationEnvironment:
    depth: int
    num_children: int

    @property
    def num_actions(self) -> int:
        return self.num_children

    @property
    def num_states(self) -> int:
        return (self.num_children ** (self.depth + 1) - 1) // (self.num_children - 1)

    @property
    def start_state(self) -> State:
        return 0

    @property
    def target_action(self) -> int:
        """Child index along the path to the target leaf (0, i.e., the leftmost child)."""
        return 0

    @property
    def goal_state(self) -> State:
        """The target leaf: the leftmost leaf, reached by taking target_action at every depth."""
        return self.first_state_at_level(self.depth)

    def step(self, state: State, action: int) -> State:
        if self.is_leaf(state):
            return state
        return self.first_child(state) + action

    def teacher_distribution_np(self, state: State, correct_prob: float) -> np.ndarray:
        """Teacher next-token distribution at a state (the teacher analyzed in the paper).

        At the root and every node in the subtree of the correct root child, the teacher
        assigns correct_prob to target_action (the child toward the target leaf) and
        spreads the remaining mass uniformly over the other children. Elsewhere, i.e.,
        if the initial token is incorrect, it is uniform over children.
        """
        uniform_probs = np.full(self.num_actions, 1.0 / self.num_actions, dtype=np.float32)
        if self.is_leaf(state) or not self.has_biased_teacher(state):
            return uniform_probs

        probs = np.full(self.num_actions, (1.0 - correct_prob) / (self.num_actions - 1), dtype=np.float32)
        probs[self.target_action] = correct_prob
        return probs

    def has_biased_teacher(self, state: State) -> bool:
        if state == self.start_state:
            return True
        return self.root_child_ancestor(state) == self.correct_root_child()

    def reference_path(self) -> List[State]:
        """States where actions are taken along the path to the goal (all but the goal)."""
        return self.path_to_goal()[:-1]

    def path_to_goal(self) -> List[State]:
        state = self.goal_state
        path = [state]
        while state != self.start_state:
            state = self.parent(state)
            path.append(state)
        return list(reversed(path))

    def all_states(self) -> List[State]:
        return list(range(self.num_states))

    def first_child(self, state: State) -> State:
        return self.num_children * state + 1

    def correct_root_child(self) -> State:
        return self.step(self.start_state, self.target_action)

    def root_child_ancestor(self, state: State) -> State:
        while self.parent(state) != self.start_state:
            state = self.parent(state)
        return state

    def parent(self, state: State) -> State:
        return (state - 1) // self.num_children

    def first_state_at_level(self, level: int) -> State:
        return (self.num_children ** level - 1) // (self.num_children - 1)

    def is_leaf(self, state: State) -> bool:
        return self.level(state) == self.depth

    def level(self, state: State) -> int:
        level = 0
        next_level_start = 1
        while level < self.depth and state >= next_level_start:
            level += 1
            next_level_start = self.first_state_at_level(level + 1)
        return level
