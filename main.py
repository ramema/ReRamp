from __future__ import annotations

import argparse
import bisect
import os
import sys
import tracemalloc
from collections import deque
from copy import deepcopy
from enum import Enum
from fractions import Fraction
from itertools import combinations, chain
from types import FrameType
from typing import TypeAlias, Any, Optional, Literal, Iterable, Generator, Callable, TypeVar

import colorama
import numpy as np
from tqdm import tqdm
from typing_extensions import Unpack
from yaml import load

import pickle

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

sys.setrecursionlimit(int(1e6))

tracemalloc.start()

VISUALISE = False
LOG = False

BuildingArea: TypeAlias = np.ndarray[tuple[Any, Any], np.dtype[np.int_]]
Position: TypeAlias = tuple[int, int]
APPROXIMATION = Fraction(1, 12)

ActionType: TypeAlias = Literal["M"] | Literal["P"] | Literal["D"]
Direction: TypeAlias = Literal["N"] | Literal["E"] | Literal["W"] | Literal["S"]

T = TypeVar('T')


def powerset(s: Iterable[T]) -> chain[tuple[Unpack[T]]]:
    item_list = list(s)
    return chain.from_iterable(combinations(item_list, r) for r in range(len(item_list) + 1))


def clear():
    os.system('cls' if os.name == 'nt' else 'clear')


class RampState(Enum):
    FORWARD = "forward"
    """ Ramp facing forward, can be reversed if there is enough space """
    BACKWARD = "backward"
    """ Ramp facing backward, can be reversed if there is enough space """
    FLOOR_MOVING = "high"
    """ Ramp cannot be reversed, floor too high on some parts """


class ActionEncodings(Enum):
    ENTER = "e"
    LEAVE = "r"
    WAIT = "q"
    MOVE_NORTH = "w"
    MOVE_WEST = "a"
    MOVE_SOUTH = "s"
    MOVE_EAST = "d"
    JUMP_MOVE_NORTH = "W"
    JUMP_MOVE_WEST = "A"
    JUMP_MOVE_SOUTH = "S"
    JUMP_MOVE_EAST = "D"
    DOWN_MOVE_NORTH = "T"
    DOWN_MOVE_WEST = "F"
    DOWN_MOVE_SOUTH = "G"
    DOWN_MOVE_EAST = "H"
    DELIVER_BLOCK_NORTH = "t"
    DELIVER_BLOCK_WEST = "f"
    DELIVER_BLOCK_SOUTH = "g"
    DELIVER_BLOCK_EAST = "h"
    PICK_UP_BLOCK_NORTH = "i"
    PICK_UP_BLOCK_WEST = "j"
    PICK_UP_BLOCK_SOUTH = "k"
    PICK_UP_BLOCK_EAST = "l"


PlanActionEncoding: TypeAlias = (ActionEncodings | tuple[ActionEncodings, bool, int, int]
                                 | tuple[ActionEncodings.WAIT, int])
""" Short representation

action ENTER and LEAVE encoded as tuple (ActionEncodings.ENTER|ActionEncodings.LEAVE, c, x, y)
action WAIT encoded as tuple (ActionEncodings.WAIT, duration)

for the rest uses ActionEncodings
"""


class PlanAction:
    __slots__ = ["t_s", "t_e", "x", "y", "z", "c", "k", "x2", "y2", "z2"]

    def __init__(self, t_s: int, t_e: int, x: str, y: str, z: str, c: bool, k: ActionType, x2: str, y2: str, z2: str):
        self.t_s = t_s
        self.t_e = t_e
        self.x = x
        self.y = y
        self.z = z
        self.c = c
        self.k = k
        self.x2 = x2
        self.y2 = y2
        self.z2 = z2

    def __str__(self):
        s = self
        return f"({s.t_s}, {s.t_e}, {s.x}, {s.y}, {s.z}, {int(s.c)}, {s.k}, {s.x2}, {s.y2}, {s.z2})"

    @staticmethod
    def enter(t_s: int, t_e: int, c: bool, xy2: Position, z2: int):
        return PlanAction(t_s, t_e, 'S', 'S', 'S', c, 'M', str(xy2[0]), str(xy2[1]), str(z2))

    @staticmethod
    def leave(t_s: int, t_e: int, xy: Position, z: int, c: bool):
        return PlanAction(t_s, t_e, str(xy[0]), str(xy[1]), str(z), c, 'M', 'E', 'E', 'E')

    @staticmethod
    def move(t_s: int, t_e: int, xy: Position, z: int, c: bool, xy2: Position, z2: int):
        return PlanAction(t_s, t_e, str(xy[0]), str(xy[1]), str(z), c, 'M', str(xy2[0]), str(xy2[1]), str(z2))

    @staticmethod
    def move_block(t_s: int, t_e: int, xy: Position, z: int, xy2: Position, z2: int):
        return PlanAction.move(t_s, t_e, xy, z, True, xy2, z2)

    @staticmethod
    def move_empty(t_s: int, t_e: int, xy: Position, z: int, xy2: Position, z2: int):
        return PlanAction.move(t_s, t_e, xy, z, False, xy2, z2)

    @staticmethod
    def pick_up(t_s: int, t_e: int, xy: Position, z: int, xy2: Position, z2: int):
        return PlanAction(t_s, t_e, str(xy[0]), str(xy[1]), str(z), False, 'P', str(xy2[0]), str(xy2[1]), str(z2))

    @staticmethod
    def deliver(t_s: int, t_e: int, xy: Position, z: int, xy2: Position, z2: int):
        return PlanAction(t_s, t_e, str(xy[0]), str(xy[1]), str(z), True, 'D', str(xy2[0]), str(xy2[1]), str(z2))

    @staticmethod
    def wait(t_s: int, t_e: int, xy: Position, z: int, c: bool):
        return PlanAction(t_s, t_e, str(xy[0]), str(xy[1]), str(z), c, 'M', str(xy[0]), str(xy[1]), str(z))

    @staticmethod
    def get_direction(x: str, y: str, x2: str, y2: str) -> Direction:
        dx = int(x2) - int(x)
        dy = int(y2) - int(y)
        match (dx, dy):
            case (0, -1):
                return 'N'
            case (-1, 0):
                return 'W'
            case (0, 1):
                return 'S'
            case (1, 0):
                return 'E'
            case _:
                raise Exception("Invalid move")

    @staticmethod
    def jump_move_encoded(xy: Position, xy2: Position) -> ActionEncodings:
        dx = xy2[0] - xy[0]
        dy = xy2[1] - xy[1]
        match (dx, dy):
            case (0, -1):
                return ActionEncodings.JUMP_MOVE_NORTH
            case (-1, 0):
                return ActionEncodings.JUMP_MOVE_WEST
            case (0, 1):
                return ActionEncodings.JUMP_MOVE_SOUTH
            case (1, 0):
                return ActionEncodings.JUMP_MOVE_EAST
            case _:
                raise Exception("Invalid move")

    @staticmethod
    def down_move_encoded(xy: Position, xy2: Position) -> ActionEncodings:
        dx = xy2[0] - xy[0]
        dy = xy2[1] - xy[1]
        match (dx, dy):
            case (0, -1):
                return ActionEncodings.DOWN_MOVE_NORTH
            case (-1, 0):
                return ActionEncodings.DOWN_MOVE_WEST
            case (0, 1):
                return ActionEncodings.DOWN_MOVE_SOUTH
            case (1, 0):
                return ActionEncodings.DOWN_MOVE_EAST
            case _:
                raise Exception("Invalid move")

    @staticmethod
    def move_encoded(xy: Position, xy2: Position) -> ActionEncodings:
        dx = xy2[0] - xy[0]
        dy = xy2[1] - xy[1]
        match (dx, dy):
            case (0, -1):
                return ActionEncodings.MOVE_NORTH
            case (-1, 0):
                return ActionEncodings.MOVE_WEST
            case (0, 1):
                return ActionEncodings.MOVE_SOUTH
            case (1, 0):
                return ActionEncodings.MOVE_EAST
            case _:
                raise Exception("Invalid move")

    @staticmethod
    def deliver_encoded(xy: Position, xy2: Position) -> ActionEncodings:
        dx = xy2[0] - xy[0]
        dy = xy2[1] - xy[1]
        match (dx, dy):
            case (0, -1):
                return ActionEncodings.DELIVER_BLOCK_NORTH
            case (-1, 0):
                return ActionEncodings.DELIVER_BLOCK_WEST
            case (0, 1):
                return ActionEncodings.DELIVER_BLOCK_SOUTH
            case (1, 0):
                return ActionEncodings.DELIVER_BLOCK_EAST
            case _:
                raise Exception("Invalid move")

    @staticmethod
    def pick_up_encoded(xy: Position, xy2: Position) -> ActionEncodings:
        dx = xy2[0] - xy[0]
        dy = xy2[1] - xy[1]
        match (dx, dy):
            case (0, -1):
                return ActionEncodings.PICK_UP_BLOCK_NORTH
            case (-1, 0):
                return ActionEncodings.PICK_UP_BLOCK_WEST
            case (0, 1):
                return ActionEncodings.PICK_UP_BLOCK_SOUTH
            case (1, 0):
                return ActionEncodings.PICK_UP_BLOCK_EAST
            case _:
                raise Exception("Invalid move")

    def export(self, building_material="iron_block") -> str:
        building_material = f" {building_material}"
        match self:
            case PlanAction(x=x, y=y, k="D", x2=x2, y2=y2):
                return f"place_block {PlanAction.get_direction(x, y, x2, y2)}"
            case PlanAction(x=x, y=y, k="P", x2=x2, y2=y2):
                return f"break_block {PlanAction.get_direction(x, y, x2, y2)}"
            case PlanAction(t_s=t_s, t_e=t_e, x=x, y=y, z=z, k='M', x2=x2, y2=y2, z2=z2) \
                if x == x2 and y == y2 and z == z2:
                return f"wait {t_e - t_s}"
            case PlanAction(k='M', x2='E', y2='E', z2='E'):
                return "leave"
            case PlanAction(x='S', y='S', z='S', c=c, k='M', x2=x2, y2=y2):
                return f"enter {x2} {y2}{building_material if c else ''}"
            case PlanAction(x=x, y=y, z=z, k='M', x2=x2, y2=y2, z2=z2):
                direction = PlanAction.get_direction(x, y, x2, y2)
                if int(z) < int(z2):
                    return f"jump_move {direction}"
                else:
                    return f"move {direction}"
            case _:
                raise Exception("Unknown action")

    def reverse(self, t_s: int, action_durations: ActionDurations) -> PlanAction:
        match self:
            case PlanAction(x=x, y=y, z=z, c=True, k="D", x2=x2, y2=y2, z2=z2) if z == z2:
                return PlanAction(t_s, t_s + action_durations.pick_up, x, y, z, False, "P", x2, y2, z2)
            case PlanAction(x=x, y=y, z=z, c=False, k="P", x2=x2, y2=y2, z2=z2) if z == z2:
                return PlanAction(t_s, t_s + action_durations.deliver, x, y, z, True, "D", x2, y2, z2)
            case PlanAction(x=x, y=y, z=z, c=c, k="M", x2='E', y2='E', z2='E'):
                return PlanAction(t_s, t_s + action_durations.enter, 'S', 'S', 'S', c, "M", x, y, z)
            case PlanAction(x='S', y='S', z='S', c=c, k="M", x2=x2, y2=y2, z2=z2):
                return PlanAction(t_s, t_s + action_durations.leave, x2, y2, z2, c, "M", 'E', 'E', 'E')
            case PlanAction(x=x, y=y, z=z, c=c, k="M", x2=x2, y2=y2, z2=z2) if x == x2 and y == y2 and z == z2:
                return PlanAction(t_s, t_s + action_durations.wait, x2, y2, z2, c, "M", x, y, z)
            case PlanAction(x=x, y=y, z=z, c=c, k="M", x2=x2, y2=y2, z2=z2):
                duration = action_durations.move_block if c else action_durations.move_empty
                return PlanAction(t_s, t_s + duration, x2, y2, z2, c, "M", x, y, z)
            case _:
                raise Exception("Unknown action")

    def to_tuple(self) -> tuple[int, int, str, str, str, bool, ActionType, str, str, str]:
        return self.t_s, self.t_e, self.x, self.y, self.z, self.c, self.k, self.x2, self.y2, self.z2


class ActionDurations:
    def __init__(self, enter=1, leave=1, move_block=1, move_empty=1, pick_up=1, deliver=1):
        self.enter = enter
        self.leave = leave
        self.move_block = move_block
        self.move_empty = move_empty
        self.pick_up = pick_up
        self.deliver = deliver
        self.wait = 1

    def reverse(self) -> ActionDurations:
        return ActionDurations(self.leave, self.enter, self.move_block, self.move_empty, self.deliver, self.pick_up)

    def __eq__(self, other):
        if not isinstance(other, ActionDurations):
            return False
        return (self.enter == other.enter
                and self.leave == other.leave
                and self.move_block == other.move_block
                and self.move_empty == other.move_empty
                and self.pick_up == other.pick_up
                and self.deliver == other.deliver
                and self.wait == other.wait)


class AgentPlan:
    def __init__(self,
                 t_s: int,
                 action_durations: ActionDurations,
                 remaining_structure: BuildingArea,
                 without_instructions: bool = False):
        # assert t_s >= 0
        self.initial_structure = remaining_structure.copy()
        self.remaining_structure = remaining_structure
        self.t_s = t_s
        self.t_e = t_s
        self.sum_of_costs = 0
        self.agent_position: Optional[Position] = None
        """ None marks positions outside the grid """
        self.agent_holding_a_block: bool = False
        """ Can be changed only when entering the grid, delivering a block, picking up a block """
        self.action_durations = action_durations
        self.without_instructions = without_instructions
        self.instructions: list[PlanActionEncoding] = []

    def get_next_agent_position_in_direction(self, direction: Direction) -> Position:
        if self.agent_position is None:
            raise Exception("Agent not on grid")
        x, y = self.agent_position
        match direction:
            case 'N':
                return x, y - 1
            case 'W':
                return x - 1, y
            case 'S':
                return x, y + 1
            case 'E':
                return x + 1, y
            case _:
                raise Exception("Unknown direction")

    def reverse(self, t_s: int) -> AgentPlan:
        """
        Provides a copy of the plan in reverse (picking a block instead of delivering and so on)
        :return: a copy of the plan in reverse
        """
        # assert self.agent_position is None  # only allow when agent off grid
        plan = AgentPlan(t_s, self.action_durations.reverse(), self.remaining_structure)
        plan.remaining_structure = self.initial_structure.copy()
        plan.t_s = t_s
        plan.t_e = t_s + self.t_e - self.t_s
        plan.sum_of_costs = self.sum_of_costs
        for action in tqdm(reversed(self.instructions), total=len(self.instructions), desc="Reversing plan"):
            match action:
                case (ActionEncodings.ENTER, c, x, y):
                    plan.instructions.append((ActionEncodings.LEAVE, c, x, y))
                case (ActionEncodings.LEAVE, c, x, y):
                    plan.instructions.append((ActionEncodings.ENTER, c, x, y))
                case (ActionEncodings.WAIT, duration):
                    plan.instructions.append((ActionEncodings.WAIT, duration))
                case ActionEncodings.MOVE_NORTH:
                    plan.instructions.append(ActionEncodings.MOVE_SOUTH)
                case ActionEncodings.MOVE_WEST:
                    plan.instructions.append(ActionEncodings.MOVE_EAST)
                case ActionEncodings.MOVE_SOUTH:
                    plan.instructions.append(ActionEncodings.MOVE_NORTH)
                case ActionEncodings.MOVE_EAST:
                    plan.instructions.append(ActionEncodings.MOVE_WEST)
                case ActionEncodings.JUMP_MOVE_NORTH:
                    plan.instructions.append(ActionEncodings.DOWN_MOVE_SOUTH)
                case ActionEncodings.JUMP_MOVE_WEST:
                    plan.instructions.append(ActionEncodings.DOWN_MOVE_EAST)
                case ActionEncodings.JUMP_MOVE_SOUTH:
                    plan.instructions.append(ActionEncodings.DOWN_MOVE_NORTH)
                case ActionEncodings.JUMP_MOVE_EAST:
                    plan.instructions.append(ActionEncodings.DOWN_MOVE_WEST)
                case ActionEncodings.DOWN_MOVE_NORTH:
                    plan.instructions.append(ActionEncodings.JUMP_MOVE_SOUTH)
                case ActionEncodings.DOWN_MOVE_WEST:
                    plan.instructions.append(ActionEncodings.JUMP_MOVE_EAST)
                case ActionEncodings.DOWN_MOVE_SOUTH:
                    plan.instructions.append(ActionEncodings.JUMP_MOVE_NORTH)
                case ActionEncodings.DOWN_MOVE_EAST:
                    plan.instructions.append(ActionEncodings.JUMP_MOVE_WEST)
                case ActionEncodings.DELIVER_BLOCK_NORTH:
                    plan.instructions.append(ActionEncodings.PICK_UP_BLOCK_NORTH)
                case ActionEncodings.DELIVER_BLOCK_WEST:
                    plan.instructions.append(ActionEncodings.PICK_UP_BLOCK_WEST)
                case ActionEncodings.DELIVER_BLOCK_SOUTH:
                    plan.instructions.append(ActionEncodings.PICK_UP_BLOCK_SOUTH)
                case ActionEncodings.DELIVER_BLOCK_EAST:
                    plan.instructions.append(ActionEncodings.PICK_UP_BLOCK_EAST)
                case ActionEncodings.PICK_UP_BLOCK_NORTH:
                    plan.instructions.append(ActionEncodings.DELIVER_BLOCK_NORTH)
                case ActionEncodings.PICK_UP_BLOCK_WEST:
                    plan.instructions.append(ActionEncodings.DELIVER_BLOCK_WEST)
                case ActionEncodings.PICK_UP_BLOCK_SOUTH:
                    plan.instructions.append(ActionEncodings.DELIVER_BLOCK_SOUTH)
                case ActionEncodings.PICK_UP_BLOCK_EAST:
                    plan.instructions.append(ActionEncodings.DELIVER_BLOCK_EAST)
                case _:
                    raise Exception("Invalid action encoding")
        return plan

    def enter(self, c: bool, xy: Position) -> PlanAction:
        # assert self.agent_position is None
        self.agent_holding_a_block = c
        z = int(self.remaining_structure[xy])
        # assert z == 0
        if not self.without_instructions:
            self.instructions.append((ActionEncodings.ENTER, c, xy[0], xy[1]))
        self.agent_position = xy
        t_s = self.t_e
        self.t_e += self.action_durations.enter
        self.sum_of_costs += self.t_e - t_s
        return PlanAction.enter(t_s, self.t_e, c, xy, z)

    def leave(self) -> PlanAction:
        # assert self.agent_position is not None
        xy = self.agent_position
        c = self.agent_holding_a_block
        z = int(self.remaining_structure[xy])
        # assert z == 0
        if not self.without_instructions:
            self.instructions.append((ActionEncodings.LEAVE, c, xy[0], xy[1]))
        self.agent_position = None
        t_s = self.t_e
        self.t_e += self.action_durations.leave
        self.sum_of_costs += self.t_e - t_s
        return PlanAction.leave(t_s, self.t_e, xy, z, c)

    def move(self, xy2: Position) -> PlanAction:
        # assert self.agent_position is not None
        xy1 = self.agent_position
        c = self.agent_holding_a_block
        z1 = int(self.remaining_structure[xy1])
        z2 = int(self.remaining_structure[xy2])
        # assert -1 <= z2 - z1 <= 1
        duration = self.action_durations.move_block if c else self.action_durations.move_empty
        if not self.without_instructions:
            if z1 < z2:
                self.instructions.append(PlanAction.jump_move_encoded(xy1, xy2))
            elif z1 > z2:
                self.instructions.append(PlanAction.down_move_encoded(xy1, xy2))
            else:
                self.instructions.append(PlanAction.move_encoded(xy1, xy2))
        self.agent_position = xy2
        t_s = self.t_e
        self.t_e += duration
        self.sum_of_costs += self.t_e - t_s
        return PlanAction.move(t_s, self.t_e, xy1, z1, c, xy2, z2)

    def move_block(self, xy2: Position) -> PlanAction:
        # assert self.agent_holding_a_block
        return self.move(xy2)

    def move_empty(self, xy2: Position) -> PlanAction:
        # assert not self.agent_holding_a_block
        return self.move(xy2)

    def pick_up(self, xy2: Position) -> PlanAction:
        # assert self.agent_position is not None
        # assert not self.agent_holding_a_block
        xy1 = self.agent_position
        z1 = int(self.remaining_structure[xy1])
        z2 = int(self.remaining_structure[xy2]) - 1
        # assert z1 == z2
        if not self.without_instructions:
            self.instructions.append(PlanAction.pick_up_encoded(xy1, xy2))
        self.remaining_structure[xy2] -= 1
        self.agent_holding_a_block = True
        t_s = self.t_e
        self.t_e += self.action_durations.pick_up
        self.sum_of_costs += self.t_e - t_s
        return PlanAction.pick_up(t_s, self.t_e, xy1, z1, xy2, z2)

    def deliver(self, xy2: Position) -> PlanAction:
        # assert self.agent_position is not None
        # assert self.agent_holding_a_block
        xy1 = self.agent_position
        z1 = int(self.remaining_structure[xy1])
        z2 = int(self.remaining_structure[xy2])
        # assert z1 == z2
        if not self.without_instructions:
            self.instructions.append(PlanAction.deliver_encoded(xy1, xy2))
        self.remaining_structure[xy2] += 1
        self.agent_holding_a_block = False
        t_s = self.t_e
        self.t_e += self.action_durations.deliver
        self.sum_of_costs += self.t_e - t_s
        return PlanAction.deliver(t_s, self.t_e, xy1, z1, xy2, z2)

    def wait(self, duration: int) -> PlanAction:
        # assert self.agent_position is not None
        # assert duration > 0
        xy = self.agent_position
        c = self.agent_holding_a_block
        z = int(self.remaining_structure[xy])
        if not self.without_instructions:
            self.instructions.append((ActionEncodings.WAIT, duration))
        t_s = self.t_e
        self.t_e += duration
        if self.agent_position is not None:  # if agent on the grid
            self.sum_of_costs += self.t_e - t_s
        return PlanAction.wait(t_s, self.t_e, xy, z, c)

    @staticmethod
    def export_encoded_instruction(encoded_instruction: PlanActionEncoding, building_material="iron_block") -> str:
        building_material = f" {building_material}"
        match encoded_instruction:
            case (ActionEncodings.ENTER, c, x, y):
                return f"enter {x} {y}{building_material if c else ''}"
            case (ActionEncodings.LEAVE, _, _, _):
                return "leave"
            case (ActionEncodings.WAIT, duration):
                return f"wait {duration}"
            case ActionEncodings.MOVE_NORTH | ActionEncodings.DOWN_MOVE_NORTH:
                return f"move N"
            case ActionEncodings.MOVE_WEST | ActionEncodings.DOWN_MOVE_WEST:
                return f"move W"
            case ActionEncodings.MOVE_SOUTH | ActionEncodings.DOWN_MOVE_SOUTH:
                return f"move S"
            case ActionEncodings.MOVE_EAST | ActionEncodings.DOWN_MOVE_EAST:
                return f"move E"
            case ActionEncodings.JUMP_MOVE_NORTH:
                return f"jump_move N"
            case ActionEncodings.JUMP_MOVE_WEST:
                return f"jump_move W"
            case ActionEncodings.JUMP_MOVE_SOUTH:
                return f"jump_move S"
            case ActionEncodings.JUMP_MOVE_EAST:
                return f"jump_move E"
            case ActionEncodings.DELIVER_BLOCK_NORTH:
                return f"place_block N"
            case ActionEncodings.DELIVER_BLOCK_WEST:
                return f"place_block W"
            case ActionEncodings.DELIVER_BLOCK_SOUTH:
                return f"place_block S"
            case ActionEncodings.DELIVER_BLOCK_EAST:
                return f"place_block E"
            case ActionEncodings.PICK_UP_BLOCK_NORTH:
                return f"break_block N"
            case ActionEncodings.PICK_UP_BLOCK_WEST:
                return f"break_block W"
            case ActionEncodings.PICK_UP_BLOCK_SOUTH:
                return f"break_block S"
            case ActionEncodings.PICK_UP_BLOCK_EAST:
                return f"break_block E"
            case _:
                raise Exception("Invalid action encoding")

    def export_instructions(self, building_material: str = "iron_block") -> Iterable[str]:
        wait_duration = self.t_s
        for instruction in self.instructions:
            exported_instruction = AgentPlan.export_encoded_instruction(instruction, building_material)
            if exported_instruction.startswith("wait "):
                wait_duration += int(exported_instruction[5:])
            elif wait_duration > 0:
                yield f"wait {wait_duration}"
                wait_duration = 0
                yield exported_instruction
            else:
                yield exported_instruction

        if wait_duration > 0:
            yield f"wait {wait_duration}"


class MultiAgentPlan:
    def __init__(self, target_structure: BuildingArea, action_durations: ActionDurations):
        self.target_structure = target_structure
        self.action_durations = action_durations
        self.agents: list[AgentPlan] = []

    def add_agents(self, *agents: AgentPlan):
        self.agents.extend(agents)
        return self

    @property
    def mission_ticks(self) -> int:
        return max([agent.t_e for agent in self.agents])

    @property
    def sum_of_costs(self) -> int:
        sum_of_costs = sum([agent.sum_of_costs for agent in self.agents])
        # agents work their entire duration on the grid, there are no wait actions outside the grid
        assert sum_of_costs == sum([agent.t_e - agent.t_s for agent in self.agents])
        return sum_of_costs

    def write_to_file(self, file_path: str, show_held_blocks: bool = False):
        with open(file_path, mode='w') as f:
            print("agents:", file=f)
            for agent_index, agent in enumerate(self.agents):
                agent_name = f"Agent{str(agent_index).rjust(len(str(len(self.agents) - 1)), '0')}"
                print(f"- agent_name: {agent_name}", file=f)
                print("  instructions:", file=f)
                for instruction in tqdm(agent.export_instructions(), total=len(agent.instructions),
                                        desc=f"Writing results to: {file_path}"):
                    print(f"  - {instruction}", file=f)
            print("building_area:", file=f)
            print(f"  size_x: {self.target_structure.shape[0] + 2}", file=f)
            print(f"  size_z: {self.target_structure.shape[1] + 2}", file=f)
            print(f"deliver_time: '{str(self.action_durations.deliver)}'", file=f)
            print(f"entry_time: '{str(self.action_durations.enter)}'", file=f)
            print(f"leave_time: '{str(self.action_durations.leave)}'", file=f)
            print(f"move_time_with_block: '{str(self.action_durations.move_block)}'", file=f)
            print(f"move_time_without_block: '{str(self.action_durations.move_empty)}'", file=f)
            print(f"pick_up_time: '{str(self.action_durations.pick_up)}'", file=f)
            print(f"mission_ticks: {self.mission_ticks}", file=f)
            print(f"show_held_blocks: {str(show_held_blocks).lower()}", file=f)
            print("building_heights:", file=f)
            for row in self.target_structure.tolist():
                print(f"- {row}", file=f)

    def __eq__(self, other: Any) -> bool:
        """
        Determines, if the two items are equal
        :param other: other item
        :return:  if the two items are equal
        """
        if not isinstance(other, MultiAgentPlan):
            return False
        if self.mission_ticks != other.mission_ticks:
            return False
        if self.sum_of_costs != other.sum_of_costs:
            return False
        if len(self.agents) != len(other.agents):
            return False
        for agent1, agent2 in zip(self.agents, other.agents):
            if agent1.instructions != agent2.instructions:
                return False
        return True


class TreeNode:
    def __init__(self, xy: Position, parent: Optional[TreeNode]):
        self.position = xy
        self.distance = 0
        if parent is not None:
            self.distance = parent.distance + 1
        self.parent = parent
        self.__children: list[TreeNode] = []
        self.deepest_node = (self, self.distance)

    @property
    def children(self) -> list[TreeNode]:
        return self.__children

    def add_child(self, child: TreeNode):
        self.children.append(child)
        node = self
        while node is not None and node.deepest_node[1] < child.deepest_node[1]:
            node.deepest_node = child.deepest_node
            node = node.parent

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        parent = "" if self.parent is None else str(self.parent.position)
        children = ', '.join(map(lambda child: str(child.position), self.__children))
        return f"[{parent}] -> {str(self.position)} -> [{children}]"


class Ramp:
    def __init__(self, plan: AgentPlan, central_path: list[Position], side_ramps: dict[Position, list[ReversibleRamp]],
                 floor_height: int = 0, number_of_blocks: Optional[int] = 0):
        self._plan = plan
        self._central_path = central_path
        self._side_ramps = side_ramps
        self._state = RampState.FORWARD
        self._floor_height = floor_height

        self._number_of_blocks = number_of_blocks if number_of_blocks is not None else self.compute_number_of_blocks()

        self.__next_block_modification_position_index = None

        self._capacity_and_maximum_height: Optional[tuple[int, int]] = None

        self._required_block_placements: deque[int] = deque()
        """
        Queue of node indices, where blocks need to be placed, before the ramp can return to its FORWARD state

        The indices belong to the central path; the blocks need to be first placed into the side ramps at the node index
        """
        self._required_block_removals: deque[int] = deque()
        """
        Queue of node indices, where blocks need to be removed, before the ramp can return to its FORWARD state

        The indices belong to the central path; the blocks need to be removed from the side ramps at the node index,
        afterwards
        """
        # assert all([self._floor_height <= self._plan.remaining_structure[xy] for xy in self.get_all_positions()])
        # assert self._number_of_blocks == self.compute_number_of_blocks()

    def compute_number_of_blocks(self):
        block_counts = {xy: self._plan.remaining_structure[xy] - self._floor_height for xy in self.get_all_positions()}
        return sum(block_counts.values())

    @classmethod
    def get_central_path_and_side_ramps(cls,
                                        start: Optional[TreeNode],
                                        end: TreeNode,
                                        plan: AgentPlan,
                                        sub_side_ramp_count: int) -> tuple[
        list[Position], dict[Position, list[ReversibleRamp]]
    ]:
        node = end
        path_child = None
        central_path: deque[Position] = deque()
        side_ramps: dict[Position, list[ReversibleRamp]] = dict()
        while node is not start and node is not None:
            central_path.appendleft(node.position)
            for child in node.children:
                if child is not path_child and sub_side_ramp_count > 0:
                    ramp = ReversibleRamp.from_tree(node, child.deepest_node[0], plan, sub_side_ramp_count - 1)
                    if ramp.reversible_capacity > 0:  # do not include side ramps that cannot be reversed
                        if node.position not in side_ramps:
                            side_ramps[node.position] = []
                        side_ramps[node.position].append(ramp)
            path_child = node
            node = node.parent
        if start is not None:
            central_path.appendleft(start.position)

        return list(central_path), side_ramps

    @classmethod
    def from_tree(cls, start: Optional[TreeNode], end: TreeNode, plan: AgentPlan, sub_side_ramp_count: int) -> Ramp:
        central_path, side_ramps = Ramp.get_central_path_and_side_ramps(start, end, plan, sub_side_ramp_count)

        return Ramp(plan, central_path, side_ramps, 0, 0)

    def ramp_absolute_height(self, xy: Position) -> int:
        return int(self._plan.remaining_structure[xy])

    def ramp_relative_height(self, xy: Position) -> int:
        # assert all([side_ramp.state == RampState.FORWARD for side_ramp in self._side_ramps.get(xy, [])])
        return self.ramp_absolute_height(xy) - self.floor_height

    @property
    def connection_point(self) -> Position:
        return self._central_path[0]

    @property
    def end_point(self) -> Position:
        return self._central_path[-1]

    @property
    def side_ramp_areas(self) -> Generator[int, None, None]:
        for ramp_list in self._side_ramps.values():
            for ramp in ramp_list:
                yield ramp.area

    @property
    def area(self) -> int:
        return len(self._central_path) - 1 + sum(self.side_ramp_areas)

    def _compute_capacity_and_maximum_height(self) -> tuple[int, int]:
        column_height = -1  # first node of central path is at height 0
        capacity = 0
        for node_index, node in enumerate(self._central_path):
            column_height += 1
            capacity += column_height
            for ramp in self.get_side_ramps(node_index):
                ramp_floor_height = column_height
                column_height += ramp.maximum_reversible_height
                capacity += ramp.reversible_capacity
                capacity += ramp_floor_height * ramp.area

        return capacity, column_height

    @property
    def maximum_height(self) -> int:
        if self._capacity_and_maximum_height is None:
            self._capacity_and_maximum_height = self._compute_capacity_and_maximum_height()
        return self._capacity_and_maximum_height[1]

    @property
    def capacity(self) -> int:
        if self._capacity_and_maximum_height is None:
            self._capacity_and_maximum_height = self._compute_capacity_and_maximum_height()
        return self._capacity_and_maximum_height[0]

    @property
    def floor_height(self) -> int:
        return self._floor_height

    @property
    def path_with_next(self) -> Generator[tuple[Position, Optional[Position]], None, None]:
        """ Ramp path with the next node (if any) """
        for node_index in range(len(self._central_path) - 1):
            yield self._central_path[node_index], self._central_path[node_index + 1]
        if len(self._central_path) >= 1:
            yield self._central_path[-1], None

    def get_side_ramps(self, node_index: int) -> list[ReversibleRamp]:
        node = self._central_path[node_index]
        return self._side_ramps.get(node, [])

    def ramp_relative_height_with_side_ramp_reaches(self, height: int, node_index: int) -> bool:
        # assert self._state in [RampState.FORWARD, RampState.FLOOR_MOVING]
        node = self._central_path[node_index]
        node_height = self.ramp_relative_height(node)
        side_ramps = self.get_side_ramps(node_index)
        for side_ramp in reversed(side_ramps):
            if side_ramp.state == RampState.FLOOR_MOVING:
                return False
            # assert side_ramp.state == RampState.FORWARD
            # assert side_ramp.can_reverse()
            if side_ramp.is_reaching_height(side_ramp.maximum_reversible_height):
                node_height += side_ramp.maximum_reversible_height
            elif node_height >= height:
                return True
            else:
                return side_ramp.is_reaching_height(height - node_height)

        return node_height >= height

    def _get_floor_raise_block_addition_list(self) -> Iterable[int]:
        block_placements: deque[int] = deque()

        skip_node = False
        for node_index, (node, next_node) in enumerate(self.path_with_next):
            if node_index == 0:
                continue  # skip the connection point
            if skip_node:
                skip_node = False
                continue
            _node_index = node_index
            next_node_index = node_index + 1
            if self.ramp_relative_height(node) == 0:  # if on floor
                if next_node is None or self.ramp_relative_height(next_node) == 0:  # if not at the start of the ramp
                    block_placements.appendleft(_node_index)
                elif self.ramp_relative_height_with_side_ramp_reaches(
                        self.ramp_relative_height(next_node), node_index
                ):  # if the first ramp column is at the same height as side ramp end
                    block_placements.appendleft(_node_index)
                    block_placements.appendleft(next_node_index)
                    skip_node = True
                else:
                    block_placements.appendleft(_node_index)
            elif next_node is None or not self.ramp_relative_height_with_side_ramp_reaches(
                    self.ramp_relative_height(next_node), node_index
            ):  # if there is no next node, or it is not at the same elevation as the end of the ramp
                block_placements.append(_node_index)
            else:  # expects the longest flat area on ramp to be two blocks long
                block_placements.append(next_node_index)
                block_placements.append(_node_index)
                skip_node = True

        return block_placements

    def move_floor_up(self):
        """
        Moves floor upwards (computes block placement positions, places first block)

        Assumes the robot stands at the connection point at the start
        Leaves the robot in the same state at the connection point at the end
        """
        # assert self._state == RampState.FORWARD
        # assert not self._required_block_placements

        self._required_block_placements.extend(self._get_floor_raise_block_addition_list())

        # assert self._required_block_placements
        self._state = RampState.FLOOR_MOVING
        self.add_block()

    def move_floor_down(self):
        # assert self._state == RampState.FORWARD
        # assert not self._required_block_removals
        # assert self._floor_height > 0

        self._required_block_removals.extendleft(self._get_floor_raise_block_addition_list())

        if self._required_block_removals:
            self._state = RampState.FLOOR_MOVING

        self._floor_height -= 1

    @staticmethod
    def _climb_side_ramps(side_ramps: list[ReversibleRamp], up: bool):
        if up:
            for side_ramp in side_ramps:
                side_ramp.pass_up()
        else:
            for side_ramp in reversed(side_ramps):
                side_ramp.pass_down()

    def _climb_side_ramps_at(self, node_index: int, up: bool):
        """
        Climbs all side ramps connected to node index

        :param node_index: central path node index

        Note: assumes the robot is standing at node index, all side ramps have the same state (forward or backward)
        """
        Ramp._climb_side_ramps(self.get_side_ramps(node_index), up)

    def _move_on_floor(self, from_index: int, to_index: int):
        delta = 1 if from_index <= to_index else -1
        for index in range(from_index + delta, to_index + delta, delta):
            # assert self.ramp_absolute_height(self._central_path[index]) == self.floor_height
            self._plan.move(self._central_path[index])

    def _move(self, from_index: int, to_index: int, climb_destination_ramps: bool, climb_origin_ramps: bool, up: bool):
        """
        Moves from one central path node (given by index) to the second (also given by index)
        :param from_index: first central path node
        :param to_index: second central path node

        Note: does not climb the side ramps at the destination
        """
        if from_index == to_index:
            if climb_destination_ramps and climb_origin_ramps:
                self._climb_side_ramps_at(from_index, up)
            return

        delta = 1 if from_index <= to_index else -1
        for index in range(from_index + delta, to_index + delta, delta):
            if from_index != index - delta or climb_origin_ramps:
                self._climb_side_ramps_at(index - delta, up)
            self._plan.move(self._central_path[index])

        if climb_destination_ramps:
            self._climb_side_ramps_at(to_index, up)

    def _deliver_at(self, target_index: int):
        """
        Starts with the robot at the connection point, goes over the ramp path until the robot reaches field next to
        the target position, delivers the block at the target position and returns back to the connection point
        :param target_index: where to place the block
        """
        self._perform_action_at(lambda xy: self._plan.deliver(xy), target_index)

    def _pick_up_at(self, target_index: int):
        """
        Starts with the robot at the connection point, goes over the ramp path until the robot reaches field next to
        the target position, picks up the block at the target position and returns back to the connection point
        :param target_index: position of the block to be picked up
        """
        self._perform_action_at(lambda xy: self._plan.pick_up(xy), target_index)

    def _perform_action_at(self, action: Callable[[Position], Any], target_index: int):
        """
        Starts with the robot at the connection point, goes over the ramp path until the robot reaches field next to
        the target position, performs the action at the target position and returns back to the connection point
        :param action: action to perform
        :param target_index: index of central path position, where to perform the action

        Note: assumes the target index is > 0
        """
        # assert target_index > 0
        self._move(0, target_index - 1, True, True, True)
        action(self._central_path[target_index])
        self._move(target_index - 1, 0, True, True, False)

    @staticmethod
    def _perform_action_at_highest_side_ramp(side_ramps: list[ReversibleRamp], action: Callable[[ReversibleRamp], Any]):
        # assert len(side_ramps) > 0
        Ramp._climb_side_ramps(side_ramps[:-1], True)
        action(side_ramps[-1])
        Ramp._climb_side_ramps(side_ramps[:-1], False)

    @staticmethod
    def _add_block_to_highest_side_ramp(side_ramps: list[ReversibleRamp], move_floor_up: bool = False):
        if move_floor_up:
            Ramp._perform_action_at_highest_side_ramp(side_ramps, lambda side_ramp: side_ramp.move_floor_up())
        else:
            Ramp._perform_action_at_highest_side_ramp(side_ramps, lambda side_ramp: side_ramp.add_block())

    @staticmethod
    def _remove_block_from_highest_side_ramp(side_ramps: list[ReversibleRamp]):
        Ramp._perform_action_at_highest_side_ramp(side_ramps, lambda side_ramp: side_ramp.remove_block())

    def _add_block_to_required_placement(self):
        """
        Adds a block to a place specified by self._required_block_placements
        (first its connected ramps from top to bottom, then the position at the central ramp)

        Assumes the robot stands with a block at the connection point at the start of the action.
        Leaves the robot without its block at the connection point at the end of the action.
        """
        node_index = self._required_block_placements[0]
        if not self._required_block_removals or node_index != self._required_block_removals[0]:
            self._required_block_removals.appendleft(node_index)

        if self.add_block_at(node_index, True):
            self._required_block_placements.popleft()

    def ramps_connect(self, first_ramp: Optional[ReversibleRamp], second_ramp: ReversibleRamp) -> bool:
        """
        Determines, if the first ramp, when reversed, connects to the second ramp
        (i.e. if the first ramp is_reaching_height of the floor difference between the ramps)

        If the first ramp is None, determines if the second ramp has floor height equal to the height
        of the central path (when all side ramps are facing FORWARD)
        :param first_ramp:
        :param second_ramp:
        :return: if the first ramp, when reversed, connects to the second ramp
        """
        node = second_ramp.connection_point
        if first_ramp is None:
            return second_ramp.floor_height == self.ramp_absolute_height(node)

        return first_ramp.is_reaching_height(second_ramp.floor_height - first_ramp.floor_height)

    def ramps_at_connect(self, node_index: int, second_ramp_index: int):
        # assert 0 <= node_index < len(self._central_path)
        side_ramps = self.get_side_ramps(node_index)
        # assert 0 <= second_ramp_index < len(side_ramps)

        first_ramp: Optional[ReversibleRamp] = side_ramps[second_ramp_index - 1] if second_ramp_index > 0 else None
        second_ramp: ReversibleRamp = side_ramps[second_ramp_index]

        return self.ramps_connect(first_ramp, second_ramp)

    def add_block_at(self, node_index: int, just_floor: bool = False) -> bool:
        """

        :param node_index:
        :param just_floor: if blocks should be added just to raise the floor
        :return: if the addition is the last one required to bring the position at node index one block up
        """
        # assert 0 <= node_index < len(self._central_path)
        side_ramps = self.get_side_ramps(node_index)

        ramp_for_addition_index = None
        move_floor_up = False
        # find the highest unfinished side ramp (or current node, counting as side_ramp -1)
        # if all floors of ramps after it are raised, add block at the ramp and finish with True
        # else add block to the lowest ramp above it without raised floor
        for reverse_index, side_ramp in enumerate(reversed(side_ramps)):
            side_ramp_index = len(side_ramps) - reverse_index - 1
            if side_ramp.state == RampState.FLOOR_MOVING:
                ramp_for_addition_index = side_ramp_index
                break
            if not self.ramps_at_connect(node_index, side_ramp_index):
                ramp_for_addition_index = side_ramp_index - 1 if side_ramp_index > 0 else None
                # ramp may not connect for two reasons:
                # 1. the ramp at ramp_for_addition_index is growing in height
                # 2. the ramp at ramp_for_addition_index is elevating its floor
                # we distinguish these two by using the just_floor argument
                # i.e. we grow the ramp at ramp_for_addition_index if and only if
                #                                                   it is not finished growing and just_floor is false
                move_floor_up = True
                grow_ramp = (not just_floor
                             and ramp_for_addition_index is not None
                             and not side_ramps[ramp_for_addition_index].at_reversible_capacity())
                if (grow_ramp or ramp_for_addition_index is None
                        or side_ramps[ramp_for_addition_index].state == RampState.FLOOR_MOVING):
                    move_floor_up = False  # do not initiate move_floor_up when growing the ramp, or after it was used
                break
            if not just_floor and not side_ramp.at_reversible_capacity():
                # assert side_ramp.floor_height == self.floor_height  # the only ramps not at capacity are at the floor
                next_side_ramps = side_ramps[side_ramp_index + 1:]
                if not next_side_ramps:
                    ramp_for_addition_index = side_ramp_index
                else:
                    ramp_for_addition_index = len(side_ramps) - 1
                    move_floor_up = True
                break

        if ramp_for_addition_index is None:
            if not side_ramps or not self.ramps_connect(None, side_ramps[0]):
                # assert 0 < node_index
                self._deliver_at(node_index)
                return True
            else:
                ramp_for_addition_index = len(side_ramps) - 1
                move_floor_up = True

        ramp_for_addition = side_ramps[ramp_for_addition_index]

        self._move(0, node_index, False, True, True)

        ramp_for_addition_added_to_floor = ramp_for_addition.state == RampState.FLOOR_MOVING or move_floor_up
        Ramp._add_block_to_highest_side_ramp(side_ramps[:ramp_for_addition_index + 1], move_floor_up)

        if ramp_for_addition_added_to_floor:
            return_value = False
        else:
            next_side_ramp_index = ramp_for_addition_index + 1
            next_side_ramp = side_ramps[next_side_ramp_index] if next_side_ramp_index < len(side_ramps) else None

            if next_side_ramp is None or self.ramps_connect(ramp_for_addition, next_side_ramp):
                return_value = True
            else:
                return_value = False

        self._move(node_index, 0, True, False, False)

        return return_value

    def _get_next_block_addition_index(self) -> int:
        # assert self.state == RampState.FORWARD
        # assert self._number_of_blocks < self.capacity

        for node_index in range(len(self._central_path) - 1, -1, -1):
            node_height = self.ramp_relative_height(self._central_path[node_index])
            if self.ramp_relative_height_with_side_ramp_reaches(node_height, node_index - 1):
                return node_index  # ramp is not at capacity, so this must occur

        raise Exception("No place for block addition")

    def add_block(self):
        """
        Adds a block to the ramp

        Assumes the robot stands with a block at the connection point at the start of the action.
        Leaves the robot without its block at the connection point at the end of the action.
        """
        if self._required_block_placements:
            self._add_block_to_required_placement()

            if not self._required_block_placements:
                self._floor_height += 1
                self._state = RampState.FORWARD
                self._required_block_removals.clear()
            return

        if self.__next_block_modification_position_index is None:
            self.__next_block_modification_position_index = self._get_next_block_addition_index()

        if self.add_block_at(self.__next_block_modification_position_index):
            self.__next_block_modification_position_index = None
            # self._assert_ramp_assumption_ok()

        self._number_of_blocks += 1

    def _remove_required_block(self):
        """
        Removes a block from a place specified by self._required_block_removals
        (first the position at the central ramp, then its connected ramps from bottom to the top)

        Assumes the robot stands without a block at the connection point at the start of the action.
        Leaves the robot with its block at the connection point at the end of the action.
        """
        node_index = self._required_block_removals[0]
        if not self._required_block_placements or node_index != self._required_block_placements[0]:
            self._required_block_placements.appendleft(node_index)

        if self.remove_block_at(node_index, True):
            self._required_block_removals.popleft()

    def remove_block_at(self, node_index: int, just_floor: bool = False) -> bool:
        """
        :param node_index:
        :param just_floor: if blocks should be removed just to lower the floor
        :return: if the removal is the last one required to bring the position at node index one block down
        """
        # assert 0 <= node_index < len(self._central_path)
        side_ramps = self.get_side_ramps(node_index)

        ramp_for_subtraction_index = None

        for side_ramp_index, side_ramp in enumerate(side_ramps):

            if side_ramp.state == RampState.FLOOR_MOVING:
                ramp_for_subtraction_index = side_ramp_index
                break

            if not self.ramps_at_connect(node_index, side_ramp_index):
                ramp_for_subtraction_index = side_ramp_index
                side_ramp.move_floor_down()
                break

            next_ramp_index = side_ramp_index + 1
            next_ramp = side_ramps[next_ramp_index] if next_ramp_index < len(side_ramps) else None
            if not just_floor and side_ramp._number_of_blocks > 0 and side_ramp.floor_height == self.floor_height:
                if next_ramp is None or self.ramps_connect(side_ramp, next_ramp):
                    ramp_for_subtraction_index = side_ramp_index
                    break

        if ramp_for_subtraction_index is None:
            # assert 0 < node_index
            self._pick_up_at(node_index)
            return len(side_ramps) == 0  # if there are no ramps, the ramp is in stable state
        else:
            self._move(0, node_index, False, True, True)

            Ramp._remove_block_from_highest_side_ramp(side_ramps[:ramp_for_subtraction_index + 1])

            self._move(node_index, 0, True, False, False)

            if side_ramps[ramp_for_subtraction_index].state == RampState.FORWARD:
                next_ramp_index = ramp_for_subtraction_index + 1
                if next_ramp_index >= len(side_ramps):
                    return True
        return False

    def _assert_ramp_assumption_ok(self):
        # the maximum size of flat area (without floor) is 2 blocks long
        if self.state != RampState.FORWARD:
            return  # everything ok, if not finished
        flat_area_index = None
        for node_index in range(len(self._central_path) - 1, 0, -1):
            prev_node_index = node_index - 1
            node_height = self.ramp_relative_height(self._central_path[node_index])
            if node_height >= 1 and self.ramp_relative_height_with_side_ramp_reaches(node_height, prev_node_index):
                # assert flat_area_index is None  # allow at most one flat area
                flat_area_index = prev_node_index

    def _get_next_block_removal_index(self) -> int:
        # assert self.state == RampState.FORWARD
        # assert self._number_of_blocks > 0

        for node_index in range(len(self._central_path) - 1, -1, -1):
            node_height = self.ramp_relative_height(self._central_path[node_index])
            prev_node_index = node_index - 1
            if node_height == 0:  # all remaining blocks in a side ramp at the end of main ramp
                break
            if self.ramp_relative_height_with_side_ramp_reaches(node_height, prev_node_index):
                return prev_node_index
            prev_node_height = self.ramp_relative_height(self._central_path[prev_node_index])
            if prev_node_height == 0:
                break

        return len(self._central_path) - 1

    def remove_block(self):
        """
        Removes a block from the ramp

        Assumes the robot stands without a block at the connection point at the start of the action.
        Leaves the robot with a block at the connection point at the end of the action.
        """
        if self._required_block_removals:
            self._remove_required_block()

            if not self._required_block_removals:
                self._state = RampState.FORWARD
                self._required_block_placements.clear()
            return

        if self.__next_block_modification_position_index is None:
            self.__next_block_modification_position_index = self._get_next_block_removal_index()

        if self.remove_block_at(self.__next_block_modification_position_index):
            self.__next_block_modification_position_index = None
            # self._assert_ramp_assumption_ok()

        self._number_of_blocks -= 1

    @property
    def state(self) -> RampState:
        return self._state

    def get_all_positions(self) -> set[Position]:
        side_ramp_positions = set()
        for side_ramps in self._side_ramps.values():
            for side_ramp in side_ramps:
                side_ramp_positions |= side_ramp.get_all_positions()
        return set(self._central_path) | side_ramp_positions

    def visualise(self, new_node: Position, l_just_width=3):
        clear()
        central_path_set = set(self._central_path)
        side_ramp_set = self.get_all_positions() - central_path_set
        for x in range(self._plan.remaining_structure.shape[0]):
            for y in range(self._plan.remaining_structure.shape[1]):
                xy = (x, y)
                color = f"{colorama.Fore.RESET}{colorama.Back.RESET}"
                if xy in central_path_set:
                    color = f"{colorama.Fore.BLACK}{colorama.Back.CYAN}"
                elif xy in side_ramp_set:
                    color = f"{colorama.Fore.WHITE}{colorama.Back.BLUE}"
                elif xy == new_node:
                    color = f"{colorama.Fore.BLACK}{colorama.Back.YELLOW}"
                print(f"{color}{str(self._plan.remaining_structure[xy]).ljust(l_just_width)}", end="")
            print()
        print(colorama.Fore.RESET, end="")
        print(colorama.Back.RESET)
        print(f"Plan length: {self._plan.t_e:_} ticks")
        input()

    def add_edge(self, ramp_node: Position, outside_node: Position, access_point: Position) -> bool:
        # assert outside_node not in self.get_all_positions()
        # assert self.state == RampState.FORWARD
        # assert self.connection_point == access_point  # must have access to block storage
        # assert self.floor_height == 0

        if VISUALISE:
            self.visualise(outside_node)

        if ramp_node != self._central_path[-1]:
            return False

        ramp_node_index = len(self._central_path) - 1

        old_end_height = self.ramp_relative_height(ramp_node)
        new_node_height = self.ramp_relative_height(outside_node)

        if (old_end_height <= new_node_height <= 0
                and not self.ramp_relative_height_with_side_ramp_reaches(1, ramp_node_index)):
            self._central_path.append(outside_node)
            self._capacity_and_maximum_height = None
            return True  # if on the floor, just append

        if new_node_height > self.maximum_height + 1:
            return False

        while (self._number_of_blocks > 0
               and (self.__next_block_modification_position_index is not None
                    or self.ramp_relative_height_with_side_ramp_reaches(new_node_height, ramp_node_index))):
            self._plan.enter(False, self.connection_point)
            self.remove_block()
            self._plan.leave()

        while (self.__next_block_modification_position_index is not None
               or not self.ramp_relative_height_with_side_ramp_reaches(new_node_height - 1, ramp_node_index)):
            self._plan.enter(True, self.connection_point)
            self.add_block()
            self._plan.leave()

        self._central_path.append(outside_node)
        self._number_of_blocks += new_node_height - self.floor_height
        self._capacity_and_maximum_height = None

        return True

    def remove_all_blocks(self, access_point: Position):
        """ Must have access to access point """
        # assert self.connection_point == access_point
        while self._number_of_blocks > 0:
            self._plan.enter(False, self.connection_point)
            self.remove_block()
            self._plan.leave()

    def remove_ramp_end_up_to(self, access_point: Position, new_ramp_end: Position):
        self.remove_all_blocks(access_point)

        # assert self._state == RampState.FORWARD
        # assert self._floor_height == 0
        # assert self._number_of_blocks == 0
        # assert self.__next_block_modification_position_index is None

        removed_count = 0
        for node in reversed(self._central_path):
            if node == new_ramp_end:
                break
            # assert node != self.connection_point
            removed_count += 1
            if node in self._side_ramps:
                del self._side_ramps[node]

        self._central_path = self._central_path[:len(self._central_path) - removed_count]
        self._capacity_and_maximum_height = None
        self._required_block_placements: deque[int] = deque()
        self._required_block_removals: deque[int] = deque()

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return ", ".join(map(lambda xy: f"({xy[0]}, {xy[1]}, [{self.ramp_relative_height(xy)}])", self._central_path))

    def get_side_ramp_stats(self) -> list[int]:
        """
        :return: 0th index how many side ramps, 1st index how many side ramps of side ramps and so on
        """
        side_ramp_stats: list[int] = []

        for side_ramps in self._side_ramps.values():
            for side_ramp in side_ramps:
                if len(side_ramp_stats) == 0:
                    side_ramp_stats.append(0)
                side_ramp_stats[0] += 1
                for i, number in enumerate(side_ramp.get_side_ramp_stats(), start=1):
                    if i >= len(side_ramp_stats):
                        side_ramp_stats.append(0)
                    side_ramp_stats[i] += number

        return side_ramp_stats

    def get_stats(self) -> str:
        return f"R[side ramp stats {self.get_side_ramp_stats()}]"


class ReversibleRampInfo:
    def __init__(self, capacity: int, maximum_height: int, pivot_index: int, area: int, use_side_ramps: bool):
        self.use_side_ramps = use_side_ramps
        self.capacity = capacity
        self.maximum_height = maximum_height
        self.pivot_index = pivot_index
        self.area = area  # does not count the connection point
        self.pivot_side_ramp_indices: list[int] = []

    def advance(self, delta: int, central_path: list[Position],
                side_ramps: dict[Position, list[ReversibleRamp]]) -> ReversibleRampInfo:
        pivot = central_path[self.pivot_index]

        info = self.advance_by_side_ramps(side_ramps.get(pivot, [])) if self.use_side_ramps else self

        return ReversibleRampInfo(
            info.capacity + info.area + 1,
            info.maximum_height + 1,
            info.pivot_index + delta,
            info.area + 1,
            self.use_side_ramps
        )

    def advance_by_side_ramp(self, side_ramp: ReversibleRamp) -> ReversibleRampInfo:
        if not self.use_side_ramps:
            return self
        return ReversibleRampInfo(
            self.capacity + self.area * side_ramp.maximum_reversible_height + side_ramp.reversible_capacity,
            self.maximum_height + side_ramp.maximum_reversible_height,
            self.pivot_index,
            self.area + side_ramp.area,
            True
        )

    def advance_by_side_ramps(self, side_ramps: Iterable[ReversibleRamp]) -> ReversibleRampInfo:
        info = self
        for side_ramp in side_ramps:
            info = info.advance_by_side_ramp(side_ramp)
        return info


class ReversibleRamp(Ramp):
    def __init__(self, plan: AgentPlan, central_path: list[Position], side_ramps: dict[Position, list[ReversibleRamp]],
                 floor_height: int, number_of_blocks: Optional[int] = 0):
        super().__init__(plan, central_path, side_ramps, floor_height, number_of_blocks)
        self.reversible_ramp_info: ReversibleRampInfo = self._compute_reversible_ramp_info()

    @classmethod
    def from_tree(cls,
                  start: Optional[TreeNode],
                  end: TreeNode,
                  plan: AgentPlan,
                  sub_side_ramp_count: int) -> ReversibleRamp:
        central_path, side_ramps = Ramp.get_central_path_and_side_ramps(start, end, plan, sub_side_ramp_count)

        return ReversibleRamp(plan, central_path, side_ramps, 0, 0)

    def get_pivot(self) -> Position:
        return self._central_path[self.reversible_ramp_info.pivot_index]

    def _compute_reversible_ramp_info(self, use_side_ramps: bool = True) -> ReversibleRampInfo:
        # starts with accepted first column; does not count the index node
        backward_ramp = ReversibleRampInfo(0, 0, 0, 0, use_side_ramps)
        # starts with accepted last column; does not count the index node
        forward_ramp = ReversibleRampInfo(0, 0, len(self._central_path) - 1, 0, True)

        # go from both sides of the ramp, until both pivot indices meet
        while backward_ramp.pivot_index < forward_ramp.pivot_index:
            new_backward_ramp = backward_ramp.advance(1, self._central_path, self._side_ramps)
            # if the ramp can be reversed, try to advance the backward ramp - if the new ramp is too big, undo change
            if new_backward_ramp.capacity <= forward_ramp.capacity:
                backward_ramp = new_backward_ramp
            else:
                forward_ramp = forward_ramp.advance(-1, self._central_path, self._side_ramps)

        # now the only side ramps potentially remaining unassigned are those at the pivot index
        # these side ramps can be used in both forward and backward ramps

        pivot = self._central_path[backward_ramp.pivot_index]
        side_ramps = self._side_ramps.get(pivot, [])
        side_ramp_indices = list(range(len(side_ramps)))
        max_backward_ramp = backward_ramp
        """ Backward ramp with maximum capacity """

        for s in powerset(side_ramp_indices):
            backward_side_ramp_indices = list(s)
            forward_side_ramp_indices = [i for i in side_ramp_indices if i not in backward_side_ramp_indices]

            test_backward_ramp = backward_ramp.advance_by_side_ramps(
                [side_ramps[i] for i in backward_side_ramp_indices])
            test_forward_ramp = forward_ramp.advance_by_side_ramps([side_ramps[i] for i in forward_side_ramp_indices])

            if max_backward_ramp.capacity < test_backward_ramp.capacity <= test_forward_ramp.capacity:
                max_backward_ramp = test_backward_ramp
                max_backward_ramp.pivot_side_ramp_indices = backward_side_ramp_indices

        backward_ramp = max_backward_ramp

        if use_side_ramps:
            reversible_info_without_ramps = self._compute_reversible_ramp_info(False)
            if reversible_info_without_ramps.maximum_height >= backward_ramp.maximum_height:
                return reversible_info_without_ramps

        return backward_ramp

    @property
    def maximum_reversible_height(self) -> int:
        """ Maximum height, measured from the floor_height, where the ramp is still reversible """
        return self.reversible_ramp_info.maximum_height

    @property
    def reversible_capacity(self) -> int:
        """ Maximum capacity, counting blocks above the floor_height, where the ramp is still reversible """
        return self.reversible_ramp_info.capacity

    def at_reversible_capacity(self) -> bool:
        return self._number_of_blocks == self.reversible_capacity and self.can_reverse()

    @property
    def backward_side_ramps(self) -> dict[Position, list[ReversibleRamp]]:
        return {self._central_path[node_index]: self.backward_side_ramps_at(node_index)
                for node_index in range(self.reversible_ramp_info.pivot_index, -1, -1)}

    def backward_side_ramps_at(self, node_index: int) -> list[ReversibleRamp]:
        node = self._central_path[node_index]
        side_ramps = self._side_ramps.get(node, [])
        if node_index < self.reversible_ramp_info.pivot_index:
            return side_ramps
        elif node_index > self.reversible_ramp_info.pivot_index:
            return []
        else:
            return [side_ramps[j] for j in self.reversible_ramp_info.pivot_side_ramp_indices]

    @property
    def forward_side_ramps(self) -> dict[Position, list[ReversibleRamp]]:
        return {self._central_path[node_index]: self.forward_side_ramps_at(node_index)
                for node_index in range(self.reversible_ramp_info.pivot_index, len(self._central_path))}

    def forward_side_ramps_at(self, node_index: int) -> list[ReversibleRamp]:
        node = self._central_path[node_index]
        side_ramps = self._side_ramps.get(node, [])
        if node_index > self.reversible_ramp_info.pivot_index:
            return side_ramps
        elif node_index < self.reversible_ramp_info.pivot_index:
            return []
        else:
            backward_side_ramp_indices = set(self.reversible_ramp_info.pivot_side_ramp_indices)
            return [side_ramps[i] for i in range(len(side_ramps)) if i not in backward_side_ramp_indices]

    def get_side_ramps(self, node_index: int) -> list[ReversibleRamp]:
        node = self._central_path[node_index]
        side_ramps = self._side_ramps.get(node, [])
        if node_index == self.reversible_ramp_info.pivot_index:
            backward_side_ramp_indices = set(self.reversible_ramp_info.pivot_side_ramp_indices)
            side_ramps = ([side_ramps[j] for j in self.reversible_ramp_info.pivot_side_ramp_indices]
                          + [side_ramps[i] for i in range(len(side_ramps)) if i not in backward_side_ramp_indices])

        return side_ramps

    def ramp_relative_height_with_side_ramp_reaches(self, height: int, node_index: int):
        # assert self._state in [RampState.FORWARD, RampState.FLOOR_MOVING]
        node = self._central_path[node_index]
        node_height = self.ramp_relative_height(node)
        side_ramps = self.forward_side_ramps_at(node_index)
        for side_ramp in reversed(side_ramps):
            if side_ramp.is_reaching_height(side_ramp.maximum_reversible_height) and side_ramp.can_reverse():
                node_height += side_ramp.maximum_reversible_height
            elif node_height >= height:
                return True
            else:
                return side_ramp.is_reaching_height(height - node_height)

        return node_height >= height

    def can_reverse(self) -> bool:
        if self.state == RampState.BACKWARD:
            return True
        if self.state == RampState.FLOOR_MOVING:
            return False
        return self._number_of_blocks <= self.reversible_capacity and self.ramp_relative_height(self.get_pivot()) == 0

    def is_reaching_height(self, height: int, number_of_blocks: Optional[int] = None) -> bool:
        """
        Determines, if the ramp in its BACKWARD state reaches given height (relative to the floor height)

        (i.e. the top block of the BACKWARD ramp is at the given height or higher)
        :param height: given height
        :param number_of_blocks: number of ramp block to use (uses all the current ramp blocks by default)
        :return: if the ramp in its BACKWARD state reaches given height
        """
        if number_of_blocks is None:
            number_of_blocks = self._number_of_blocks

        if number_of_blocks == self.reversible_ramp_info.capacity:
            return True

        if number_of_blocks > self.reversible_ramp_info.capacity:
            return False

        backward_ramp = ReversibleRampInfo(0, 0, 0, 0, self.reversible_ramp_info.use_side_ramps)

        while (backward_ramp.capacity < number_of_blocks and backward_ramp.maximum_height < height
               and backward_ramp.pivot_index != self.reversible_ramp_info.pivot_index):
            new_backward_ramp = backward_ramp.advance(1, self._central_path, self._side_ramps)
            pivot = self._central_path[backward_ramp.pivot_index]
            side_ramps = self._side_ramps.get(pivot, [])

            # if there is a chance the ramp is found by partial usage of ramps, do not accept, break the loop
            if (self.reversible_ramp_info.use_side_ramps
                    and len(side_ramps) > 0 and new_backward_ramp.capacity > number_of_blocks):
                break
            else:
                backward_ramp = new_backward_ramp

        pivot = self._central_path[backward_ramp.pivot_index]
        side_ramps = self._side_ramps.get(pivot, [])
        if backward_ramp.pivot_index == self.reversible_ramp_info.pivot_index:
            side_ramps = [side_ramps[i] for i in self.reversible_ramp_info.pivot_side_ramp_indices]
        else:
            side_ramps = deepcopy(side_ramps)

        while (self.reversible_ramp_info.use_side_ramps and len(side_ramps) > 0
               and backward_ramp.capacity < number_of_blocks and backward_ramp.maximum_height < height):
            side_ramp = side_ramps.pop()
            new_backward_ramp = backward_ramp.advance_by_side_ramp(side_ramp)
            remaining_height = height - backward_ramp.maximum_height
            remaining_capacity = number_of_blocks - backward_ramp.capacity

            # if there is a chance the ramp is found by partial usage of the side ramp, try it
            if new_backward_ramp.capacity > number_of_blocks and remaining_height < side_ramp.maximum_reversible_height:
                return side_ramp.is_reaching_height(remaining_height, remaining_capacity)
            else:
                backward_ramp = new_backward_ramp

        return backward_ramp.capacity <= number_of_blocks and backward_ramp.maximum_height >= height

    def pass_up(self):
        """

        Assumes the ramp is in the FORWARD state at the start of the action.
        Assumes the robot stands with or without a block at the connection point at the start of the action.
        Leaves the robot at the connection point at the end of the action with the ramp in the BACKWARD state.
        """
        # assert self.state == RampState.FORWARD
        # assert self.can_reverse()

        pivot_index = self.reversible_ramp_info.pivot_index

        self._move_on_floor(0, pivot_index)

        forward_ramp = Ramp(self._plan, self._central_path[pivot_index:], self.forward_side_ramps, self.floor_height,
                            self._number_of_blocks)
        backward_ramp = Ramp(self._plan, self._central_path[pivot_index::-1],
                             self.backward_side_ramps if self.reversible_ramp_info.use_side_ramps else dict(),
                             self.floor_height, 0)

        while forward_ramp._number_of_blocks > 0:
            if self._plan.agent_holding_a_block:
                backward_ramp.add_block()
                forward_ramp.remove_block()
            else:
                forward_ramp.remove_block()
                backward_ramp.add_block()

        self._state = RampState.BACKWARD

        self._climb_side_ramps(self.backward_side_ramps_at(pivot_index), True)
        self._move(pivot_index, 0, True, False, True)

    def pass_down(self):
        # assert self.state == RampState.BACKWARD
        # assert self.can_reverse()

        pivot_index = self.reversible_ramp_info.pivot_index

        self._move(0, pivot_index, False, True, False)
        self._climb_side_ramps(self.backward_side_ramps_at(pivot_index), False)

        forward_ramp = Ramp(self._plan, self._central_path[pivot_index:], self.forward_side_ramps, self.floor_height, 0)
        backward_ramp = Ramp(self._plan, self._central_path[pivot_index::-1],
                             self.backward_side_ramps if self.reversible_ramp_info.use_side_ramps else dict(),
                             self.floor_height, self._number_of_blocks)

        while backward_ramp._number_of_blocks > 0:
            if self._plan.agent_holding_a_block:
                forward_ramp.add_block()
                backward_ramp.remove_block()
            else:
                backward_ramp.remove_block()
                forward_ramp.add_block()

        self._state = RampState.FORWARD

        self._move_on_floor(pivot_index, 0)


class DummyTQDM:
    def __init__(self, total: int):
        self.n = 0
        self.total: int = total

    def update(self, value: int):
        self.n += value
        if LOG:
            print(f"{str(self.n).rjust(len(str(self.total)))} of {self.total} complete")

    def close(self):
        pass

    def set_description(self, description: str):
        pass


def neighbors(node: Position) -> Iterable[Position]:
    for dx, dy in [(-1, 0), (0, -1), (0, 1), (1, 0)]:
        yield node[0] + dx, node[1] + dy


def filtered_neighbors(node: Position, closed: set[Position], remaining_structure: BuildingArea) -> Iterable[Position]:
    _neighbors = [n for n in neighbors(node) if index_in_array(n, remaining_structure) and n not in closed]

    def smallest_height_difference(neighbor: Position) -> int:
        return abs(int(remaining_structure[node] - remaining_structure[neighbor]))

    while _neighbors:
        m = min(_neighbors, key=smallest_height_difference)
        yield m
        _neighbors.remove(m)


def index_in_array(index: tuple[int, int], array: np.ndarray):
    if len(index) != len(array.shape):
        return False
    return all([0 <= i < s for i, s in zip(index, array.shape)])


def dfs_tree(node: TreeNode, closed: set[Position], remaining_structure: BuildingArea, plan: AgentPlan,
             access_point: Position, unvisited_nodes: dict[Position, dict[TreeNode, None]], ramp: Ramp, _tqdm: tqdm) -> bool:
    updated_remaining_structure = False
    for neighbor in filtered_neighbors(node.position, closed, remaining_structure):
        if index_in_array(neighbor, remaining_structure) and neighbor not in closed:
            can_deconstruct_neighbor = False
            if ramp.end_point != node.position:
                ramp.remove_ramp_end_up_to(access_point, node.position)
            if ramp.add_edge(node.position, neighbor, access_point):
                can_deconstruct_neighbor = True
            else:
                unvisited_nodes[neighbor][node] = None
            if can_deconstruct_neighbor:
                if neighbor in unvisited_nodes:
                    _tqdm.update(1)
                    # snapshot = tracemalloc.take_snapshot()
                    # top_stats = snapshot.statistics('lineno')
                    # stat = top_stats[0]
                    # print(stat)
                    _tqdm.set_description(f"Plan length: {plan.t_e:_} ticks (running DFS; {ramp.get_stats()})")

                    del unvisited_nodes[neighbor]
                    updated_remaining_structure = True
                closed |= {neighbor}
                neighbor_node = TreeNode(neighbor, node)
                node.add_child(neighbor_node)
                updated_neighbor = dfs_tree(neighbor_node, closed, remaining_structure,
                                            plan, access_point, unvisited_nodes, ramp, _tqdm)
                updated_remaining_structure = updated_remaining_structure or updated_neighbor

    return updated_remaining_structure


def single_access_re_ramp(remaining_structure: BuildingArea,
                          sub_side_ramp_count: int,
                          access_point: Position,
                          blocked_points: set[Position],
                          unvisited_nodes: dict[Position, dict[TreeNode, None]],
                          plan: AgentPlan,
                          _tqdm: tqdm) -> bool:
    """

    :param remaining_structure: height-map of the structure, that the agents must build
    :param sub_side_ramp_count: maximum number of sub in sub-ramps
    :param access_point: access point for the agents
    :param blocked_points: inaccessible points for the agents
    :param unvisited_nodes: dictionary of not-yet visited nodes and their found neighbors
    :param plan: partial plan for target structure removal, that this algorithm will continue
    :param _tqdm: handle for progress bar
    :return: if any of the target structure columns was removed
    """
    if len(unvisited_nodes) == 0:
        return False  # there are no columns to remove
    closed = deepcopy(blocked_points) | {access_point}
    ramp = Ramp(plan, [access_point], dict())
    access_tree_node = TreeNode(access_point, None)
    _tqdm.set_description(f"Plan length: {plan.t_e:_} ticks (running DFS)")
    updated = dfs_tree(access_tree_node, closed, remaining_structure, plan, access_point, unvisited_nodes, ramp, _tqdm)
    _tqdm.set_description(f"Plan length: {plan.t_e:_} ticks (removing all blocks)")
    ramp.remove_all_blocks(access_point)

    for unvisited_node, neighbor_ordered_set in list(unvisited_nodes.items()):
        for neighbor in neighbor_ordered_set:
            ramp = Ramp.from_tree(access_tree_node, neighbor, plan, sub_side_ramp_count)
            _tqdm.set_description(f"Plan length: {plan.t_e:_} ticks (running DFS)")
            updated |= dfs_tree(neighbor, closed, remaining_structure, plan, access_point, unvisited_nodes, ramp, _tqdm)
            _tqdm.set_description(f"Plan length: {plan.t_e:_} ticks (removing all blocks)")
            ramp.remove_all_blocks(access_point)

    for _, neighbor_ordered_set in list(unvisited_nodes.items()):
        neighbor_ordered_set.clear()

    _tqdm.set_description(f"Plan length: {plan.t_e:_} ticks")

    return updated


def get_default_available_area(access_point: Position, structure: BuildingArea) -> tuple[set[Position], Position]:
    X = structure.shape[0]
    Y = structure.shape[1]

    if X < Y:
        area, end_point = get_default_available_area((access_point[1], access_point[0]), structure.T)
        return {(y, x) for x, y in area}, end_point

    # X >= Y
    match access_point:
        case (x, 0):
            end_y = min(x, X - 1 - x, (Y + 1) // 2)
            return {(x, y) for y in range(0, end_y)}, (x, end_y - 1)
        case (x, y) if y == Y - 1:
            end_y = max(Y - X + x, Y - 1 - x, (Y - 1) // 2)
            return {(x, y) for y in range(Y - 1, end_y, -1)}, (x, end_y + 1)
        case (0, y):
            end_x = min(y, Y - 1 - y) + 1
            return {(x, y) for x in range(0, end_x)}, (end_x - 1, y)
        case (x, y) if x == X - 1:
            end_x = max(X - Y + y, X - 1 - y, (Y + 1) // 2) - 1
            return {(x, y) for x in range(X - 1, end_x, -1)}, (end_x + 1, y)
        case _:
            raise Exception("Unsupported access point position")


def l1_distance(xy1: Position, xy2: Position) -> int:
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])


class AgentAssignedArea:
    def __init__(self, plan: AgentPlan, access_and_end_points: list[tuple[Position, Position]], area: set[Position]):
        self.plan = plan
        self.access_point_end_point_list = access_and_end_points
        self.area = area

    def get_neighbor(self, areas: Iterable[tuple[int, AgentAssignedArea]]) -> Optional[tuple[int, AgentAssignedArea]]:
        """
        Returns first of given areas, which neighbors this agent assigned area, None of none neighbor this area
        :param areas: given areas to check
        :return: first of given areas, which neighbors this agent assigned area, None of none neighbor this area
        """
        for area_index, area in areas:
            for xy1, end_xy1 in self.access_point_end_point_list:
                for xy2, end_xy2 in area.access_point_end_point_list:
                    if l1_distance(xy1, xy2) == 1 or l1_distance(end_xy1, end_xy2) == 1:
                        return area_index, area
        return None

    def connect(self, area: AgentAssignedArea) -> AgentAssignedArea:
        if self.plan.t_e < area.plan.t_e:
            area.connect(self)
            return area

        # self.plan.t_e >= area.plan.t_e; continue just with self.plan; no need to synchronise
        self.access_point_end_point_list += area.access_point_end_point_list
        self.area |= area.area
        return self


def re_ramp_init(target_structure: BuildingArea,
                 blocked_points: Optional[list[Position]]):
    if blocked_points is None:
        blocked_points = []

    # the structure is accessible from an added ring of border cells
    target_structure = np.pad(target_structure, 1)
    X = target_structure.shape[0]
    Y = target_structure.shape[1]
    access_points = \
        [(x, 0) for x in range(X)] + \
        [(x, Y - 1) for x in range(X)] + \
        [(0, y) for y in range(1, Y - 1)] + \
        [(X - 1, y) for y in range(1, Y - 1)]

    # check all access and blocked points are within the target structure
    for p in access_points + blocked_points:
        if not index_in_array(p, target_structure):
            raise f"Point {p} is not in the target structure"

    # check that access points are not blocked, remove the blocked ones
    if len(set(access_points).intersection(set(blocked_points))) > 0:
        print(f"The following access points are blocked:")
        print(set(access_points).intersection(set(blocked_points)))
        access_points = [access_point for access_point in access_points if access_point not in blocked_points]

    access_and_blocked_points = set(access_points + blocked_points)

    return target_structure.copy(), access_and_blocked_points, access_points


class MultiAgentPickleInfo:
    def __init__(self,
                 target_structure: BuildingArea,
                 action_type_durations: ActionDurations,
                 blocked_points: Optional[list[Position]],
                 remaining_structure: BuildingArea,
                 access_and_blocked_points: set[Position],
                 access_points: list[Position],
                 plans_and_available_areas: list[AgentAssignedArea],
                 unvisited_nodes: dict[Position, dict[TreeNode, None]],
                 plan_queue: list[tuple[int, int, int]]):
        self.VERSION = 1,
        self.target_structure = target_structure
        self.action_type_durations = action_type_durations
        self.blocked_points = blocked_points
        self.remaining_structure = remaining_structure
        self.access_and_blocked_points = access_and_blocked_points
        self.access_points = access_points
        self.plans_and_available_areas = plans_and_available_areas
        self.unvisited_nodes = unvisited_nodes
        self.plan_queue = plan_queue

    def check_input_same(self,
                         target_structure: BuildingArea,
                         action_type_durations: ActionDurations,
                         blocked_points: Optional[list[Position]]) -> bool:
        return (target_structure.shape == self.target_structure.shape
                and np.all(target_structure == self.target_structure)
                and action_type_durations == self.action_type_durations
                and (blocked_points is self.blocked_points or blocked_points == self.blocked_points))

    def to_tuple(self):
        return (self.remaining_structure, self.access_and_blocked_points, self.access_points,
                self.plans_and_available_areas, self.unvisited_nodes, self.plan_queue)

    @staticmethod
    def load(pickle_file_name: Optional[str],
             target_structure: BuildingArea,
             action_type_durations: ActionDurations,
             blocked_points: Optional[list[Position]]) -> Optional[MultiAgentPickleInfo]:
        if pickle_file_name is not None and os.path.exists(pickle_file_name):
            with (open(pickle_file_name, mode='rb') as pickle_file):
                state = pickle.load(pickle_file)
                if (isinstance(state, MultiAgentPickleInfo)
                        and state.check_input_same(target_structure, action_type_durations, blocked_points)):
                    return state
        return None

    def save(self, pickle_file_name: Optional[str]):
        if pickle_file_name is not None:
            with (open(pickle_file_name, mode='wb') as pickle_file):
                pickle.dump(self, pickle_file)


def re_ramp_multi_agent(target_structure: BuildingArea,
                        sub_side_ramp_count: int,
                        action_type_durations: ActionDurations = ActionDurations(),
                        blocked_points: Optional[list[Position]] = None,
                        pickle_file_name: Optional[str] = None) -> Optional[MultiAgentPlan]:
    """
    Creates single-agent plan for building the target structure

    :param target_structure: height-map of the structure, that the agents must build
    :param sub_side_ramp_count: maximum number of sub in sub-ramps
    :param action_type_durations: action durations
    :param blocked_points: inaccessible points for the agents
    :param pickle_file_name: optional file, where to safe progress
    :return: single-agent plan for building the target structure
    """
    remaining_structure, access_and_blocked_points, access_points = re_ramp_init(target_structure, blocked_points)

    available_positions: set[Position] = set(zip(*np.where(np.ones_like(remaining_structure))))
    plans_and_available_areas: list[AgentAssignedArea] = []

    for access_point_index, access_point in enumerate(access_points):
        agent_plan = AgentPlan(0, action_type_durations.reverse(), remaining_structure)
        area, end_point = get_default_available_area(access_point, remaining_structure)
        plans_and_available_areas.append(AgentAssignedArea(agent_plan, [(access_point, end_point)], area))

    unvisited_nodes: dict[Position, dict[TreeNode, None]] = {
        xy: dict() for xy in zip(*np.where(remaining_structure > 0))
    }

    pbar = DummyTQDM(total=len(unvisited_nodes)) if VISUALISE or LOG else tqdm(total=len(unvisited_nodes), smoothing=0)

    plan_queue: list[tuple[int, int, int]] = []
    """ list of: agent_plan.t_e, -number_of_remaining_columns_within_assigned_area, agent_assigned_area_index tuples """

    def run_agent_plan_while_modifications_done(_agent_assigned_area: AgentAssignedArea):
        while any([single_access_re_ramp(remaining_structure, sub_side_ramp_count, _access_point,
                                         (access_and_blocked_points - {_access_point}) | (
                                                 available_positions - _agent_assigned_area.area),
                                         unvisited_nodes, _agent_assigned_area.plan, pbar)
                   for _access_point, _end_point in _agent_assigned_area.access_point_end_point_list]):
            pass

    def create_priority_queue_tuple(_area_index: int) -> tuple[int, int, int]:
        """
        Creates tuple (area.plan.t_e, -sum(mask(area.plan.remaining_structure) > 0), _area_index)
        for area at _area_index
        """
        _area = plans_and_available_areas[_area_index]
        _assigned_area = _area.area
        masked_remaining_structure = _area.plan.remaining_structure[*zip(*_assigned_area)]
        return _area.plan.t_e, -np.sum(masked_remaining_structure > 0), _area_index

    def insert_into_plan_queue(_plan_index: int):
        """
        Creates tuple using create_priority_queue_tuple at given _plan_index and inserts it into the priority queue
        to keep it sorted
        :param _plan_index: index of item in plans_and_available_areas array
        """
        bisect.insort(plan_queue, create_priority_queue_tuple(_plan_index), key=lambda x: x[0:1])

    state = MultiAgentPickleInfo.load(pickle_file_name, target_structure, action_type_durations, blocked_points)
    if state is not None:
        (remaining_structure, access_and_blocked_points, access_points,
         plans_and_available_areas, unvisited_nodes, plan_queue) = state.to_tuple()
        pbar.update(pbar.total - len(unvisited_nodes) - pbar.n)
    else:
        state = MultiAgentPickleInfo(target_structure, action_type_durations, blocked_points, remaining_structure,
                                     access_and_blocked_points, access_points, plans_and_available_areas,
                                     unvisited_nodes, plan_queue)

        # deconstruct what you can, using all the agents
        for plan_index, agent_assigned_area in enumerate(plans_and_available_areas):
            run_agent_plan_while_modifications_done(agent_assigned_area)
            insert_into_plan_queue(plan_index)

        state.save(pickle_file_name)

    def to_areas(_not_connected_area_indices: list[int]) -> Iterable[tuple[int, AgentAssignedArea]]:
        for area_index in _not_connected_area_indices:
            yield area_index, plans_and_available_areas[area_index]

    while len(plan_queue) > 1:
        not_connected_area_indices: list[int] = [plan_queue.pop(0)[-1]]
        area2_index: int = plan_queue.pop(0)[-1]
        area2 = plans_and_available_areas[area2_index]

        while (area1_with_index := area2.get_neighbor(to_areas(not_connected_area_indices))) is None:
            not_connected_area_indices.append(area2_index)
            area2_index = plan_queue.pop(0)[-1]
            area2 = plans_and_available_areas[area2_index]

        area2 = area2.connect(area1_with_index[1])

        run_agent_plan_while_modifications_done(area2)

        for index in not_connected_area_indices:
            if index != area1_with_index[0]:
                insert_into_plan_queue(index)
            else:
                insert_into_plan_queue(area2_index)

        state.save(pickle_file_name)

    max_t_e = max(map(lambda a: a.plan.t_e, plans_and_available_areas))
    pbar.set_description(f"Plan length: {max_t_e:_} ticks")
    pbar.close()

    if not np.all(remaining_structure == 0):
        print("Remaining un-buildable structure:")
        print(remaining_structure)
        return None

    plan = MultiAgentPlan(target_structure, action_type_durations)
    for area in plans_and_available_areas:
        if area.plan.instructions:
            plan.add_agents(area.plan.reverse(max_t_e - area.plan.t_e))
    return plan


class SingleAgentPickleInfo:
    def __init__(self,
                 target_structure: BuildingArea,
                 action_type_durations: ActionDurations,
                 blocked_points: Optional[list[Position]],
                 remaining_structure: BuildingArea,
                 access_and_blocked_points: set[Position],
                 access_points: list[Position],
                 plan: AgentPlan,
                 unvisited_nodes: dict[Position, dict[TreeNode, None]]):
        self.VERSION = 1,
        self.target_structure = target_structure
        self.action_type_durations = action_type_durations
        self.blocked_points = blocked_points
        self.remaining_structure = remaining_structure
        self.access_and_blocked_points = access_and_blocked_points
        self.access_points = access_points
        self.plan = plan
        self.unvisited_nodes = unvisited_nodes

    def check_input_same(self,
                         target_structure: BuildingArea,
                         action_type_durations: ActionDurations,
                         blocked_points: Optional[list[Position]]) -> bool:
        return (target_structure.shape == self.target_structure.shape
                and np.all(target_structure == self.target_structure)
                and action_type_durations == self.action_type_durations
                and (blocked_points is self.blocked_points or blocked_points == self.blocked_points))

    def to_tuple(self)\
            -> tuple[BuildingArea, set[Position], list[Position], AgentPlan, dict[Position, dict[TreeNode, None]]]:
        s = self
        return s.remaining_structure, s.access_and_blocked_points, s.access_points, s.plan, s.unvisited_nodes

    @staticmethod
    def load(pickle_file_name: Optional[str],
             target_structure: BuildingArea,
             action_type_durations: ActionDurations,
             blocked_points: Optional[list[Position]]) -> Optional[SingleAgentPickleInfo]:
        if pickle_file_name is not None and os.path.exists(pickle_file_name):
            with (open(pickle_file_name, mode='rb') as pickle_file):
                state = pickle.load(pickle_file)
                if (isinstance(state, SingleAgentPickleInfo)
                        and state.check_input_same(target_structure, action_type_durations, blocked_points)):
                    return state
        return None

    def save(self, pickle_file_name: Optional[str]):
        if pickle_file_name is not None:
            with (open(pickle_file_name, mode='wb') as pickle_file):
                pickle.dump(self, pickle_file)


def re_ramp(target_structure: BuildingArea,
            sub_side_ramp_count: int,
            action_type_durations: ActionDurations = ActionDurations(),
            blocked_points: Optional[list[Position]] = None,
            pickle_file_name: Optional[str] = None) -> Optional[MultiAgentPlan]:
    """
    Creates single-agent plan for building the target structure

    :param target_structure: height-map of the structure, that the agents must build
    :param sub_side_ramp_count: maximum number of sub in sub-ramps
    :param action_type_durations: action durations
    :param blocked_points: inaccessible points for the agents
    :param pickle_file_name: optional file, where to safe progress
    :return: single-agent plan for building the target structure
    """
    remaining_structure, access_and_blocked_points, access_points = re_ramp_init(target_structure, blocked_points)

    plan = AgentPlan(0, action_type_durations.reverse(), remaining_structure)
    unvisited_nodes: dict[Position, dict[TreeNode, None]] = {
        xy: dict() for xy in zip(*np.where(remaining_structure > 0))
    }

    state = SingleAgentPickleInfo.load(pickle_file_name, target_structure, action_type_durations, blocked_points)
    if state is not None:
        remaining_structure, access_and_blocked_points, access_points, plan, unvisited_nodes = state.to_tuple()
    else:
        state = SingleAgentPickleInfo(target_structure, action_type_durations, blocked_points, remaining_structure,
                                      access_and_blocked_points, access_points, plan, unvisited_nodes)

    pbar = DummyTQDM(total=len(unvisited_nodes)) if VISUALISE or LOG else tqdm(total=len(unvisited_nodes), smoothing=0)
    while any([
        single_access_re_ramp(remaining_structure, sub_side_ramp_count, access_point,
                              access_and_blocked_points - {access_point}, unvisited_nodes, plan, pbar)
        for access_point in access_points]
    ):
        state.save(pickle_file_name)

    pbar.close()

    if not np.all(remaining_structure == 0):
        print("Remaining un-buildable structure:")
        print(remaining_structure)
        return None

    return MultiAgentPlan(target_structure, action_type_durations).add_agents(plan.reverse(0))


def re_ramp_wrapper(multi_agent: bool,
                    sub_side_ramp_count: int,
                    input_file_name: str,
                    pickle_file_name: Optional[str] = None) -> Optional[MultiAgentPlan]:
    re_ramp_algorithm = re_ramp_multi_agent if multi_agent else re_ramp

    with open(input_file_name, mode='r') as input_file:
        input_object = load(input_file, Loader=Loader)
        building = np.array(input_object["building_heights"])
        durations = ActionDurations(
            enter=int(input_object["entry_time"]),
            leave=int(input_object["leave_time"]),
            pick_up=int(input_object["pick_up_time"]),
            deliver=int(input_object["deliver_time"]),
            move_empty=int(input_object["move_time_without_block"]),
            move_block=int(input_object["move_time_with_block"]),
        )

        return re_ramp_algorithm(building, sub_side_ramp_count, durations, None, pickle_file_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="ReRamp algorithm",
        description="""Creates a plan for building given structure using TERMES robots
        
        --sub-side-ramp-count
        0 = no side ramps
        1 = side ramps
        2 = side ramps with side ramps
        3 = side ramps with side ramps with side ramps
        ...
        """
    )
    parser.add_argument("input_file")
    parser.add_argument("-m", "--multi-agent", action='store_true', default=False)
    parser.add_argument("-v", "--visualise", action='store_true', default=False)
    parser.add_argument("-l", "--log", action='store_true', default=False)
    parser.add_argument("-o", "--output-file", default=None)
    parser.add_argument("-p", "--pickle-file", default=None)
    parser.add_argument("-s", "--sub-side-ramp-count", default=1)  #

    args = parser.parse_args()

    VISUALISE = args.visualise
    LOG = args.log
    file_name = args.input_file

    output_file_name = args.output_file
    if output_file_name is None:
        output_file_name = f"./outputs/{os.path.basename(file_name)}"

    pickle_file_path = args.pickle_file
    if pickle_file_path is not None:
        print(f"Writing progress to pickle file: {pickle_file_path}")

    print(f"Loading input: {file_name}")

    building_plan = re_ramp_wrapper(args.multi_agent, args.sub_side_ramp_count, file_name, pickle_file_path)

    if building_plan is not None:
        print(f"\nmakespan: {building_plan.mission_ticks}", flush=True)
        print(f"sum_of_costs: {building_plan.sum_of_costs}\n", flush=True)
        output_dir = os.path.dirname(output_file_name)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        building_plan.write_to_file(output_file_name)
