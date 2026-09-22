from typing import Iterabl
from dataclasses import dataclass
import heapq
from itertools import count
from time import perf_counter

SIZE = 3
TILE_COUNT = SIZE * SIZE
GOAL_STATE = (0,1,2,3,4,5,6,7,8)

def validate_state(state: Iterable[int]) -> tuple[int, ...]:
    #return a normalized state, or raise ValueError if it's invalid
    normalized = tuple(state)

    if len(normalized) != TILE_COUNT:
        raise ValueError("A puzzle state must contain exactly 9 numbers")
    
    if set(normalized) != set(GOAL_STATE):
        raise ValueError("A puzzle state must contain each number 0 through 8 once")

    return normalized

def count_inversions(state: Iterable[int]) -> int:
    #Count inversions after ignoring blank tile: 0
    checked_state = validate_state(state)
    numbered_tiles = [tile for tile in checked_state if tile != 0]

    inversion_count = 0
    for left_index in range (len(numbered_tiles)):
        for right_index in range (left_index + 1, len(numbered_tiles)):
            if numbered_tiles[left_index] > numbered_tiles[right_index]:
                inversion_count += 1

    return inversion_count

def is_solvable(state: Iterable[int]) -> bool:
    #return true when the state can reach the specified goal state
    return count_inversions(state) % 2 == 0

def misplaced_tiles(state: Iterable[int]) -> int:
    #h1: count tiles that are not in their goal positions (blank doesnt count as misplaced tile)

    checked_state = validate_state(state)
    return sum(
        tile != 0 and tile != goal_tile
        for tile, goal_tile in zip(checked_state, GOAL_STATE)
    )

def manhattan_distance(state: Iterable[int]) -> int:
    #h2: sum each numbered tile's row or column distance from its goal
    checked_state = validate_state(state)
    total_distance = 0

    for current_index, tile in enumerate(checked_state):
        if tile == 0:
            continue

        current_row, current_column = divmod(current_index, SIZE)
        goal_row, goal_column = divmod(tile, SIZE)
        total_distance += abs(current_row - goal_row) + abs(current_column - goal_column)

    return total_distance

def format_state(state: Iterable[int]) -> str:
    #format a flat state as the required three row display
    checked_state = validate_state(state)
    rows = []
    for row_start in range(0, TILE_COUNT, SIZE):
        row = checked_state[row_start : row_start + SIZE]
        rows.append(" ".join(str(tile) for tile in row))
    return "\n".join(rows)

def get_neighbors(state: Iterable[int]) -> list[tuple[int, ...]]:
    #return all states reachable in one move
    checked_state = validate_state(state)
    blank_index = checked_state.index(0)
    blank_row, blank_column = divmod(blank_index, SIZE)

    possible_moves = (
        (-1, 0), #up
        (1, 0),  #down
        (0, -1), #left
        (0, 1),  #right
    )

    neighbors = []
    for row_change, column_change in possible_moves:
        neighbor_row = blank_row + row_change
        neighbor_column = blank_column + column_change

        if not (0 <= neighbor_row < SIZE and 0 <= neighbor_column < SIZE):
            continue
        
        neighbor_index = neighbor_row * SIZE + neighbor_column
        changed_state = list(checked_state)
        changed_state[blank_index], changed_state[neighbor_index] = (
            changed_state[neighbor_index],
            changed_state[blank_index],
        )
        neighbors.append(tuple(changed_state))

    return neighbors

def run_phase1_checks() -> None:
    #Run small checks before adding the A* algorithm
    goal = GOAL_STATE
    one_move_away = (1, 0, 2, 3, 4, 5, 6, 7, 8)
    sample_solvable = (3, 1, 2, 6, 4, 5, 7, 8, 0)
    sample_unsolvable = (1, 2, 3, 4, 5, 6, 8, 7, 0)

    assert count_inversions(goal) == 0
    assert is_solvable(goal)
    assert misplaced_tiles(goal) == 0
    assert manhattan_distance(goal) == 0
    assert len(get_neighbors(goal)) == 2

    # The blank is intentionally excluded from h1.
    assert misplaced_tiles(one_move_away) == 1
    assert manhattan_distance(one_move_away) == 1

    assert is_solvable(sample_solvable)
    assert not is_solvable(sample_unsolvable)

    print("Phase 1 checks passed.")
    print("\nSample state:")
    print(format_state(sample_solvable))
    print("\nInversions:", count_inversions(sample_solvable))
    print("Solvable:", is_solvable(sample_solvable))
    print("h1 misplaced tiles:", misplaced_tiles(sample_solvable))
    print("h2 Manhattan distance:", manhattan_distance(sample_solvable))
    print("\nLegal neighboring states:")
    for neighbor_number, neighbor in enumerate(get_neighbors(sample_solvable), start=1):
        print(f"\nMove {neighbor_number}:")
        print(format_state(neighbor))

@dataclass
class SearchResult:
    #info returned after solving one puzzle with A*
    
    path: list[tuple[int, ...]]
    nodes_generated: int
    runtime_ms: float

    @property
    def solution_depth(self) -> int:
        #return the number of moves from the initial state to the goal
        return len(self.path) -1

def reconstruct_path(
    parents: dict[tuple[int, ...], tuple[int, ...] | None],
    goal_state: tuple[int, ...],
) -> list[tuple[int, ...]]:
    #Follow parent pointers backward and return the path in forward order
    path = []
    current_state: tuple[int, ...] | None = goal_state

    while current_state is not None:
        path.append(current_state)
        current_state = parents[current_state]

    path.reverse()
    return path

if __name__ == "__main__":
    run_phase1_checks()
