from typing import Iterable
from dataclasses import dataclass
import heapq
from itertools import count
from time import perf_counter

SIZE = 3
TILE_COUNT = SIZE * SIZE
GOAL_STATE = (0,1,2,3,4,5,6,7,8)

def validate_state(state: Iterable[int]) -> tuple[int, ...]:
    #make sure puzzle contains exaclty one copy of every tile
    #otherwise throw error
    normalized = tuple(state)

    if len(normalized) != TILE_COUNT:
        raise ValueError("A puzzle state must contain exactly 9 numbers")
    
    if set(normalized) != set(GOAL_STATE):
        raise ValueError("A puzzle state must contain each number 0 through 8 once")

    return normalized

def count_inversions(state: Iterable[int]) -> int:
    #Count pairs of numbered tiles that appear in the wrong order
    checked_state = validate_state(state)
    numbered_tiles = [tile for tile in checked_state if tile != 0]

    inversion_count = 0
    #for every pair of tiles, add to counts of inversions when tile on left has bigger number than tile on right (blank ignored)
    for left_index in range (len(numbered_tiles)):
        for right_index in range (left_index + 1, len(numbered_tiles)):
            if numbered_tiles[left_index] > numbered_tiles[right_index]:
                inversion_count += 1

    return inversion_count

def is_solvable(state: Iterable[int]) -> bool:
    #return true when puzzle can reach goal state
    return count_inversions(state) % 2 == 0

def misplaced_tiles(state: Iterable[int]) -> int:
    #h1: count tiles that are not in their goal positions (blank doesnt count as misplaced tile)

    checked_state = validate_state(state)

    #compare each tile with tile that belongs in same position in goal 
    #blank not included in h1
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

        #the quotient is the row and the remainder is the column
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

    #every legal move swaps the blank with one adjacent tile
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
    goal_state: tuple[int, ...], ) -> list[tuple[int, ...]]:
    #Follow parent pointers backward and return the path in forward order
    path = []
    current_state: tuple[int, ...] | None = goal_state

    #search stores each state's predecessor instead of storing full path inside every frontier entry 
    while current_state is not None:
        path.append(current_state)
        current_state = parents[current_state]

    path.reverse()
    return path

def a_star_h1(initial_state: tuple[int, ...]) -> SearchResult | None:
    #solve a puzzle using A* and h1
    #A* chooses state with smallest estimated total cost

    checked_state = validate_state(initial_state)

    if not is_solvable(checked_state):
        return None

    start_time = perf_counter()

    #frontier is a min-heap so state with lowest f-cost comes out first
    #counter only breaks ties. it's not part of the score
    states_waiting_to_be_explored = []
    tie_breaker = count()

    moves_taken_so_far = 0
    estimated_total_cost = moves_taken_so_far + misplaced_tiles(checked_state)
    heapq.heappush(states_waiting_to_be_explored, (estimated_total_cost, next(tie_breaker), checked_state),)

    #best_number_of_moves records cheapest route found so far to each state
    #previous_state lets us rebuild final state at end 
    best_number_of_moves = {checked_state: moves_taken_so_far}
    previous_state = {checked_state: None}
    already_explored = set()
    nodes_generated = 0

    while states_waiting_to_be_explored:
        current_f_cost, _, current_state = heapq.heappop(states_waiting_to_be_explored)
        current_number_of_moves = best_number_of_moves[current_state]
        expected_f_cost = current_number_of_moves + misplaced_tiles(current_state)

        #a state can appear in the heap more than once
        #ignore old entry if shorter route to the same state was found later
        if current_f_cost != expected_f_cost:
            continue

        if current_state in already_explored:
            continue
        already_explored.add(current_state)

        if current_state == GOAL_STATE:
            runtime_ms = (perf_counter() - start_time) * 1000
            return SearchResult (path=reconstruct_path(previous_state, current_state), nodes_generated=nodes_generated, runtime_ms=runtime_ms,)

        for neighbor_state in get_neighbors(current_state):
            nodes_generated += 1

            if neighbor_state in already_explored:
                continue

            number_of_moves_to_neighbor = current_number_of_moves + 1
            old_number_of_moves = best_number_of_moves.get(neighbor_state)

            #add this neighbor only if it's new or we found cheaper path
            if old_number_of_moves is None or number_of_moves_to_neighbor < old_number_of_moves:
                best_number_of_moves[neighbor_state] = number_of_moves_to_neighbor
                previous_state[neighbor_state] = current_state
                neighbor_f_cost = number_of_moves_to_neighbor + misplaced_tiles(neighbor_state)
                heapq.heappush(states_waiting_to_be_explored, (neighbor_f_cost, next(tie_breaker), neighbor_state),)

    return None 

def print_solution(result: SearchResult | None) -> None:
    #print each state after the inital state and the search stats
    if result is None:
        print("This puzzle isn't solvable")
        return

    for step_number, state in enumerate(result.path[1:], start=1):
        print(f"Step: {step_number}")
        print(format_state(state))

    print(f"Solution depth: {result.solution_depth}")
    print(f"Search cost: {result.nodes_generated}")
    print(f"Time: {result.runtime_ms:.3f}ms")

def run_phase2_checks() -> None:
    #check A* against puzzles whose depths are easy to verify
    test_cases = [("Already solved", GOAL_STATE, 0), ("One move", (1,0,2,3,4,5,6,7,8), 1), ("Two moves", (1,4,2,3,0,5,6,7,8), 2), ("Four moves", (1,2,5,3,4,8,6,7,0), 4), ("Unsolvable", (1,2,3,4,5,6,8,7,0), None),]

    for name, state, excepted_depth in test_cases:
        result = a_star_h1(state)

        if excepted_depth is None:
            assert result is None, f"{name} should be unsolvable"
        else:
            assert result is not None, f"{name} should have a solution"
            assert result.solution_depth == excepted_depth, (f"{name}: expected depth {excepted_depth}, " f"got{result.solution_depth}")
            assert result.path[0] == state
            assert result.path[-1] == GOAL_STATE

        print(f"Passed: {name}")

def run_demo() -> None:
    #solve sample state using h1
    sample_state = (3,1,2,6,4,5,7,8,0)

    print("\nInitial state:")
    print(format_state(sample_state))
    print("\nA* solution using h1:")
    print_solution(a_star_h1(sample_state))


if __name__ == "__main__":
    run_phase1_checks()
    print("\nA* h1 checks:")
    run_phase2_checks()
    run_demo()
