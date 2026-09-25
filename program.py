from typing import Iterable
from dataclasses import dataclass
import heapq
from itertools import count
from time import perf_counter
import sys
import random
from pathlib import Path
from statistics import mean
import csv
from contextlib import redirect_stdout

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

def a_star_h2(initial_state: tuple[int, ...]) -> SearchResult | None:
    #solve puzzle using A* and h2, the manhattan distance heuristic
    #same A* process as a_star_h1, but h2 estimates remaining work by adding the row and column distance of every numbered tile from its goal position

    checked_state = validate_state(initial_state)

    if not is_solvable(checked_state):
        return None
    
    start_time = perf_counter()

    #each heap item stores estimated total cost, tie number, and board
    states_waiting_to_be_explored = []
    tie_breaker = count()

    moves_taken_so_far = 0
    estimated_total_cost = moves_taken_so_far + manhattan_distance(checked_state)
    heapq.heappush(
        states_waiting_to_be_explored, (estimated_total_cost, next(tie_breaker), checked_state),
    )

    best_number_of_moves = {checked_state: moves_taken_so_far}
    previous_state = {checked_state: None}
    already_explored = set()
    nodes_generated = 0

    while states_waiting_to_be_explored:
        current_f_cost, _, current_state = heapq.heappop(
            states_waiting_to_be_explored
        )

        current_number_of_moves = best_number_of_moves[current_state]
        expected_f_cost = current_number_of_moves + manhattan_distance(current_state)

        # ignore outdated entry if cheaper path was found later
        if current_f_cost != expected_f_cost:
            continue
        
        if current_state in already_explored:
            continue
        already_explored.add(current_state)

        if current_state == GOAL_STATE:
            runtime_ms = (perf_counter() - start_time) * 1000
            return SearchResult(
                path=reconstruct_path(previous_state, current_state),
                nodes_generated=nodes_generated, 
                runtime_ms=runtime_ms,
            )

        for neighbor_state in get_neighbors(current_state):
            #this counts every neighbor produced while expanding a state
            nodes_generated += 1
            if neighbor_state in already_explored:
                continue
            
            number_of_moves_to_neighbor = current_number_of_moves + 1
            old_number_of_moves = best_number_of_moves.get(neighbor_state)

            if (old_number_of_moves is None or number_of_moves_to_neighbor < old_number_of_moves):
                best_number_of_moves[neighbor_state] = number_of_moves_to_neighbor
                previous_state[neighbor_state] = current_state
                neighbor_f_cost = (number_of_moves_to_neighbor + manhattan_distance(neighbor_state))
                heapq.heappush(states_waiting_to_be_explored, (neighbor_f_cost, next(tie_breaker), neighbor_state),)

    return None

def print_solution(result: SearchResult | None) -> None:
    #print each state after the initial state and the search stats
    if result is None:
        print("This puzzle isn't solvable")
        return

    for step_number, state in enumerate(result.path[1:], start=1):
        print(f"Step: {step_number}")
        print(format_state(state))

    print(f"Solution depth: {result.solution_depth}")
    print(f"Search Cost: {result.nodes_generated}")
    print(f"Time: {result.runtime_ms:.6f} ms")

class TranscriptStream:
    #send normal program output to both screen and a transcript output file

    def __init__(self, screen_stream, transcript_file):
        self.screen_stream = screen_stream
        self.transcript_file = transcript_file

    def write(self, text: str) -> int:
        self.screen_stream.write(text)
        self.transcript_file.write(text)
        self.transcript_file.flush()
        return len(text)

    def flush(self) -> None:
        self.screen_stream.flush()
        self.transcript_file.flush()

    def isatty(self) -> bool:
        return self.screen_stream.isatty()

def get_next_transcript_path() -> Path:
    #choose the next unused program_output(number).txt filename

    file_number = 1
    while True:
        possible_path = Path(f"program_output({file_number}).txt")
        if not possible_path.exists():
            return possible_path
        file_number += 1

def ask_user(prompt: str) -> str:
    # display an input prompt and save the user's response in the transcript

    # print the prompt separately so the transcript receives it before input
    print(prompt, end="", flush=True)
    answer = input()
    sys.stdout.write(f"{answer}\n")
    return answer.strip()

def create_random_puzzle() -> tuple[int, ...]:
    #create random, solvable puzzle board
    #random board generated first, if it's unsolvable, then 2 tiles are swapped

    random_puzzle = list(GOAL_STATE)
    random.shuffle(random_puzzle)

    if not is_solvable(random_puzzle):
        #these positions are guaranteed to contain numbered tiles because blank can appear in only on position 
        first_tile_index = 0
        second_tile_index = 1

        if random_puzzle[first_tile_index] == 0:
            first_tile_index = 2
        if random_puzzle[second_tile_index] == 0:
            second_tile_index = 2
        
        random_puzzle[first_tile_index], random_puzzle[second_tile_index] = (
            random_puzzle[second_tile_index],
            random_puzzle[first_tile_index],
        )

    return tuple(random_puzzle)

def read_user_input_puzzle() -> tuple[int, ...]:
    #read one puzzle from three input rows and validate it
    print("Enter your puzzle:")
    print("Use 0 for the blank tile")

    entered_numbers = []
    while len(entered_numbers) < TILE_COUNT:
        row_number = len(entered_numbers) // SIZE + 1
        row_text = ask_user(f"Row {row_number}: ")

        try:
            numbers_in_row = [int(value) for value in row_text.split()]
        except ValueError:
            print("Please enter numbers only, separated by spaces")
            continue
        
        if len(numbers_in_row) != SIZE:
            print("Each row must contain exactly three numbers")
            continue
        
        entered_numbers.extend(numbers_in_row)

    return validate_state(entered_numbers)

def read_puzzles_from_file(file_name: str) -> list[tuple[int, ...]]:
    #read every three row puzzle stored in one sample text file
    #collect every group of three valid rows
    file_path = Path(file_name.strip().strip('"'))

    if not file_path.is_file():
        raise FileNotFoundError(f"Could not find the file: {file_path}")

    puzzles_from_file = []
    current_puzzle_numbers = []

    with file_path.open("r", encoding="utf-8") as puzzle_file:
        for line in puzzle_file:
            stripped_line = line.strip()

            if not stripped_line or set(stripped_line) <= {"/"}:
                continue
            
            try:
                numbers_on_line = [int(value) for value in stripped_line.split()]
            except ValueError:
                #ignore labels or other non puzzle text 
                continue
            
            if len(numbers_on_line) != SIZE:
                continue
            
            current_puzzle_numbers.extend(numbers_on_line)

            if len(current_puzzle_numbers) == TILE_COUNT:
                puzzles_from_file.append(validate_state(current_puzzle_numbers))
                current_puzzle_numbers = []

    if current_puzzle_numbers:
        raise ValueError("The file ended with incomplete puzzle")

    if not puzzles_from_file:
        raise ValueError("No valid 8-puzzle configurations were found in file")

    return puzzles_from_file


def choose_puzzle_from_file() -> tuple[int, ...]:
    #Load a file and let user select one puzzle from it
    file_name = ask_user("Enter the sample file path: ")

    try:
        puzzles_in_file = read_puzzles_from_file(file_name)
    except (FileNotFoundError, ValueError) as error:
        print(f"Could not load file: {error}")
        raise ValueError from error

    print(f"Loaded {len(puzzles_in_file)} puzzles.")

    while True:
        puzzle_number_text = ask_user(
            f"Which puzzle do you want to solve? (1-{len(puzzles_in_file)}): "
        )

        try:
            puzzle_number = int(puzzle_number_text)
        except ValueError:
            print("Please enter a whole number.")
            continue

        if 1 <= puzzle_number <= len(puzzles_in_file):
            selected_puzzle = puzzles_in_file[puzzle_number - 1]
            print("\nSelected puzzle:")
            print(format_state(selected_puzzle))
            return selected_puzzle

        print(f"Please enter a number from 1 to {len(puzzles_in_file)}.")

def print_batch_analysis(results_by_depth: dict[int, dict[str, list[float]]]) -> None:
    #Print averages that can be copied into the project report
    print("\n100-Case Analysis Results")
    print("Depth | Cases | h1 Avg Nodes | h1 Avg ms | h2 Avg Nodes | h2 Avg ms")
    print("------|-------|--------------|-----------|--------------|----------")

    for solution_depth in sorted(results_by_depth):
        depth_results = results_by_depth[solution_depth]
        print(
            f"{solution_depth:5} | "
            f"{len(depth_results['h1_nodes']):5} | "
            f"{mean(depth_results['h1_nodes']):12.1f} | "
            f"{mean(depth_results['h1_time']):9.3f} | "
            f"{mean(depth_results['h2_nodes']):12.1f} | "
            f"{mean(depth_results['h2_time']):8.3f}"
        )

def save_analysis_csv(results_by_depth: dict[int, dict[str, list[float]]]) -> None:
    #save grouped averages into csv file

    with Path("analysis_results.csv").open( 
        "w", newline="", encoding="utf-8"
    ) as results_file:
        csv_writer = csv.writer(results_file)
        csv_writer.writerow(
            [
                "solution_depth", 
                "cases_tested",
                "h1_average_nodes",
                "h1_average_time_ms",
                "h2_average_nodes",
                "h2_average_time_ms",
            ]
        )

        for solution_depth in sorted(results_by_depth):
            depth_results = results_by_depth[solution_depth]
            csv_writer.writerow(
                [
                    solution_depth,
                    len(depth_results["h1_nodes"]),
                    f"{mean(depth_results['h1_nodes']):.2f}",
                    f"{mean(depth_results['h1_time']):.3f}",
                    f"{mean(depth_results['h2_nodes']):.2f}",
                    f"{mean(depth_results['h2_time']):.3f}",
                ]
            )
    print("Saved to analysis_results.csv")


def run_100_case_analysis() -> None:
    #analyze sample file puzzles and random puzzles
    
    sample_file_names = [
        "Length4(2).txt",
        "Length8(2).txt",
        "Length12(2).txt",
        "Length16(2).txt",
        "Length20(2).txt",
    ]

    all_test_puzzles = []
    expected_lengths = (4,8,12,16,20)

    for expected_length in expected_lengths:
        #accept both original names and uploaded names with (2)
        possible_file_names = (f"Length{expected_length}.txt", f"Length{expected_length}(2).txt")
        file_name = next(
            (
                possible_name 
                for possible_name in possible_file_names
                if Path(possible_name).is_file()
            ),
            possible_file_names[0],
        )

        try:
            puzzles_in_file = read_puzzles_from_file(file_name)
            all_test_puzzles.extend(puzzles_in_file)
            print(f"Loaded {len(puzzles_in_file)} puzzles from {file_name}")
        except (FileNotFoundError, ValueError) as error:
            print(f"Skipping {file_name}: {error}")

    while len(all_test_puzzles) < 100:
        all_test_puzzles.append(create_random_puzzle())

    # keep exactly 100 cases even if extra files are added later
    all_test_puzzles = all_test_puzzles[:100]
    results_by_depth = {}

    for case_number, starting_state in enumerate(all_test_puzzles, start=1):
        h1_result = a_star_h1(starting_state)
        h2_result = a_star_h2(starting_state)

        if h1_result is None or h2_result is None:
            print(f"Skipping unsolvable case {case_number}.")
            continue

        solution_depth = h1_result.solution_depth
        if solution_depth not in results_by_depth:
            results_by_depth[solution_depth] = {
                "h1_nodes": [],
                "h1_time": [],
                "h2_nodes": [],
                "h2_time": [],
            }

        results_by_depth[solution_depth]["h1_nodes"].append(
            h1_result.nodes_generated
        )
        results_by_depth[solution_depth]["h1_time"].append(h1_result.runtime_ms)
        results_by_depth[solution_depth]["h2_nodes"].append(
            h2_result.nodes_generated
        )
        results_by_depth[solution_depth]["h2_time"].append(h2_result.runtime_ms)

        if case_number % 10 == 0:
            print(f"Finished {case_number}/100 cases.")

    print_batch_analysis(results_by_depth)
    save_analysis_csv(results_by_depth)

def choose_heuristic() -> str:
    #ask user which heuristic A* should use
    while True:
        print("\nSelect the heuristic function:")
        print("(1) h1 - Misplaced tiles")
        print("(2) h2 - Manhattan distance")
        selected_option = ask_user("Enter your choice: ")

        if selected_option == "1":
            return "h1"
        if selected_option == "2":
            return "h2"
        
        print("Please enter 1 or 2")

def run_program() -> None:
    # Create one numbered transcript for this entire run
    transcript_path = get_next_transcript_path()

    with transcript_path.open("w", encoding="utf-8") as transcript_file:
        screen_and_transcript = TranscriptStream(
            sys.stdout,
            transcript_file
        )

        # keep displaying the program on screen while also saving it
        with redirect_stdout(screen_and_transcript):
            print(f"Program transcript: {transcript_path.name}")
            print("8-Puzzle A* Solver")
            print("Goal State:")
            print(format_state(GOAL_STATE))

            while True:
                while True:
                    print("\nChoose how to enter the puzzle:")
                    print("[1] Random")
                    print("[2] Manual Input")
                    print("[3] Load Puzzle from .txt File")
                    print("[4] Run 100-Case Analysis")
                    print("[5] Exit")

                    selected_option = ask_user("Enter your choice: ")

                    if selected_option == "1":
                        starting_state = create_random_puzzle()
                        print("\nPuzzle:")
                        print(format_state(starting_state))
                        break

                    if selected_option == "2":
                        try:
                            starting_state = read_user_input_puzzle()
                        except ValueError as error:
                            print(f"Invalid puzzle: {error}")
                            continue
                        break

                    if selected_option == "3":
                        try:
                            starting_state = choose_puzzle_from_file()
                        except ValueError:
                            continue
                        break

                    if selected_option == "4":
                        run_100_case_analysis()
                        continue

                    if selected_option == "5":
                        print("Exiting...")
                        print(f"Transcript saved to {transcript_path.name}")
                        return

                    print("Please enter 1, 2, 3, 4, or 5.")

                if not is_solvable(starting_state):
                    print(
                        "\nThis puzzle is not solvable because "
                        "it has odd inversion parity"
                    )
                    print("Returning to main menu")
                    continue

                selected_heuristic = choose_heuristic()

                if selected_heuristic == "h1":
                    solution_result = a_star_h1(starting_state)
                else:
                    solution_result = a_star_h2(starting_state)

                if solution_result is not None:
                    solution_depth = solution_result.solution_depth
                    if solution_result.solution_depth < 6:
                        print("\nDepth < 6")
                    elif 9 <= solution_depth <= 15:
                        print("\nDepth 9-15")
                    elif solution_depth > 18:
                        print("\nDepth > 18")
                    else:
                        print(f"\nSolution depth: {solution_depth}")

                print(f"\nA* solution using {selected_heuristic}:")
                print_solution(solution_result)

if __name__ == "__main__":
    run_program()
