# CS 4200 P1

This project implements A* search for the 8-puzzle using two heuristic functions:

* `h1` : number of misplaced numbered tiles
* `h2` : total Manhattan distance of the numbered tiles from their goal positions

The goal state is:

```
0 1 2
3 4 5
6 7 8
```

The `0` represents the blank tile.

Files

* `program.py` - main source code
* `Length4.txt` - sample puzzles with a solution length of 4
* `Length8.txt` - sample puzzles with a solution length of 8
* `Length12.txt` - sample puzzles with a solution length of 12
* `Length16.txt` - sample puzzles with a solution length of 16
* `Length20.txt` - sample puzzles with a solution length of 20
* `analysis_results.csv` - table with generated averages from the 100 case analysis
* `program_output(#).txt` - numbered transcript of interactive program run

# How to run the program?

Run the program with:

`python program.py`

The menu provides the following choices:

1. Generate a random solvable puzzle
2. Enter a puzzle manually as three rows of three numbers
3. Load a puzzle from one of the sample .txt files and select a puzzle
4. Run the 100 case analysis and generate its result table
5. Exit

For option 2 (manual input), reference the following example for format:

```
3 1 2
6 4 5
7 8 0
```

After selecting a puzzle, choose `1` for h1 or `2` for h2. The program prints every state in the solution, the solution depth, the number of generated nodes, and the runtime.

### 100 case analysis

The anaylsis uses 50 supplied sample puzzles and adds 50 randomly generated solvable puzzles. It groups results by actual solution depth and calculates the average search cost and average runtime for h1 and h2. The results are printed to the console and saved to `analysis_results.csv`

Each time the program starts, it creates the next unused transcript file, such as `program_output(1).txt` or `program_output(2).txt`. The transcript records the menu conversation, the user's entries, selected puzzle, heuristic, every solution step, search statistics, analysis messages, and the exit message. The program returns to the main menu after each puzzle, file load, or analysis run and keeps running until the user selects option `5`.
