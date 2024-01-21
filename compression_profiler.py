from enum import Enum
import json
import itertools
import time
import subprocess
import re
from pathlib import Path
import os
import signal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from profiler_arguments import *
from tqdm import tqdm

ABC_BINARY = Path("build/_deps/abc-build/abc")
ABC_ALIAS_SOURCE = Path("build/_deps/abc-src/abc.rc")
KNOR_BINARY = Path("build/knor")

UNMINIMIZED_AIG_FOLDER = Path("aigs_unminimized")
MINIMIZED_AIG_FOLDER = Path("aigs_minimized")
PROBLEM_FILES_FOLDER = Path("examples")

PROFILER_SOURCE = Path("profiler.json")

class SolveResult(Enum):
	TIMED_OUT = 1
	UNREALIZABLE = 2
	SOLVED = 3
	CRASHED = 4

class ProfilerData:
	def __init__(self, source: Path):
		self.source: Path = source
		if source.is_file():
			with open(source, "r") as file:
				self.data = json.load(file)
				print("Loaded profiler data from '{}'".format(source))
		else:
			self.data = { "problem_files": [] }
			print("Created new profiler data")
			self.save()

	# Saves current data to source file
	def save(self):
		print("Saving profiler data. Do not quit... ", end="", flush=True)
		with open(self.source, 'w') as file:
			json.dump(self.data, file, indent=3)
		print("Saved results in '{}'".format(self.source))

# ====================== Plotters ======================== #

def get_solve_attempt_with_args(solve_attempts: list[dict], args: list[str]) -> dict | None:
	""" Finds the solve attempt that matches the given optimization arguments """
	for attempt in solve_attempts:
		if attempt["args_used"] == args:
			return attempt
	return None

def get_optimization_with_args(optimizations: list[dict], args: list[str]) -> dict | None:
	""" Finds the optimization that matches the given optimization arguments """
	for opt in optimizations:
		if opt["args_used"] == args:
			return opt
	return None

def calculate_optimization_gain(previous_AND_gate_count: int, new_AND_gate_count: int) -> float:
	""" Calculates percentage of decrease in size, a.k.a. our view of what "gain" is """
	difference = previous_AND_gate_count - new_AND_gate_count
	gain = difference / previous_AND_gate_count
	return gain

# Returns dictionary collecting exploded results from repetition optimizations
def collect_duplication_data(problem_files: list[dict]) -> dict:
	duplication_data = {"head": [], "tail": [], "repetition": [], "gain": []}
	for arg_head in ABC_OPTIMIZATION_ARGUMENTS:
		for arg_tail in ABC_OPTIMIZATION_ARGUMENTS:
			if arg_head == arg_tail: continue

			for problem_file in problem_files:
				if problem_file["known_unrealizable"] == True: continue

				for solve_attempt in problem_file["solve_attempts"]:
					if solve_attempt["timed_out"] or solve_attempt["crashed"]: continue

					AND_count_history = [solve_attempt["data"]["and_gates"]]

					for repetition in range(REPETITION_TEST_MAX_REPETITION):
						test = [arg_head] * repetition + [arg_tail]
						optimization = get_optimization_with_args(solve_attempt["optimizations"], test)
						
						if not optimization: continue
						
						if optimization["timed_out"]:
							raise Exception("Data not available due to previous timed-out calculation")

						previous = AND_count_history[-1]
						current = optimization["data"]["and_gates"]
						AND_count_history.append(current)

						gain = 100 * calculate_optimization_gain(previous, current)

						duplication_data["head"].append(arg_head)
						duplication_data["tail"].append(arg_tail)
						duplication_data["repetition"].append(repetition)
						duplication_data["gain"].append(gain)

	return duplication_data

# Plots the repetition minimization results into a separate window
def plot_repetition_minimization_results():
	profiler = ProfilerData(PROFILER_SOURCE)
	sns.set_theme()
	data = collect_duplication_data(profiler.data["problem_files"])
	figure = sns.catplot(data=data, col="head", x="repetition", y="gain", hue="tail", kind="boxen")
	plt.show()

def collect_cleanup_data(problem_files: list[dict], solve_attempt_args: list[list[str]]) -> dict:
	cleanup_data = {"head"}


# =========================== Solvers ================================ #

def get_problem_file_paths() -> list[Path]:
	""" Gets list of all problem file paths.
		- Searches in the PROBLEM_FILES_FOLDER for '.ehoa' files
		- Returns list of Path objects to each file """
	problem_file_paths = []
	for item in Path(PROBLEM_FILES_FOLDER).iterdir():
		if item.is_file() and item.suffix == ".ehoa":
			problem_file_paths.append(item)
	return problem_file_paths

def initialize_problem_files(profiler: ProfilerData):
	""" Initializes all problem file entries in the given profiler data
		based on the problem files it can find."""
	problem_file_paths = get_problem_file_paths()

	for problem_file_path in problem_file_paths:

		problem_file_already_initialized = False
		for problem_file in profiler.data["problem_files"]:
			if problem_file["source"] == str(problem_file_path):
				problem_file_already_initialized = True
				break # This file is already initialized. No need to search further
		
		if not problem_file_already_initialized:
			profiler.data["problem_files"].append({
				"source": str(problem_file_path),
				"known_unrealizable": False,
				"solve_attempts": []
			})

# Checks if performing the given solve attempt will gain us new information
def is_solve_attempt_worth_it(problem_file: dict, knor_argument_combo: list[str], solve_timeout_seconds: int) -> bool:
	""" Tells if attempting to solve the given problem with given arguments will give new data.\n
		Returns True unless:
		- The problem is unrealizable
		- Previous solve attempt crashed
		- Previous solve attempt timed out, and this time no more time is allowed
		- Successfully solved it already"""
	if problem_file["known_unrealizable"]: return False

	previous_solve_attempt: dict | None = None
	for solve_attempt in problem_file["solve_attempts"]:
		if solve_attempt["args_used"] == knor_argument_combo:
			previous_solve_attempt = solve_attempt
			break
	
	# If we have not tried this yet, we will gain new information by trying
	if not previous_solve_attempt:
		return True
	else:
		# If we have tried before:
		# Do not retry if previous time crashed
		if previous_solve_attempt["crashed"]: return False
		
		# If previous try timed out, only retry if we have bigger timeout this time
		if previous_solve_attempt["timed_out"] and solve_timeout_seconds > previous_solve_attempt["solve_time"]: return True

		# Otherwise, we should not try again
		return False

def run_shell_command(cmd: str, timeout_seconds: float | None) -> subprocess.Popen[bytes] | None:
	""" Runs linux shell command with given timeout in seconds.
		No timeout is indicated with None. \n
		Returns process result or None on timeout. """
	# Runs the given shell command in terminal, timeout in seconds
	try:
		p = subprocess.Popen(cmd, start_new_session=True, shell=True, stdout=subprocess.PIPE)
		p.wait(timeout=timeout_seconds)
		return p
	except subprocess.TimeoutExpired:
		os.killpg(os.getpgid(p.pid), signal.SIGTERM)
		return None

def parse_aig_read_stats_output(cmd_output: str, verbose: bool=False) -> dict | None:
	""" Parses AIG stats from given output of a shell command.\n
	 	Looks for the first ABC 'print_stats' output and returns tuple of:
		- Number of AND gates,
		- Number of logic levels (a.k.a. depth of network),
		- Number of latches,
		- Number of inputs,
		- Number of outputs"""
	sanitized_output = cmd_output.replace(" ","")
	try:
		and_gates = int(re.findall("and=[0-9]*", sanitized_output)[0].split("=")[1])
		latches = int(re.findall("lat=[0-9]*", sanitized_output)[0].split("=")[1])
		levels = int(re.findall("lev=[0-9]*", sanitized_output)[0].split("=")[1])
		inputs, outputs = map(int, re.findall("i/o=[0-9]*/[0-9]*",sanitized_output)[0].split("=")[1].split("/"))
		
		return {
			"and_gates": and_gates,
			"levels": levels,
			"latches": latches,
			"inputs": inputs,
			"outputs": outputs
		}
	except:
		if verbose:
			tqdm.write("Things that went wrong parsing: \n")
			tqdm.write(cmd_output)
		else:
			tqdm.write("Failed to parse aiger output.")

def get_aig_stats_from_file(file: Path) -> dict | None:
	""" Reads given AIG file with ABC and returns its stats."""
	abc_read_cmd = "./{} -c 'read {}; print_stats'".format(ABC_BINARY, file)
	command_result = run_shell_command(abc_read_cmd, 10)
	if not command_result or not command_result.stdout: return None
	cmd_output = command_result.stdout.read().decode()
	return parse_aig_read_stats_output(cmd_output)

def solve_problem_files(
		problem_files: list[dict],
		knor_argument_combos: list[list[str]],
		verbose=True, # TODO: Implement this
		solve_timeout=MAX_TIME_SECONDS_FOR_KNOR_COMMAND):
	""" Solves each gven problem file with all given Knor argument combinations.
		Given timeout applies to the timeout of each solve attempt."""
	
	# Prepare the general output folder for solutions
	if not UNMINIMIZED_AIG_FOLDER.is_dir():
		UNMINIMIZED_AIG_FOLDER.mkdir()

	for problem_file in tqdm(problem_files, desc="problem_file", position=0, leave=False):
		problem_file_source = Path(problem_file["source"])
		if not problem_file_source.is_file():
			tqdm.write("Error: problem file '{}' not available".format(problem_file_source))
			#TODO: Check if this works

		# Skip this problem file if we already know it is unrealizable
		if problem_file["known_unrealizable"]:
			tqdm.write("Skipping '{}' because its unrealizable".format(problem_file_source))
			continue
		
		# Prepare the specific output folder for this problem file
		solution_output_folder = UNMINIMIZED_AIG_FOLDER / problem_file_source.name.rstrip("".join(problem_file_source.suffixes))
		if not solution_output_folder.is_dir(): solution_output_folder.mkdir()

		for knor_argument_combo in tqdm(knor_argument_combos, desc="arg_combination", position=1, leave=False):
			if not is_solve_attempt_worth_it(problem_file, knor_argument_combo, solve_timeout):
				tqdm.write("Skipping solving '{}' with args {} because not worth it".format(problem_file_source, knor_argument_combo))
				# TODO: Better feedback message
				continue
			
			output_file_name = problem_file_source.with_stem(problem_file_source.stem + "_args" + "".join(knor_argument_combo)).with_suffix(".aag" if "-a" in knor_argument_combo else ".aig").name
			output_file = solution_output_folder / output_file_name

			solve_command = "./{} {} {} > {}".format(KNOR_BINARY, problem_file_source, " ".join(knor_argument_combo), output_file)

			command_start = time.time()
			try:
				command_result = run_shell_command(solve_command, solve_timeout)
			except KeyboardInterrupt:
				tqdm.write("Aborted solving '{}' with {} after: {:.2f}s".format(problem_file_source, knor_argument_combo, time.time() - command_start))
				return
			command_time = time.time() - command_start

			solution_data: dict | None = None
			solve_result = SolveResult.SOLVED
			if not command_result:
				tqdm.write("Timeout for solving '{}' with args{} in: {:.2f}s".format(problem_file_source, knor_argument_combo, command_time))
				solve_result = SolveResult.TIMED_OUT
			elif command_result.returncode == 10:
				# Successfully solved problem, so read solution AIG data
				tqdm.write("Solved '{}' with args {} in {:.2f}s".format(problem_file_source, knor_argument_combo, command_time))
				solution_data = get_aig_stats_from_file(output_file)
			elif command_result.returncode == 20:
				tqdm.write("Unrealizable to solve '{}'. Took: {:.2f}s".format(problem_file_source, command_time))
				solve_result = SolveResult.UNREALIZABLE
			else:
				tqdm.write("Crashed from solving '{}' with args {} after: {:.2f}s".format(problem_file_source, knor_argument_combo, command_time))
				solve_result = SolveResult.CRASHED

			# === Store the new result
			
			if solve_result == SolveResult.UNREALIZABLE:
				problem_file["known_unrealizable"] = True
				# Do not try new knor solve combinations as this problem is unrealizable
				break
			
			# Check if given solve already happened
			previous_solve_attempt = None
			for solve_attempt in problem_file["solve_attempts"]:
				if solve_attempt["args_used"] == knor_argument_combo:
					previous_solve_attempt = solve_attempt
					break
			
			if not previous_solve_attempt:
				# This is a new attempt, so just append to existing list
				problem_file["solve_attempts"].append({
					"args_used": knor_argument_combo,
					"output_file": str(output_file),
					"command_used": solve_command,
					"timed_out": solve_result == SolveResult.TIMED_OUT,
					"crashed": solve_result == SolveResult.CRASHED,
					"solve_time": command_time,
					"data": solution_data,
					"optimizations": []
				})
			else:
				tqdm.write("Updated previous solve attempt of: '{}' with args: {}".format(problem_file_source, knor_argument_combo))
				# Retry attempt, so update previous attempt
				previous_solve_attempt["data"] = solution_data

				if solve_result == SolveResult.SOLVED or solve_result == SolveResult.UNREALIZABLE:
					previous_solve_attempt["timed_out"] = False
					previous_solve_attempt["crashed"] = False
					previous_solve_attempt["solve_time"] = command_time
				else: # Crashed or timed out
					previous_solve_attempt["timed_out"] = solve_result == SolveResult.TIMED_OUT
					previous_solve_attempt["crashed"] = solve_result == SolveResult.CRASHED
					# Set solve time to longest we have tried
					previous_solve_attempt["solve_time"] = max(previous_solve_attempt["solve_time"], command_time)

def find_new_optimization_id(solution: dict) -> int:
	""" Searches all existing optimization ids of the given solution and returns a new, highest one.
		This ID is used for unique optimization file names."""
	if not "optimizations" in solution: solution["optimizations"] = []

	# Get highest optimization id
	highest_optimization_id = 0
	for optimization in solution["optimizations"]:
		if optimization["id"] > highest_optimization_id:
			highest_optimization_id = optimization["id"]
	# Increment for next optimization
	new_optimization_id = highest_optimization_id + 1
	return new_optimization_id

def find_best_subset_optimization(optimizations: list[dict], wanted_args: list[str]) -> dict | None:	
	""" Goes through given optimizations, and finds the best one
		that can serve as a base for the wanted optimization.\n
		Example -> Solution1 has following optimizations:\n
			- ('A')
			- ('A', 'B')
		If target optimization is ('A', 'C'), this function should return the optimization
		with 'A', as now only the 'C' optimization has to be done, reusing previously
		calculated data.\n
		Returns None if no subset of given optimization could be found
		"""
	most_similar_optimization = None

	for optimization in optimizations:
		used_args = optimization["args_used"]
		# If more used args then wanted, this optimization is of no use
		if len(used_args) > len(wanted_args): continue
		
		found_inconsistency = False
		for i in range(len(used_args)):
			# If we encounter different argument, this optimization is not a subset
			if wanted_args[i] != used_args[i]:
				found_inconsistency = True
				break

		# If we looped through entire used_args, this optimization is a subset
		if not found_inconsistency:
			# If we do not have one yet, set it
			if not most_similar_optimization: most_similar_optimization = optimization
			# If we looped through all wanted args as well, this is exactly the same optimization
			elif len(used_args) > len(most_similar_optimization["args_used"]):
				most_similar_optimization = optimization

				if used_args == wanted_args: break
	return most_similar_optimization

def create_abc_optimization_command(
		source_file: Path,
		optimization_arguments: list[str],
		output_file: Path) -> str:
	""" Creates an ABC optimization command.
		- Source file is a path to an AIG file
		- Optimization_arguments is a list of optimize arguments that will be applied to the source
		- The result will be written to given output_file\n
		The created command also prints data of optimized result in terminal\n
		Returns the command"""

	all_arguments = [
		"read {}".format(source_file)			# Read the source file
	] + optimization_arguments + [				# Apply optimizations
		"time",									# Print time it took to optimize
		"write_aiger {}".format(output_file),	# Write to output file
		"print_stats"							# Print stats of optimized file
	]

	arguments_string = "; ".join(all_arguments)
	command = "./{} -c '{}'".format(ABC_BINARY, arguments_string)
	return command

def write_comment_to_aig_file(aig_file: Path, comment: str):
	""" Appends given comment to given AIG file."""
	with open(aig_file, "a") as file:
		file.write("c\n")
		file.write(comment)
		file.write("\n")

def execute_optimization_on_solution(solution: dict, arguments: list[str], output_folder: Path, optimize_timeout: float=MAX_TIME_SECONDS_FOR_OPTIMIZE_COMMAND):
	""" Executes the optimizations of given arguments if they have not been done before on the given solution.
		Ensures intermediate optimization results are stored as well for reuse later.
		Stores results in given output folder."""
	
	new_optimization_id = find_new_optimization_id(solution)

	arguments_build_up = []

	# Go through each argument and try to optimize the solution with the argument chain up till then 
	for argument in arguments:
		arguments_build_up.append(argument)

		# Find previous optimization attempt for these built-up arguments chain
		previous_optimize_attempt = get_optimization_with_args(solution["optimizations"], arguments_build_up)
		
		# If previous optimization attempt exists, only continue if it makes sense to try it again
		if previous_optimize_attempt:
			# If previous optimization was successful, we can go to the next argument
			if not previous_optimize_attempt["timed_out"]:
				# tqdm.write("Already completed optimization {} for solution {}".format(arguments_build_up, solution["args_used"]))
				continue
			# If previous attempt timed out and we do not have more time than then, this optimization cannot be done
			if optimize_timeout <= previous_optimize_attempt["optimize_time_python"]:
				tqdm.write("Cannot improve timed out optimize attempt {} on {}".format(arguments_build_up, solution["args_used"]))
				return

		# The file that needs to be optimized
		source_file: Path

		origin_arguments = arguments_build_up[:-1]

		# If no arguments, we must optimize the source problem file solution
		if not origin_arguments:
			source_file = Path(solution["output_file"])
		else:
			# Otherwise optimize a previous optimization with args "origin_arguments"
			origin_optimization = get_optimization_with_args(solution["optimizations"], origin_arguments)

			# If we failed to find the origin optimization, return
			if not origin_optimization:
				tqdm.write("Failed to find optimization base for {} ({}) on solution {}".format(arguments_build_up, origin_arguments, solution["args_used"]))
				return
			
			source_file = Path(origin_optimization["output_file"])

		# Check if source file actually still exists
		if not source_file.exists():
			tqdm.write("Source file missing: '{}'".format(source_file))
			return

		output_file = output_folder / "m{}.aig".format(new_optimization_id)
		command = create_abc_optimization_command(source_file, [argument], output_file)

		optimize_start = time.time()
		result = run_shell_command(command, MAX_TIME_SECONDS_FOR_OPTIMIZE_COMMAND)
		optimize_time = time.time() - optimize_start
		
		# Being unable to read the AIG data indicates an error occurred
		stats: dict | None = None
		if result and result.stdout:
			output = result.stdout.read().decode()
			stats = parse_aig_read_stats_output(output, True)
		
		if not result:
			tqdm.write("Timed out optimizing with {} on solution {} after {:.2f}s".format(arguments_build_up, solution["args_used"], optimize_time))

		# TODO: Check if optimization crashed
		if result and not stats:
			# That means we did not time out, but ABC gave an error
			tqdm.write("ABC command crashed on: \n{}".format(command))
		
		# If we retry optimization, update previous one. Otherwise, append it
		if previous_optimize_attempt:
			previous_optimize_attempt["command_used"] = command
			previous_optimize_attempt["output_file"] = output_file
			previous_optimize_attempt["args_used"] = arguments_build_up
			previous_optimize_attempt["optimize_time_python"] = optimize_time
			previous_optimize_attempt["actual_optimize_time"] = None
			previous_optimize_attempt["timed_out"] = result == None
			previous_optimize_attempt["data"] = stats
			previous_optimize_attempt["id"] = new_optimization_id
			tqdm.write("Retried optimization {} of {}".format(arguments_build_up, solution["args_used"]))
		else:
			optimization = {
				"command_used": command,
				"output_file": str(output_file),
				"args_used": arguments_build_up.copy(),
				"optimize_time_python": optimize_time,
				"actual_optimize_time": None,
				"timed_out": result == None,
				"data": stats,
				"id": new_optimization_id
			}
			solution["optimizations"].append(optimization)
			# tqdm.write("Did optimization {} of {}".format(arguments_build_up, solution["args_used"]))

		write_comment_to_aig_file(output_file, "Optimized with ABC arguments:\n{}".format(arguments))

		new_optimization_id += 1

def execute_optimizations_on_solutions(
		problem_files: list[dict],
		solve_argument_combos: list[list[str]] | None,
		optimization_argument_combos: list[list[str]],
		verbose=False,
		timeout=MAX_TIME_SECONDS_FOR_OPTIMIZE_COMMAND):
	""" Optimizes the given solutions (or all if "None") of the given problem files with the given optimization arguments"""
	# Ensure base folder for optimization results exists
	if not MINIMIZED_AIG_FOLDER.is_dir(): MINIMIZED_AIG_FOLDER.mkdir()
	
	for problem_file in tqdm(problem_files, desc="problem files", position=0):
		if problem_file["known_unrealizable"]:
			tqdm.write("Unrealizable problem file cannot be optimized: '{}'".format(problem_file["source"]))
			continue # We cannot optimize unrealizable solutions

		# Ensure problem file specific output folder
		problem_file_source = Path(problem_file["source"])
		problem_folder = MINIMIZED_AIG_FOLDER / problem_file_source.name.rstrip("".join(problem_file_source.suffixes))
		if not problem_folder.is_dir(): problem_folder.mkdir()

		# Get all solve attempts that would fit our wanted "solve_argument" combos
		matching_solve_attempts: list[dict] = []
		if not solve_argument_combos:
			# Select all solve attempts
			matching_solve_attempts = problem_file["solve_attempts"]
		else:
			# Find the solve attempts that match
			for solve_argument_combo in solve_argument_combos:
				matching = get_solve_attempt_with_args(problem_file["solve_attempts"], solve_argument_combo)
				if matching: matching_solve_attempts.append(matching)
				else: tqdm.write("Warning: Missing solve attempt {} for '{}'".format(solve_argument_combo, problem_file_source))

		# Filter out invalid solutions
		target_solutions = []
		for matching_solve_attempt in matching_solve_attempts:
			if matching_solve_attempt["crashed"] or matching_solve_attempt["timed_out"]:
				# We cannot optimize this solution, so skip it
				tqdm.write("Cannot optimize solution of '{}' with {} because the solve attempt {}".format(
					problem_file_source,
					matching_solve_attempt["args_used"],
					"crashed" if matching_solve_attempt["crashed"] else "timed out"))
			else:
				target_solutions.append(matching_solve_attempt)

		# Then apply optimizations to target solutions
		for target_solution in tqdm(target_solutions, desc="solution", position=1, leave=False):
			# Ensure solution specific output folder exists
			arguments_used_for_solution = target_solution["args_used"]
			solution_output_folder = problem_folder / "_".join(map(lambda x: x.replace("-",""), arguments_used_for_solution))
			if not solution_output_folder.is_dir(): solution_output_folder.mkdir()

			for optimization_argument_combo in tqdm(optimization_argument_combos, desc="optimization", position=2, leave=False):				
				try:
					execute_optimization_on_solution(target_solution, optimization_argument_combo, solution_output_folder)
				except KeyboardInterrupt:
					tqdm.write("Aborted optimizing '{}' with optimizations {}.".format(target_solution["output_file"], optimization_argument_combo))
					return

# ======================== Solver initiators ================ #

def get_problem_files(profiler: ProfilerData, regex: str | None = None) -> list[dict]:
	""" Returns a list of problem files whose name match the given regex, or all files if regex is None."""
	all_problem_files = profiler.data["problem_files"]

	if not regex: return all_problem_files

	matching_problem_files = []
	for problem_file in all_problem_files:
		problem_file_name = Path(problem_file["source"]).name

		if re.match(regex, problem_file_name):
			matching_problem_files.append(problem_file)

	return matching_problem_files

def get_knor_arguments_combinations(knor_strategy_args: list[str], oink_solve_args: list[str], binary_out: bool = True) -> list[list[str]]:
	""" Returns a list of all argument combinations to give to Knor.
		If binary_out is False, '-a' will be appended instead '-b'."""
	# Get all possible combinations of knor args
	knor_arg_combinations = []
	for i in range(len(knor_strategy_args) + 1):
		l = []
		for c in itertools.combinations(knor_strategy_args, i):
			l.append(c)
		knor_arg_combinations.extend(l)

	all_arg_combinations = []
	# Now, combine knor arg combos with every possible oink arg
	for oink_arg in oink_solve_args:
		for knor_arg_combo in knor_arg_combinations:
			new_combo = list(knor_arg_combo)
			new_combo.append(oink_arg)
			new_combo.append("-b" if binary_out else "-a")
			all_arg_combinations.append(new_combo)
	
	return all_arg_combinations

def solve_arbiter_problems():
	""" Solves all problem files with 'arbiter' in their name. """
	profiler = ProfilerData(PROFILER_SOURCE)
	initialize_problem_files(profiler)
	target_knor_args = get_knor_arguments_combinations(KNOR_ARGS, OINK_SOLVER_ARGS)
	target_problem_files = get_problem_files(profiler, ".*arbiter.*")
	solve_problem_files(target_problem_files, target_knor_args, solve_timeout=11)
	profiler.save()

def get_duplication_optimization_arguments(arguments: list[str], depth: int) -> list[list[str]]:
	""" Returns a list of arguments used for checking if duplicate sequential
		arguments have positive effect. """
	arguments_list = []
	for argument in arguments:
		arguments_list.append([argument])

		for heading_argument in arguments:
			if heading_argument == argument: continue
			for heading_length in range(1, depth):
				arguments_list.append([heading_argument] * heading_length + [argument])
	return arguments_list

def test_duplication_optimizations_on_arbiter_solutions():
	""" Optimizes all arbiter solutions to see if duplicate sequential optimizations make sense """
	profiler = ProfilerData(PROFILER_SOURCE)
	target_problem_files = get_problem_files(profiler, ".*arbiter.*")
	target_solve_attempts = get_knor_arguments_combinations(KNOR_ARGS, OINK_SOLVER_ARGS)
	target_optimization_arguments = get_duplication_optimization_arguments(list(ABC_OPTIMIZATION_ARGUMENTS.keys()), 4)
	execute_optimizations_on_solutions(target_problem_files, target_solve_attempts, target_optimization_arguments, timeout=50)
	profiler.save()

def solve_all_problem_files():
	""" Will create a solution for every possible example problem file. This takes looong..."""
	profiler = ProfilerData(PROFILER_SOURCE)
	problem_files = get_problem_files(profiler)
	arg_combos = get_knor_arguments_combinations(KNOR_ARGS, OINK_SOLVER_ARGS)
	solve_problem_files(problem_files, arg_combos)
	profiler.save()

def get_cleanup_optimizations_test_arguments():
	arg_combos = []
	for abc_arg in ABC_OPTIMIZATION_ARGUMENTS:
		arg_combos.append([abc_arg])
		arg_combos.append(["cleanup", abc_arg])
		arg_combos.append([abc_arg, "cleanup"])
	return arg_combos

def solve_cleanup_optimization_tests():
	profiler = ProfilerData(PROFILER_SOURCE)
	target_problem_files = get_problem_files(profiler, None)
	target_solutions = get_knor_arguments_combinations(KNOR_ARGS, OINK_SOLVER_ARGS)
	optimization_args = get_cleanup_optimizations_test_arguments()
	execute_optimizations_on_solutions(target_problem_files, target_solutions, optimization_args, timeout=50)
	profiler.save()

def get_range_of_numbers(minimum: int, maximum: int, steps: int) -> list[int]:
	""" Divides the given range in given steps. Returns list of integers within. """
	if steps <= 1: return [minimum]
	step_size = (maximum - minimum) / (steps - 1)
	nums = sorted(list(set([minimum+round(step_size*i) for i in range(steps)])))
	return nums

def get_parameter_flag_variations(flag: str, mutate: int, exponential: bool = False) -> list[str]:
	""" Gets variations of a single ABC optimization parameter flag. 
		Mutate indicates how many numbers requested for lower and higher flag values.
		Example: flag = '-N #8>=4<=16', mutate = 3
		Results in ['-N 4', '-N 6', '-N 8', '-N 12', '-N 16'] """
	base_flag = flag.split(" ")[0]
	if not '#' in flag: return [base_flag]

	nums = []
	default = int(re.findall("#-?[0-9]*", flag)[0][1:])
	minimum = int(re.findall(">=-?[0-9]*", flag)[0][2:])
	nums.extend(get_range_of_numbers(minimum, default, mutate))

	maxes = re.findall("<=-?[0-9]*", flag)
	maximum = int(maxes[0][2:]) if maxes else None

	if maximum:
		nums.extend(get_range_of_numbers(default, maximum, mutate))
	else:
		for i in range(mutate):
			if exponential:
				# Add exponential -> 10, 100, 10000, 100000000...
				nums.append(max(10, max(nums))**2)
			else:
				# Add double -> 2, 4, 8, 16, 32, 64...
				nums.append(max(1, 2 * max(nums)))

	nums = sorted(list(set(nums))) # Remove duplicates and sorts for convenience
	return ["{} {}".format(base_flag, x) for x in nums]

def get_abc_argument_flag_combinations(argument_base: str, flags: list[str], mutate: int, exponential: bool = False) -> list[str]:
	""" Gets variations of a single ABC optimization argument.\n
		Argument base example: 'b'\n
		Flags example: ['-l', '-d', '-s']"""
	if not flags: return [argument_base]

	flags_variations: list[list[str]] = [get_parameter_flag_variations(flag, mutate, exponential) for flag in flags]
	
	combos = flags_variations[0]
	for flag_variations in flags_variations[1:]:
		combos.extend(map(lambda x: " ".join(x), itertools.product(combos, flag_variations)))
		combos.extend(flag_variations)
	
	argument_combos = [argument_base] + ["{} {}".format(argument_base, combo) for combo in combos]
	return argument_combos

def get_all_abc_opt_arguments(mutate: int) -> list[str]:
	args = []
	for abc_arg in ABC_OPTIMIZATION_ARGUMENTS:
		args.extend(get_abc_argument_flag_combinations(abc_arg, ABC_OPTIMIZATION_ARGUMENTS[abc_arg], mutate))
	return args

# p = ProfilerData(PROFILER_SOURCE)

# def optimize_for_test_1(profiler):
# 	""" See if trim optimization has effect. """
# 	a = get_target_problem_files(profiler)
# 	b = get_target_solve_attempt_arguments()
# 	trim_variants = get_abc_argument_flag_combinations("trim", CLEANUPS["trim"], 0)
# 	abc_args = []
# 	for trim_variant in trim_variants:
# 		abc_args.append([trim_variant])
# 		abc_args.append([trim_variant, "drw"])
# 		abc_args.append(["drw", trim_variant])
# 	execute_optimizations_on_solutions(a, b, abc_args, timeout=60)

# def show_test_1(profiler):
# 	a = get_target_problem_files(profiler)
# 	b = get_target_solve_attempt_arguments()

# 	for problem_file in a:
# 		for solve_attempt in problem_file["solve_attempts"]:
# 			if not solve_attempt["args_used"] in b: continue
# 			pass

def get_target_problem_files(profiler: ProfilerData) -> list[dict]:
	return get_problem_files(profiler, "full_arbiter_[0-9]")
def get_target_solve_attempt_arguments() -> list[list[str]]:
	return get_knor_arguments_combinations(KNOR_ARGS, OINK_SOLVER_ARGS)

# Test 1: Compare rw and drw optimizations performance
def solve_for_test_1():
	profiler = ProfilerData(PROFILER_SOURCE)
	a = get_target_problem_files(profiler)
	b = get_target_solve_attempt_arguments()
	solve_problem_files(a, b, solve_timeout=60)
	profiler.save()

def optimize_for_test_1():
	""" See rewrite(rw) or dag-aware rewrite(drw) is better. """
	profiler = ProfilerData(PROFILER_SOURCE)
	a = get_target_problem_files(profiler)
	b = get_target_solve_attempt_arguments()
	rw_variants = get_abc_argument_flag_combinations("rw", ABC_OPTIMIZATION_ARGUMENTS["rw"], 3)
	drw_variants = get_abc_argument_flag_combinations("drw", ABC_OPTIMIZATION_ARGUMENTS["drw"], 3)
	joint_args = list(map(lambda x: [x], rw_variants + drw_variants))
	execute_optimizations_on_solutions(a, b, joint_args, timeout=60)
	profiler.save()

def show_test_1(profiler: ProfilerData, n: int = 5):
	""" Plots the top n best average arguments for rw and drw. """
	problem_files = get_target_problem_files(profiler)
	solve_attempt_arg_combos = get_target_solve_attempt_arguments()
	rw_variants = get_abc_argument_flag_combinations("rw", ABC_OPTIMIZATION_ARGUMENTS["rw"], 3)
	drw_variants = get_abc_argument_flag_combinations("drw", ABC_OPTIMIZATION_ARGUMENTS["drw"], 3)
	optimize_arg_combos = list(map(lambda x: [x], rw_variants + drw_variants))
	
	raw_data = {
		"solve_args": [],
		"opt_arg_bases": [],
		"opt_arg_flags": [],
		"gains": [],
		"times": []
	}

	for problem_file in problem_files:
		for solve_attempt in problem_file["solve_attempts"]:
			if not solve_attempt["args_used"] in solve_attempt_arg_combos: continue
			if not solve_attempt["data"]: continue

			solve_args_used = " ".join(solve_attempt["args_used"])

			base_and_count = solve_attempt["data"]["and_gates"]

			for optimization in solve_attempt["optimizations"]:
				if not optimization["args_used"] in optimize_arg_combos: continue
				if len(optimization["args_used"]) != 1: raise Exception("Expected only one argument")

				argument = optimization["args_used"][0]
				arg_base: str = argument.split(" ")[0]
				flags: str = " ".join(argument.split(" ")[1:])

				new_and_count = optimization["data"]["and_gates"]
				gain = base_and_count / new_and_count

				raw_data["solve_args"].append(solve_args_used)
				raw_data["opt_arg_bases"].append(arg_base)
				raw_data["opt_arg_flags"].append(flags)
				raw_data["gains"].append(gain)
				raw_data["times"].append(optimization["optimize_time_python"])

	data = pd.DataFrame(raw_data)

	# Get average of each argument over each solution
	averaged_gain_over_files = data.groupby(["opt_arg_bases", "opt_arg_flags"]).agg({"gains": "mean"})
	# Get the top 5 of each argument base with the highest gain
	top_argument_df = averaged_gain_over_files.sort_values("gains", ascending=False).groupby(["opt_arg_bases"]).head(5).sort_values("opt_arg_bases").reset_index()

	top_arguments_found = list(top_argument_df["opt_arg_bases"] + " " + top_argument_df["opt_arg_flags"])
	
	# Add complete optimization argument as column
	data["argument"] = (data["opt_arg_bases"] + " " + data["opt_arg_flags"]).apply(lambda x: x.strip())
	
	plot_data = data[data["argument"].isin(top_arguments_found)]

	sns.set_theme()
	sns.catplot(data=plot_data, kind="violin", x="opt_arg_flags", y="gains", hue="opt_arg_bases")
	plt.xticks(rotation=45)
	plt.show()

# Test 2: Compare performance of rf and drf optimizations
def solve_for_test_2():
	profiler = ProfilerData(PROFILER_SOURCE)
	a = get_target_problem_files(profiler)
	b = get_target_solve_attempt_arguments()
	solve_problem_files(a, b, solve_timeout=60)
	profiler.save()

def optimize_for_test_2():
	""" See if refactor (rf) or dag-aware refactor (drf) is more effective """
	profiler = ProfilerData(PROFILER_SOURCE)
	a = get_target_problem_files(profiler)
	b = get_target_solve_attempt_arguments()
	rw_variants = get_abc_argument_flag_combinations("rf", ABC_OPTIMIZATION_ARGUMENTS["rf"], 3)
	drw_variants = get_abc_argument_flag_combinations("drf", ABC_OPTIMIZATION_ARGUMENTS["drf"], 3)
	joint_args = list(map(lambda x: [x], rw_variants + drw_variants))
	execute_optimizations_on_solutions(a, b, joint_args, timeout=60)
	profiler.save()

def show_test_2(profiler: ProfilerData, n: int = 5):
	""" Plots the top n best average rf and top n best average drf """
	problem_files = get_target_problem_files(profiler)
	solve_attempt_arg_combos = get_target_solve_attempt_arguments()
	rf_variants = get_abc_argument_flag_combinations("rf", ABC_OPTIMIZATION_ARGUMENTS["rf"], 3)
	drf_variants = get_abc_argument_flag_combinations("drf", ABC_OPTIMIZATION_ARGUMENTS["drf"], 3)
	optimize_arg_combos = list(map(lambda x: [x], rf_variants + drf_variants))
	
	raw_data = {
		"solve_args": [],
		"opt_arg_bases": [],
		"opt_arg_flags": [],
		"gains": [],
		"times": []
	}

	for problem_file in problem_files:
		for solve_attempt in problem_file["solve_attempts"]:
			if not solve_attempt["args_used"] in solve_attempt_arg_combos: continue
			if not solve_attempt["data"]: continue

			solve_args_used = " ".join(solve_attempt["args_used"])

			base_and_count = solve_attempt["data"]["and_gates"]

			for optimization in solve_attempt["optimizations"]:
				if not optimization["args_used"] in optimize_arg_combos: continue
				if len(optimization["args_used"]) != 1: raise Exception("Expected only one argument")

				argument = optimization["args_used"][0]
				arg_base: str = argument.split(" ")[0]
				flags: str = " ".join(argument.split(" ")[1:])

				new_and_count = optimization["data"]["and_gates"]
				gain = base_and_count / new_and_count

				raw_data["solve_args"].append(solve_args_used)
				raw_data["opt_arg_bases"].append(arg_base)
				raw_data["opt_arg_flags"].append(flags)
				raw_data["gains"].append(gain)
				raw_data["times"].append(optimization["optimize_time_python"])

	data = pd.DataFrame(raw_data)

	# Get average of each argument over each solution
	averaged_gain_over_files = data.groupby(["opt_arg_bases", "opt_arg_flags"]).agg({"gains": "mean"})
	# Get the top 5 of each argument base with the highest gain
	top_argument_df = averaged_gain_over_files.sort_values("gains", ascending=False).groupby(["opt_arg_bases"]).head(5).sort_values("opt_arg_bases").reset_index()

	top_arguments_found = list(top_argument_df["opt_arg_bases"] + " " + top_argument_df["opt_arg_flags"])
	
	# Add complete optimization argument as column
	data["argument"] = (data["opt_arg_bases"] + " " + data["opt_arg_flags"]).apply(lambda x: x.strip())
	
	plot_data = data[data["argument"].isin(top_arguments_found)]

	sns.set_theme()
	sns.catplot(data=plot_data, kind="violin", x="opt_arg_flags", y="gains", hue="opt_arg_bases")
	plt.xticks(rotation=45)
	plt.show()

# Test 3: Test if other optimizations are any good
def solve_for_test_3():
	profiler = ProfilerData(PROFILER_SOURCE)
	a = get_target_problem_files(profiler)
	b = get_target_solve_attempt_arguments()
	solve_problem_files(a, b, solve_timeout=60)
	profiler.save()

def optimize_for_test_3():
	""" Test if one of the other optimization arguments are good. """
	profiler = ProfilerData(PROFILER_SOURCE)
	a = get_target_problem_files(profiler)
	b = get_target_solve_attempt_arguments()

	opt_args = []
	for arg in ["drwsat", "rs", "dc2", "irw", "irws", "iresyn"]:
		opt_args.extend(get_abc_argument_flag_combinations(arg, ABC_OPTIMIZATION_ARGUMENTS[arg], 3))
	opt_args = list(map(lambda x: [x], opt_args))
	
	execute_optimizations_on_solutions(a, b, opt_args, timeout=60)
	profiler.save()

def show_test_3(profiler: ProfilerData):
	pass # TODO: Implement this

# Test 4: Test balance performance. Probably not amazing
def solve_for_test_4():
	profiler = ProfilerData(PROFILER_SOURCE)
	a = get_target_problem_files(profiler)
	b = get_target_solve_attempt_arguments()
	solve_problem_files(a, b, solve_timeout=60)
	profiler.save()

def optimize_for_test_4():
	""" Optimizes all target solutions with the balance optimization """
	profiler = ProfilerData(PROFILER_SOURCE)
	a = get_target_problem_files(profiler)
	b = get_target_solve_attempt_arguments()
	balances = list(map(lambda x: [x], get_abc_argument_flag_combinations("b", ABC_OPTIMIZATION_ARGUMENTS["b"], 3)))
	execute_optimizations_on_solutions(a, b, balances, timeout=60)
	profiler.save()

def show_test_4(profiler: ProfilerData):
	pass # TODO: Implement this

# Test 5: Are extreme drw flags better than small normal drw flags?
def solve_for_test_5():
	profiler = ProfilerData(PROFILER_SOURCE)
	a = get_target_problem_files(profiler)
	b = get_target_solve_attempt_arguments()
	solve_problem_files(a, b, solve_timeout=60)
	profiler.save()

def optimize_for_test_5():
	""" See if drw performs better with extremely big parameters """
	profiler = ProfilerData(PROFILER_SOURCE)
	a = get_target_problem_files(profiler)
	b = get_target_solve_attempt_arguments()
	extreme_drw_variants = get_abc_argument_flag_combinations("drw", ABC_OPTIMIZATION_ARGUMENTS["drw"], 3, True)
	joint_args = list(map(lambda x: [x], extreme_drw_variants))
	execute_optimizations_on_solutions(a, b, joint_args, timeout=60)
	profiler.save()

def show_test_5(profiler: ProfilerData):
	pass # TODO: Implement

# Test 6: Are extreme drf flags better than small normal drf flags?
def solve_for_test_6():
	profiler = ProfilerData(PROFILER_SOURCE)
	a = get_target_problem_files(profiler)
	b = get_target_solve_attempt_arguments()
	solve_problem_files(a, b, solve_timeout=60)
	profiler.save()

def optimize_for_test_6():
	""" See if drf performs better with extremely big parameters """
	profiler = ProfilerData(PROFILER_SOURCE)
	a = get_target_problem_files(profiler)
	b = get_target_solve_attempt_arguments()
	extreme_drw_variants = get_abc_argument_flag_combinations("drf", ABC_OPTIMIZATION_ARGUMENTS["drf"], 3, True)
	joint_args = list(map(lambda x: [x], extreme_drw_variants))
	execute_optimizations_on_solutions(a, b, joint_args, timeout=60)
	profiler.save()

def show_test_6(profiler: ProfilerData):
	pass # TODO: Implement

# # Chaining tests
# def solve_for_test_5():
# 	pass

# def optimize_for_test_5():
# 	""" Optimizes all to see if duplicate sequential optimizations make sense """
# 	profiler = ProfilerData(PROFILER_SOURCE)
# 	target_problem_files = get_problem_files(profiler, ".*arbiter.*")
# 	target_solve_attempts = get_knor_arguments_combinations(KNOR_ARGS, OINK_SOLVER_ARGS)
# 	target_optimization_arguments = get_duplication_optimization_arguments(list(ABC_OPTIMIZATION_ARGUMENTS.keys()), 4)
# 	execute_optimizations_on_solutions(target_problem_files, target_solve_attempts, target_optimization_arguments, timeout=50)
# 	profiler.save()

# ========================================================================================

def check_aig_data_structure(data: dict, source: str):
	if not "and_gates" in data: raise Exception("AIG data missing 'and_gates' attribute of source: {}".format(source))
	if not isinstance(data["and_gates"], int): raise Exception("AIG data had invalid 'and_gates' attribute of source: {}".format(source))
	if not "levels" in data: raise Exception("AIG data missing 'levels' attribute of source: {}".format(source))
	if not isinstance(data["levels"], int): raise Exception("AIG data had invalid 'levels' attribute of source: {}".format(source))
	if not "latches" in data: raise Exception("AIG data missing 'latches' attribute of source: {}".format(source))
	if not isinstance(data["latches"], int): raise Exception("AIG data had invalid 'latches' attribute of source: {}".format(source))
	if not "inputs" in data: raise Exception("AIG data missing 'inputs' attribute of source: {}".format(source))
	if not isinstance(data["inputs"], int): raise Exception("AIG data had invalid 'inputs' attribute of source: {}".format(source))
	if not "outputs" in data: raise Exception("AIG data missing 'outputs' attribute of source: {}".format(source))
	if not isinstance(data["outputs"], int): raise Exception("AIG data had invalid 'outputs' attribute of source: {}".format(source))

def check_optimization_structure(optimization: dict,
								 handled_optimization_args: list[list[str]],
								 handled_optimization_ids: list[int],
								 source: str):
	if not "args_used" in optimization: raise Exception("Missing 'args_used' attribute in optimization of source: {}".format(source))
	if not isinstance(optimization["args_used"], list): raise Exception("Optimization of solution had invalid 'args_used' of source: {}".format(source))
	if not all(isinstance(x, str) for x in optimization["args_used"]): raise Exception("Invalid argument of 'args_used' in optimization of source: {}".format(source))

	opt_args_used = optimization["args_used"]
	if opt_args_used in handled_optimization_args: raise Exception("Optimization duplicated: {} of source: {}".format(opt_args_used, source))
	handled_optimization_args.append(opt_args_used)

	if not "command_used" in optimization: raise Exception("Missing 'command_used' attrbibute in optimization of source: {}".format(opt_args_used, source))
	if not isinstance(optimization["command_used"], str): raise Exception("Invalid type for 'command_used' in opt {} of source: {}".format(opt_args_used, source))
	if not "output_file" in optimization: raise Exception("Missing 'output_file' attrbibute in optimization {} of source: {}".format(opt_args_used, source))
	if not isinstance(optimization["output_file"], str): raise Exception("Invalid type for 'output_file' in opt {} of source: {}".format(opt_args_used, source))
	if not "optimize_time_python" in optimization: raise Exception("Missing 'optimize_time_python' attrbibute in optimization {} of source: {}".format(opt_args_used, source))
	if not isinstance(optimization["optimize_time_python"], float): raise Exception("Invalid type for 'optimize_time_python' in opt {} of source: {}".format(opt_args_used, source))
	if not "actual_optimize_time" in optimization: raise Exception("Missing 'actual_optimize_time' attrbibute in optimization {} of source: {}".format(opt_args_used, source))
	# Actual optimize time may be None
	if not "timed_out" in optimization: raise Exception("Missing 'timed_out' attrbibute in optimization {} of source: {}".format(opt_args_used, source))
	if not isinstance(optimization["timed_out"], bool): raise Exception("Invalid type for 'timed_out' in opt {} of source: {}".format(opt_args_used, source))
	if not "id" in optimization: raise Exception("Missing 'id' attrbibute in optimization {} of source: {}".format(opt_args_used, source))
	if not isinstance(optimization["id"], int): raise Exception("Invalid type for 'id' in opt {} of source: {}".format(opt_args_used, source))

	if optimization["id"] in handled_optimization_ids: raise Exception("Duplicate optimization id: {} in optimization {} of source: {}".format(optimization["id"], opt_args_used, source))
	handled_optimization_ids.append(optimization["id"])

	if not "data" in optimization: raise Exception("Missing 'data' attribute in optimization {} of source: {}".format(opt_args_used, source))
	
	if optimization["timed_out"]:
		if isinstance(optimization["data"], dict): raise Exception("Failed optimization still has data in optimization {} of source: {}".format(opt_args_used, source))
	else:
		if not isinstance(optimization["data"], dict): raise Exception("Successful optimization does not have data in optimization {} of source: {}".format(opt_args_used, source))
		check_aig_data_structure(optimization["data"], "optimization {} of {}".format(opt_args_used, source))

def check_solve_attempt_structure(solve_attempt: dict,
								  handled_solve_attempt_args: list[list[str]],
								  source: str):
	if not "args_used" in solve_attempt: raise Exception("Missing 'args_used' attribute in solve attempt of '{}'".format(source))
	if not type(solve_attempt["args_used"]) is list: raise Exception("Problem file '{}' had solution with invalid type for 'args_used' instead of list".format(source))
	
	solve_args_used = solve_attempt["args_used"]
	if solve_args_used in handled_solve_attempt_args: raise Exception("Solve attempt duplicated: {} for '{}'".format(solve_args_used, source))
	handled_solve_attempt_args.append(solve_args_used)

	if not all(isinstance(x, str) for x in solve_args_used): raise Exception("Problem file '{}' had solve attempt with invalid type in 'args_used' instead of str".format(source))

	if not "timed_out" in solve_attempt: raise Exception("Missing 'timed_out' attribute in solve attempt of '{}' with args {}".format(source, solve_args_used))
	if not isinstance(solve_attempt["timed_out"], bool): raise Exception("Solve attempt of '{}' with args {} had invalid type for 'timed_out' instead of bool".format(source, solve_args_used))
	if not "crashed" in solve_attempt: raise Exception("Missing 'crashed' attribute in solve attempt of '{}' with args {}".format(source, solve_args_used))
	if not isinstance(solve_attempt["crashed"], bool): raise Exception("Solve attempt of '{}' with args {} had invalid type for 'crashed' instead of bool".format(source, solve_args_used))
	if not "solve_time" in solve_attempt: raise Exception("Missing 'solve_time' attribute in solve attempt of '{}' with args {}".format(source, solve_args_used))
	if not isinstance(solve_attempt["solve_time"], float): raise Exception("Solve attempt of '{}' with args {} had invalid type for 'solve_time' instead of float".format(source, solve_args_used))
	if not "data" in solve_attempt: raise Exception("Missing 'data' attribute in solve attempt of '{}' with args {}".format(source, solve_args_used))
	
	# If this solve attempt crashed or timed out, there is not much else to see
	if solve_attempt["crashed"] or solve_attempt["timed_out"]: return
	
	check_aig_data_structure(solve_attempt["data"], "solution {} of problem file '{}'".format(solve_args_used, source))

	if not "optimizations" in solve_attempt: raise Exception("Missing 'optimizations' attribute in solve attempt of '{}' with args {}".format(source, solve_args_used))
	if not isinstance(solve_attempt["optimizations"], list): raise Exception("Solve attempt of '{}' with args {} had invalid type for 'optimizations' instead of list".format(source, solve_args_used))

	handled_optimization_args: list[list[str]] = []
	handled_optimization_ids: list[int] = []

	for optimization in solve_attempt["optimizations"]:
		check_optimization_structure(optimization, handled_optimization_args, handled_optimization_ids, "solution {} of problem file '{}'".format(solve_args_used, source))

def check_problem_file_structure_correctness(problem_files: list[dict]):
	""" Prints all errors of the given profilers data."""
	handled_problem_file_sources: list[str] = []

	for problem_file in problem_files:
		if not "source" in problem_file: raise Exception("Missing 'source' attribute in problem file")
		if not type(problem_file["source"]) is str: raise Exception("Problem file 'source' type was not str")

		source = problem_file["source"]
		if source in handled_problem_file_sources: raise Exception("Problem file is duplicate: '{}'".format(source))
		handled_problem_file_sources.append(source)

		if not "known_unrealizable" in problem_file: raise Exception("Missing 'known_unrealizable' attribute in '{}'".format(source))
		if not type(problem_file["known_unrealizable"]) is bool: raise Exception("Problem file '{}' had invalid type for 'known_unrealizable' instead of boolean".format(source))
		if not "solve_attempts" in problem_file: raise Exception("Missing 'solve_attempts' attribute in '{}'".format(source))
		if not type(problem_file["solve_attempts"]) is list: raise Exception("Problem file '{}' had invalid type for 'solve_attempts' instead of list".format(source))

		handled_solve_attempt_args: list[list[str]] = []

		if not problem_file["known_unrealizable"]:
			for solve_attempt in problem_file["solve_attempts"]:
				check_solve_attempt_structure(solve_attempt, handled_solve_attempt_args, "problem file '{}'".format(source))

	print("Done")

def check_profiler_structure():
	profiler = ProfilerData(PROFILER_SOURCE)
	check_problem_file_structure_correctness(profiler.data["problem_files"])

def fix_missing_attributes(profiler: ProfilerData):
	problem_files = get_problem_files(profiler)
	for problem_file in tqdm(problem_files, desc="problem files", position=0):
		if problem_file["known_unrealizable"]: continue

		source = problem_file["source"]
		for solve_attempt in tqdm(problem_file["solve_attempts"], desc="solve_attempt", position=1):
			if not "data" in solve_attempt:
				solve_attempt["data"] = None
				tqdm.write("Added 'data' attribute to: {} of {}".format(solve_attempt["args_used"], source))

			if solve_attempt["crashed"] or solve_attempt["timed_out"]: continue

			if not solve_attempt["data"]:
				# Then this solve attempt needs to have data!
				stats = get_aig_stats_from_file(solve_attempt["output_file"])
				if not stats: raise Exception("Should have data, but failed to get stats in solve attempt: {} of problem '{}'".format(solve_attempt["args_used"], source))
				solve_attempt["data"] = stats
				tqdm.write("Read and set AIG stats of solve attempt {} of problem '{}'".format(solve_attempt["args_used"], source))

			if not "optimizations" in solve_attempt:
				solve_attempt["optimizations"] = []
				tqdm.write("Added empty optimizations list to solve attempt {} of problem {}".format(solve_attempt["args_used"], source))

			for optimization in solve_attempt["optimizations"]:
				if optimization["timed_out"]: continue
				
				if not "data" in optimization:
					optimization["data"] = None

				if not optimization["data"]:
					# Then this optimization needs to have data!
					stats = get_aig_stats_from_file(optimization["output_file"])
					if not stats: print("Should have data, but failed to get stats in optimization: {} of solution {} of problem '{}'".format(optimization["args_used"], solve_attempt["args_used"], source))
					optimization["data"] = stats
					tqdm.write("Read and set AIG stats of opt {} of solution {} of problem '{}'".format(optimization["args_used"], solve_attempt["args_used"], source))
		