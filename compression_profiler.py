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

	# Inserts given command result into data
	def handle_command_result(
			self,
			problem_source: Path,
			args_used: list[str],
			command_used: str,
			output_file: Path,
			solve_time: float,
			result: SolveResult
			):
		for problem_file in self.data["problem_files"]:
			if problem_file["source"] == str(problem_source):
				# If we know problem is unrealizable, set it
				problem_file["known_unrealizable"] |= result == SolveResult.UNREALIZABLE
				
				# Check if given solve already happened
				for attempt in problem_file["solve_attempts"]:
					if attempt["args_used"] == args_used:
						attempt["command_used"] = command_used
						attempt["output_file"] = str(output_file)
						# We already did this command before
						if result == SolveResult.SOLVED or result == SolveResult.UNREALIZABLE:
							attempt["timed_out"] = False
							attempt["crashed"] = False
							# Save fastest solve/unrealizable
							attempt["solve_time"] = min(attempt["solve_time"], solve_time)
						else: # Crashed or timed out
							attempt["timed_out"] = result == SolveResult.TIMED_OUT
							attempt["crashed"] = result == SolveResult.CRASHED
							# Set solve time to longest we have tried
							attempt["solve_time"] = max(attempt["solve_time"], solve_time)
						return
				# Otherwise, add solve to problem file
				problem_file["solve_attempts"].append({
					"args_used": args_used,
					"output_file": str(output_file),
					"command_used": command_used,
					"timed_out": result == SolveResult.TIMED_OUT,
					"crashed": result == SolveResult.CRASHED,
					"solve_time": solve_time
				})
				return
		# Otherotherwise, create new probem file with given solve data
		self.data["problem_files"].append({
			"source": str(problem_source),
			"known_unrealizable": result == SolveResult.UNREALIZABLE,
			"solve_attempts": [
				{
					"args_used": args_used,
					"output_file": str(output_file),
					"command_used": command_used,
					"timed_out": result == SolveResult.TIMED_OUT,
					"crashed": result == SolveResult.CRASHED,
					"solve_time": solve_time
				}
			]
		})

	# Returns whether running given commmand will give new results
	def is_new_command(self, problem_source: Path, args: list[str], solve_time: float):
		for problem_file in self.data["problem_files"]:
			if problem_file["source"] == str(problem_source):
				if problem_file["known_unrealizable"]: return False
				for solve_attempt in problem_file["solve_attempts"]:
					if solve_attempt["args_used"] == args:
						# Only try again if not crashed, but timed out with less time than available this time
						is_new = \
							not solve_attempt["crashed"] \
							and solve_attempt["timed_out"] \
							and solve_attempt["solve_time"] < solve_time
						return is_new
				# We have not attempted to solve with given args yet
				return True
		# We have not attempted to solve the given problem yet
		return True
	
# # Creates knor command that solves the given problem file with given arguments
# # Returns command and target output file
# def create_knor_command(file: Path, args: list[str], output_folder: Path = None) -> tuple[str, Path]:
# 	# First, ensure the default output folder if none is specified
# 	if not output_folder:
# 		if not UNMINIMIZED_AIG_FOLDER.is_dir():
# 			UNMINIMIZED_AIG_FOLDER.mkdir()

# 		output_folder = UNMINIMIZED_AIG_FOLDER / file.name.rstrip("".join(file.suffixes))
		
# 		if not output_folder.exists():
# 			output_folder.mkdir()
# 	# And make sure the output folder exists (whether specified or not)
# 	if not output_folder.is_dir():
# 		raise FileNotFoundError("Could not find or create output folder: {}".format(output_folder))
	
# 	output_file_name = file.with_stem(file.stem + "_args" + "".join(args)).with_suffix(".aag" if "-a" in args else ".aig").name
# 	output_file = output_folder / output_file_name
	
# 	command = "./{} {} {} > {}".format(KNOR_BINARY, file, " ".join(args), output_file)
# 	return command, output_file

# Returns a list of all argument combinations to give to Knor
def create_knor_arguments_combinations(knor_args: list[str], oink_args: list[str], binary_out: bool = True) -> list[list[str]]:
	# Get all possible combinations of knor args
	knor_arg_combinations = []
	for i in range(len(knor_args) + 1):
		l = []
		for c in itertools.combinations(knor_args, i):
			l.append(c)
		knor_arg_combinations.extend(l)

	all_arg_combinations = []
	# Now, combine knor arg combos with every possible oink arg
	for oink_arg in oink_args:
		for knor_arg_combo in knor_arg_combinations:
			new_combo = list(knor_arg_combo)
			new_combo.append(oink_arg)
			new_combo.append("-b" if binary_out else "-a")
			all_arg_combinations.append(new_combo)
	
	return all_arg_combinations

# # Returns list of problem files in the PROBLEM_FILES_FOLDER
# # Optional: regex to only select matching file names
# def get_problem_files(regex: str=None) -> list[Path]:
# 	problem_file_list = []
	
# 	for item in Path(PROBLEM_FILES_FOLDER).iterdir():
# 		if item.is_file():
# 			if regex:
# 				if re.match(regex, item.name):
# 					problem_file_list.append(item)
# 			else:
# 				problem_file_list.append(item)
# 	return problem_file_list

# # Solves the given problem files with the given argument combinations and stores stats in given profiling data
# def solve_problem_files(files: list[Path], arg_combinations: list[list[str]], profiler_data: ProfilerData, timeout=MAX_TIME_SECONDS_FOR_KNOR_COMMAND):
# 	total_commands = len(files) * len(arg_combinations)
	
# 	for file_num, file in enumerate(files):
# 		for arg_combo_num, arg_combo in enumerate(arg_combinations):
# 			progress_percentage = (file_num * len(arg_combinations) + arg_combo_num) / total_commands * 100
# 			print("{:.1f}% (File: {}/{}, arg: {}/{})... ".format(progress_percentage, file_num, len(files), arg_combo_num, len(arg_combinations)), end="", flush=True)

# 			if not profiler_data.is_new_command(file, arg_combo, timeout):
# 				print("Already computed. Skipping...")
# 				continue

# 			command, output_file = create_knor_command(file, arg_combo)

# 			command_start = time.time()
# 			try:
# 				result = run_shell_command(command, timeout)
# 			except KeyboardInterrupt:
# 				print("Aborted after: {:.2f}s".format(time.time() - command_start))
# 				return
# 			command_time = time.time() - command_start

# 			solve_result = SolveResult.SOLVED
# 			if not result:
# 				print("Timed out in: {:.2f}s ({} with {})".format(command_time, file, arg_combo))
# 				solve_result = SolveResult.TIMED_OUT
# 			elif result.returncode == 10:
# 				print("Solved in: {:.2f}s".format(command_time))
# 			elif result.returncode == 20:
# 				print("Unrealizable in: {:.2f}s".format(command_time))
# 				solve_result = SolveResult.UNREALIZABLE
# 			else:
# 				print("Crashed in: {:.2f}s".format(command_time))
# 				solve_result = SolveResult.CRASHED
			
# 			profiler_data.handle_command_result(
# 				file,
# 				arg_combo,
# 				command,
# 				output_file,
# 				command_time,
# 				solve_result
# 			)

# # Reads all AIG output files in profiler and adds the read data to the profiler
# def add_aig_stats_to_profiler(profiler: ProfilerData, reread: bool=False):
# 	problem_files_count = len(profiler.data["problem_files"])
# 	for count, problem_file in enumerate(profiler.data["problem_files"]):
# 		percentage = count / problem_files_count * 100
# 		print("{:.2f}% Reading solutions of problem {}/{}... ".format(percentage, count + 1, problem_files_count), flush=True, end="")
# 		read_solution_counter = 0
# 		start_time = time.time()
# 		if not problem_file["known_unrealizable"]:
# 			for solve_attempt in problem_file["solve_attempts"]:
# 				if not solve_attempt["timed_out"] and not solve_attempt["crashed"]:
# 					# This solve attempt was successful
# 					if "data" in solve_attempt and not reread: continue # Data already calculated
# 					solution_file = Path(solve_attempt["output_file"])
# 					aig_stats = get_aig_stats_from_file(solution_file)
# 					solve_attempt["data"] = aig_stats
# 					read_solution_counter += 1
# 			read_time = time.time() - start_time
# 			print("Read {} solutions in {:.2f}s".format(read_solution_counter, read_time))
# 		else: print("Unrealizable, no solutions.")

# # Runs ABC 'read' and 'print_stats' command on fiven file and returns the parsed stats
# def get_aig_stats_from_file(file: Path):
# 	abc_read_cmd = "./{} -c 'read {}; print_stats'".format(ABC_BINARY, file)
# 	cmd_output = run_shell_command(abc_read_cmd, 10).stdout.read().decode()
# 	return parse_aig_read_stats_output(cmd_output)

# Will create a solution for every possible example problem file. This takes looong...
def solve_all_problem_files():
	profiler = ProfilerData(PROFILER_SOURCE)
	problem_files = get_problem_files()
	arg_combos = create_knor_arguments_combinations(KNOR_ARGS, OINK_SOLVER_ARGS)
	solve_problem_files(problem_files, arg_combos, profiler)
	profiler.save()

# Solves all arbiter problems
def solve_all_arbiter_problem_files():
	profiler = ProfilerData(PROFILER_SOURCE)
	problem_files = get_problem_files(".*arbiter.*")
	arg_combos = create_knor_arguments_combinations(KNOR_ARGS, OINK_SOLVER_ARGS)
	solve_problem_files(problem_files, arg_combos, profiler)
	profiler.save()


# # Prepares profiler data structure and returns first new optimization ID
# def prepare_solution_optimizations(solution_data: dict, abc_args: list[str], output_folder: Path) -> int:
# 	if not "optimizations" in solution_data: solution_data["optimizations"] = []

# 	# Get highest optimization id
# 	highest_optimization_id = 0
# 	for optimization in solution_data["optimizations"]:
# 		if optimization["id"] > highest_optimization_id:
# 			highest_optimization_id = optimization["id"]
# 	# Increment for next optimization
# 	new_optimization_id = highest_optimization_id + 1

# 	# Make sure all single optimization arguments have been executed
# 	for abc_arg in abc_args:
# 		single_abc_arg_in_optimizations = False
# 		for optimization in solution_data["optimizations"]:
# 			if optimization["args_used"] == [abc_arg]:
# 				single_abc_arg_in_optimizations = True
# 				break
		
# 		# If we already made this optimization, check the next one
# 		if single_abc_arg_in_optimizations: continue

# 		# Create this single argument optimization
# 		command, output_file = create_abc_optimization_command(
# 			solution_data["output_file"],
# 			[abc_arg],
# 			new_optimization_id,
# 			output_folder)
		
# 		optimize_start = time.time()
# 		result = run_shell_command(command, MAX_TIME_SECONDS_FOR_OPTIMIZE_COMMAND)
# 		optimize_time = time.time() - optimize_start
		
# 		optimized_data = parse_aig_read_stats_output(result.stdout.decode())

# 		solution_data["optimizations"].append({
# 			"command_used": command,
# 			"output_file": str(output_file),
# 			"args_used": [abc_arg],
# 			"compress_time_python": optimize_time,
# 			"actual_compress_time": None,
# 			"timed_out": optimized_data == None,
# 			"data": optimized_data,
# 			"id": new_optimization_id
# 		})
		
# 		new_optimization_id += 1

# 		write_comment_to_aig_file(output_file, "This file was optimized with the following ABC arguments:\n{}".format(", ".join([abc_arg])))
			
# 	return new_optimization_id
	
# def get_optimizations_of_depth(solution: dict, depth: int) -> list[dict]:
# 	optimizations = []
# 	for optimization in solution["optimizations"]:
# 		if len(optimization["args_used"]) == depth:
# 			optimizations.append(optimization)
# 	return optimizations

# def branch_off_optimizations(solution: dict, abc_args: list[str], output_folder: Path, next_id: int):
# 	# Now iterate over compression methods
# 	progress_made = True
# 	optimize_depth = 1
# 	while progress_made:
# 		progress_made = False

# 		total_best_gain = 1

# 		branch_depth_start = time.time()
# 		for optimization in get_optimizations_of_depth(solution, optimize_depth):
# 			previous_optimize_args = optimization["args_used"]
# 			previous_AND_gate_count = optimization["data"]["and_gates"]
# 			min_AND_gate_gain = 1

# 			for abc_arg in abc_args:
# 				new_optimize_args = previous_optimize_args + [abc_arg]

# 				# Check if this has already been calculated
# 				is_calculated = False
# 				for opt in solution["optimizations"]:
# 					if opt["args_used"] == new_optimize_args:
# 						is_calculated = True
# 				if is_calculated: continue

# 				command, output_file = create_abc_optimization_command(optimization["output_file"], [abc_arg], next_id, output_folder)

# 				optimize_start = time.time()
# 				result = run_shell_command(command, 60)
# 				optimize_time = time.time() - optimize_start

# 				output_data = None
# 				if result != None:
# 					output_data = parse_aig_read_stats_output(result.stdout.decode())

# 				solution["optimizations"].append({
# 					"command_used": command,
# 					"output_file": str(output_file),
# 					"args_used": new_optimize_args,
# 					"compress_time_python": optimize_time,
# 					"actual_compress_time": None,
# 					"timed_out": output_data == None,
# 					"data": output_data,
# 					"id": next_id
# 				})

# 				gain = output_data["and_gates"] / previous_AND_gate_count
# 				min_AND_gate_gain = min(min_AND_gate_gain, gain)

# 				write_comment_to_aig_file(output_file, "This file was optimized with the following ABC arguments:\n{}".format(", ".join(new_optimize_args)))

# 				next_id += 1

# 			if min_AND_gate_gain < 0.99:
# 				# There has been an optimization of more than 1%
# 				progress_made = True
# 			total_best_gain = min(total_best_gain, min_AND_gate_gain)
		
		
# 		branch_depth_time = time.time() - branch_depth_start
# 		print("Branched depth level: {} in {:.2f} seconds with gain: {}".format(optimize_depth, branch_depth_time, total_best_gain))

# 		optimize_depth += 1
# 		# progress_made = False
# 		if optimize_depth > 4:
# 			progress_made = False

# def try_minimization_methods(profiler: ProfilerData, abc_args: list[str]):
# 	if not MINIMIZED_AIG_FOLDER.is_dir():
# 		MINIMIZED_AIG_FOLDER.mkdir()
	
# 	for problem_file in profiler.data["problem_files"]:
# 		# If unrealizable, there can be no minimization
# 		if problem_file["known_unrealizable"]: continue

# 		problem_file_source = Path(problem_file["source"])
# 		problem_folder = MINIMIZED_AIG_FOLDER / problem_file_source.name.rstrip("".join(problem_file_source.suffixes))

# 		if not problem_folder.is_dir(): problem_folder.mkdir()

# 		for solve_attempt in problem_file["solve_attempts"]:
# 			if solve_attempt["crashed"] or solve_attempt["timed_out"]: continue

# 			args_used: list[str] = solve_attempt["args_used"]
# 			solution_folder = problem_folder / "_".join(map(lambda x: x.replace("-",""), args_used))
# 			if not solution_folder.is_dir(): solution_folder.mkdir()

# 			new_highest_id = prepare_solution_optimizations(solve_attempt, abc_args, solution_folder)

# 			branch_start = time.time()
# 			branch_off_optimizations(solve_attempt, abc_args, solution_folder, new_highest_id)
# 			branch_time = time.time() - branch_start
# 			print("Branched for solution {} in {:.2f}s".format(args_used, branch_time))

# 	profiler.save()

# Returns a list of arguments used for checking if duplicate sequential arguments have positive effect
# def get_duplication_optimization_arguments(arguments: list[str], depth: int) -> list[list[str]]:
# 	arguments_list = []
# 	for argument in arguments:
# 		arguments_list.append([argument])

# 		for heading_argument in arguments:
# 			if heading_argument == argument: continue
# 			for heading_length in range(1, depth):
# 				arguments_list.append([heading_argument] * heading_length + [argument])
# 	return arguments_list

# # Will optimize all solutions (that match regex) with each given optimization
# def execute_optimizations(profiler: ProfilerData, optimizations: list[list[str]], problem_regex: str = None):
# 	if not MINIMIZED_AIG_FOLDER.is_dir(): MINIMIZED_AIG_FOLDER.mkdir()
	
# 	total_problem_files = len(profiler.data["problem_files"])
# 	total_optimizations = len(optimizations)

# 	for problem_file_count, problem_file in enumerate(profiler.data["problem_files"]):
# 		# If regex specified and not matching, skip this problem file
# 		if problem_regex and not re.match(problem_regex, problem_file["source"]): continue

# 		# If unrealizable, there can be no minimization
# 		if problem_file["known_unrealizable"]: continue

# 		# Ensure primary output folder
# 		problem_file_source = Path(problem_file["source"])
# 		problem_folder = MINIMIZED_AIG_FOLDER / problem_file_source.name.rstrip("".join(problem_file_source.suffixes))
# 		if not problem_folder.is_dir(): problem_folder.mkdir()

# 		total_solutions = len(problem_file["solve_attempts"])

# 		for solve_attempt_count, solve_attempt in enumerate(problem_file["solve_attempts"]):
# 			for optimization_count, optimization in enumerate(optimizations):

# 				total_things_todo = total_problem_files * total_solutions * total_optimizations
# 				current_total_progress_percentage = (problem_file_count * total_solutions * total_optimizations + solve_attempt_count * total_optimizations + optimization_count) / total_things_todo * 100
				
# 				print("{:.2f}% Optimizing file {}/{}, solution {}/{}, optimization {}/{}: ".format(current_total_progress_percentage, problem_file_count+1, total_problem_files, solve_attempt_count+1, total_solutions, optimization_count+1, total_optimizations), end="", flush=True)
# 				# If solving did not happen, go to next solve attempt
# 				if solve_attempt["crashed"] or solve_attempt["timed_out"]:
# 					print("Previous timeout or crash. Skipping...")
# 					continue

# 				# Ensure seconday output folder
# 				args_used: list[str] = solve_attempt["args_used"]
# 				solution_folder = problem_folder / "_".join(map(lambda x: x.replace("-",""), args_used))
# 				if not solution_folder.is_dir(): solution_folder.mkdir()

# 				execute_optimization_for_solution(solve_attempt, optimization, solution_folder)

# # Executes all duplication optimizations for all "arbiter" problem solutions
# def do_duplication_optimizations_for_arbiter_problems():
# 	profiler = ProfilerData(Path("profiler.json"))
# 	optimizations = get_duplication_optimization_arguments(ABC_OPTIMIZATION_ARGUMENTS, REPETITION_TEST_MAX_REPETITION)

# 	try:
# 		execute_optimizations(profiler, optimizations, problem_regex=".*arbiter.*")
# 	except KeyboardInterrupt:
# 		print("Aborted by user")
	
# 	profiler.save()

# def do_duplication_optimizations_for_all_problems():
# 	profiler = ProfilerData(Path("profiler.json"))
# 	optimizations = get_duplication_optimization_arguments(ABC_OPTIMIZATION_ARGUMENTS, REPETITION_TEST_MAX_REPETITION)

# 	try:
# 		execute_optimizations(profiler, optimizations)
# 	except KeyboardInterrupt:
# 		print("Aborted by user")

# 	profiler.save()

# Searches for optimization with given arguments in list of optimizations
def get_optimization(optimizations: list[dict], args: list[str]):
	for opt in optimizations:
		if opt["args_used"] == args:
			return opt
	return None

# Calculates percentage of decrease in size, a.k.a. our view of what "gain" is
def calculate_optimization_gain(previous_AND_gate_count: int, new_AND_gate_count: int) -> float:
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
						optimization = get_optimization(solve_attempt["optimizations"], test)
						
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
	
# def do_all_arguments_once_in_pairs():
# 	argument_combos = []
# 	for arg1 in ABC_OPTIMIZATION_ARGUMENTS:
# 		for arg2 in ABC_OPTIMIZATION_ARGUMENTS:
# 			if arg2 == arg1: continue

# 			argument_combos.append((arg1, arg2))

# 			# for arg3 in ABC_OPTIMIZATION_ARGUMENTS:
# 			# 	if arg3 == arg2 or arg3 == arg1: continue
# 			# 	argument_combos.append((arg1, arg2, arg3))

# 	print(len(argument_combos))
# 	input()
# 	for x in argument_combos:
# 		print(x)



# def test_if_cleanup_has_effect():
# 	profiler = ProfilerData(PROFILER_SOURCE)
# 	for problem_file in profiler.data["problem_files"]:

# 		# If unrealizable, there can be no minimization
# 		if problem_file["known_unrealizable"]: continue

# 		# Ensure primary output folder
# 		problem_file_source = Path(problem_file["source"])
# 		problem_folder = MINIMIZED_AIG_FOLDER / problem_file_source.name.rstrip("".join(problem_file_source.suffixes))
# 		if not problem_folder.is_dir(): problem_folder.mkdir()

# 		for solve_attempt in problem_file["solve_attempts"]:

# 			# Skip timed out or crashed ones
# 			if solve_attempt["timed_out"] or solve_attempt["crashed"]: continue

# 			# Ensure seconday output folder
# 			args_used: list[str] = solve_attempt["args_used"]
# 			solution_folder = problem_folder / "_".join(map(lambda x: x.replace("-",""), args_used))
# 			if not solution_folder.is_dir(): solution_folder.mkdir()

# 			execute_optimization_for_solution(solve_attempt, ["cleanup"], solution_folder)

# ======================================================================================================
# ======================================================================================================
# ======================================================================================================
# ======================================================================================================
# ======================================================================================================
# ======================================================================================================
# ======================================================================================================
# ======================================================================================================
# ======================================================================================================

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
def is_solve_attempt_worth_it(problem_file: dict, solve_argument_combo: list[str], solve_timeout_seconds: int) -> bool:
	""" Tells if attempting to solve the given problem with given arguments will give new data.\n
		Returns True unless:
		- The problem is unrealizable
		- Previous solve attempt crashed
		- Previous solve attempt timed out, and this time no more time is allowed"""
	if problem_file["known_unrealizable"]: return False

	previous_solve_attempt = None
	for solve_attempt in problem_file["solve_attempts"]:
		if solve_attempt["args_used"] == solve_argument_combo:
			previous_solve_attempt = solve_attempt
			break
	
	# If we have not tried this yet, we will gain new information by trying
	if not previous_solve_attempt: return True

	# If solve attempt crashed, do not retry
	if previous_solve_attempt["crashed"]: return False
	# If solve attempt timed out, only retry if we have more time
	if previous_solve_attempt["timed_out"] and solve_attempt["solve_time"] >= solve_timeout_seconds: return False

	return True

def run_shell_command(cmd: str, timeout_seconds: float | None) -> subprocess.Popen[bytes] | None:
	""" Runs linux shell command with given timeout in seconds.
		No timeout is indicated with None."""
	# Runs the given shell command in terminal, timeout in seconds
	try:
		p = subprocess.Popen(cmd, start_new_session=True, shell=True, stdout=subprocess.PIPE)
		p.wait(timeout=timeout_seconds)
		return p
	except subprocess.TimeoutExpired:
		print("Timed out!")
		os.killpg(os.getpgid(p.pid), signal.SIGTERM)
		return None

def solve_problem_files(
		problem_files: list[dict], 
		solve_argument_combos: list[list[str]], 
		profiler: ProfilerData, 
		verbose=True, 
		solve_timeout=MAX_TIME_SECONDS_FOR_KNOR_COMMAND):
	""" Solves each gven problem file with all given Knor argument combinations.
		Given timeout applies to the timeout of each solve attempt."""
	
	total_planned_solves = len(problem_files) * len(solve_argument_combos)

	# Prepare the general output folder for solutions
	if not UNMINIMIZED_AIG_FOLDER.is_dir():
		UNMINIMIZED_AIG_FOLDER.mkdir()

	for problem_file_index, problem_file in enumerate(problem_files):
		
		# Prepare the specific output folder for this problem file
		problem_file_source = Path(problem_file["source"])
		# TODO: Check if this problem file still exists
		solution_output_folder = UNMINIMIZED_AIG_FOLDER / problem_file_source.name.rstrip("".join(problem_file_source.suffixes))

		for arg_combo_index, solve_argument_combo in enumerate(solve_argument_combos):
			if verbose:
				progress_percentage = (problem_file_index * len(solve_argument_combos) + arg_combo_index) / total_planned_solves * 100
				print("{:.1f}% (File: {}/{}, arg: {}/{})... ".format(progress_percentage, problem_file_index, len(problem_files), arg_combo_index, len(solve_argument_combos)), end="", flush=True)

			if not is_solve_attempt_worth_it(problem_file, solve_argument_combo, solve_timeout):
				if verbose: print("Solve attempt not worth it. Skipping...")
				continue
			
			output_file_name = problem_file_source.with_stem(problem_file_source.stem + "_args" + "".join(solve_argument_combo)).with_suffix(".aag" if "-a" in solve_argument_combo else ".aig").name
			output_file = solution_output_folder / output_file_name

			solve_command = "./{} {} {} > {}".format(KNOR_BINARY, problem_file_source, " ".join(solve_argument_combo), output_file)

			command_start = time.time()
			try:
				result = run_shell_command(solve_command, solve_timeout)
			except KeyboardInterrupt:
				print("Aborted after: {:.2f}s".format(time.time() - command_start))
				return
			command_time = time.time() - command_start
			
			solve_result = SolveResult.SOLVED
			if not result:
				if verbose: print("Timed out in: {:.2f}s ({} with {})".format(command_time, problem_file_source, solve_argument_combo))
				solve_result = SolveResult.TIMED_OUT
			elif result.returncode == 10:
				if verbose: print("Solved in: {:.2f}s".format(command_time))
				# TODO Read solution stats and store these as well
			elif result.returncode == 20:
				if verbose: print("Unrealizable in: {:.2f}s".format(command_time))
				solve_result = SolveResult.UNREALIZABLE
			else:
				if verbose: print("Crashed in: {:.2f}s".format(command_time))
				solve_result = SolveResult.CRASHED

			profiler.handle_command_result(
				problem_file_source,
				solve_argument_combo,
				solve_command,
				output_file,
				command_time,
				solve_result
			)

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

def find_best_subset_optimization(solution: dict, wanted_args: list[str]) -> dict | None:	
	""" Goes through optimizations of given solution, and finds the best one
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

	for optimization in solution["optimizations"]:
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
		output_file: Path) -> tuple[str, Path]:
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

def parse_aig_read_stats_output(cmd_output: str) -> dict:
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
		print("Failed to parse: \n\n{}".format(sanitized_output))

def write_comment_to_aig_file(aig_file: Path, comment: str):
	""" Appends given comment to given AIG file."""
	with open(aig_file, "a") as file:
		file.write("c\n")
		file.write(comment)
		file.write("\n")


def execute_optimization_on_solution(solution: dict, arguments: list[str], output_folder: Path, verbose: bool=True):
	""" Executes the optimizations of given arguments if they have not been done before on the given solution.
		Ensures intermediate optimization results are stored as well for reuse later.
		Stores results in given output folder."""
	
	new_optimization_id = find_new_optimization_id(solution)

	arguments_build_up = []

	if verbose: print("[", end="", flush=True)
	for argument in arguments:
		arguments_build_up.append(argument)

		closest_base = find_best_subset_optimization(solution, arguments_build_up)

		redo_previous_try = False

		if closest_base and closest_base["timed_out"]:
			# Best base timed out, so we either try again or skip this one
			if closest_base["optimize_time_python"] < MAX_TIME_SECONDS_FOR_OPTIMIZE_COMMAND:
				# We can try again
				closest_base = find_best_subset_optimization(solution, arguments_build_up[:-1])
				redo_previous_try = True
			else:
				if verbose: print("X", end="", flush=True)

		source_file = None

		if closest_base:
			if closest_base["args_used"] == arguments_build_up:
				# If we already calculated this, continue to next arguments
				if verbose: print("*", end="", flush=True)
				continue
			# Otherwise, branch off closest optimization
			source_file = closest_base["output_file"]
			if verbose: print("a", end="", flush=True)
		else:
			# Then we need to branch off of the original solution
			source_file = solution["output_file"]
			if verbose: print("o", end="", flush=True)

		output_file = output_folder / "m{}.aig".format(new_optimization_id)

		command, output_file = create_abc_optimization_command(source_file, [argument], output_file)

		optimize_start = time.time()
		result = run_shell_command(command, MAX_TIME_SECONDS_FOR_OPTIMIZE_COMMAND)
		optimize_time = time.time() - optimize_start
		
		stats = None
		if result:
			output = result.stdout.read().decode()
			stats = parse_aig_read_stats_output(output)

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

		write_comment_to_aig_file(output_file, "Optimized with ABC arguments:\n{}".format(arguments))

		if redo_previous_try:
			for existing_opt in solution["optimizations"]:
				if existing_opt["args_used"] == arguments_build_up:
					for key in optimization:
						existing_opt[key] = optimization[key]
		else:
			solution["optimizations"].append(optimization)

		new_optimization_id += 1
	
	if verbose: print("]", flush=True)

def execute_optimizations_on_solutions(
		problem_files: list[dict],
		solve_argument_combos: list[list[str]] | None,
		optimization_argument_combos: list[list[str]],
		verbose=False,
		timeout=MAX_TIME_SECONDS_FOR_OPTIMIZE_COMMAND):
	""" Optimizes the given solutions (or all if "None") of the given problem files with the given optimization arguments"""
	# Ensure base folder for optimization results exists
	if not MINIMIZED_AIG_FOLDER.is_dir(): MINIMIZED_AIG_FOLDER.mkdir()
	
	problem_files_count = len(problem_files)
	optimizations_count = len(optimization_argument_combos)

	for problem_file_index, problem_file in enumerate(problem_files):
		if problem_file["known_unrealizable"]: continue # We cannot optimize unrealizable solutions

		# Ensure problem file specific output folder
		problem_file_source = Path(problem_file["source"])
		problem_folder = MINIMIZED_AIG_FOLDER / problem_file_source.name.rstrip("".join(problem_file_source.suffixes))
		if not problem_folder.is_dir(): problem_folder.mkdir()

		# Collect solutions to apply optimizations on
		target_solutions = []
		for solve_attempt in problem_file["solve_attempts"]:
			# Failed solve attempts cannot be optimized
			if solve_attempt["crashed"] or solve_attempt["timed_out"]: continue

			# If we want to optimize all solutions, add this solution anyways
			if not solve_argument_combos:
				target_solutions.append(solve_attempt)
			# If we specified solutions to optimize, see if this solution is in that list
			elif solve_argument_combos and solve_attempt["args_used"] in solve_argument_combos:
				target_solutions.append(solve_attempt)

		solutions_count = len(target_solutions)

		for target_solution_index, target_solution in enumerate(target_solutions):
			# Ensure solution specific output folder
			arguments_used_for_solution = target_solution["args_used"]
			solution_output_folder = problem_folder / "_".join(map(lambda x: x.replace("-",""), arguments_used_for_solution))
			if not solution_output_folder.is_dir(): solution_output_folder.mkdir()

			total_things_todo = problem_files_count * solutions_count * optimizations_count

			for arg_combo_index, optimization_argument_combo in enumerate(optimization_argument_combos):
				
				current_total_progress_percentage = (problem_file_index * solutions_count * optimizations_count + target_solution_index * solutions_count + arg_combo_index) / total_things_todo * 100
				if verbose: print("{:.2f}% Optimizing file {}/{}, solution {}/{}, optimization {}/{}: ".format(current_total_progress_percentage, problem_file_index+1, problem_files_count, target_solution_index+1, solutions_count, arg_combo_index+1, optimizations_count), end="", flush=True)
				
				execute_optimization_on_solution(target_solution, optimization_argument_combo, solution_output_folder, verbose=verbose)

def get_problem_files(profiler: ProfilerData, regex: str = None) -> list[dict]:
	""" Returns a list of problem files whose name match the given regex, or all files if regex is None."""
	all_problem_files = profiler.data["problem_files"]

	if not regex: return all_problem_files

	matching_problem_files = []
	for problem_file in all_problem_files:
		problem_file_name = Path(problem_file["source"]).name

		if re.match(regex, problem_file_name):
			matching_problem_files.append(problem_file)

	return matching_problem_files

# ========================================================================================


# def test_data():
# 	profiler = ProfilerData(PROFILER_SOURCE)
# 	a = collect_duplication_data(profiler.data["problem_files"])
# 	return a

# def add_aig():
# 	profiler = ProfilerData(PROFILER_SOURCE)
# 	add_aig_stats_to_profiler(profiler)
# 	profiler.save()

# TODO: Check if all entries are not malformed
# TODO: TQDM


test_cmd1 = './build/knor examples/full_arbiter_8.tlsf.ehoa --sym -b > test.aig'
test_cmd2 = './build/_deps/abc-build/abc -c "read test.aig; print_stats"'

# def test1():
# 	p = ProfilerData(Path("profiler.json"))
# 	try:
# 		try_minimization_methods(p, ABC_OPTIMIZATION_ARGUMENTS[0:2])
# 	except KeyboardInterrupt:
# 		print("Aborted by user")
# 	p.save()
	
# def test2():
# 	p = ProfilerData(Path("profiler.json"))
# 	try:
# 		try_minimization_methods(p, ABC_OPTIMIZATION_ARGUMENTS)
# 	except KeyboardInterrupt:
# 		print("Aborted by user")
# 	p.save()

# if __name__ == "__main__":
	# solve_all_arbiter_problem_files()
	# add_aig_stats_to_profiler(ProfilerData(Path("profiler.json")), True)
	# try_minimization_methods(ProfilerData(Path("profiler.json")))
