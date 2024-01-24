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
import threading
import signal

ABC_BINARY = Path("build/_deps/abc-build/abc")
ABC_ALIAS_SOURCE = Path("build/_deps/abc-src/abc.rc")
KNOR_BINARY = Path("build/knor")

UNMINIMIZED_AIG_FOLDER = Path("aigs_unminimized")
MINIMIZED_AIG_FOLDER = Path("aigs_minimized")
PROBLEM_FILES_FOLDER = Path("examples")

PROFILER_SOURCE = Path("profiler.json")

AIG_PARSE_TIMEOUT_SECONDS = 20 # seconds

# Threads used for executing optimizations in parralell
THREAD_COUNT = 3
CLUSTER_THREAD_COUNT = 16
CLUSTER_SOLVE_TIMEOUT = 120 # seconds
CLUSTER_OPTIMIZE_TIMEOUT = 60 # seconds

# To ensure the user does not stop the program during a critical part of the code,
# KeyboardInterrupt exceptions will be handled through the following event flag
KEYBOARD_INTERRUPT_HAS_BEEN_CALLED: threading.Event = threading.Event()
def keyboard_interrupt_handler(signum, frame):
	global KEYBOARD_INTERRUPT_HAS_BEEN_CALLED
	KEYBOARD_INTERRUPT_HAS_BEEN_CALLED.set()
signal.signal(signal.SIGINT, keyboard_interrupt_handler)

class SolveResult(Enum):
	TIMED_OUT = 1
	UNREALIZABLE = 2
	SOLVED = 3
	CRASHED = 4

class ShellCommandResult(Enum):
	SUCCESS = 1
	TIMEOUT = 2
	INTERRUPTED = 3

class VerbosityLevel(Enum):
	OFF = 1
	ERROR = 2
	WARNING = 3
	INFO = 4
	DETAIL = 5

class SolveProblemFileWorth(Enum):
	YES = "yes"
	NO_UNREALIZABLE = "unrealizable"
	NO_CRASHED = "crashed"
	NO_TIMED_OUT = "timed-out"
	NO_ALREADY_SOLVED = "already solved"

LOG_VERBOSITY_LEVEL = VerbosityLevel.WARNING
LOG_TO_TQDM = True

def LOG(message: str, message_verbosity_level: VerbosityLevel):
	global LOG_VERBOSITY_LEVEL, LOG_TO_TQDM

	prefix = ""
	if message_verbosity_level == VerbosityLevel.ERROR: prefix = "ERROR"
	if message_verbosity_level == VerbosityLevel.WARNING: prefix = "WARNING"
	if message_verbosity_level == VerbosityLevel.INFO: prefix = "INFO"
	if message_verbosity_level == VerbosityLevel.DETAIL: prefix = "DETAIL"

	if message_verbosity_level.value <= LOG_VERBOSITY_LEVEL.value:
		# Then we print it!
		log_text = "[{}]: {}".format(prefix, message)
		if LOG_TO_TQDM: tqdm.write(log_text)
		else: print(log_text)

class ProfilerData:
	def __init__(self, source: Path):
		self.source: Path = source
		if source.is_file():
			with open(source, "r") as file:
				LOG("Loading profiler data from '{}'... ".format(source), VerbosityLevel.INFO)

				self.data = json.load(file)
				LOG("Successfully loaded profiler.", VerbosityLevel.INFO)
		else:
			self.data = { "problem_files": [] }
			LOG("Created new profiler data", VerbosityLevel.INFO)
			self.save()

	# Saves current data to source file
	def save(self):
		LOG("WARNING: Saving profiler data. Do not quit... ", VerbosityLevel.INFO)
		with open(self.source, 'w') as file:
			json.dump(self.data, file, indent=3)
		LOG("Saved results in '{}'".format(self.source), VerbosityLevel.INFO)

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

# # ====================== Plotters ======================== #
			
# def calculate_optimization_gain(previous_AND_gate_count: int, new_AND_gate_count: int) -> float:
# 	""" Calculates percentage of decrease in size, a.k.a. our view of what "gain" is """
# 	difference = previous_AND_gate_count - new_AND_gate_count
# 	gain = difference / previous_AND_gate_count
# 	return gain

# # Returns dictionary collecting exploded results from repetition optimizations
# def collect_duplication_data(problem_files: list[dict]) -> dict:
# 	duplication_data = {"head": [], "tail": [], "repetition": [], "gain": []}
# 	for arg_head in ABC_OPTIMIZATION_ARGUMENTS:
# 		for arg_tail in ABC_OPTIMIZATION_ARGUMENTS:
# 			if arg_head == arg_tail: continue

# 			for problem_file in problem_files:
# 				if problem_file["known_unrealizable"] == True: continue

# 				for solve_attempt in problem_file["solve_attempts"]:
# 					if solve_attempt["timed_out"] or solve_attempt["crashed"]: continue

# 					AND_count_history = [solve_attempt["data"]["and_gates"]]

# 					for repetition in range(REPETITION_TEST_MAX_REPETITION):
# 						test = [arg_head] * repetition + [arg_tail]
# 						optimization = get_optimization_with_args(solve_attempt["optimizations"], test)
						
# 						if not optimization: continue
						
# 						if optimization["timed_out"]:
# 							raise Exception("Data not available due to previous timed-out calculation")

# 						previous = AND_count_history[-1]
# 						current = optimization["data"]["and_gates"]
# 						AND_count_history.append(current)

# 						gain = 100 * calculate_optimization_gain(previous, current)

# 						duplication_data["head"].append(arg_head)
# 						duplication_data["tail"].append(arg_tail)
# 						duplication_data["repetition"].append(repetition)
# 						duplication_data["gain"].append(gain)

# 	return duplication_data

# # Plots the repetition minimization results into a separate window
# def plot_repetition_minimization_results():
# 	profiler = ProfilerData(PROFILER_SOURCE)
# 	sns.set_theme()
# 	data = collect_duplication_data(profiler.data["problem_files"])
# 	figure = sns.catplot(data=data, col="head", x="repetition", y="gain", hue="tail", kind="boxen")
# 	plt.show()


# =========================== Solvers ================================ #

def get_optimization_with_args(optimizations: list[dict], args: list[str]) -> dict | None:
	""" Finds the optimization that matches the given optimization arguments """
	for opt in optimizations:
		if opt["args_used"] == args:
			return opt
	return None

def get_solve_attempt_with_args(solve_attempts: list[dict], args: list[str]) -> dict | None:
	""" Finds the solve attempt that matches the given optimization arguments """
	for attempt in solve_attempts:
		if attempt["args_used"] == args:
			return attempt
	return None

def is_solve_attempt_worth_it(problem_file: dict, knor_argument_combo: list[str], solve_timeout_seconds: float) -> SolveProblemFileWorth:
	""" Tells if attempting to solve the given problem with given arguments will give new data.\n
		Returns True unless:
		- The problem is unrealizable
		- Previous solve attempt crashed
		- Previous solve attempt timed out, and this time no more time is allowed
		- Successfully solved it already"""
	if problem_file["known_unrealizable"]: return SolveProblemFileWorth.NO_UNREALIZABLE

	previous_solve_attempt: dict | None = None
	for solve_attempt in problem_file["solve_attempts"]:
		if solve_attempt["args_used"] == knor_argument_combo:
			previous_solve_attempt = solve_attempt
			break
	
	# If we have not tried this yet, we will gain new information by trying
	if not previous_solve_attempt:
		return SolveProblemFileWorth.YES
	else:
		# If we have tried before:
		# Do not retry if previous time crashed
		if previous_solve_attempt["crashed"]: return SolveProblemFileWorth.NO_CRASHED
		
		# If previous try timed out, only retry if we have bigger timeout this time
		if previous_solve_attempt["timed_out"] and solve_timeout_seconds > previous_solve_attempt["solve_time"]: return SolveProblemFileWorth.YES

		# Otherwise, we should not try again
		return SolveProblemFileWorth.NO_TIMED_OUT

def run_shell_command(cmd: str, timeout_seconds: float | None, allow_keyboard_interrupts: bool) -> tuple[ShellCommandResult, subprocess.Popen[bytes] | None]:
	""" Runs linux shell command with given timeout in seconds.\n
		If allow_keyboard_interrupts is False, function will block untill command finished or timed out.\n
		No timeout is indicated with None.\n
		Do only set allow_keyboard_interrupts to false if a timeout has been specified!\n
		Returns the ShellCommandResult type with corresponding value."""
	# Run the given shell command in terminal
	process: subprocess.Popen[bytes] = subprocess.Popen(cmd, start_new_session=True, shell=True, stdout=subprocess.PIPE)
	
	# Record when we started the command
	time_started = time.time()

	while True:
		# Check if we have a return code yet
		return_code = process.poll()

		# If process finished, return the result
		if return_code is not None:
			return ShellCommandResult.SUCCESS, process
		
		# Test if we timed out
		is_timed_out = time.time() - time_started > timeout_seconds if timeout_seconds is not None else False
		is_interrupted = KEYBOARD_INTERRUPT_HAS_BEEN_CALLED.is_set() if allow_keyboard_interrupts else False

		# If we reached the timeout time OR keyboard interrupt has been signalled
		if is_timed_out or is_interrupted:
			# Kill the process and return None
			os.killpg(os.getpgid(process.pid), signal.SIGTERM)
			result_type = ShellCommandResult.TIMEOUT if is_timed_out else ShellCommandResult.INTERRUPTED
			return result_type, None
		
		# Otherwise, we just need to wait a tiny bit longer
		time.sleep(0.01)


def parse_aig_read_stats_output(cmd_output: str) -> dict | None:
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
	except (IndexError, ValueError):
		LOG("Failed to parse AIG stats: {}".format(cmd_output), VerbosityLevel.DETAIL)
		return None

def get_aig_stats_from_file(file: Path) -> dict | None:
	""" Reads given AIG file with ABC and returns its stats.\n
		Returns None if failed to parse AIG.\n
		Cannot be interrupted by KeyboardInterrupts. """
	abc_read_cmd = "./{} -c 'read {}; print_stats'".format(ABC_BINARY, file)
	_, result = run_shell_command(abc_read_cmd, AIG_PARSE_TIMEOUT_SECONDS, allow_keyboard_interrupts=False)
	
	if result is None:
		LOG("Reading AIG timed out. Please increase AIG_PARSE_TIMEOUT_SECONDS.", VerbosityLevel.ERROR)
		return None
	
	if not result.stdout:
		LOG("Failed to get STDOUT of AIG parse command.", VerbosityLevel.ERROR)
		return None

	shell_output = result.stdout.read().decode()
	return parse_aig_read_stats_output(shell_output)

def solve_problem_files(
		problem_files: list[dict],
		knor_argument_combos: list[list[str]],
		solve_timeout_seconds: float):
	""" Solves each gven problem file with all given Knor argument combinations.
		Given timeout applies to the timeout of each solve attempt."""
	
	# Prepare the general output folder for solutions
	if not UNMINIMIZED_AIG_FOLDER.is_dir():
		UNMINIMIZED_AIG_FOLDER.mkdir()

	for problem_file in tqdm(problem_files, desc="problem_file", position=0, leave=False):
		problem_file_source = Path(problem_file["source"])
		if not problem_file_source.is_file():
			LOG("Failed to open problem file '{}'.".format(problem_file_source), VerbosityLevel.ERROR)
			continue

		# Skip this problem file if we already know it is unrealizable
		if problem_file["known_unrealizable"]:
			LOG("Skipping problem file '{}' because it's unrealizable.".format(problem_file_source), VerbosityLevel.DETAIL)
			continue
		
		# Prepare the specific output folder for this problem file
		solution_output_folder = UNMINIMIZED_AIG_FOLDER / problem_file_source.name.rstrip("".join(problem_file_source.suffixes))
		if not solution_output_folder.is_dir(): solution_output_folder.mkdir()

		for knor_argument_combo in tqdm(knor_argument_combos, desc="arg_combination", position=1, leave=False):
			attempt_worth_it = is_solve_attempt_worth_it(problem_file, knor_argument_combo, solve_timeout_seconds)
			if not attempt_worth_it == SolveProblemFileWorth.YES:
				LOG("Skipping solving '{}' with args {} because: {}.".format(problem_file_source, knor_argument_combo, attempt_worth_it.value), VerbosityLevel.DETAIL)
				continue
			
			output_file_name = problem_file_source.with_stem(problem_file_source.stem + "_args" + "".join(knor_argument_combo)).with_suffix(".aag" if "-a" in knor_argument_combo else ".aig").name
			output_file = solution_output_folder / output_file_name

			solve_command = "./{} {} {} > {}".format(KNOR_BINARY, problem_file_source, " ".join(knor_argument_combo), output_file)

			command_start = time.time()
			command_status, command_result = run_shell_command(solve_command, solve_timeout_seconds, True)
			command_time = time.time() - command_start

			solution_data: dict | None = None
			solve_result = SolveResult.SOLVED

			if command_status == ShellCommandResult.INTERRUPTED:
				LOG("Aborted solving '{}' with {} after: {:.2f}s.".format(problem_file_source, knor_argument_combo, time.time() - command_start), VerbosityLevel.OFF)
				return

			elif command_status == ShellCommandResult.TIMEOUT:
				LOG("Timeout for solving '{}' with args {} in: {:.2f}s.".format(problem_file_source, knor_argument_combo, command_time), VerbosityLevel.INFO)
				solve_result = SolveResult.TIMED_OUT

			elif command_result is None:
				LOG("Should have gotten command result, but received None.", VerbosityLevel.ERROR)
				return

			elif command_result.returncode == 10:
				# Successfully solved problem, so read solution AIG data
				LOG("Solved '{}' with args {} in {:.2f}s".format(problem_file_source, knor_argument_combo, command_time), VerbosityLevel.DETAIL)
				solution_data = get_aig_stats_from_file(output_file)
				if not solution_data:
					LOG("Successfully solved '{}' with args {} into '{}' but failed to parse AIG data.".format(problem_file_source, knor_argument_combo, output_file), VerbosityLevel.ERROR)
				
			elif command_result.returncode == 20:
				LOG("Unrealizable to solve '{}' after {:.2f}s.".format(problem_file_source, command_time), VerbosityLevel.INFO)
				solve_result = SolveResult.UNREALIZABLE
			else:
				LOG("Warning: Crashed from solving '{}' with args {} after: {:.2f}s".format(problem_file_source, knor_argument_combo, command_time), VerbosityLevel.WARNING)
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
				LOG("Added new solve attempt of '{}' with args {} to profiler.".format(problem_file_source, knor_argument_combo), VerbosityLevel.DETAIL)
			else:
				LOG("Updated previous solve attempt of: '{}' with args: {}".format(problem_file_source, knor_argument_combo), VerbosityLevel.INFO)
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

def execute_optimization_on_solution(
		solution: dict,
		arguments: list[str],
		output_folder: Path,
		optimize_timeout_seconds: float):
	""" Executes the optimizations of given arguments if they have not been done before on the given solution.
		Ensures intermediate optimization results are stored as well for reuse later.
		Stores results in given output folder."""
	
	new_optimization_id = find_new_optimization_id(solution)

	arguments_build_up = []

	# Go through each argument and try to optimize the solution with the argument chain up till then 
	for argument in arguments:
		# Return when user has interrupted
		if KEYBOARD_INTERRUPT_HAS_BEEN_CALLED.is_set(): return
		
		arguments_build_up.append(argument)

		# Find previous optimization attempt for these built-up arguments chain
		previous_optimize_attempt = get_optimization_with_args(solution["optimizations"], arguments_build_up)
		
		# If previous optimization attempt exists, only continue if it makes sense to try it again
		if previous_optimize_attempt:
			# If previous optimization was successful, we can go to the next argument
			if not previous_optimize_attempt["timed_out"]: continue

			# If previous attempt timed out and we do not have more time than then, this optimization cannot be done
			if optimize_timeout_seconds <= previous_optimize_attempt["optimize_time_python"]:
				LOG("Cannot improve timed out optimize attempt {} on {}".format(arguments_build_up, solution["args_used"]), VerbosityLevel.INFO)
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
				LOG("Failed to find optimization base for {} ({}) on solution {}".format(arguments_build_up, origin_arguments, solution["args_used"]), VerbosityLevel.ERROR)
				return
			
			source_file = Path(origin_optimization["output_file"])

		# Check if source file actually still exists
		if not source_file.exists():
			LOG("Source file missing: '{}'".format(source_file), VerbosityLevel.ERROR)
			return

		output_file = output_folder / "m{}.aig".format(new_optimization_id)
		command = create_abc_optimization_command(source_file, [argument], output_file)

		optimize_start = time.time()
		result_type, result = run_shell_command(command, optimize_timeout_seconds, False)
		optimize_time = time.time() - optimize_start

		if result_type == ShellCommandResult.TIMEOUT:
			# We can no longer continue this optimization path, so return
			LOG("Timed out optimizing with {} on solution {} after {:.2f}s.".format(arguments_build_up, solution["args_used"], optimize_time), VerbosityLevel.WARNING)
			return
		
		stats: dict | None = None
		if result and result.stdout:
			output = result.stdout.read().decode()
			stats = parse_aig_read_stats_output(output)
			if not stats:
				# That means we did not time out, but ABC gave an error
				LOG("ABC command crashed on: \n{}".format(command), VerbosityLevel.ERROR)
		
		if not result:
			LOG("Did not get result from optimizing with {} on solution {} after {:.2f}s".format(arguments_build_up, solution["args_used"], optimize_time), VerbosityLevel.ERROR)
			return
		
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
			LOG("Retried optimization {} of {}".format(arguments_build_up, solution["args_used"]), VerbosityLevel.INFO)
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

		write_comment_to_aig_file(output_file, "Optimized with ABC arguments:\n{}".format(arguments))

		new_optimization_id += 1

def execute_optimizations_on_solutions(
		problem_files: list[dict],
		solve_argument_combos: list[list[str]] | None,
		optimization_argument_combos: list[list[str]],
		timeout_seconds: float,
		n_threads: int = 1):
	""" Optimizes the given solutions (or all if "None") of the given problem files with the given optimization arguments.\n
		Performs optimizations on different threads on solution-level. """
	global KEYBOARD_INTERRUPT_HAS_BEEN_CALLED

	if n_threads < 1: raise Exception("Cannot perform optimization on no threads.")

	# Ensure base folder for optimization results exists
	if not MINIMIZED_AIG_FOLDER.is_dir(): MINIMIZED_AIG_FOLDER.mkdir()
	
	for problem_file in tqdm(problem_files, desc="problem files", position=0):
		if problem_file["known_unrealizable"]:
			LOG("Skipping unrealizable problem file: '{}', as it cannot be optimized".format(problem_file["source"]), VerbosityLevel.INFO)
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
				else: LOG("Missing solve attempt {} for '{}'".format(solve_argument_combo, problem_file_source), VerbosityLevel.WARNING)

		# Filter out invalid solutions
		target_solutions = []
		for matching_solve_attempt in matching_solve_attempts:
			if matching_solve_attempt["crashed"] or matching_solve_attempt["timed_out"]:
				# We cannot optimize this solution, so skip it
				LOG("Cannot optimize solution of '{}' with {} because the solve attempt {}".format(
					problem_file_source,
					matching_solve_attempt["args_used"],
					"crashed" if matching_solve_attempt["crashed"] else "timed out"), VerbosityLevel.INFO)
			else:
				target_solutions.append(matching_solve_attempt)

		# Then apply optimizations to target solutions
		if n_threads == 1:
			for target_solution in tqdm(target_solutions, desc="solution", position=1, leave=False):
				# Ensure solution specific output folder exists
				arguments_used_for_solution = target_solution["args_used"]
				solution_output_folder = problem_folder / "_".join(map(lambda x: x.replace("-",""), arguments_used_for_solution))
				if not solution_output_folder.is_dir(): solution_output_folder.mkdir()

				for optimization_argument_combo in tqdm(optimization_argument_combos, desc="optimization", position=2, leave=False):				
					try:
						execute_optimization_on_solution(target_solution, optimization_argument_combo, solution_output_folder, timeout_seconds)
					except KeyboardInterrupt:
						LOG("User aborted optimizing '{}' with optimizations {}.".format(target_solution["output_file"], optimization_argument_combo), VerbosityLevel.OFF)
						return
		else:
			# Spawn n_threads amount of thread workers that will perform the optimizations
			worker_threads: list[threading.Thread] = []

			# Workers pick a solution to perform all given optimization commands on, and then insert it into the "finished" list
			remaining_unoptimized_solutions: list[dict] = [solution for solution in target_solutions] # Clone to not mess up the profiler's list
			remaining_list_mutex: threading.Lock = threading.Lock()

			finished_optimized_solutions: list[dict] = []
			finished_list_mutex: threading.Lock = threading.Lock()

			# Spawn the thread for just progress bar of solutions
			progress_thread = threading.Thread(target=status_worker_function, args=(
				finished_optimized_solutions,
				len(target_solutions),
				1
			))
			progress_thread.start()

			# Now spawn the threads
			for i in range(n_threads):
				worker_i = threading.Thread(target=optimize_worker_function, args=(
					remaining_unoptimized_solutions,
					remaining_list_mutex,
					finished_optimized_solutions,
					finished_list_mutex,
					optimization_argument_combos,
					problem_folder,
					timeout_seconds,
					"worker_{}".format(i),
					2 + i
					))
				worker_i.start()
				worker_threads.append(worker_i)

			# Join worker threads. Will happen if they are done or if they see KeyboardInterrupt has happened
			for worker in worker_threads:
				worker.join()
			progress_thread.join()

			# If the threads stopped becasuse of the KeyboardInterrupt signal, return
			if KEYBOARD_INTERRUPT_HAS_BEEN_CALLED.is_set():
				LOG("User aborted optimizing at problem file: '{}'".format(problem_file_source), VerbosityLevel.OFF)
				return

def status_worker_function(
		finished_solutions: list[dict],
		total_solutions: int,
		tqdm_bar_position: int,
		):
	""" Displays the progress bar of solutions on which all optimizations have been performed.\n
		Runs until KeyboardInterrupt happens or all optimiztions have been performed on all solutions. 
		Meant for separate thread for when multiple worker threads are used for optimizing. """
	global KEYBOARD_INTERRUPT_HAS_BEEN_CALLED

	with tqdm(total=total_solutions, mininterval=0.3, position=tqdm_bar_position, desc="solutions", leave=False) as pbar:
		while not KEYBOARD_INTERRUPT_HAS_BEEN_CALLED.is_set():
			actual_progress = len(finished_solutions)
			bar_progress = pbar.n
			new_progress = actual_progress - bar_progress
			pbar.update(new_progress)
			if actual_progress == total_solutions: break # We have finished our progress, so we can stop
			time.sleep(0.5)

def optimize_worker_function(
		unsolved_solutions: list[dict],
		unsolved_list_mutex: threading.Lock,
		finished_solutions: list[dict],
		finished_list_mutex: threading.Lock,
		optimization_argument_combos: list[list[str]],
		problem_folder: Path,
		optimize_timeout_seconds: float,
		tqdm_bar_title: str,
		tqdm_bar_position: int
		):
	""" Picks solution from unsolved_solutions and performs all optimizations on it.\n
		If all optimizations have been done on picked solution, solution is appended to finished_solutions. \n
	 	Meant as function for optimizations worker thread. Runs until
			* KeyboardInterrupt has been signalled
			* or all optimizations have been performed. """
	global KEYBOARD_INTERRUPT_HAS_BEEN_CALLED

	while not KEYBOARD_INTERRUPT_HAS_BEEN_CALLED.is_set():
		# First, pick the first solution to perform optimizations on:
		with unsolved_list_mutex:
			if not unsolved_solutions: return # There is no solution we can optimize left
			target_solution: dict = unsolved_solutions.pop(0)

		# Ensure solution specific output folder exists
		arguments_used_for_solution = target_solution["args_used"]
		solution_output_folder = problem_folder / "_".join(map(lambda x: x.replace("-",""), arguments_used_for_solution))
		if not solution_output_folder.is_dir(): solution_output_folder.mkdir()

		for argument_combo in tqdm(optimization_argument_combos, mininterval=0.3, desc=tqdm_bar_title, position=tqdm_bar_position, leave=False):
			# If KeyboardInterrupt has been singalled, stop
			if KEYBOARD_INTERRUPT_HAS_BEEN_CALLED.is_set(): break

			# This argument combo needs to be applied on the picked solution
			execute_optimization_on_solution(target_solution, argument_combo, solution_output_folder, optimize_timeout_seconds)

		# Now add the solution we were optimizing to the finished pile
		with finished_list_mutex:
			finished_solutions.append(target_solution)

# ======================== Argument Creators ===================== #
def get_knor_flag_combinations(knor_synthesize_flags: list[str], knor_solve_flags: list[str], binary_out: bool = True) -> list[list[str]]:
	""" Returns a list of all argument combinations to give to Knor.
		If binary_out is False, '-a' will be appended instead '-b'."""
	# Get all possible combinations of knor args
	knor_flag_combinations = []
	for i in range(len(knor_synthesize_flags) + 1):
		l = []
		for c in itertools.combinations(knor_synthesize_flags, i):
			l.append(c)
		knor_flag_combinations.extend(l)

	all_flag_combinations = []
	# Now, combine knor synthesize flag combos with every possible knor solve flag
	for oink_arg in knor_solve_flags:
		for knor_arg_combo in knor_flag_combinations:
			new_combo = list(knor_arg_combo)
			new_combo.append(oink_arg)
			new_combo.append("-b" if binary_out else "-a")
			all_flag_combinations.append(new_combo)
	
	return all_flag_combinations

def get_range_of_numbers(minimum: int, maximum: int, steps: int) -> list[int]:
	""" Divides the given range in given steps. Returns list of integers within. """
	if steps <= 1: return [minimum]
	step_size = (maximum - minimum) / (steps - 1)
	# Sort and remove duplicates
	nums = sorted(list(set([minimum+round(step_size*i) for i in range(steps)])))
	return nums

def get_ABC_flag_number_variations(flag_number_data: dict | None, mutate: int) -> list[int]:
	""" Gets variations of a single ABC optimization parameter flag number. 
		Mutate indicates how many numbers requested for lower and higher flag values.
		Example: number_data = {max: 16, min: 4, default: 8}, mutate = 3
		Results in [4, 6, 8, 12, 16] """
	if flag_number_data is None: return []

	nums = []
	default = flag_number_data["default"]
	minimum = flag_number_data["min"]
	nums.extend(get_range_of_numbers(minimum, default, mutate))

	maximum = None
	if "max_pseudo" in flag_number_data: maximum = flag_number_data["max_pseudo"]
	elif "max" in flag_number_data: maximum = flag_number_data["max"]

	if maximum:
		nums.extend(get_range_of_numbers(default, maximum, mutate))
	else:
		for i in range(mutate):
			# Add exponential -> 10, 100, 10000, 100000000...
			new_highest = max(1, max(nums)) * 10
			nums.append(min(new_highest, ABC_FLAG_LIMIT))

	nums = sorted(list(set(nums))) # Remove duplicates and sorts for convenience
	return nums

def get_ABC_argument_variations(argument_base: str, flags: dict | None, mutate: int) -> list[str]:
	""" Gets variations of a single ABC optimization argument.\n
		Argument base example: 'b'\n
		Flags example: ['-l', '-d', '-s']"""
	if not flags: return [argument_base]

	flags_variations: list[list[str]] = []
	for flag in flags:
		flag_number_data = flags[flag]
		numbers = get_ABC_flag_number_variations(flag_number_data, mutate)

		if not numbers:
			# This flag has no number, so just add it
			flags_variations.append([flag])

		flag_variations: list[str] = []
		for number in numbers:
			flag_variations.append("{} {}".format(flag, number))
		flags_variations.append(flag_variations)

	combos = flags_variations[0]
	for flag_variations in flags_variations[1:]:
		combos.extend(map(lambda x: " ".join(x), itertools.product(combos, flag_variations)))
		combos.extend(flag_variations)
	
	argument_combos = [argument_base] + ["{} {}".format(argument_base, combo) for combo in combos]
	return argument_combos

def get_all_ABC_optimization_arguments(mutate: int) -> list[str]:
	""" Returns list of all individual ABC optimization commands and their variations. """
	arguments: list[str] = []
	for argument in ABC_OPTIMIZATION_ARGUMENTS:
		arguments.extend(get_ABC_argument_variations(argument, ABC_OPTIMIZATION_ARGUMENTS[argument], mutate))
	return arguments

def get_all_ABC_optimization_duos(mutate: int) -> list[list[str]]:
	""" Creates pairs of every ABC optimization command. """
	duos = []
	all_optimizations = get_all_ABC_optimization_arguments(mutate)
	for first in all_optimizations:
		for second in all_optimizations:
			duo = [first, second]
			duos.append(duo)
	return duos

def get_ABC_optimization_duplication_combos(mutate: int, depth: int) -> list[list[str]]:
	""" Creates a list of optimization argument combos for performing duplication effectiveness tests. 
		A, BA, BBA, BBBA, B, AB, AAB, AAB, AAAB etc. """
	duplication_combos: list[list[str]] = []

	all_argument_variations = get_all_ABC_optimization_arguments(mutate)
	for head_of_chain in all_argument_variations:
		duplication_combos.append([head_of_chain])

		for tail in all_argument_variations:
			if head_of_chain.split(" ")[0] == tail.split(" ")[0]: continue
			for depth_level in range(1, depth):
				chain = [tail] * depth_level + [head_of_chain]
				duplication_combos.append(chain)

	return duplication_combos

def get_ABC_cleanup_arguments(mutate: int) -> list[str]:
	""" Returns list of all individual ABC cleanup commands and their variants. """
	arguments: list[str] = []
	for argument in ABC_CLEANUP_ARGUMENTS:
		arguments.extend(get_ABC_argument_variations(argument, ABC_CLEANUP_ARGUMENTS[argument], mutate))
	return arguments

def get_all_ABC_cleanup_arguments(mutate: int) -> list[str]:
	""" Returns list of all individual ABC cleanup commands and their variations. """
	arguments: list[str] = []
	for argument in ABC_CLEANUP_ARGUMENTS:
		arguments.extend(get_ABC_argument_variations(argument, ABC_CLEANUP_ARGUMENTS[argument], mutate))
	return arguments

def get_ABC_cleanup_arguments_sandwiched_in_optimization_duos(mutate: int) -> list[list[str]]:
	""" Returns a list of all optimization duo argument combos with a cleanup argument in between. """
	duos = get_all_ABC_optimization_duos(3)
	cleanup_arguments = get_all_ABC_cleanup_arguments(mutate)

	sandwiched_triples: list[list[str]] = []
	for cleanup_argument in cleanup_arguments:
		for first, second in duos:
			triple = [first, cleanup_argument, second]
			sandwiched_triples.append(triple)
	return sandwiched_triples

def get_ABC_premade_optimization_strategies() -> list[list[str]]:
	""" Returns a list of all argument combos from already existing optimization strategies. """
	strategies: list[list[str]] = [
		# c2rs
		["b -l", "rs -K 6 -l", "rw -l", "rs -K 6 -N 2 -l", "rf -l", "rs -K 8 -l", "b -l", "rs -K 8 -N 2 -l", "rw -l", "rs -K 10 -l", "rwz -l", "rs -K 10 -N 2 -l", "b -l", "rs -K 12 -l", "rfz -l", "rs -K 12 -N 2 -l", "rwz -l", "b -l"],
		# compress
		["b -l", "rw -l", "rwz -l", "b -l", "rwz -l", "b -l"],
		# compress2
		["b -l", "rw -l", "rf -l", "b -l", "rw -l", "rwz -l", "b -l", "rfz -l", "rwz -l", "b -l"],
		# compress2rs
		["b -l", "rs -K 6 -l", "rw -l", "rs -K 6 -N 2 -l", "rf -l", "rs -K 8 -l", "b -l", "rs -K 8 -N 2 -l", "rw -l", "rs -K 10 -l", "rwz -l", "rs -K 10 -N 2 -l", "b -l", "rs -K 12 -l", "rfz -l", "rs -K 12 -N 2 -l", "rwz -l", "b -l"],
		# drwsat2
		["st", "drw", "b -l", "drw", "drf", "ifraig -C 20", "drw", "b -l", "drw", "drf"],
		# r2rs
		["b", "rs -K 6", "rw", "rs -K 6 -N 2", "rf", "rs -K 8", "b", "rs -K 8 -N 2", "rw", "rs -K 10", "rwz", "rs -K 10 -N 2", "b", "rs -K 12", "rfz", "rs -K 12 -N 2", "rwz", "b"],
		# resyn
		["b", "rw", "rwz", "b", "rwz", "b"],
		# resyn2
		["b", "rw", "rf", "b", "rw", "rwz", "b", "rfz", "rwz", "b"],
		# resyn2a
		["b", "rw", "b", "rw", "rwz", "b", "rwz", "b"],
		# resyn2rs
		["b", "rs -K 6", "rw", "rs -K 6 -N 2", "rf", "rs -K 8", "b", "rs -K 8 -N 2", "rw", "rs -K 10", "rwz", "rs -K 10 -N 2", "b", "rs -K 12", "rfz", "rs -K 12 -N 2", "rwz", "b"],
		# resyn3
		["b", "rs", "rs -K 6", "b", "rsz", "rsz -K 6", "b", "rsz -K 5", "b"],
		# rwsat
		["st", "rw -l", "b -l", "rw -l", "rf -l"],
		# src_rs
		["st", "rs -K 6 -N 2 -l", "rs -K 9 -N 2 -l", "rs -K 12 -N 2 -l"],
		# src_rw
		["st", "rw -l", "rwz -l", "rwz -l"],
		# src_rws
		["st", "rw -l", "rs -K 6 -N 2 -l", "rwz -l", "rs -K 9 -N 2 -l", "rwz -l", "rs -K 12 -N 2 -l"],
	]
	return strategies


# TODO: Implement this during one of the proper tests
# def get_duplication_optimization_arguments(arguments: list[str], depth: int) -> list[list[str]]:
# 	""" Returns a list of arguments used for checking if duplicate sequential
# 		arguments have positive effect. """
# 	arguments_list = []
# 	for argument in arguments:
# 		arguments_list.append([argument])

# 		for heading_argument in arguments:
# 			if heading_argument == argument: continue
# 			for heading_length in range(1, depth):
# 				arguments_list.append([heading_argument] * heading_length + [argument])
# 	return arguments_list

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

# def solve_arbiter_problems():
# 	""" Solves all problem files with 'arbiter' in their name. """
# 	profiler = ProfilerData(PROFILER_SOURCE)
# 	initialize_problem_files(profiler)
# 	target_knor_args = get_knor_flag_combinations(KNOR_SYNTHESIZE_FLAGS, KNOR_SOLVE_FLAGS)
# 	target_problem_files = get_problem_files(profiler, ".*arbiter.*")
# 	solve_problem_files(target_problem_files, target_knor_args, solve_timeout=11)
# 	profiler.save()

# def test_duplication_optimizations_on_arbiter_solutions():
# 	""" Optimizes all arbiter solutions to see if duplicate sequential optimizations make sense """
# 	profiler = ProfilerData(PROFILER_SOURCE)
# 	target_problem_files = get_problem_files(profiler, ".*arbiter.*")
# 	target_solve_attempts = get_knor_flag_combinations(KNOR_SYNTHESIZE_FLAGS, KNOR_SOLVE_FLAGS)
# 	target_optimization_arguments = get_duplication_optimization_arguments(list(ABC_OPTIMIZATION_ARGUMENTS.keys()), 4)
# 	execute_optimizations_on_solutions(target_problem_files, target_solve_attempts, target_optimization_arguments, timeout=50)
# 	profiler.save()

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
	target_solutions = get_knor_flag_combinations(KNOR_SYNTHESIZE_FLAGS, KNOR_SOLVE_FLAGS)
	optimization_args = get_cleanup_optimizations_test_arguments()
	execute_optimizations_on_solutions(target_problem_files, target_solutions, optimization_args, timeout_seconds=50)
	profiler.save()

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

def get_small_problem_files(profiler: ProfilerData) -> list[dict]:
	small = [
		"ActionConverter",	# AIG size: 8
		"amba_decomposed_arbiter_2",	# AIG size: 70
		"amba_decomposed_arbiter_3",	# AIG size: 300
		"amba_decomposed_decode",	# AIG size: 6
		"arbiter_with_buffer",	# AIG size: 1
		"Automata",	# AIG size: 70
		"Automata32S",	# AIG size: 110
		"Cockpitboard",	# AIG size: 15
		"detector",	# AIG size: 50
		"EnemeyModule",	# AIG size: 7
		"EscalatorBidirectional",	# AIG size: 140
		"EscalatorBidirectionalInit",	# AIG size: 150
		"full_arbiter",	# AIG size: 150
		"full_arbiter_2",	# AIG size: 50
		"full_arbiter_3",	# AIG size: 300
		"Gamelogic",	# AIG size: 80
		"GamemodeChooser",	# AIG size: 100
		"lilydemo10",	# AIG size: 7
		"lilydemo17",	# AIG size: 100
		"loadcomp3",	# AIG size: 200
		"loadfull2",	# AIG size: 50
		"ltl2dba_alpha",	# AIG size: 10
		"ltl2dba_beta",	# AIG size: 140
		"ltl2dba01",	# AIG size: 10
		"MusicAppFeedback",	# AIG size: 30
		"OneCounter",	# AIG size: 500
		"prioritized_arbiter",	# AIG size: 30
		"Scoreboard",	# AIG size: 10
		"Sensor",	# AIG size: 300
		"simple_arbiter",	# AIG size: 30
		"Zoo0",	# AIG size: 20
	]
	targets = []
	for wanted in small:
		for problem_file in profiler.data["problem_files"]:
			source = Path(problem_file["source"])
			current = source.name.removesuffix("".join(source.suffixes))
			if current == wanted:
				targets.append(problem_file)
				break

	LOG("Selected: {} out of {} wanted problem files".format(len(targets), len(small)), VerbosityLevel.INFO)
	return targets

# def solve_for_test_0():
# 	""" Solves each problem file with each Knor argument combination. """
# 	profiler = ProfilerData(PROFILER_SOURCE)
# 	a = get_problem_files(profiler)
# 	b = get_knor_flag_combinations(KNOR_SYNTHESIZE_FLAGS, KNOR_SOLVE_FLAGS)
# 	solve_problem_files(a, b)
# 	profiler.save()

# def show_test_0():
# 	profiler = ProfilerData(PROFILER_SOURCE)
# 	solve_attempt_arg_combos = get_small_problem_files(profiler)

# 	raw_data = {
# 		"problem_file_source": [],
# 		"solve_args": [],
# 		"synthesize_args": [],
# 		"solve_time": [],
# 		"and_gates": [],
# 		"increase": [] 	# Increase in size compared to baseline (smallest AIG per problem file)
# 	}

# 	# Collect the minimums
# 	minimum_and_gates = {}
# 	for problem_file in profiler.data["problem_files"]:
# 		if problem_file["known_unrealizable"]: continue
# 		for solve_attempt in problem_file["solve_attempts"]:
# 			if not solve_attempt["args_used"] in solve_attempt_arg_combos: continue
# 			if not solve_attempt["data"]: continue

# 			source = problem_file["source"]
# 			current_and_count = solve_attempt["data"]["and_gates"]
# 			if source not in minimum_and_gates:
# 				minimum_and_gates[source] = current_and_count
# 			else:
# 				minimum_and_gates[source] = min(minimum_and_gates[source], current_and_count) # At least 1 so we can compare
	
# 	# Compare each solution with the minimum (but not the 0 bests)
# 	for problem_file in profiler.data["problem_files"]:
# 		if problem_file["known_unrealizable"]: continue
# 		for solve_attempt in problem_file["solve_attempts"]:
# 			if not solve_attempt["args_used"] in solve_attempt_arg_combos: continue
# 			if not solve_attempt["data"]: continue

# 			source = problem_file["source"]
# 			args_used = solve_attempt["args_used"]
# 			solve_args = args_used[-2]
# 			synthesize_args = args_used[:-2]
# 			and_count = solve_attempt["data"]["and_gates"]
# 			solve_time = solve_attempt["solve_time"]

# 			minimum_of_this_file = minimum_and_gates[source]
# 			if minimum_of_this_file == 0: continue

# 			increase = and_count / minimum_and_gates[source]

# 			raw_data["problem_file_source"].append(source)
# 			raw_data["solve_args"].append(solve_args)
# 			raw_data["synthesize_args"].append(synthesize_args)
# 			raw_data["solve_time"].append(solve_time)
# 			raw_data["and_gates"].append(and_count)
# 			raw_data["increase"].append(increase)

# 	df = pd.DataFrame(raw_data)

# 	df["synthesize_str"] = df["synthesize_args"].apply(lambda x: str(x))

# 	sns.set_theme()
# 	sns.catplot(data=df, kind="boxen", x="solve_args", y="increase", hue="synthesize_str")
# 	plt.xticks(rotation=45)
# 	plt.show()


# 	return df


# # Test 1: Compare rw and drw optimizations performance
# def solve_for_test_1():
# 	profiler = ProfilerData(PROFILER_SOURCE)
# 	a = get_small_problem_files(profiler)
# 	b = get_knor_flag_combinations(KNOR_SYNTHESIZE_FLAGS, KNOR_SOLVE_FLAGS)
# 	solve_problem_files(a, b, solve_timeout=60)
# 	profiler.save()

# def optimize_for_test_1():
# 	""" See rewrite(rw) or dag-aware rewrite(drw) is better. """
# 	profiler = ProfilerData(PROFILER_SOURCE)
# 	a = get_target_problem_files(profiler)
# 	b = get_target_solve_attempt_arguments()
# 	rw_variants = get_abc_argument_flag_combinations("rw", ABC_OPTIMIZATION_ARGUMENTS["rw"], 3)
# 	drw_variants = get_abc_argument_flag_combinations("drw", ABC_OPTIMIZATION_ARGUMENTS["drw"], 3)
# 	joint_args = list(map(lambda x: [x], rw_variants + drw_variants))
# 	execute_optimizations_on_solutions(a, b, joint_args, timeout=60, n_threads=THREAD_COUNT)
# 	profiler.save()

# def show_test_1(profiler: ProfilerData, n: int = 5):
# 	""" Plots the top n best average arguments for rw and drw. """
# 	problem_files = get_target_problem_files(profiler)
# 	solve_attempt_arg_combos = get_target_solve_attempt_arguments()
# 	rw_variants = get_abc_argument_flag_combinations("rw", ABC_OPTIMIZATION_ARGUMENTS["rw"], 3)
# 	drw_variants = get_abc_argument_flag_combinations("drw", ABC_OPTIMIZATION_ARGUMENTS["drw"], 3)
# 	optimize_arg_combos = list(map(lambda x: [x], rw_variants + drw_variants))
	
# 	raw_data = {
# 		"solve_args": [],
# 		"opt_arg_bases": [],
# 		"opt_arg_flags": [],
# 		"gains": [],
# 		"times": []
# 	}

# 	for problem_file in problem_files:
# 		for solve_attempt in problem_file["solve_attempts"]:
# 			if not solve_attempt["args_used"] in solve_attempt_arg_combos: continue
# 			if not solve_attempt["data"]: continue

# 			solve_args_used = " ".join(solve_attempt["args_used"])

# 			base_and_count = solve_attempt["data"]["and_gates"]

# 			for optimization in solve_attempt["optimizations"]:
# 				if not optimization["args_used"] in optimize_arg_combos: continue
# 				if len(optimization["args_used"]) != 1: raise Exception("Expected only one argument")

# 				argument = optimization["args_used"][0]
# 				arg_base: str = argument.split(" ")[0]
# 				flags: str = " ".join(argument.split(" ")[1:])

# 				new_and_count = optimization["data"]["and_gates"]
# 				gain = base_and_count / new_and_count

# 				raw_data["solve_args"].append(solve_args_used)
# 				raw_data["opt_arg_bases"].append(arg_base)
# 				raw_data["opt_arg_flags"].append(flags)
# 				raw_data["gains"].append(gain)
# 				raw_data["times"].append(optimization["optimize_time_python"])

# 	data = pd.DataFrame(raw_data)

# 	# Get average of each argument over each solution
# 	averaged_gain_over_files = data.groupby(["opt_arg_bases", "opt_arg_flags"]).agg({"gains": "mean"})
# 	# Get the top 5 of each argument base with the highest gain
# 	top_argument_df = averaged_gain_over_files.sort_values("gains", ascending=False).groupby(["opt_arg_bases"]).head(5).sort_values("opt_arg_bases").reset_index()

# 	top_arguments_found = list(top_argument_df["opt_arg_bases"] + " " + top_argument_df["opt_arg_flags"])
	
# 	# Add complete optimization argument as column
# 	data["argument"] = (data["opt_arg_bases"] + " " + data["opt_arg_flags"]).apply(lambda x: x.strip())
	
# 	plot_data = data[data["argument"].isin(top_arguments_found)]

# 	sns.set_theme()
# 	sns.catplot(data=plot_data, kind="violin", x="opt_arg_flags", y="gains", hue="opt_arg_bases")
# 	plt.xticks(rotation=45)
# 	plt.show()

# # Test 2: Compare performance of rf and drf optimizations
# def solve_for_test_2():
# 	profiler = ProfilerData(PROFILER_SOURCE)
# 	a = get_target_problem_files(profiler)
# 	b = get_target_solve_attempt_arguments()
# 	solve_problem_files(a, b, solve_timeout=60)
# 	profiler.save()

# def optimize_for_test_2():
# 	""" See if refactor (rf) or dag-aware refactor (drf) is more effective """
# 	profiler = ProfilerData(PROFILER_SOURCE)
# 	a = get_target_problem_files(profiler)
# 	b = get_target_solve_attempt_arguments()
# 	rw_variants = get_abc_argument_flag_combinations("rf", ABC_OPTIMIZATION_ARGUMENTS["rf"], 3)
# 	drw_variants = get_abc_argument_flag_combinations("drf", ABC_OPTIMIZATION_ARGUMENTS["drf"], 3)
# 	joint_args = list(map(lambda x: [x], rw_variants + drw_variants))
# 	execute_optimizations_on_solutions(a, b, joint_args, timeout=60, n_threads=THREAD_COUNT)
# 	profiler.save()

# def show_test_2(profiler: ProfilerData, n: int = 5):
# 	""" Plots the top n best average rf and top n best average drf """
# 	problem_files = get_target_problem_files(profiler)
# 	solve_attempt_arg_combos = get_target_solve_attempt_arguments()
# 	rf_variants = get_abc_argument_flag_combinations("rf", ABC_OPTIMIZATION_ARGUMENTS["rf"], 3)
# 	drf_variants = get_abc_argument_flag_combinations("drf", ABC_OPTIMIZATION_ARGUMENTS["drf"], 3)
# 	optimize_arg_combos = list(map(lambda x: [x], rf_variants + drf_variants))
	
# 	raw_data = {
# 		"solve_args": [],
# 		"opt_arg_bases": [],
# 		"opt_arg_flags": [],
# 		"gains": [],
# 		"times": []
# 	}

# 	for problem_file in problem_files:
# 		for solve_attempt in problem_file["solve_attempts"]:
# 			if not solve_attempt["args_used"] in solve_attempt_arg_combos: continue
# 			if not solve_attempt["data"]: continue

# 			solve_args_used = " ".join(solve_attempt["args_used"])

# 			base_and_count = solve_attempt["data"]["and_gates"]

# 			for optimization in solve_attempt["optimizations"]:
# 				if not optimization["args_used"] in optimize_arg_combos: continue
# 				if len(optimization["args_used"]) != 1: raise Exception("Expected only one argument")

# 				argument = optimization["args_used"][0]
# 				arg_base: str = argument.split(" ")[0]
# 				flags: str = " ".join(argument.split(" ")[1:])

# 				new_and_count = optimization["data"]["and_gates"]
# 				gain = base_and_count / new_and_count

# 				raw_data["solve_args"].append(solve_args_used)
# 				raw_data["opt_arg_bases"].append(arg_base)
# 				raw_data["opt_arg_flags"].append(flags)
# 				raw_data["gains"].append(gain)
# 				raw_data["times"].append(optimization["optimize_time_python"])

# 	data = pd.DataFrame(raw_data)

# 	# Get average of each argument over each solution
# 	averaged_gain_over_files = data.groupby(["opt_arg_bases", "opt_arg_flags"]).agg({"gains": "mean"})
# 	# Get the top 5 of each argument base with the highest gain
# 	top_argument_df = averaged_gain_over_files.sort_values("gains", ascending=False).groupby(["opt_arg_bases"]).head(5).sort_values("opt_arg_bases").reset_index()

# 	top_arguments_found = list(top_argument_df["opt_arg_bases"] + " " + top_argument_df["opt_arg_flags"])
	
# 	# Add complete optimization argument as column
# 	data["argument"] = (data["opt_arg_bases"] + " " + data["opt_arg_flags"]).apply(lambda x: x.strip())
	
# 	plot_data = data[data["argument"].isin(top_arguments_found)]

# 	sns.set_theme()
# 	sns.catplot(data=plot_data, kind="violin", x="opt_arg_flags", y="gains", hue="opt_arg_bases")
# 	plt.xticks(rotation=45)
# 	plt.show()

# # Test 3: Test if other optimizations are any good
# def solve_for_test_3():
# 	profiler = ProfilerData(PROFILER_SOURCE)
# 	a = get_target_problem_files(profiler)
# 	b = get_target_solve_attempt_arguments()
# 	solve_problem_files(a, b, solve_timeout=60)
# 	profiler.save()

# def optimize_for_test_3():
# 	""" Test if one of the other optimization arguments are good. """
# 	profiler = ProfilerData(PROFILER_SOURCE)
# 	a = get_target_problem_files(profiler)
# 	b = get_target_solve_attempt_arguments()

# 	opt_args = []
# 	for arg in ["drwsat", "rs", "dc2", "irw", "irws", "iresyn"]:
# 		opt_args.extend(get_abc_argument_flag_combinations(arg, ABC_OPTIMIZATION_ARGUMENTS[arg], 3))
# 	opt_args = list(map(lambda x: [x], opt_args))
	
# 	execute_optimizations_on_solutions(a, b, opt_args, timeout=60, n_threads=THREAD_COUNT)
# 	profiler.save()

# def show_test_3(profiler: ProfilerData):
# 	pass # TODO: Implement this

# # Test 4: Test balance performance. Probably not amazing
# def solve_for_test_4():
# 	profiler = ProfilerData(PROFILER_SOURCE)
# 	a = get_target_problem_files(profiler)
# 	b = get_target_solve_attempt_arguments()
# 	solve_problem_files(a, b, solve_timeout=60)
# 	profiler.save()

# def optimize_for_test_4():
# 	""" Optimizes all target solutions with the balance optimization """
# 	profiler = ProfilerData(PROFILER_SOURCE)
# 	a = get_problem_files(profiler, ".*arbiter.*")
# 	b = get_target_solve_attempt_arguments()
# 	balances = list(map(lambda x: [x], get_abc_argument_flag_combinations("b", ABC_OPTIMIZATION_ARGUMENTS["b"], 3)))
# 	execute_optimizations_on_solutions(a, b, balances, timeout=60, n_threads=THREAD_COUNT)
# 	profiler.save()

# def show_test_4(profiler: ProfilerData):
# 	pass # TODO: Implement this

# # Test 5: Are extreme drw flags better than small normal drw flags?
# def solve_for_test_5():
# 	profiler = ProfilerData(PROFILER_SOURCE)
# 	a = get_target_problem_files(profiler)
# 	b = get_target_solve_attempt_arguments()
# 	solve_problem_files(a, b, solve_timeout=60)
# 	profiler.save()

# def optimize_for_test_5():
# 	""" See if drw performs better with extremely big parameters """
# 	profiler = ProfilerData(PROFILER_SOURCE)
# 	a = get_target_problem_files(profiler)
# 	b = get_target_solve_attempt_arguments()
# 	extreme_drw_variants = get_abc_argument_flag_combinations("drw", ABC_OPTIMIZATION_ARGUMENTS["drw"], 3, True)
# 	joint_args = list(map(lambda x: [x], extreme_drw_variants))
# 	execute_optimizations_on_solutions(a, b, joint_args, timeout=60, n_threads=THREAD_COUNT)
# 	profiler.save()

# def show_test_5(profiler: ProfilerData):
# 	pass # TODO: Implement

# # Test 6: Are extreme drf flags better than small normal drf flags?
# def solve_for_test_6():
# 	profiler = ProfilerData(PROFILER_SOURCE)
# 	a = get_target_problem_files(profiler)
# 	b = get_target_solve_attempt_arguments()
# 	solve_problem_files(a, b, solve_timeout=60)
# 	profiler.save()

# def optimize_for_test_6():
# 	""" See if drf performs better with extremely big parameters """
# 	profiler = ProfilerData(PROFILER_SOURCE)
# 	a = get_target_problem_files(profiler)
# 	b = get_target_solve_attempt_arguments()
# 	extreme_drw_variants = get_abc_argument_flag_combinations("drf", ABC_OPTIMIZATION_ARGUMENTS["drf"], 3, True)
# 	joint_args = list(map(lambda x: [x], extreme_drw_variants))
# 	execute_optimizations_on_solutions(a, b, joint_args, timeout=60, n_threads=THREAD_COUNT)
# 	profiler.save()

# def show_test_6(profiler: ProfilerData):
# 	pass # TODO: Implement

# # Chaining tests
# def solve_for_test_5():
# 	pass

# def optimize_for_test_5():
# 	""" Optimizes all to see if duplicate sequential optimizations make sense """
# 	profiler = ProfilerData(PROFILER_SOURCE)
# 	target_problem_files = get_problem_files(profiler, ".*arbiter.*")
# 	target_solve_attempts = get_knor_flag_combinations(KNOR_SYNTHESIZE_FLAGS, KNOR_SOLVE_FLAGS)
# 	target_optimization_arguments = get_duplication_optimization_arguments(list(ABC_OPTIMIZATION_ARGUMENTS.keys()), 4)
# 	execute_optimizations_on_solutions(target_problem_files, target_solve_attempts, target_optimization_arguments, timeout=50)
# 	profiler.save()

# ================================= Profiler Checkers ==================================== #

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

	LOG("Finished checking profiler structure correctness", VerbosityLevel.INFO)

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
				LOG("Added 'data' attribute to: {} of {}".format(solve_attempt["args_used"], source), VerbosityLevel.INFO)

			if solve_attempt["crashed"] or solve_attempt["timed_out"]: continue

			if not solve_attempt["data"]:
				# Then this solve attempt needs to have data!
				stats = get_aig_stats_from_file(solve_attempt["output_file"])
				if not stats: raise Exception("Should have data, but failed to get stats in solve attempt: {} of problem '{}'".format(solve_attempt["args_used"], source))
				solve_attempt["data"] = stats
				LOG("Read and set AIG stats of solve attempt {} of problem '{}'".format(solve_attempt["args_used"], source), VerbosityLevel.INFO)

			if not "optimizations" in solve_attempt:
				solve_attempt["optimizations"] = []
				LOG("Added empty optimizations list to solve attempt {} of problem {}".format(solve_attempt["args_used"], source), VerbosityLevel.INFO)

			for optimization in solve_attempt["optimizations"]:
				if optimization["timed_out"]: continue
				
				if not "data" in optimization:
					optimization["data"] = None

				if not optimization["data"]:
					# Then this optimization needs to have data!
					stats = get_aig_stats_from_file(optimization["output_file"])
					if not stats: print("Should have data, but failed to get stats in optimization: {} of solution {} of problem '{}'".format(optimization["args_used"], solve_attempt["args_used"], source))
					optimization["data"] = stats
					LOG("Read and set AIG stats of opt {} of solution {} of problem '{}'".format(optimization["args_used"], solve_attempt["args_used"], source), VerbosityLevel.INFO)


# =================== CLUSTER TESTS ======================= #

def do_cluster_stuff():
	# 1. Initialize profiler
	profiler = ProfilerData(PROFILER_SOURCE)
	initialize_problem_files(profiler)
	profiler.save()
	LOG("Initialized profiler!", VerbosityLevel.INFO)

	if KEYBOARD_INTERRUPT_HAS_BEEN_CALLED.is_set(): return

	# 2. Solve all problem files
	solve_all_problem_files(profiler)
	profiler.save()
	LOG("Solved all problem files!", VerbosityLevel.INFO)

	if KEYBOARD_INTERRUPT_HAS_BEEN_CALLED.is_set(): return

	# 3. Perform test 1
	cluster_test_1(profiler)
	profiler.save()
	LOG("Performed test 1: all optimizations once!", VerbosityLevel.INFO)

	if KEYBOARD_INTERRUPT_HAS_BEEN_CALLED.is_set(): return

	# 4. Perform test 2
	cluster_test_2(profiler)
	profiler.save()
	LOG("Performed test 2: all duos!", VerbosityLevel.INFO)

	if KEYBOARD_INTERRUPT_HAS_BEEN_CALLED.is_set(): return

	# 5. Perform test 3
	cluster_test_3(profiler)
	profiler.save()
	LOG("Performed test 3: duplications!", VerbosityLevel.INFO)

	if KEYBOARD_INTERRUPT_HAS_BEEN_CALLED.is_set(): return
	
	# 6. Perform test 4
	cluster_test_4(profiler)
	profiler.save()
	LOG("Performed test 4: cleanups on solutions!", VerbosityLevel.INFO)

	if KEYBOARD_INTERRUPT_HAS_BEEN_CALLED.is_set(): return

	# 7. Perform test 5
	cluster_test_5(profiler)
	profiler.save()
	LOG("Performed test 5: sandwiched cleanups!", VerbosityLevel.INFO)

	if KEYBOARD_INTERRUPT_HAS_BEEN_CALLED.is_set(): return

	# 8. Perform test 6
	cluster_test_6(profiler)
	profiler.save()
	LOG("Performed test 6: premade strategies!", VerbosityLevel.INFO)

	if KEYBOARD_INTERRUPT_HAS_BEEN_CALLED.is_set(): return


def solve_all_problem_files(profiler: ProfilerData):
	""" Will create a solution for every possible example problem file. """
	problem_files = get_problem_files(profiler)
	knor_flag_combos = get_knor_flag_combinations(KNOR_SYNTHESIZE_FLAGS, KNOR_SOLVE_FLAGS)
	solve_problem_files(problem_files, knor_flag_combos, solve_timeout_seconds=CLUSTER_SOLVE_TIMEOUT)

def cluster_test_1(profiler: ProfilerData):
	""" Do all ABC optimizations once on each solution. """
	problem_files = get_problem_files(profiler)
	knor_flag_combos = get_knor_flag_combinations(KNOR_SYNTHESIZE_FLAGS, KNOR_SOLVE_FLAGS)
	optimizations = get_all_ABC_optimization_arguments(3)
	optimization_combos = list(map(lambda x: [x], optimizations))
	execute_optimizations_on_solutions(problem_files, knor_flag_combos, optimization_combos, timeout_seconds=CLUSTER_OPTIMIZE_TIMEOUT, n_threads=CLUSTER_THREAD_COUNT)

def cluster_test_2(profiler: ProfilerData):
	""" Do all ABC optimization duos on each solution. """
	problem_files = get_problem_files(profiler)
	knor_flag_combos = get_knor_flag_combinations(KNOR_SYNTHESIZE_FLAGS, KNOR_SOLVE_FLAGS)
	optimization_duos = get_all_ABC_optimization_duos(3)
	execute_optimizations_on_solutions(problem_files, knor_flag_combos, optimization_duos, timeout_seconds=CLUSTER_OPTIMIZE_TIMEOUT, n_threads=CLUSTER_THREAD_COUNT)

def cluster_test_3(profiler: ProfilerData):
	""" Do the duplication test: See if repeating same argument is effective. """
	problem_files = get_problem_files(profiler)
	knor_flag_combos = get_knor_flag_combinations(KNOR_SYNTHESIZE_FLAGS, KNOR_SOLVE_FLAGS)
	optimization_combos = get_ABC_optimization_duplication_combos(2, 4)
	execute_optimizations_on_solutions(problem_files, knor_flag_combos, optimization_combos, timeout_seconds=CLUSTER_OPTIMIZE_TIMEOUT, n_threads=CLUSTER_THREAD_COUNT)

def cluster_test_4(profiler: ProfilerData):
	""" Performs cleanup optimization commands on unoptimized solutions. """
	problem_files = get_problem_files(profiler)
	knor_flag_combos = get_knor_flag_combinations(KNOR_SYNTHESIZE_FLAGS, KNOR_SOLVE_FLAGS)
	cleanup_arguments = get_ABC_cleanup_arguments(3)
	cleanup_combos = list(map(lambda x: [x], cleanup_arguments))
	execute_optimizations_on_solutions(problem_files, knor_flag_combos, cleanup_combos, timeout_seconds=CLUSTER_OPTIMIZE_TIMEOUT, n_threads=CLUSTER_THREAD_COUNT)

def cluster_test_5(profiler: ProfilerData):
	""" Sandwich cleanup arguments between optimization duos to see if it has actual effect. """
	problem_files = get_problem_files(profiler)
	knor_flag_combos = get_knor_flag_combinations(KNOR_SYNTHESIZE_FLAGS, KNOR_SOLVE_FLAGS)
	sandwiched_cleanups = get_ABC_cleanup_arguments_sandwiched_in_optimization_duos(3)
	execute_optimizations_on_solutions(problem_files, knor_flag_combos, sandwiched_cleanups, timeout_seconds=CLUSTER_OPTIMIZE_TIMEOUT, n_threads=CLUSTER_THREAD_COUNT)

def cluster_test_6(profiler: ProfilerData):
	""" Perform all premade optimization strategies. """
	problem_files = get_problem_files(profiler)
	knor_flag_combos = get_knor_flag_combinations(KNOR_SYNTHESIZE_FLAGS, KNOR_SOLVE_FLAGS)
	strategy_combos = get_ABC_premade_optimization_strategies()
	execute_optimizations_on_solutions(problem_files, knor_flag_combos, strategy_combos, timeout_seconds=CLUSTER_OPTIMIZE_TIMEOUT, n_threads=CLUSTER_THREAD_COUNT)


"""

===== All these arguments are OPTIMIZE arguments ====

1. All arguments once (A, B, C, D, E)
- See which optimizes best
- See which ones do not optimize at all

2. All duos of each argument, including dupes (AA, AB, AC, BA, BB, BC, CA, CB, CC)
- See which pair of arguments works best
- See if 'preparations' like balancing improves effectiveness of other commands

3. Duplication test: Each argument succeeded by a multitude of the same other argument
(A, BA, BBA, BBBA, B, AB, AAB, AAAB, AAAAB)
- See if repeating the same argument makes sense

====== Intertwine cleanup commands through all other commands ======
4. Perform cleanup optimizations on each unopimized AIG
- See if the conversion from BDD to AIG is "clean"
- Able to exclude of other optimizations create cleanup possibilities

5. Perform cleanup between each optimization duo from before
- See if the optimizations create cleanup possibilities

====== Now the big chaining strategy tests come into play.... ========

6. Figure out best premade optimization strategy

"""

# ///////// # TODO: Collect better argument flag combinations (perhaps in reproducible way)
# ///////// # TODO: Collect argument combinations per plan.
# ///////// # TODO: Figure out why some argumen flag combinations do not work
# TODO: Improve logging for better debugging

# TODO: Make solving also parrallel


# ============================================================================
 
# Plot sample example AIG sizes to see the distribution
