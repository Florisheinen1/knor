from __future__ import annotations # To make it run on 3.8 as well
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
from datetime import datetime

ABC_BINARY = Path("build/_deps/abc-build/abc")
ABC_ALIAS_SOURCE = Path("build/_deps/abc-src/abc.rc")
KNOR_BINARY = Path("build/knor")

UNMINIMIZED_AIG_FOLDER = Path("aigs_unminimized")
MINIMIZED_AIG_FOLDER = Path("aigs_minimized")
PROBLEM_FILES_FOLDER = Path("examples")

PROFILER_SOURCE = Path("profiler_final.json")
PROGRESS_SOURCE = Path("progress_final.txt")
LOG_FILE_SOURCE = Path("logs_final.txt")

AIG_PARSE_TIMEOUT_SECONDS = 20 # seconds

LAST_TIME_SAVED = time.time()

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

class OptimizeResult(Enum):
	SUCCESS = 1
	CRASHED = 2
	TIMEOUT = 3
	INTERRUPTED = 4

# Indicates how extensive the tests need to be
class TestSize(Enum):
	Small = 1			# Few problem files,		1 mutation,  3 repetitions
	Medium = 2			# Quarter of problem files,	2 mutations, 3 repetitions
	Big = 3				# Half of problem files, 	3 mutations, 4 repetitions
	Everything = 4		# All problem files, 		4 mutations, 5 repetitions

LOG_VERBOSITY_LEVEL = VerbosityLevel.INFO
LOG_TO_TQDM = True

def LOG(message: str, message_verbosity_level: VerbosityLevel):
	global LOG_VERBOSITY_LEVEL, LOG_TO_TQDM

	prefix = ""
	if message_verbosity_level == VerbosityLevel.ERROR: prefix = "ERROR"
	if message_verbosity_level == VerbosityLevel.WARNING: prefix = "WARNING"
	if message_verbosity_level == VerbosityLevel.INFO: prefix = "INFO"
	if message_verbosity_level == VerbosityLevel.DETAIL: prefix = "DETAIL"
	log_text = "[{}]: {}".format(prefix, message)

	if message_verbosity_level.value <= LOG_VERBOSITY_LEVEL.value:
		# Then we print it!
		if LOG_TO_TQDM: tqdm.write(log_text)
		else: print(log_text)

	if message_verbosity_level.value <= VerbosityLevel.INFO.value:
		with open(LOG_FILE_SOURCE, "a") as file:
			file.write("{}\n".format(log_text))

def LOG_PROGRESS(message: str):
	now = datetime.now()
	prefix = "[{}:{}:{}]:".format(now.hour, now.minute, now.second)
	with open(PROGRESS_SOURCE, "a") as file:
		file.write("{} {}\n".format(prefix, message))

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
			LOG("Created new profiler data: {}".format(self.source), VerbosityLevel.INFO)
			self.save()

	def save(self):
		""" Saves current data to source file. """
		global LAST_TIME_SAVED
		LOG("WARNING: Saving profiler data: {}. Do not quit... ".format(self.source), VerbosityLevel.INFO)
		LOG_PROGRESS("Saving profiler data: {}. Do not stop program now...".format(self.source))
		with open(self.source, 'w') as file:
			json.dump(self.data, file)
		LOG("Saved results in '{}'".format(self.source), VerbosityLevel.INFO)
		LOG_PROGRESS("Done saving profiler data: {}".format(self.source))
		LAST_TIME_SAVED = time.time()

	def backup(self, name: str):
		""" Creates backup of current data. """
		now = datetime.now()
		moment_str: str = "{}-{}-{}_{}-{}-{}".format(now.year, now.month, now.day, now.hour, now.minute, now.second)
		output_file = Path("backup_{}_{}.json".format(name, moment_str))
		LOG("Creating backup at: '{}'...".format(output_file), VerbosityLevel.INFO)
		LOG_PROGRESS("Creating backup at: '{}'...".format(output_file))
		with open(output_file, "w") as file:
			json.dump(self.data, file)
		LOG("Created backup.", VerbosityLevel.INFO)
		LOG_PROGRESS("Created backup.")


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

def run_ABC_optimization_command(command: str, timeout_seconds: float | None) -> tuple[OptimizeResult, dict | None]:
	""" Runs the given ABC optimization command. Returns after timeout or keyboard interrupt. """
	result_type, result_data = run_shell_command(command, timeout_seconds, True)

	if result_type == ShellCommandResult.INTERRUPTED: return OptimizeResult.INTERRUPTED, None
	elif result_type == ShellCommandResult.TIMEOUT: return OptimizeResult.TIMEOUT, None
	
	# At this point, the command executed without giving back error code. Whether it successfully optimized is something we need to determine

	# If we do not have return data, something terrible happened!
	if not result_data or not result_data.stdout:
		LOG("Optimize command executed properly, but failed to retrieve command result!", VerbosityLevel.ERROR)
		return OptimizeResult.CRASHED, None
	
	output = result_data.stdout.read().decode()
	parsed_stats = parse_aig_read_stats_output(output)

	# If the output does not contain stats of an AIG, the optimization command crashed for some reason (perhaps incorrect flags)
	if not parsed_stats: return OptimizeResult.CRASHED, None

	return OptimizeResult.SUCCESS, parsed_stats

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
	result_type, result = run_shell_command(abc_read_cmd, AIG_PARSE_TIMEOUT_SECONDS, allow_keyboard_interrupts=False)
	
	if result_type == ShellCommandResult.INTERRUPTED: return None
	elif result_type == ShellCommandResult.TIMEOUT: return None
	
	if not result or not result.stdout:
		LOG("Reading AIG command succeeded, but failed to get output.", VerbosityLevel.ERROR)
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
			# If previous attempt crashed, there is no use trying again
			if previous_optimize_attempt["crashed"]: return
			# Continue to next argument if we already did it
			if not previous_optimize_attempt["timed_out"] and previous_optimize_attempt["data"]: continue
			# Otherwise, retry iff we have bigger timeout this time

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
		result_type, stats = run_ABC_optimization_command(command, optimize_timeout_seconds)
		optimize_time = time.time() - optimize_start

		if result_type == OptimizeResult.INTERRUPTED:
			LOG("Aborted while optimizing with {} on solution {} after {:.2f}s.".format(arguments_build_up, solution["args_used"], optimize_time), VerbosityLevel.OFF)
			return
		elif result_type == OptimizeResult.TIMEOUT:
			LOG("Timed out optimizing with {} on solution {} after {:.2f}s.".format(arguments_build_up, solution["args_used"], optimize_time), VerbosityLevel.WARNING)
		elif result_type == OptimizeResult.CRASHED:
			LOG("Crashed optimizing with {} on solution {} after {:.2f}s.".format(arguments_build_up, solution["args_used"], optimize_time), VerbosityLevel.ERROR)
		else:
			LOG("Successfully optimized with {} on solution {} after {:.2f}s.".format(arguments_build_up, solution["args_used"], optimize_time), VerbosityLevel.DETAIL)
		
		# If we retry optimization, update previous one. Otherwise, append it
		if previous_optimize_attempt:
			previous_optimize_attempt["command_used"] = command
			previous_optimize_attempt["output_file"] = output_file
			previous_optimize_attempt["args_used"] = arguments_build_up
			previous_optimize_attempt["optimize_time_python"] = optimize_time
			previous_optimize_attempt["actual_optimize_time"] = None
			previous_optimize_attempt["timed_out"] = result_type == OptimizeResult.TIMEOUT
			previous_optimize_attempt["crashed"] = result_type == OptimizeResult.CRASHED
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
				"timed_out": result_type == OptimizeResult.TIMEOUT,
				"crashed": result_type == OptimizeResult.CRASHED,
				"data": stats,
				"id": new_optimization_id
			}
			solution["optimizations"].append(optimization)

		write_comment_to_aig_file(output_file, "Optimized with ABC arguments:\n{}".format(arguments))

		new_optimization_id += 1

def execute_optimizations_on_solutions(
		profiler: ProfilerData,
		problem_files: list[dict],
		solve_argument_combos: list[list[str]] | None,
		optimization_argument_combos: list[list[str]],
		timeout_seconds: float,
		n_threads: int = 1):
	""" Optimizes the given solutions (or all if "None") of the given problem files with the given optimization arguments.\n
		Performs optimizations on different threads on solution-level. """
	global KEYBOARD_INTERRUPT_HAS_BEEN_CALLED, LAST_TIME_SAVED

	if n_threads < 1: raise Exception("Cannot perform optimization on no threads.")

	# Ensure base folder for optimization results exists
	if not MINIMIZED_AIG_FOLDER.is_dir(): MINIMIZED_AIG_FOLDER.mkdir()
	
	_N_PROBLEMS = len(problem_files)
	_I_PROBLEM = 0
	LOG_PROGRESS("Starting to optimize solutions...")
	for problem_file in tqdm(problem_files, desc="problem files", position=0):
		_I_PROBLEM += 1
		if problem_file["known_unrealizable"]:
			LOG("Skipping unrealizable problem file: '{}', as it cannot be optimized".format(problem_file["source"]), VerbosityLevel.INFO)
			continue # We cannot optimize unrealizable solutions


		# Ensure problem file specific output folder
		problem_file_source = Path(problem_file["source"])
		LOG_PROGRESS(" -> {}/{}: Optimizing: '{}'".format(_I_PROBLEM, _N_PROBLEMS, problem_file_source))
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
		
		LOG_PROGRESS(" -> {}/{}: Stopped optimizing: '{}'".format(_I_PROBLEM, _N_PROBLEMS, problem_file_source))
		time_since_last_save = time.time() - LAST_TIME_SAVED
		if time_since_last_save > 120:
			LOG("More than 120 seconds passed: {:.2f}s. Saving".format(time_since_last_save), VerbosityLevel.OFF)
			profiler.save()
		else:
			LOG("No 120 seconds have passed yet: {:.2f}s".format(time_since_last_save), VerbosityLevel.OFF)

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
		
		LOG_PROGRESS("    -> Optimized all of solution: {}".format(arguments_used_for_solution))

		# Now add the solution we were optimizing to the finished pile
		with finished_list_mutex:
			finished_solutions.append(target_solution)

# ======================== Argument Creators ===================== #
def get_knor_flag_combinations(test_size: TestSize, binary_out: bool = True) -> list[list[str]]:
	""" Returns a list of all argument combinations to give to Knor.
		If binary_out is False, '-a' will be appended instead '-b'."""
	if test_size == TestSize.Small: return [["--sym", "-b"]] # Only with small test sizes, immediately return '--sym'

	# Get all possible combinations of knor args
	knor_flag_combinations = []
	for i in range(len(KNOR_SYNTHESIZE_FLAGS) + 1):
		l = []
		for c in itertools.combinations(KNOR_SYNTHESIZE_FLAGS, i):
			l.append(c)
		knor_flag_combinations.extend(l)

	all_flag_combinations = []
	# Now, combine knor synthesize flag combos with every possible knor solve flag
	for oink_arg in KNOR_SOLVE_FLAGS:
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

def get_all_ABC_optimization_arguments(test_size: TestSize) -> list[str]:
	""" Returns list of all individual ABC optimization commands and their variations. """
	mutate_level = min(test_size.value, TestSize.Medium.value)

	target_arguments = list(ABC_OPTIMIZATION_ARGUMENTS.keys())
	if test_size == TestSize.Small: target_arguments = ["b", "drw", "drf"]

	arguments: list[str] = []
	for argument in target_arguments:
		arguments.extend(get_ABC_argument_variations(argument, ABC_OPTIMIZATION_ARGUMENTS[argument], mutate_level))
	return arguments

def get_all_ABC_optimization_duos(test_size: TestSize) -> list[list[str]]:
	""" Creates pairs of every ABC optimization command. """
	duos = []
	all_optimizations = get_all_ABC_optimization_arguments(test_size)
	for first in all_optimizations:
		for second in all_optimizations:
			duo = [first, second]
			duos.append(duo)
	return duos

def get_ABC_optimization_duplication_combos(test_size: TestSize) -> list[list[str]]:
	""" Creates a list of optimization argument combos for performing duplication effectiveness tests. 
		A, BA, BBA, BBBA, B, AB, AAB, AAB, AAAB etc. """
	duplication_combos: list[list[str]] = []

	repetition_depth = min(2, test_size.value)

	all_argument_variations = get_all_ABC_optimization_arguments(test_size)
	for head_of_chain in all_argument_variations:
		duplication_combos.append([head_of_chain])

		for tail in all_argument_variations:
			if head_of_chain.split(" ")[0] == tail.split(" ")[0]: continue
			for depth_level in range(1, repetition_depth):
				chain = [tail] * depth_level + [head_of_chain]
				duplication_combos.append(chain)

	return duplication_combos

def get_all_ABC_cleanup_arguments(test_size: TestSize) -> list[str]:
	""" Returns list of all individual ABC cleanup commands and their variations. """
	mutate_level = min(test_size.value, TestSize.Medium.value)

	target_arguments = list(ABC_CLEANUP_ARGUMENTS.keys())
	if test_size == TestSize.Small: target_arguments = ["trim", "cleanup"]

	arguments: list[str] = []
	for argument in target_arguments:
		arguments.extend(get_ABC_argument_variations(argument, ABC_CLEANUP_ARGUMENTS[argument], mutate_level))
	return arguments

def get_ABC_cleanup_arguments_sandwiched_in_optimization_duos(test_size: TestSize) -> list[list[str]]:
	""" Returns a list of all optimization duo argument combos with a cleanup argument in between. """
	duos = get_all_ABC_optimization_duos(test_size)
	cleanup_arguments = get_all_ABC_cleanup_arguments(test_size)

	sandwiched_triples: list[list[str]] = []
	for cleanup_argument in cleanup_arguments:
		for first, second in duos:
			triple = [first, cleanup_argument, second]
			sandwiched_triples.append(triple)
	return sandwiched_triples

def get_ABC_premade_optimization_strategies() -> dict[str, list[str]]:
	""" Returns a dict of all already existing optimization strategies names with their list of optimization commands. """
	return ABC_PREMADE_STRATEGIES

def get_ABC_custom_optimization_strategies() -> list[list[str]]:
	""" Returns a list of optimization strategies this research came up with. """
	return ABC_CUSTOM_STRATEGIES

def get_balance_optimize_combos(test_size: TestSize, index) -> list[list[str]]:
	""" Get all [balance, optimize] ABC argument combinations. """
	optimize_arguments = get_all_ABC_optimization_arguments(test_size)
	mutate_level = min(test_size.value, TestSize.Medium.value)

	balance_arguments = get_ABC_argument_variations("b", ABC_CLEANUP_ARGUMENTS["b"], mutate_level)[index*4:index*4+4]

	balance_optimize_combos = []
	for balance_argument in balance_arguments:
		for optimize_argument in optimize_arguments:
			balance_optimize_combos.append([balance_argument, optimize_argument])
	return balance_optimize_combos

def get_optimize_cleanup_combos(test_size: TestSize) -> list[list[str]]:
	""" Get all [optimize, cleanup] ABC argument combinations. """
	optimize_arguments = get_all_ABC_optimization_arguments(test_size)
	mutate_level = min(test_size.value, TestSize.Medium.value)
	cleanup_arguments = get_all_ABC_cleanup_arguments(test_size)

	combos = []
	for opt in optimize_arguments:
		for clean in cleanup_arguments:
			combos.append([opt, clean])
	return combos


def get_problem_files(profiler: ProfilerData, test_size: TestSize) -> list[dict]:
	""" Returns a list of problem files whose name match the given regex, or all files if regex is None."""
	all_problem_files = profiler.data["problem_files"]

	target_problem_files = []
	for problem_file in all_problem_files[::10][:len(all_problem_files[::10])]:
		if problem_file["known_unrealizable"]: continue
		target_problem_files.append(problem_file)

	return target_problem_files

	# if test_size == TestSize.Everything: return all_problem_files # All files
	# if test_size == TestSize.Big: return all_problem_files[::2][:len(all_problem_files[::2])] # Every other file
	# if test_size == TestSize.Medium: return all_problem_files[::5][:len(all_problem_files[::5])] # Every fifth file
	# if test_size == TestSize.Small: return all_problem_files[::20][:len(all_problem_files[::20])] # Every 20th file

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
								 handled_optimization_args: list[str],
								 handled_optimization_ids: list[int],
								 source: str):
	if not "args_used" in optimization: raise Exception("Missing 'args_used' attribute in optimization of source: {}".format(source))
	if not isinstance(optimization["args_used"], list): raise Exception("Optimization of solution had invalid 'args_used' of source: {}".format(source))
	if not all(isinstance(x, str) for x in optimization["args_used"]): raise Exception("Invalid argument of 'args_used' in optimization of source: {}".format(source))

	opt_args_used = optimization["args_used"]
	if "".join(opt_args_used) in handled_optimization_args: raise Exception("Optimization duplicated: {} of source: {}".format(opt_args_used, source))
	handled_optimization_args.append("".join(opt_args_used))

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
	if not "crashed" in optimization: raise Exception("Missing 'crashed' attrbibute in optimization {} of source: {}".format(opt_args_used, source))
	if not isinstance(optimization["crashed"], bool): raise Exception("Invalid type for 'crashed' in opt {} of source: {}".format(opt_args_used, source))
	if not "id" in optimization: raise Exception("Missing 'id' attrbibute in optimization {} of source: {}".format(opt_args_used, source))
	if not isinstance(optimization["id"], int): raise Exception("Invalid type for 'id' in opt {} of source: {}".format(opt_args_used, source))

	if optimization["id"] in handled_optimization_ids: raise Exception("Duplicate optimization id: {} in optimization {} of source: {}".format(optimization["id"], opt_args_used, source))
	handled_optimization_ids.append(optimization["id"])

	if not "data" in optimization: raise Exception("Missing 'data' attribute in optimization {} of source: {}".format(opt_args_used, source))
	
	if optimization["timed_out"] or optimization["crashed"]:
		if isinstance(optimization["data"], dict): raise Exception("Failed optimization still has data in optimization {} of source: {}".format(opt_args_used, source))
	else:
		if not isinstance(optimization["data"], dict): raise Exception("Successful optimization does not have data in optimization {} of source: {}".format(opt_args_used, source))
		check_aig_data_structure(optimization["data"], "optimization {} of {}".format(opt_args_used, source))

def check_solve_attempt_structure(solve_attempt: dict,
								  handled_solve_attempt_args: list[str],
								  source: str):
	if not "args_used" in solve_attempt: raise Exception("Missing 'args_used' attribute in solve attempt of '{}'".format(source))
	if not type(solve_attempt["args_used"]) is list: raise Exception("Problem file '{}' had solution with invalid type for 'args_used' instead of list".format(source))
	
	solve_args_used = solve_attempt["args_used"]
	if "".join(solve_args_used) in handled_solve_attempt_args: raise Exception("Solve attempt duplicated: {} for '{}'".format(solve_args_used, source))
	handled_solve_attempt_args.append("".join(solve_args_used))

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

	handled_optimization_args: list[str] = []
	handled_optimization_ids: list[int] = []

	for optimization in solve_attempt["optimizations"]:
		if KEYBOARD_INTERRUPT_HAS_BEEN_CALLED.is_set(): return
		check_optimization_structure(optimization, handled_optimization_args, handled_optimization_ids, "solution {} of problem file '{}'".format(solve_args_used, source))

def check_profiler_structure_correctness(profiler: ProfilerData):
	""" Prints all errors of the given profilers data."""
	handled_problem_file_sources: list[str] = []

	for problem_file in profiler.data["problem_files"]:
		if KEYBOARD_INTERRUPT_HAS_BEEN_CALLED.is_set(): return

		if not "source" in problem_file: raise Exception("Missing 'source' attribute in problem file")
		if not type(problem_file["source"]) is str: raise Exception("Problem file 'source' type was not str")

		source = problem_file["source"]
		if source in handled_problem_file_sources: raise Exception("Problem file is duplicate: '{}'".format(source))
		handled_problem_file_sources.append(source)

		if not "known_unrealizable" in problem_file: raise Exception("Missing 'known_unrealizable' attribute in '{}'".format(source))
		if not type(problem_file["known_unrealizable"]) is bool: raise Exception("Problem file '{}' had invalid type for 'known_unrealizable' instead of boolean".format(source))
		if not "solve_attempts" in problem_file: raise Exception("Missing 'solve_attempts' attribute in '{}'".format(source))
		if not type(problem_file["solve_attempts"]) is list: raise Exception("Problem file '{}' had invalid type for 'solve_attempts' instead of list".format(source))

		handled_solve_attempt_args: list[str] = []

		if not problem_file["known_unrealizable"]:
			for solve_attempt in problem_file["solve_attempts"]:
				if KEYBOARD_INTERRUPT_HAS_BEEN_CALLED.is_set(): return
				check_solve_attempt_structure(solve_attempt, handled_solve_attempt_args, "problem file '{}'".format(source))

	LOG("Finished checking profiler structure correctness", VerbosityLevel.INFO)

def fix_profiler_structure(profiler: ProfilerData, use_tqdm: bool = False):
	for problem_file in tqdm(profiler.data["problem_files"], desc="problem files", position=0, leave=False, disable=not use_tqdm):
		if KEYBOARD_INTERRUPT_HAS_BEEN_CALLED.is_set(): return
		
		if problem_file["known_unrealizable"]: continue

		source = problem_file["source"]
		for solve_attempt in tqdm(problem_file["solve_attempts"], desc="solve_attempt", leave=False, position=1, disable=not use_tqdm):
			if KEYBOARD_INTERRUPT_HAS_BEEN_CALLED.is_set(): return
			
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

			for optimization in tqdm(solve_attempt["optimizations"], desc="optimization", position=2, leave=False, disable=not use_tqdm):
				if KEYBOARD_INTERRUPT_HAS_BEEN_CALLED.is_set(): return

				if "crashed" not in optimization:
					optimization["crashed"] = False

				if optimization["timed_out"] or optimization["crashed"]: continue
				
				if not "data" in optimization:
					optimization["data"] = None

				if not optimization["data"]:
					# Then this optimization needs to have data!
					stats = get_aig_stats_from_file(optimization["output_file"])
					if not stats:
						# If we failed to parse it, it probably was a crashed optimization
						LOG("Added crashed: True", VerbosityLevel.INFO)
						optimization["crashed"] = True
					else:
						optimization["data"] = stats
						LOG("Read and set AIG stats of opt {} of solution {} of problem '{}'".format(optimization["args_used"], solve_attempt["args_used"], source), VerbosityLevel.INFO)


# =================== TESTS ======================= #
# TEST: Balance before each opt command -> 6 hours
# TEST: Cleanup after each opt command -> 6 hours

# TOnight:
# - Fix two graphs into paper
# - Fix todos in paper
# - Improve and add references

def test_8(thread_count: int, optimize_timeout_s: float, test_size: TestSize):
	profiler = ProfilerData(PROFILER_SOURCE)
	target_problem_files = get_problem_files(profiler, test_size)
	target_knor_arg_combos = get_knor_flag_combinations(test_size)

	target_opt_combos = list(get_ABC_premade_optimization_strategies().values())
	target_opt_combos.extend(get_ABC_custom_optimization_strategies())

	execute_optimizations_on_solutions(profiler, target_problem_files, target_knor_arg_combos, target_opt_combos, optimize_timeout_s, n_threads=thread_count)
	profiler.save()
	profiler.backup("8_AFTER_8")

def fifth_try(thread_count: int, optimize_timeout_s: float, test_size: TestSize, index: int):
	profiler = ProfilerData(PROFILER_SOURCE)
	test_7(profiler, test_size, thread_count, optimize_timeout_s, index)
	profiler.save()
	profiler.backup("5_AFTER_TEST_7_i{}".format(index))

def test_7(profiler: ProfilerData, test_size: TestSize, thread_count: int, optimize_timeout_s: float, index: int):
	""" Do balancing before optimizations """
	LOG_PROGRESS("Starting the test 7 with i: {}".format(index))
	target_problem_files = get_problem_files(profiler, test_size)
	target_knor_arg_combos = get_knor_flag_combinations(test_size)
	balance_optimization_combos = get_balance_optimize_combos(test_size, index)
	LOG_PROGRESS("Found: {}".format(len(balance_optimization_combos)))
	execute_optimizations_on_solutions(profiler, target_problem_files, target_knor_arg_combos, balance_optimization_combos, timeout_seconds=optimize_timeout_s, n_threads=thread_count)


def third_try(thread_count: int, optimize_timeout_s: float, test_size: TestSize):
	profiler = ProfilerData(PROFILER_SOURCE)
	a: tuple[pd.DataFrame, pd.DataFrame, list[str]] = get_test_1_data(profiler, test_size)

	# Do cleanup bois
	test_4(profiler, TestSize.Big, thread_count, optimize_timeout_s)
	profiler.backup("3_AFTER_TEST_4")
	profiler.save()

	if KEYBOARD_INTERRUPT_HAS_BEEN_CALLED.is_set(): return

	# Do strategies
	test_6(profiler, TestSize.Big, thread_count, optimize_timeout_s)
	profiler.backup("3_AFTER_TEST_6")
	profiler.save()

	if KEYBOARD_INTERRUPT_HAS_BEEN_CALLED.is_set(): return

	# Do duos
	test_2_decreased(profiler, a[2], test_size, thread_count, optimize_timeout_s)
	profiler.backup("3_AFTER_TEST_2")
	profiler.save()

	if KEYBOARD_INTERRUPT_HAS_BEEN_CALLED.is_set(): return

def do_tests(thread_count: int, solve_timeout_s: float, optimize_timeout_s: float, test_size: TestSize):
	# 1. Initialize profiler
	LOG("Intializing profiler...", VerbosityLevel.INFO)
	profiler = ProfilerData(PROFILER_SOURCE)
	initialize_problem_files(profiler)
	profiler.save()
	LOG("Initialized profiler!", VerbosityLevel.INFO)

	if KEYBOARD_INTERRUPT_HAS_BEEN_CALLED.is_set(): return

	LOG("Solving files...", VerbosityLevel.INFO)
	# 2. Solve all problem files
	solve_all_problem_files(profiler, test_size, solve_timeout_s)
	profiler.save()
	LOG("Solved all problem files!", VerbosityLevel.INFO)

	if KEYBOARD_INTERRUPT_HAS_BEEN_CALLED.is_set(): return

	# 3. Perform test 1
	LOG("Performing opts!", VerbosityLevel.INFO)
	test_1(profiler, test_size, thread_count, optimize_timeout_s)
	profiler.save()
	profiler.backup("TEST1")
	LOG("Performed test 1: all optimizations once!", VerbosityLevel.INFO)

	if KEYBOARD_INTERRUPT_HAS_BEEN_CALLED.is_set(): return

	LOG("Performing opts2!", VerbosityLevel.INFO)
	# 4. Perform test 2
	test_2(profiler, test_size, thread_count, optimize_timeout_s)
	profiler.save()
	profiler.backup("TEST2")
	LOG("Performed test 2: all duos!", VerbosityLevel.INFO)

	if KEYBOARD_INTERRUPT_HAS_BEEN_CALLED.is_set(): return

	# 5 is procrastinated downwards!

	# 6. Perform test 4
	test_4(profiler, test_size, thread_count, optimize_timeout_s)
	profiler.save()
	profiler.backup("TEST4")
	LOG("Performed test 4: cleanups on solutions!", VerbosityLevel.INFO)

	if KEYBOARD_INTERRUPT_HAS_BEEN_CALLED.is_set(): return

	# 7. Perform test 5
	test_5(profiler, test_size, thread_count, optimize_timeout_s)
	profiler.save()
	profiler.backup("TEST5")
	LOG("Performed test 5: sandwiched cleanups!", VerbosityLevel.INFO)

	if KEYBOARD_INTERRUPT_HAS_BEEN_CALLED.is_set(): return

	# 8. Perform test 6
	test_6(profiler, test_size, thread_count, optimize_timeout_s)
	profiler.save()
	profiler.backup("TEST6")
	LOG("Performed test 6: premade strategies!", VerbosityLevel.INFO)

	if KEYBOARD_INTERRUPT_HAS_BEEN_CALLED.is_set(): return

	# 5. Perform test 3
	test_3(profiler, test_size, thread_count, optimize_timeout_s)
	profiler.save()
	profiler.backup("TEST3")
	LOG("Performed test 3: duplications!", VerbosityLevel.INFO)

	if KEYBOARD_INTERRUPT_HAS_BEEN_CALLED.is_set(): return


def solve_all_problem_files(profiler: ProfilerData, test_size: TestSize, solve_timeout_s: float):
	""" Will create a solution for every possible example problem file. """
	target_problem_files = get_problem_files(profiler, test_size)
	target_knor_arg_combos = get_knor_flag_combinations(test_size)
	solve_problem_files(target_problem_files, target_knor_arg_combos, solve_timeout_seconds=solve_timeout_s)

def test_1(profiler: ProfilerData, test_size: TestSize, thread_count: int, optimize_timeout_s: float):
	""" Do all ABC optimizations once on each solution. """
	target_problem_files = get_problem_files(profiler, test_size)
	target_knor_arg_combos = get_knor_flag_combinations(test_size)
	optimizations = get_all_ABC_optimization_arguments(test_size)
	optimization_combos = list(map(lambda x: [x], optimizations))
	execute_optimizations_on_solutions(profiler, target_problem_files, target_knor_arg_combos, optimization_combos, timeout_seconds=optimize_timeout_s, n_threads=thread_count)

def test_2(profiler: ProfilerData, test_size: TestSize, thread_count: int, optimize_timeout_s: float):
	""" Do all ABC optimization duos on each solution. """
	target_problem_files = get_problem_files(profiler, test_size)
	target_knor_arg_combos = get_knor_flag_combinations(test_size)
	optimization_duos = get_all_ABC_optimization_duos(test_size)
	execute_optimizations_on_solutions(profiler, target_problem_files, target_knor_arg_combos, optimization_duos, timeout_seconds=optimize_timeout_s, n_threads=thread_count)

def test_2_decreased(profiler: ProfilerData, target_abc_args: list[str], test_size: TestSize, thread_count: int, optimize_timeout_s: float):
	""" Do all ABC optimization duos on each solution. """
	target_problem_files = get_problem_files(profiler, test_size)
	target_knor_arg_combos = get_knor_flag_combinations(test_size)
	
	duos: list[list[str]] = []
	for first in target_abc_args:
		for second in target_abc_args:
			duo = [first, second]
			duos.append(duo)
	
	execute_optimizations_on_solutions(profiler, target_problem_files, target_knor_arg_combos, duos, timeout_seconds=optimize_timeout_s, n_threads=thread_count)


def test_3(profiler: ProfilerData, test_size: TestSize, thread_count: int, optimize_timeout_s: float):
	""" Do the duplication test: See if repeating same argument is effective. """
	target_problem_files = get_problem_files(profiler, test_size)
	target_knor_arg_combos = get_knor_flag_combinations(test_size)
	optimization_combos = get_ABC_optimization_duplication_combos(test_size)
	execute_optimizations_on_solutions(profiler, target_problem_files, target_knor_arg_combos, optimization_combos, timeout_seconds=optimize_timeout_s, n_threads=thread_count)

def test_4(profiler: ProfilerData, test_size: TestSize, thread_count: int, optimize_timeout_s: float):
	""" Performs cleanup optimization commands on unoptimized solutions. """
	target_problem_files = get_problem_files(profiler, test_size)
	target_knor_arg_combos = get_knor_flag_combinations(test_size)
	cleanup_arguments = get_all_ABC_cleanup_arguments(test_size)
	cleanup_combos = list(map(lambda x: [x], cleanup_arguments))
	execute_optimizations_on_solutions(profiler, target_problem_files, target_knor_arg_combos, cleanup_combos, timeout_seconds=optimize_timeout_s, n_threads=thread_count)

def test_5(profiler: ProfilerData, test_size: TestSize, thread_count: int, optimize_timeout_s: float):
	""" Sandwich cleanup arguments between optimization duos to see if it has actual effect. """
	target_problem_files = get_problem_files(profiler, test_size)
	target_knor_arg_combos = get_knor_flag_combinations(test_size)
	sandwiched_cleanups = get_ABC_cleanup_arguments_sandwiched_in_optimization_duos(test_size)
	execute_optimizations_on_solutions(profiler, target_problem_files, target_knor_arg_combos, sandwiched_cleanups, timeout_seconds=optimize_timeout_s, n_threads=thread_count)

def test_6(profiler: ProfilerData, test_size: TestSize, thread_count: int, optimize_timeout_s: float):
	""" Perform all premade optimization strategies. """
	target_problem_files = get_problem_files(profiler, test_size)
	target_knor_arg_combos = get_knor_flag_combinations(test_size)
	strategy_combos = list(get_ABC_premade_optimization_strategies().values())
	execute_optimizations_on_solutions(profiler, target_problem_files, target_knor_arg_combos, strategy_combos, timeout_seconds=optimize_timeout_s, n_threads=thread_count)


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

# TODO: Make solving also parrallel


# ============================================================================
 
"""
I want to compare each indivual command



"""

def get_crashes_or_incompletes(problem_files: list[dict], knor_args: list[list[str]], opt_args: list[list[str]]) -> tuple[
		list[str], # Crashed files
		list[str], # Crashed solve args
		list[str], # Crashed opt args
		list[str], # Missing files
		list[str], # Missing solve args
		list[str], # Missing opt args
		]:
	target_problem_files = [file["source"] for file in problem_files]
	target_knor_args = ["".join(arg_combo) for arg_combo in knor_args]
	target_opt_args = ["".join(arg_combo) for arg_combo in opt_args]

	crashed_problem_files: list[str] = []
	crashed_knor_args: list[str] = []
	crashed_opt_args: list[str] = []
	missing_problem_files: list[str] = []
	missing_knor_args: list[str] = []
	missing_opt_args: list[str] = []

	# Keep track of the files that are realizable
	succeeded_problem_files: list[str] = []

	for problem_file in problem_files:
		source = problem_file["source"]
		# Only check target problem files
		if source not in target_problem_files: continue
		# If file has been unrealizable before or missing before, skip this one
		if source in crashed_problem_files or source in missing_problem_files: continue

		# If this file is unrealizable, remember it
		if problem_file["known_unrealizable"]:
			crashed_problem_files.append(source)
			LOG("Problem file: '{}' is unrealizable".format(source), VerbosityLevel.WARNING)
			continue

		# Remember that this file succeeded
		succeeded_problem_files.append(source)
		
		# Keep track of the solve attempts that succeeded
		succeeded_solve_args = []

		for solve_attempt in problem_file["solve_attempts"]:
			knor_args_used = "".join(solve_attempt["args_used"])
			# Only check target solve attempts
			if knor_args_used not in target_knor_args: continue
			# If we know these knor args have previously crashed or were missing, dont care about them then
			if knor_args_used in crashed_knor_args or knor_args_used in missing_knor_args: continue
			
			# If these knor_args have crashed or timed out, remember it
			if solve_attempt["crashed"] or solve_attempt["timed_out"]:
				crashed_knor_args.append(knor_args_used)
				reason = "crashed" if solve_attempt["crashed"] else "timed out"
				LOG("Knor args: {} have {}".format(knor_args_used, reason), VerbosityLevel.WARNING)
				continue
			
			# Remember that these solve args succeeded
			succeeded_solve_args.append(knor_args_used)

			# Keep track of successful optimization argument combos
			succeeded_opt_args: list[str] = []

			for optimization in solve_attempt["optimizations"]:
				opt_args_used = "".join(optimization["args_used"])
				# Only check target opt args
				if opt_args_used not in target_opt_args: continue
				# Skip if we already know these opt args have previously crashed or missed
				if opt_args_used in missing_opt_args or opt_args_used in crashed_opt_args: continue

				# If these opt args have crashed or timed out, remember it
				if optimization["crashed"] or optimization["timed_out"]:
					crashed_opt_args.append(opt_args_used)
					LOG("Optimization: {} crashed".format(opt_args_used), VerbosityLevel.WARNING)
					continue

				# Remember that these opt args succeeded
				succeeded_opt_args.append(opt_args_used)

			for opt_arg in target_opt_args:
				test = "".join(opt_arg)
				# If it is one of the crashes or missing ones, we do not care
				if test in crashed_opt_args or test in missing_opt_args: continue
				# Otherwise, it should have succeeded!
				if not test in succeeded_opt_args:
					LOG("Did not (yet) perform optimization: {}".format(test), VerbosityLevel.WARNING)
					missing_opt_args.append(test)
			
		
		for knor_arg in target_knor_args:
			test = "".join(knor_arg)
			# If it is one of the crashes or missing ones, we do not care
			if test in crashed_knor_args or test in missing_knor_args: continue
			# But if it did not crash and is not missing yet, it should have succeeded
			if test not in succeeded_solve_args:
				LOG("Did not (yet) perform solve attempt with: {}".format(test), VerbosityLevel.WARNING)
				missing_knor_args.append(test)

	for problem_file in problem_files:
		source = problem_file["source"]
		# If this is one of the failed or missing problem files, ignore this one
		if source in crashed_problem_files or source in missing_problem_files: continue
		# But if this one should have succeeded, check if it actually did
		if source not in succeeded_problem_files:
			LOG("Did not (yet) solve problem file: '{}'".format(source), VerbosityLevel.WARNING)
			missing_problem_files.append(source)
	
	ok_problem_files: int = len(succeeded_problem_files)
	ok_knor_args: int = len(target_knor_args) - len(crashed_knor_args) - len(missing_knor_args)
	ok_abc_args: int = len(target_opt_args) - len(crashed_opt_args) - len(missing_opt_args)

	LOG("OK problem files: {}/{}, OK knor args: {}/{}. OK ABC args: {}/{}".format(
		ok_problem_files, 
		len(target_problem_files), 
		ok_knor_args, 
		len(target_knor_args), 
		ok_abc_args, 
		len(target_opt_args)
		), VerbosityLevel.INFO)

	return crashed_problem_files, crashed_knor_args, crashed_opt_args, missing_problem_files, missing_knor_args, missing_opt_args


def get_test_1_data(profiler: ProfilerData, test_size: TestSize):
	target_problem_files: list[dict] = get_problem_files(profiler, test_size)
	target_knor_arg_combos: list[list[str]] = get_knor_flag_combinations(test_size)
	all_optimizations: list[str] = get_all_ABC_optimization_arguments(test_size)
	target_opt_arg_combos: list[list[str]] = list(map(lambda x: [x], all_optimizations))

	crashed_files, crashed_knor_args, crashed_opt_args, missing_files, missing_knor_args, missing_opt_args = get_crashes_or_incompletes(target_problem_files, target_knor_arg_combos, target_opt_arg_combos)

	succeeded_problem_files: list[str] = [file["source"] for file in target_problem_files if file["source"] not in crashed_files and file["source"] not in missing_files]
	succeeded_solve_args: list[str] = ["".join(args) for args in target_knor_arg_combos if "".join(args) not in crashed_knor_args and "".join(args) not in missing_knor_args]
	succeeded_opt_args: list[str] = ["".join(args) for args in target_opt_arg_combos if "".join(args) not in crashed_opt_args and "".join(args) not in missing_opt_args]

	# for problem_file in succeeded_problem_files:
	# 	LOG("Using problem file: '{}'".format(problem_file), VerbosityLevel.OFF)
	# for solve_arg in succeeded_solve_args:
	# 	LOG("Using solve args: {}".format(solve_arg), VerbosityLevel.OFF)
	# for opt_arg in succeeded_opt_args:
	# 	LOG("Using opt args: {}".format(opt_arg), VerbosityLevel.OFF)

	# Collect all optimization args and their corresponding gain that they created
	raw_data = {
		"opt_args": [],
		"gain": []
	}

	for problem_file in target_problem_files:
		if problem_file["source"] not in succeeded_problem_files:
			LOG("Skipping required problem file data!", VerbosityLevel.ERROR)
			continue
		for solve_attempt in problem_file["solve_attempts"]:
			if "".join(solve_attempt["args_used"]) not in succeeded_solve_args:
				LOG("Skipping required knor opt data!", VerbosityLevel.ERROR)
				continue

			initial_AND_count: int = solve_attempt["data"]["and_gates"]

			for optimization in solve_attempt["optimizations"]:
				arg_used = "".join(optimization["args_used"])
				if arg_used not in succeeded_opt_args:
					LOG("Skipping required ABC opt data!", VerbosityLevel.ERROR)
					continue

				current_AND_count: int = optimization["data"]["and_gates"]
				gain = initial_AND_count / current_AND_count

				raw_data["opt_args"].append(arg_used)
				raw_data["gain"].append(gain)

	# Now, pick the best N arguments!
	df = pd.DataFrame(raw_data)
	sorted_args_with_gains = df.groupby(["opt_args"]).agg({"gain": "median"}).sort_values(["gain"], ascending=False).reset_index()
	best_n_args = 20
	best_args = sorted_args_with_gains["opt_args"].values[:best_n_args]
	plot_data = df[df["opt_args"].isin(best_args)].reset_index()
	
	plot_data.to_csv(Path("TEST_1_RESULTS.csv"))

	test_2_n = int(round(len(sorted_args_with_gains) / 2))
	test_2_args = sorted_args_with_gains["opt_args"].values[:test_2_n]

	return df, plot_data, list(test_2_args)


def get_test_4_data(profiler: ProfilerData, test_size: TestSize):
	""" Data of cleanup optimization commands on unoptimized solutions. """
	target_problem_files = get_problem_files(profiler, test_size)
	target_knor_arg_combos = get_knor_flag_combinations(test_size)
	cleanup_arguments = get_all_ABC_cleanup_arguments(test_size)
	cleanup_combos = list(map(lambda x: [x], cleanup_arguments))

	crashed_files, crashed_knor_args, crashed_opt_args, missing_files, missing_knor_args, missing_opt_args = get_crashes_or_incompletes(target_problem_files, target_knor_arg_combos, cleanup_combos)

	succeeded_problem_files: list[str] = [file["source"] for file in target_problem_files if file["source"] not in crashed_files and file["source"] not in missing_files]
	succeeded_solve_args: list[str] = ["".join(args) for args in target_knor_arg_combos if "".join(args) not in crashed_knor_args and "".join(args) not in missing_knor_args]
	succeeded_opt_args: list[str] = ["".join(args) for args in cleanup_combos if "".join(args) not in crashed_opt_args and "".join(args) not in missing_opt_args]

	raw_data = {
		"opt_args": [],
		"gain": []
	}

	for problem_file in target_problem_files:
		source = problem_file["source"]
		if source not in succeeded_problem_files:
			continue

		for solve_attempt in problem_file["solve_attempts"]:
			knor_args_used = "".join(solve_attempt["args_used"])
			if knor_args_used not in succeeded_solve_args:
				continue

			original_AND_gate_count = solve_attempt["data"]["and_gates"]

			for optimization in solve_attempt["optimizations"]:
				abc_args_used = "".join(optimization["args_used"])
				if abc_args_used not in succeeded_opt_args:
					continue

				new_AND_gate_count = optimization["data"]["and_gates"]
				gain = original_AND_gate_count / new_AND_gate_count

				raw_data["opt_args"].append(abc_args_used)
				raw_data["gain"].append(gain)

	return pd.DataFrame(raw_data)

def get_test_6_data(profiler: ProfilerData, test_size: TestSize):
	""" Data of all premade optimization strategies. """
	target_problem_files = get_problem_files(profiler, test_size)
	target_knor_arg_combos = get_knor_flag_combinations(test_size)
	target_strategies = get_ABC_premade_optimization_strategies()

	crashed_files, crashed_knor_args, crashed_opt_args, missing_files, missing_knor_args, missing_opt_args = get_crashes_or_incompletes(target_problem_files, target_knor_arg_combos, list(target_strategies.values()))

	succeeded_problem_files: list[str] = [file["source"] for file in target_problem_files if file["source"] not in crashed_files and file["source"] not in missing_files]
	succeeded_solve_args: list[str] = ["".join(args) for args in target_knor_arg_combos if "".join(args) not in crashed_knor_args and "".join(args) not in missing_knor_args]
	succeeded_opt_args: list[str] = ["".join(args) for args in target_strategies.values() if "".join(args) not in crashed_opt_args and "".join(args) not in missing_opt_args]

	raw_data = {
		"optimization_strategy": [],
		"optimization_step": [],
		"gain_so_far": [],
		"argument_performed": []
	}

	for problem_file in target_problem_files:
		source = problem_file["source"]
		if source not in succeeded_problem_files:
			LOG("Skipping required problem file data: {}".format(source), VerbosityLevel.ERROR)
			continue
		for solve_attempt in problem_file["solve_attempts"]:
			knor_args_used = "".join(solve_attempt["args_used"])
			if knor_args_used not in succeeded_solve_args:
				continue
			
			base_AND_count = solve_attempt["data"]["and_gates"]

			for strategy in target_strategies:
				strategy_args = target_strategies[strategy]
				strategy_args_str: str = "".join(strategy_args)
				if strategy_args_str not in succeeded_opt_args:
					continue
				
				built_up = []
				for step in strategy_args:
					built_up.append(step)

					matching_opt = get_optimization_with_args(solve_attempt["optimizations"], built_up)
					if not matching_opt:
						LOG("Could not find necessary optimization: {}".format(built_up), VerbosityLevel.ERROR)
						continue
					
					current_AND_count = matching_opt["data"]["and_gates"]
					total_gain_so_far = base_AND_count / current_AND_count

					# We found it!
					raw_data["optimization_strategy"].append(strategy)
					raw_data["optimization_step"].append(len(built_up)-1)
					raw_data["gain_so_far"].append(total_gain_so_far)
					raw_data["argument_performed"].append(built_up[-1])

	df = pd.DataFrame(raw_data)
	return df

def get_test_7_data(profiler: ProfilerData, test_size: TestSize, index: int):
	target_problem_files = get_problem_files(profiler, test_size)
	target_knor_arg_combos = get_knor_flag_combinations(test_size)
	target_opt_combos = get_balance_optimize_combos(test_size, index)
	
	crashed_files, crashed_knor_args, crashed_opt_args, missing_files, missing_knor_args, missing_opt_args = get_crashes_or_incompletes(target_problem_files, target_knor_arg_combos, target_opt_combos)

	succeeded_problem_files: list[str] = [file["source"] for file in target_problem_files if file["source"] not in crashed_files and file["source"] not in missing_files]
	succeeded_solve_args: list[str] = ["".join(args) for args in target_knor_arg_combos if "".join(args) not in crashed_knor_args and "".join(args) not in missing_knor_args]
	succeeded_opt_args: list[str] = ["".join(args) for args in target_opt_combos if "".join(args) not in crashed_opt_args and "".join(args) not in missing_opt_args]

	raw_data = {
		"full_optimization": [], # The full optimization combo as str
		"full_gain": [],
		"balance_gain": [],
		"opt_gain": []
	}

	for problem_file in target_problem_files:
		source = problem_file["source"]
		if source not in succeeded_problem_files:
			LOG("Skipping required problem file data: {}".format(source), VerbosityLevel.ERROR)
			continue
		for solve_attempt in problem_file["solve_attempts"]:
			knor_args_used = "".join(solve_attempt["args_used"])
			if knor_args_used not in succeeded_solve_args:
				continue
			
			base_AND_count = solve_attempt["data"]["and_gates"]

			for target_opt_combo in target_opt_combos:
				target_opt_combo_str: str = "".join(target_opt_combo)
				if target_opt_combo_str not in succeeded_opt_args:
					continue

				first_opt, second_opt = target_opt_combo

				matching_complete_opt = get_optimization_with_args(solve_attempt["optimizations"], target_opt_combo)
				if not matching_complete_opt:
					LOG("Could not find necessary optimization: {}".format(target_opt_combo), VerbosityLevel.ERROR)
					continue

				full_gain = base_AND_count / matching_complete_opt["data"]["and_gates"]

				matching_first_opt = get_optimization_with_args(solve_attempt["optimizations"], [first_opt])
				if not matching_first_opt:
					LOG("Cannot compare 'balance' with 'balance ; opt' if 'balance' itself has not been done", VerbosityLevel.ERROR)
					continue

				balance_gain = base_AND_count / matching_first_opt["data"]["and_gates"]

				matching_second_opt = get_optimization_with_args(solve_attempt["optimizations"], [second_opt])
				if not matching_second_opt:
					LOG("Cannot compare 'opt' with 'balance ; opt' if 'opt' itself has not been done", VerbosityLevel.ERROR)
					continue

				opt_gain = base_AND_count / matching_second_opt["data"]["and_gates"]

				raw_data["full_optimization"].append(target_opt_combo_str)
				raw_data["full_gain"].append(full_gain)
				raw_data["balance_gain"].append(balance_gain)
				raw_data["opt_gain"].append(opt_gain)
				
	return pd.DataFrame(raw_data)


def get_test_8_data(profiler: ProfilerData, test_size: TestSize) -> pd.DataFrame:
	""" Data of all premade optimization strategies + our own made one """
	target_problem_files = get_problem_files(profiler, test_size)
	target_knor_arg_combos = get_knor_flag_combinations(test_size)

	premade_strategies = get_ABC_premade_optimization_strategies()
	custom_strategies = get_ABC_custom_optimization_strategies()
	all_strategies = list(premade_strategies.values())
	all_strategies.extend(custom_strategies)

	crashed_files, crashed_knor_args, crashed_strategies, missing_files, missing_knor_args, missing_strategies = get_crashes_or_incompletes(target_problem_files, target_knor_arg_combos, all_strategies)

	succeeded_problem_files: list[str] = [file["source"] for file in target_problem_files if file["source"] not in crashed_files and file["source"] not in missing_files]
	succeeded_solve_args: list[str] = ["".join(args) for args in target_knor_arg_combos if "".join(args) not in crashed_knor_args and "".join(args) not in missing_knor_args]
	succeeded_strategies: list[str] = ["".join(args) for args in all_strategies if "".join(args) not in crashed_strategies and "".join(args) not in missing_strategies]

	raw_data = {
		"optimization_strategy": [],
		"optimization_step": [],
		"gain_so_far": [],
		"argument_performed": []
	}

	for problem_file in target_problem_files:
		source = problem_file["source"]
		if source not in succeeded_problem_files:
			LOG("Skipping required problem file data: {}".format(source), VerbosityLevel.ERROR)
			continue
		for solve_attempt in problem_file["solve_attempts"]:
			knor_args_used = "".join(solve_attempt["args_used"])
			if knor_args_used not in succeeded_solve_args:
				continue
			
			base_AND_count = solve_attempt["data"]["and_gates"]

			# Now check both premade strategies and later custom ones as well
			for strategy_name in premade_strategies:
				strategy_args = premade_strategies[strategy_name]
				strategy_args_str: str = "".join(strategy_args)
				if strategy_args_str not in succeeded_strategies:
					continue
				
				built_up = []
				for step in strategy_args:
					built_up.append(step)

					matching_opt = get_optimization_with_args(solve_attempt["optimizations"], built_up)
					if not matching_opt:
						LOG("Could not find necessary optimization: {}".format(built_up), VerbosityLevel.ERROR)
						continue
					
					current_AND_count = matching_opt["data"]["and_gates"]
					total_gain_so_far = base_AND_count / current_AND_count

					# We found it!
					raw_data["optimization_strategy"].append(strategy_name)
					raw_data["optimization_step"].append(len(built_up)-1)
					raw_data["gain_so_far"].append(total_gain_so_far)
					raw_data["argument_performed"].append(step)
			# Also check the custom strategies
			for i, strategy in enumerate(custom_strategies):
				strategy_name = "custom_{}".format(i)
				strategy_str = "".join(strategy)
				if strategy_str not in succeeded_strategies:
					continue

				built_up = []
				for step in strategy:
					built_up.append(step)

					matching_opt = get_optimization_with_args(solve_attempt["optimizations"], built_up)
					if not matching_opt:
						LOG("Could not find necessary optimization: {}".format(built_up), VerbosityLevel.ERROR)
						continue
					
					current_AND_count = matching_opt["data"]["and_gates"]
					total_gain_so_far = base_AND_count / current_AND_count

					# We found it!
					raw_data["optimization_strategy"].append(strategy_name)
					raw_data["optimization_step"].append(len(built_up)-1)
					raw_data["gain_so_far"].append(total_gain_so_far)
					raw_data["argument_performed"].append(step)

	df = pd.DataFrame(raw_data)
	return df

# =============================================================


def plot_test_1(test_size: TestSize = TestSize.Big):
	# profiler = ProfilerData(PROFILER_SOURCE)
	# df, _, _ = get_test_1_data(profiler, test_size)
	df = pd.read_csv(Path("TEST_1_PLOT_DATA.csv"))
	df.groupby(["opt_args"]).agg({"gain": "median"})

	sorted_order = df.groupby(["opt_args"]).agg({"gain":"median"}).sort_values(["gain"], ascending=False).reset_index()["opt_args"].values[:15]

	fitting_data = df[df["opt_args"].isin(sorted_order)]

	sns.set_theme()

	from matplotlib.font_manager import fontManager, FontProperties
	path = Path("TimesNewRoman.ttf")
	fontManager.addfont(path)
	prop = FontProperties(fname=path)
	sns.set(font=prop.get_name(), font_scale=2)

	plotted = sns.catplot(data=fitting_data, kind="boxen", x="gain", y="opt_args", order=sorted_order, height=10, aspect=8/10)
	
	plotted.set(xlabel="Gain", ylabel="ABC optimization command")
	
	# plt.xticks(rotation=90)
	
	# TODO: Specify to use LaTeX font

	plt.show()

	return df

def plot_test_4(test_size: TestSize = TestSize.Big):
	df = pd.read_csv(Path("TEST_4_PLOT_DATA.csv"))

	sorted_top_args = df.groupby(["opt_args"]).agg({"gain": "mean"}).sort_values(["gain"], ascending=False).reset_index()["opt_args"].values[:10]

	filtered_data = df[df["opt_args"].isin(sorted_top_args)]

	sns.set_theme()

	from matplotlib.font_manager import fontManager, FontProperties
	path = Path("TimesNewRoman.ttf")
	fontManager.addfont(path)
	prop = FontProperties(fname=path)
	sns.set(font=prop.get_name(), font_scale=3)

	plotted = sns.catplot(data=filtered_data, kind="boxen", x="gain", y="opt_args", hue="opt_args", dashes="opt_args", order=sorted_top_args, height=10, aspect=10/10)
	
	plotted.set(xlabel="Gain", ylabel="ABC optimization command")
	
	plt.xticks([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
	
	# TODO: Specify to use LaTeX font

	plt.show()


	return df



def plot_test_6(test_size: TestSize = TestSize.Big):
	df = pd.read_csv(Path("TEST_6_FINAL_PLOT_DATA.csv"))

	sns.set_theme()

	from matplotlib.font_manager import fontManager, FontProperties
	path = Path("TimesNewRoman.ttf")
	fontManager.addfont(path)
	prop = FontProperties(fname=path)
	sns.set(font=prop.get_name(), font_scale=2)

	half_data = list(df["optimization_strategy"].unique())[::2]
	other = list(df["optimization_strategy"].unique())[1::2]

	plotted1 = sns.relplot(data=df[df["optimization_strategy"].isin(other)], kind="line", x="optimization_step", y="gain_so_far", hue="optimization_strategy", height=10, aspect=10/5, linewidth=3)
	
	plotted1._legend.set_title("Strategy") # type: ignore
	plotted1.set(xlabel="Optimization step", ylabel="Total gain")
	plt.xticks([x for x in range(20)])
	plt.show()

def plot_test_8(test_size: TestSize):
	pass

# Plot sample example AIG sizes to see the distribution

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
