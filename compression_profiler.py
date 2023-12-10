from enum import Enum
import json
import itertools
import time
import subprocess
import re
from pathlib import Path

ABC_BINARY = Path("build/_deps/abc-build/abc")
ABC_ALIAS_SOURCE = Path("build/_deps/abc-src/abc.rc")
KNOR_BINARY = Path("build/knor")

PROFILING_FILE = Path("profiler.json")

UNMINIMIZED_AIG_FOLDER = Path("aigs_unminimized")
MINIMIZED_AIG_FOLDER = Path("aigs_minimized")
PROBLEM_FILES_FOLDER = Path("examples")

MAX_TIME_SECONDS_FOR_KNOR_COMMAND = 60 # seconds = 2 minutes

PROFILER_SOURCE = Path("profiler.json")

# These args can be combined
# Order does not matter
# No arguments is possibility too
KNOR_ARGS = [
	"--no-bisim",	# Because adding --bisim is default
	"--binary",		# Because --onehot is default
	"--isop",		# Because ITE is default
	
	# "--best", 	# To find the combo of --bisim/no-bisim, --isop/ite and --onehot/binary
	# "--compress" # No use of compress. This will be measured later
]

# Exactly one arg should be selected at a time
OINK_SOLVER_ARGS = [
	"--sym",	# Default
	# Tangle learning family, aim of research
	"--tl",		# Recommended
	"--rtl",	# Recommended
	# "--ortl",
	# "--ptl",
	# "--spptl",
	# "--dtl",
	# "--idtl",
	# Fixpoint algorithm, artrocious behaviour
	"--fpi",	# Recommended
	"--fpj",	# Recommended
	# "--fpjg",
	# Priority promotion family
	"--npp",	# Recommended
	# "--pp",
	# "--ppp",
	# "--rr",
	# "--dp",
	# "--rrdp",
	# Zielonka's recursive algorithm
	# "--zlk",
	# "--uzlk",
	# "--zlkq",
	# "--zlkpp-std",
	# "--zlkpp-waw",
	# "--zlkpp-liv",
	# Strategy improvement
	# "--psi",
	# "--ssi",
	# Progress measures
	# "--tspm",
	# "--spm",
	# "--mspm",
	# "--sspm",
	# "--bsspm",
	# "--qpt",
	# "--bqpt",
]

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
		with open(self.source, 'w') as file:
			json.dump(self.data, file, indent=True)
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
	
# Creates knor command that solves the given problem file with given arguments
# Returns command and target output file
def create_knor_command(file: Path, args: list[str], output_folder: Path = None) -> tuple[str, Path]:
	# First, ensure the default output folder if none is specified
	if not output_folder:
		if not UNMINIMIZED_AIG_FOLDER.is_dir():
			UNMINIMIZED_AIG_FOLDER.mkdir()

		output_folder = UNMINIMIZED_AIG_FOLDER / file.name.rstrip("".join(file.suffixes))
		
		if not output_folder.exists():
			output_folder.mkdir()
	# And make sure the output folder exists (whether specified or not)
	if not output_folder.is_dir():
		raise FileNotFoundError("Could not find or create output folder: {}".format(output_folder))
	
	output_file_name = file.with_stem(file.stem + "_args" + "".join(args)).with_suffix(".aag" if "-a" in args else ".aig").name
	output_file = output_folder / output_file_name
	
	command = "./{} {} {} > {}".format(KNOR_BINARY, file, " ".join(args), output_file)
	return command, output_file

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

# Returns list of problem files in the PROBLEM_FILES_FOLDER
# Optional: regex to only select matching file names
def get_problem_files(regex: str = None) -> list[Path]:
	problem_file_list = []
	
	for item in Path(PROBLEM_FILES_FOLDER).iterdir():
		if item.is_file():
			if regex:
				if re.match(regex, item.name):
					problem_file_list.append(item)
			else:
				problem_file_list.append(item)
	return problem_file_list

# Solves the given problem files with the given argument combinations and stores stats in given profiling data
def solve_problem_files(files: list[Path], arg_combinations: list[list[str]], profiler_data: ProfilerData, timeout=MAX_TIME_SECONDS_FOR_KNOR_COMMAND):
	total_commands = len(files) * len(arg_combinations)
	
	for file_num, file in enumerate(files):
		for arg_combo_num, arg_combo in enumerate(arg_combinations):
			progress_percentage = (file_num * len(arg_combinations) + arg_combo_num) / total_commands * 100
			print("{:.1f}% (File: {}/{}, arg: {}/{})... ".format(progress_percentage, file_num, len(files), arg_combo_num, len(arg_combinations)), end="", flush=True)

			if not profiler_data.is_new_command(file, arg_combo, timeout):
				print("Already computed. Skipping...")
				continue

			command, output_file = create_knor_command(file, arg_combo)

			command_start = time.time()
			try:
				result = run_shell_command(command, timeout)
			except KeyboardInterrupt:
				print("Aborted after: {:.2f}s".format(time.time() - command_start))
				return
			command_time = time.time() - command_start

			solve_result = SolveResult.SOLVED
			if not result:
				print("Timed out in: {:.2f}s".format(command_time))
				solve_result = SolveResult.TIMED_OUT
			elif result.returncode == 10:
				print("Solved in: {:.2f}s".format(command_time))
			elif result.returncode == 20:
				print("Unrealizable in: {:.2f}s".format(command_time))
				solve_result = SolveResult.UNREALIZABLE
			else:
				print("Crashed in: {:.2f}s".format(command_time))
				solve_result = SolveResult.CRASHED
			
			profiler_data.handle_command_result(
				file,
				arg_combo,
				command,
				output_file,
				command_time,
				solve_result
			)

# Runs the given shell command in terminal, timeout in seconds
def run_shell_command(cmd: str, timeout: float):
	try:
		return subprocess.run([cmd], shell=True, timeout=timeout, stdout=subprocess.PIPE)
	except subprocess.TimeoutExpired as e:
		return None

# Reads all AIG output files in profiler and adds the read data to the profiler
def add_aig_stats_to_profiler(profiler: ProfilerData):
	problem_files_count = len(profiler.data["problem_files"])
	for count, problem_file in enumerate(profiler.data["problem_files"]):
		percentage = count / problem_files_count * 100
		print("{:.2f}% Reading solutions of problem {}/{}... ".format(percentage, count, problem_files_count), flush=True, end="")
		read_solution_counter = 0
		start_time = time.time()
		if not problem_file["known_unrealizable"]:
			for solve_attempt in problem_file["solve_attempts"]:
				if not solve_attempt["timed_out"] and not solve_attempt["crashed"]:
					# This solve attempt was successful
					if solve_attempt["data"]: continue # Data already calculated
					solution_file = Path(solve_attempt["output_file"])
					aig_stats = get_aig_stats_from_file(solution_file)
					solve_attempt["data"] = aig_stats
					read_solution_counter += 1
			read_time = time.time() - start_time
			print("Read {} solutions in {:.2f}s".format(read_solution_counter, read_time))
		else: print("Unrealizable, no solutions.")
	profiler.save()

# Runs ABC 'read' and 'print_stats' command on fiven file and returns the parsed stats
def get_aig_stats_from_file(file: Path):
	abc_read_cmd = "./{} -c 'read {}; print_stats'".format(ABC_BINARY, file)
	cmd_output = run_shell_command(abc_read_cmd, 10).stdout.decode().replace(" ","")
	and_gates = int(re.findall("and=[0-9]*", cmd_output)[0].split("=")[1])
	latches = int(re.findall("lat=[0-9]*", cmd_output)[0].split("=")[1])
	levels = int(re.findall("lev=[0-9]*", cmd_output)[0].split("=")[1])
	inputs, outputs = map(int, re.findall("i/o=[0-9]*/[0-9]*",cmd_output)[0].split("=")[1].split("/"))
	
	return {
		"and_gates": and_gates,
		"levels": levels,
		"latches": latches,
		"inputs": inputs,
		"outputs": outputs
	}

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



def generate_compile_commands_for_aigs(profiler_data):
	# ./abc -c "r i10.aig; b; ps; b; rw -l; rw -lz; b; rw -lz; b; ps; cec"
	commands = {
		"compress": "b -l; rw -l; rwz -l; b -l; rwz -l; b -l",
		"compress2": "b -l; rw -l; rf -l; b -l; rw -l; rwz -l; b -l; rfz -l; rwz -l; b -l",
		"compress2rs":
		"""
			b -l;
			rs -K 6 -l;
			rw -l;
			rs -K 6 -N 2 -l;
			rf -l;
			rs -K 8 -l;
			b -l;
			rs -K 8 -N 2 -l;
			rw -l; rs -K 10 -l;
			rwz -l;
			rs -K 10 -N 2 -l;
			b -l;
			rs -K 12 -l;
			rfz -l;
			rs -K 12 -N 2 -l;
			rwz -l;
			b -l
		"""
	}

	filename = UNMINIMIZED_AIG_FOLDER + "aigs_unminimized/arbiter_with_buffer/arbiter_with_buffer.tlsf.args--tl.aig"
	
	source_command = "source " + ABC_ALIAS_SOURCE + ""
	args = source_command + "; r " + filename + "; &topand"

	cmd = ABC_BINARY + " -c '" + args + "'"


	arg_possibilites = [
		"b -l",
		"rw -l",
		"rwz -l",
		"rf -l",
		"rfz -l",
		"rs -K 6 -l",
		"rs -K 6 -N 2 -l",
		"rs -K 8 -l",
		"rs -K 8 -N 2 -l",
		"rs -K 10 -l",
		"rs -K 10 -N 2 -l",
		"rs -K 12 -l",
		"rs -K 12 -N 2 -l",
	]

	result = run_shell_command(cmd, 20)

	pass

def permutate_BFS(options, length):
	import copy
	all_permutations = []

	# Initialize first set of permutations, which is just the options, each in a single element list
	permutations_of_previous_length: list[list[str]] = list(map(list, options))
	all_permutations.extend(permutations_of_previous_length)

	i = 1
	while i < length:
		permutations_of_current_length = []

		for permutation_of_previous_length in permutations_of_previous_length:

			for option in options:
				new_permutation = [x for x in permutation_of_previous_length] + [option]
				permutations_of_current_length.append(new_permutation)
		
		all_permutations.extend(permutations_of_current_length)
		permutations_of_previous_length = permutations_of_current_length
		i += 1
	
	return all_permutations
	
""" A B C
A
B
C

A A
A B
A C

B A
B B
B C
C A
C B
C C
A A A
A A B
A A C
A B A
A B B
A B C
A B 
"""

if __name__ == "__main__":
	solve_all_arbiter_problem_files()
	add_aig_stats_to_profiler(ProfilerData(Path("profiler.json")))
