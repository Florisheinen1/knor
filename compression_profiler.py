import json
import os
import itertools
import pathlib
import time
import subprocess

ABC_BINARY = "build/_deps/abc-build/abc"
KNOR_BINARY = "build/knor"

PROFILING_FILE = "profiler.json"

UNMINIMIZED_AIG_FOLDER = "aigs_unminimized"
MINIMIZED_AIG_FOLDER = "aigs_minimized"
EHOA_FILES_FOLDER = "examples"

MAX_TIME_SECONDS_FOR_KNOR_COMMAND = 120 # seconds = 2 minutes

def generate_solve_commands_for_file(file_path: str, file_name: str):
	# These args can be combined
	# Order does not matter
	# No arguments is possibility too
	knor_args = [
		"--no-bisim",	# Because adding --bisim is default
		"--binary",		# Because --onehot is default
		"--isop",		# Because ITE is default
		
		# "--best", 	# To find the combo of --bisim/no-bisim, --isop/ite and --onehot/binary
		# "--compress" # No use of compress. This will be measured later
	]

	# Exactly one arg should be selected at a time
	oink_solver_args = [
		"--sym",	# Default
		# Tangle learning family, aim of research
		"--tl",		# Recommended
		"--rtl",	# Recommended
		"--ortl",
		"--ptl",
		"--spptl",
		"--dtl",
		"--idtl",
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

	output_args = [
		"-a",
		"-b"
	]

	target_folder_path = UNMINIMIZED_AIG_FOLDER + "/" + os.path.splitext(os.path.splitext(file_name)[0])[0] + "/"
	if not os.path.isdir(target_folder_path):
		os.makedirs(target_folder_path)

	# Get all possible combinations of knor args
	knor_arg_combinations = []
	for i in range(len(knor_args) + 1):
		l = []
		for c in itertools.combinations(knor_args, i):
			l.append(c)
		knor_arg_combinations.extend(l)

	all_arg_combinations = []
	# Now, combine knor arg combos with every possible oink arg
	for oink_arg in oink_solver_args:
		for knor_arg_combo in knor_arg_combinations:
			new_combo = list(knor_arg_combo)
			new_combo.append(oink_arg)
			all_arg_combinations.append(tuple(new_combo))
	
	# COMMAND:
	commands = []
	# ./knor ../examples/amba_decomposed_arbiter_10.tlsf.ehoa --bisim --onehot --compress -a -v > controller.aag
	for arg_combo in all_arg_combinations:
		file_arg_text = "".join(arg_combo) # Args used in output filename
		command_arg_text = " ".join(arg_combo) # Args used in knor command

		output_file = target_folder_path + os.path.splitext(file_name)[0] + ".args" + file_arg_text + ".aag"
		
		command = "./{} {} {} -a > {}".format(KNOR_BINARY, file_path, command_arg_text, output_file)
		
		commands.append((command, arg_combo, str(file_path), output_file))
	return commands

def load_profiling_stats_file():
	try:
		with open(PROFILING_FILE, "r") as file:
			data = json.load(file)
			return data
	except FileNotFoundError as e:
		# Then create the file first
		data = {
			"problem_files": []
		}
		return data
	
def store_profiling_stats_file(data):
	with open(PROFILING_FILE, 'w') as file:
		json.dump(data, file, indent=True)

def has_done_command_before(profiling_data, source_file_path, args):
	for source_file in profiling_data["problem_files"]:
		if source_file["source"] == source_file_path:
			for command in source_file["commands"]:
				if command["args_used"] == list(args):
					return True
	return False

def is_known_unrealizable(profiling_data, source_file_path):
	for source_file in profiling_data["problem_files"]:
		if source_file["source"] == source_file_path and not source_file["realizable"]:
			return True
	return False

def set_is_unrealizable(profiling_data, source_file_path):
	for source_file in profiling_data["problem_files"]:
		if source_file["source"] == source_file_path:
			source_file["realizable"] = False
			return
	profiling_data["problem_files"].append({
		"source": source_file_path,
		"realizable": False,
		"commands": []
	})
		
def set_command_had_timeout(profiling_data, cmd, args, input, output):
	command_data = {
		"command_used": cmd,
		"args_used": list(args),
		"output_file": output,
		"solve_time": None,
		"timed_out": True
	}
	for source_file in profiling_data["problem_files"]:
		if source_file["source"] == input:
			source_file["commands"].append(command_data)
			return
	profiling_data["problem_files"].append({
		"source": input,
		"realizable": None,
		"commands": [
			command_data
		]
	})

def insert_command_result_into_profiling_data(data, cmd, args, input, output, time):
	# Prepare the data to be inserted
	command_data = {
		"command_used": cmd,
		"args_used": list(args),
		"output_file": output,
		"solve_time": time,
		"timed_out": False
	}

	# Add if source file entry already exists
	for source_file in data["problem_files"]:
		if source_file["source"] == input:
			source_file["commands"].append(command_data)
			source_file["realizable"] = True
			return
	
	# Otherwise, need to add the data first, 
	data["problem_files"].append({
		"source": input,
		"realizable": True,
		"commands": [
			command_data
		]
	})

def prepare_non_minimized_aigs():
	# First, ensure unminimized folder exists
	pathlib.Path(UNMINIMIZED_AIG_FOLDER).mkdir(parents=True, exist_ok=True)

	# List all EHOA files in examples folder
	ehoa_files_list = []
	for item in pathlib.Path(EHOA_FILES_FOLDER).iterdir():
		if item.is_file():
			ehoa_files_list.append((item, item.name))

	ehoa_file_count = len(ehoa_files_list)
	print("Found", ehoa_file_count, "ehoa files")

	# Get all command combinations of arguments
	commands = []
	for ehoa_file_item in ehoa_files_list:
		target_path, target_name = ehoa_file_item

		file_commands = generate_solve_commands_for_file(target_path, target_name)
		commands.extend(file_commands)

	commands_count = len(commands)

	print("Created:", commands_count, "commands")

	profiling_data = load_profiling_stats_file()

	script_start = time.time()

	# For every command we created
	for count, (cmd, args, source, out) in enumerate(commands):
		if not has_done_command_before(profiling_data, source, args) and not is_known_unrealizable(profiling_data, source):
			cmd_start = time.time()
			
			percentage = (100 * count) / commands_count

			try:
				result = subprocess.run([cmd], shell=True, timeout=MAX_TIME_SECONDS_FOR_KNOR_COMMAND)
			except subprocess.TimeoutExpired as e:
				set_command_had_timeout(profiling_data, cmd, args, source, out)
				print("{:.1f}%".format(percentage), "CMD:", count, "/", commands_count, "TIMEOUT: time > ", MAX_TIME_SECONDS_FOR_KNOR_COMMAND)
				continue

			cmd_end = time.time()
			diff = cmd_end - cmd_start

			script_now = time.time()
			script_duration = script_now - script_start

			print("{:.1f}%".format(percentage), "CMD:", count, "/", commands_count, "code", result.returncode, "runtime:", script_duration)
			
			if result.returncode == 10:
				insert_command_result_into_profiling_data(profiling_data, cmd, args, source, out, diff)
				store_profiling_stats_file(profiling_data)
			elif result.returncode == 20:
				# This problem is not realizable. Set that in the profiler data
				set_is_unrealizable(profiling_data, source)
				store_profiling_stats_file(profiling_data)
				print("Problem " + source + " is unrealizable")
			else:
				print("Failed to execute command: '" + cmd + "'")



if __name__ == "__main__":
	prepare_non_minimized_aigs()