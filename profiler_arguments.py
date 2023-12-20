MAX_TIME_SECONDS_FOR_KNOR_COMMAND = 120 # seconds
MAX_TIME_SECONDS_FOR_OPTIMIZE_COMMAND = 120 # seconds

REPETITION_TEST_MAX_REPETITION = 5

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

ABC_OPTIMIZATION_ARGUMENTS = [
	"b -l",
	"rw -l",
	"rwz -l",
	"rf -l",
	"rfz -l",
	# "rs -K 6 -l",
	# "rs -K 6 -N 2 -l",
	# "rs -K 8 -l",
	# "rs -K 8 -N 2 -l",
	# "rs -K 10 -l",
	# "rs -K 10 -N 2 -l",
	# "rs -K 12 -l",
	# "rs -K 12 -N 2 -l",
]
