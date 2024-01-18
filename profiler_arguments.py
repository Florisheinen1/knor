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
	# "--fpj",	# Recommended
	# "--fpjg",
	# Priority promotion family
	# "--npp",	# Recommended
	# "--pp",
	# "--ppp",
	# "--rr",
	# "--dp",
	# "--rrdp",
	# Zielonka's recursive algorithm
	"--zlk",
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

ABC_OPTIMIZATION_ARGUMENTS = {
	"b": [				# transforms the current network into a well-balanced AIG
		"-l",			# toggle minimizing the number of levels [default = yes]
		"-d",			# toggle duplication of logic [default = no]
		"-s",			# toggle duplication on the critical paths [default = no]
		"-x",			# toggle balancing multi-input EXORs [default = no]
	],
	"rw": [				# performs technology-independent rewriting of the AIG
		"-l",			# toggle preserving the number of levels [default = yes]
		"-z",			# toggle using zero-cost replacements [default = no]
	],
	"rf": [				# performs technology-independent refactoring of the AIG
		"-N #10>=0",	# the max support of the collapsed node [default = 10]
		# "-M #1>=0",	# the min number of nodes saved after one step (0 <= num) [default = 1]	// Introduced in a newer commit of abc
		"-l",			# toggle preserving the number of levels [default = yes]
		"-z",			# toggle using zero-cost replacements [default = no]
	],
	"drw": [			# performs combinational AIG rewriting
		"-C #8>=1",		# the max number of cuts at a node [default = 8]
		"-N #5>=1",		# the max number of subgraphs tried [default = 5]
		# "-M #1>=0",	# the min number of nodes saved after one step (0 <= num) [default = 1]	// This is introduced in a newer commit of ABC
		"-l",			# toggle preserving the number of levels [default = no]
		# "-f",			# toggle representing fanouts [default = yes]							// Conflicts with -l
		"-z",			# toggle using zero-cost replacements [default = no]
		"-r",			# toggle using cut recycling [default = yes]
	],
	"drf": [			# performs combinational AIG refactoring
		"-M #2>=0",		# the min MFFC size to attempt refactoring [default = 2]
		"-K #12>=0",	# the max number of cuts leaves [default = 12]
		"-C #5>=0",		# the max number of cuts to try at a node [default = 5]
		"-e",			# toggle extending tbe cut below MFFC [default = no]
		"-l",			# toggle preserving the number of levels [default = no]
		"-z",			# toggle using zero-cost replacements [default = no]
	],
	"drwsat": [			# performs combinational AIG optimization for SAT
		"-b",			# toggle internal balancing [default = no]
	],
	"rs": [				# performs technology-independent restructuring of the AIG
		"-K #8>=4<=16",	# the max cut size (4 <= num <= 16) [default = 8]
		"-N #1>=0<=3",	# the max number of nodes to add (0 <= num <= 3) [default = 1]
		# "-M #1>=0",	# the min number of nodes saved after one step (0 <= num) [default = 1]	// Introduced in newer abc commit
		"-F #0>=0",		# the number of fanout levels for ODC computation [default = 0]
		"-l",			# toggle preserving the number of levels [default = yes]
		"-z",			# toggle using zero-cost replacements [default = no]
	],
	"dc2": [			# performs combinational AIG optimization
		"-b",			# toggle internal balancing [default = no]
		"-l",			# toggle updating level [default = no]
		"-f",			# toggle representing fanouts [default = yes]
		"-p",			# toggle power-aware rewriting [default = no]
	],
	"irw": [			# perform combinational AIG rewriting
		"-l",			# toggle preserving the number of levels [default = yes]
		"-z",			# toggle using zero-cost replacements [default = no]
	],
	"irws": [			# perform sequential AIG rewriting
		# "-l",			# toggle preserving the number of levels [default = yes]	// Introduced in newer abc commit
		"-z",			# toggle using zero-cost replacements [default = no]
	],
	"iresyn": [			# performs combinational resynthesis
		"-l",			# toggle preserving the number of levels [default = yes]
	],
}

FROM_AIG = [
	("collapse", [		# collapses the network by constructing global BDDs
  		"-r",			# toggles dynamic variable reordering -> off
		"-o",			# toggles reverse variable ordering
		"-d"			# toggles dual-rail collapsing mode
	]),
	("satclp", [		# performs SAT based collapsing
		"-C (0>)",		# the limit on the SOP size of one output
		"-c",			# toggles using canonical ISOP computation
		"-r",			# toggles using reverse veriable ordering
		"-s",			# toggles shared CNF computation
	]),
	("multi", [			# transforms an AIG into a logic network by creating larger nodes
		"-F (<20>)",	# the maximum fanin size after renoding
		"-T (<1>)",		# the threshold for AIG node duplication
		"-m",			# creates multi-input AND graph
		"-s",			# creates a simple AIG (no renoding) 
		"-f",			# creates a factor-cut network
		"-c",			# performs renoding to derive the CNF
	]),
	("renode", [		# transforms the AIG into a logic network with larger nodes 
						# while minimizing the number of FF literals of the node SOPs
		"-K (2-16)",	# the max cut size for renoding
		"-C (0-4096)",	# the max number of cuts used at a node
		"-F (0>)",		# the number of area flow recovery iterations
		"-A (0>)",		# the number of exact area recovery iterations
		"-s",			# toggles minimizing SOP cubes instead of FF lits
		"-b",			# toggles minimizing BDD nodes instead of FF lits
		"-c",			# toggles minimizing CNF clauses instead of FF lits
		"-i",			# toggles minimizing MV-SOP instead of FF lits
		"-a",			# toggles area-oriented mapping
	])
]

CLEANUPS = {
	"trim": [],			# removes POs def by constants and PIs wo fanouts
	"cleanup": [		# for AIGs, removes PIs w/o fanout and POs driven by const-0
		"-i",			# toggles removing PIs without fanout [default = yes]
		"-o",			# toggles removing POs with const-0 drivers [default = yes]
	],
	"scleanup": [		# performs sequential cleanup of the current network by removing nodes and latches that do not feed into POs
		"-c",			# sweep stuck-at latches detected by ternary simulation [default = yes]
		"-e",			# merge equal latches (same data inputs and init states) [default = yes]
		"-n",			# toggle preserving latch names [default = yes]
		"-m",			# toggle using hybrid ternary/symbolic simulation [default = no]
		"-F #1>=0",		# the number of first frames simulated symbolically [default = 1]
		"-S #512>=0",	# the number of frames when symbolic saturation begins [default = 512]
	],
	"csweep": [			# performs cut sweeping using a new method
		"-C #8>=2",		# limit on the number of cuts (C >= 2) [default = 8]
		"-K #6>=3<=16",	# limit on the cut size (3 <= K <= 16) [default = 6]
	],
	"ssweep": [			# performs sequential sweep using K-step induction
		"-P #0>=0",		# max partition size (0 = no partitioning) [default = 0]
		"-Q #0>=0",		# partition overlap (0 = no overlap) [default = 0]
		# "-N #0>=0",	# number of time frames to use as the prefix [default = 0]
		# "-F #1>=0",	# number of time frames for induction (1=simple) [default = 1]
		# "-L #0>=0",	# max number of levels to consider (0=all) [default = 0]
		"-l",			# toggle latch correspondence only [default = no]
		"-r",			# toggle AIG rewriting [default = no]
		"-f",			# toggle fraiging (combinational SAT sweeping) [default = no]
		# "-e",			# toggle writing implications as assertions [default = no]
		"-t",			# toggle using one-hotness conditions [default = no]
	],
	"scorr": [			# performs sequential sweep using K-step induction
		"-P #0>=0",		# max partition size 
		"-Q #0>=0",		# partition overlap
		"-F #1>=0",		# number of time frames for induction
		"-C #0>=0",		# max number of conflicts at a node
		"-L #0>=0",		# max number of levels to consider
		# "-N (0>)",	# number of last POs treated as constraints
		"-S #2>=0",		# additional simulation frames for c-examples (0=none) [default = 2]
		"-I #-1>=0",		# iteration number to stop and output SR-model (-1=none) [default = -1]
		"-V #5000>=0",	# min var num needed to recycle the SAT solver [default = 5000]
		"-M #250>=0",	# min call num needed to recycle the SAT solver [default = 250]
		# "-N (0>)",	# set last <num> POs to be constraints (use with -c) [default = 0]
		"-X #0>=0",		# the number of iterations of little or no improvement [default = 0]
		# "-c",			# toggle using explicit constraints [default = no]
		# "-m",			# toggle full merge if constraints are present [default = no]
		"-p",			# toggle aligning polarity of SAT variables [default = no]
		"-l",			# toggle doing latch correspondence [default = no]
		"-k",			# toggle doing constant correspondence [default = no]
		"-o",			# toggle doing 'PO correspondence' [default = no]
		# "-d",			# toggle dynamic addition of constraints [default = no]
		# "-s",			# toggle local simulation in the cone of influence [default = no]
		# "-e",			# toggle dumping disproved internal equivalences [default = no]
		# "-f",			# toggle dumping proved internal equivalences [default = no]
		"-q",			# toggle quitting when PO is not a constant candidate [default = no]
	]
}

TO_AIG = [
	("strash", [	# transforms combinational logic into an AIG
		"-a",		# toggles between using all nodes and DFS nodes
		"-c",		# toggles cleanup to remove the dagling AIG nodes
		"-r",		# toggles using the record of AIG subgraphs
		"-i"		# toggles complementing the POs of the AIG
	]),
	("fraig", [			# Transforms the current network into a functionally-reduced AIG
		"-R (127-32769)",	# number of random patterns (127 < num < 32769) [default = 2048]
		"-D (127-32769)",	# number of systematic patterns (127 < num < 32769) [default = 2048]
		"-C	(<100>)",	# number of backtracks for one SAT problem [default = 100]
		"-r",			# toggle functional reduction [default = yes]
		"-s",			# toggle considering sparse functions [default = yes]
		"-c",			# toggle accumulation of choices [default = no]
		"-p",			# toggle proving the miter outputs [default = no]
		"-e",			# toggle functional sweeping using EXDC [default = no]
		"-a",			# toggle between all nodes and DFS nodes [default = dfs]
		"-t",			# toggle using partitioned representation [default = no]
	]),	
]

ON_NOT_AIG = [
	("eliminate", [		# traditional "eliminate -1", which collapses the node into its fanout
                   		# if the node's variable appears in the fanout's factored form only once
		"-V (-1>)", 	# the "value" parameter used by "eliminate" in SIS [default = -1]
		"-N	(<12>)", 	# the maximum node support after collapsing [default = 12]
		"-I (0>)", 		# the maximum number of iterations [default = 1]
		"-g", 			# toggle using greedy eliminate (without "value") [default = no]
		"-r", 			# use the reverse topological order [default = no]
		"-s", 			# toggle eliminating similar nodes [default = no]
	]),
	("logicpush"),	# performs logic pushing to reduce structural bias NOT AIG
]

# ("dfraig", [		# performs fraiging using a new method
# 	"-C (<100>)",	# limit on the number of conflicts [default = 100]
# 	"-s",			# toggle considering sparse functions [default = yes]
# 	"-p",			# toggle proving the miter outputs [default = no]
# 	"-r",			# toggle speculative reduction [default = no]
# 	"-c",			# toggle accumulation of choices [default = no]
# ]),
# ("ifraig", [		# performs fraiging using a new method
# 	"-P (0>)", 		# partition size (0 = partitioning is not used) [default = 0]
# 	"-C (<100>)", 	# limit on the number of conflicts [default = 100]
# 	"-L (0>)", 		# limit on node level to fraig (0 = fraig all nodes) [default = 0]
# 	"-s", 			# toggle considering sparse functions [default = yes]
# 	"-p", 			# toggle proving the miter outputs [default = no]
# ]),


# cleanup / &mfs


# cut		# computes k-feasible cuts for the AIG

# istrash	# perform sequential structural hashing


# ============== FAILED =============
# "mfs",			# performs don't-care-based optimization of logic networks
# "mfs2",			# performs don't-care-based optimization of logic networks
# "mfs3",			# performs don't-care-based optimization of logic networks
# "mfse",		# performs don't-care-based optimization of logic networks
# "rr",			# removes combinational redundancies in the current network
# "sweep", 		# removes dangling nodes; propagates constant, buffers, inverters
# "&scl", 		# performs structural sequential cleanup
# "&mfs", 		# performs don't-care-based optimization of logic networks
# "restructure",	# performs technology-independent restructuring of the AIG
# "dropsat",		# replaces satisfiable POs by constant 0 and cleans up the AIG
# "seq",			# converts aig into sequential aig
# "unseq",		# converts sequential AIG into an SOP logic network 
# "phase",		# performs sequential cleanup of the current network by removing nodes and latches that do not feed into POs
# "&bidec",		# performs heavy rewriting of the AIG
# "&shrink",		# performs fast shrinking using current mapping
# "&reshape",		# performs AIG resubstitution
# "&syn2",		# performs AIG optimization
# "&syn3",		# performs AIG optimization
# "&syn4",		# performs AIG optimization
# "&lnetopt",		# performs specialized AIG optimization
# scut		# computes k-feasible cuts for the AIG
# logic		# transforms an AIG into a logic network with SOPs
# sop		# Converts to SOP
# bdd		# Converts to bdd
# aig		# converts to AIG
# muxes		# converts the current network into a network derived by
# "bidec",
# "fxch",			# performs fast extract with cube hashing on the current network NOT AIG
