# These flags can be combined
# Order does not matter
# No flag is a possibility too
KNOR_SYNTHESIZE_FLAGS = [
	"--no-bisim",	# Because adding --bisim is default
	"--binary",		# Because --onehot is default
	"--isop",		# Because ITE is default
	
	# "--best",		# To find the combo of --bisim/no-bisim, --isop/ite and --onehot/binary
	# "--compress"	# No use of compress. This will be measured later
]

# Exactly one arg should be selected at a time
KNOR_SOLVE_FLAGS = [
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

ABC_FLAG_LIMIT = 100000000000 # Much higher than this will crash ABC (probably due to overflow)

ABC_OPTIMIZATION_ARGUMENTS = {
	"b": { # transforms the current network into a well-balanced AIG
		"-l": None,			# toggle minimizing the number of levels [default = yes]
		"-d": None,			# toggle duplication of logic [default = no]
		"-s": None,			# toggle duplication on the critical paths [default = no]
		"-x": None,			# toggle balancing multi-input EXORs [default = no]
	},
	"rw": { # performs technology-independent rewriting of the AIG
		"-l": None,			# toggle preserving the number of levels [default = yes]
		"-z": None,			# toggle using zero-cost replacements [default = no]
	},
	"rf": { # performs technology-independent refactoring of the AIG
		"-N": {"default": 10, "min": 2, "max": 15}, 	# the max support of the collapsed node [default = 10]
		"-l": None,				# toggle preserving the number of levels [default = yes]
		"-z": None,				# toggle using zero-cost replacements [default = no]
	},
	"drw": { # performs combinational AIG rewriting
		"-C": {"default": 8, "min": 1, "max_pseudo": 1000},	# the max number of cuts at a node [default = 8]			// Limited due to laptop memory limitations
		"-N": {"default": 5, "min": 0},	# the max number of subgraphs tried [default = 5]
		"-l": None,				# toggle preserving the number of levels [default = no]
		# "-f": None,				# toggle representing fanouts [default = yes] 										// Conflicts with -l
		"-z": None,				# toggle using zero-cost replacements [default = no]
		"-r": None,				# toggle using cut recycling [default = yes]
	},
	"drf": { # performs combinational AIG refactoring
		"-M": {"default": 2, "min": 0},	# the min MFFC size to attempt refactoring [default = 2]
		"-K": {"default": 12, "min": 4, "max": 15}, 	# the max number of cuts leaves [default = 12]
		"-C": {"default": 5, "min": 2, "max_pseudo": 1000},	# the max number of cuts to try at a node [default = 5]		// Limited due to laptop memory limitations
		"-e": None,				# toggle extending tbe cut below MFFC [default = no]
		"-l": None,				# toggle preserving the number of levels [default = no]
		"-z": None,				# toggle using zero-cost replacements [default = no]
	},
	"drwsat": { # performs combinational AIG optimization for SAT
		"-b": None,		# toggle internal balancing [default = no]
	},
	"rs": {	# performs technology-independent restructuring of the AIG
		"-K": {"default": 8, "min": 5, "max": 15},						# the max cut size (4 <= num <= 16) [default = 8]					// ABC error: actual limit is 4 < K < 16 when -F is used
		"-N": {"default": 1, "min": 0, "max": 3},						# the max number of nodes to add (0 <= num <= 3) [default = 1]
		"-F": {"default": 0, "min": 0, "max": 8, "max_pseudo": 5},		# the number of fanout levels for ODC computation [default = 0]		// Crashes on computer when used above 5, not a big fan of this one -> SUPER slow!
		"-l": None,														# toggle preserving the number of levels [default = yes]
		"-z": None,														# toggle using zero-cost replacements [default = no]
	},
	"dc2": { # performs combinational AIG optimization
		"-b": None,			# toggle internal balancing [default = no]
		"-l": None,			# toggle updating level [default = no]
		# "-f": None,			# toggle representing fanouts [default = yes]					// Crashes when used together with -l
		"-p": None,			# toggle power-aware rewriting [default = no]
	},
	"irw": { # perform combinational AIG rewriting
		"-l": None,			# toggle preserving the number of levels [default = yes]
		"-z": None,			# toggle using zero-cost replacements [default = no]
	},
	"irws": { # perform sequential AIG rewriting
		"-z": None,			# toggle using zero-cost replacements [default = no]
	},
	"iresyn": { # performs combinational resynthesis
		"-l": None,			# toggle preserving the number of levels [default = yes]
	},
}

ABC_CLEANUP_ARGUMENTS = {
	"trim": None,	# removes POs def by constants and PIs wo fanouts
	"cleanup": { # for AIGs, removes PIs w/o fanout and POs driven by const-0
		"-i": None,			# toggles removing PIs without fanout [default = yes]
		"-o": None,			# toggles removing POs with const-0 drivers [default = yes]
	},
	"scleanup": { # performs sequential cleanup of the current network by removing nodes and latches that do not feed into POs
		"-c": None,							# sweep stuck-at latches detected by ternary simulation [default = yes]
		"-e": None,							# merge equal latches (same data inputs and init states) [default = yes]
		"-n": None,							# toggle preserving latch names [default = yes]
		"-m": None,							# toggle using hybrid ternary/symbolic simulation [default = no]
		"-F": {"default": 1, "min": 0}, 	# the number of first frames simulated symbolically [default = 1]
		"-S": {"default": 512, "min": 0},	# the number of frames when symbolic saturation begins [default = 512]
	},
	"csweep": { # performs cut sweeping using a new method
		"-C": {"default": 8, "min": 2, "max_pseudo": 1000}, 	# limit on the number of cuts (C >= 2) [default = 8]						// Limited due to laptop memory limitations
		"-K": {"default": 6, "min": 3, "max": 16, "max_pseudo": 10}, 				# limit on the cut size (3 <= K <= 16) [default = 6]	// Limited due to laptop memory limitations
	},
	# Too many combinations of flags that give unexpected behaviour and/or crashes, so for now just stick with the default "ssweep"
	"ssweep": {	# performs sequential sweep using K-step induction
		# "-P": {"default": 0, "min": 2}, 	# max partition size (0 = no partitioning) [default = 0]
		# "-Q": {"default": 0, "min": 2}, 	# partition overlap (0 = no overlap) [default = 0]
		# # "-N": {"default": 0, "min": 0}, 	# number of time frames to use as the prefix [default = 0]			// No
		# # "-F": {"default": 1, "min": 0}, 	# number of time frames for induction (1=simple) [default = 1]		// No
		# # "-L": {"default": 0, "min": 0}, 	# max number of levels to consider (0=all) [default = 0]			// No
		# "-l": None,				# toggle latch correspondence only [default = no]			
		# "-r": None,				# toggle AIG rewriting [default = no]
		# "-f": None,				# toggle fraiging (combinational SAT sweeping) [default = no]
		# "-e": None,				# toggle writing implications as assertions [default = no]
		# # "-t": None,				# toggle using one-hotness conditions [default = no]			// Writes to another file
	},
	# Too many combinations of flags that give unexpected behaviour and/or crashes, so for now just stick with the default "scorr"
	"scorr": { # performs sequential sweep using K-step induction
		# "-P": {"default": 0, "min": 2},		# max partition size (0 = no partitioning) [default = 0]
		# "-Q": {"default": 0, "min": 0},		# partition overlap (0 = no overlap) [default = 0]
		# "-F": {"default": 1, "min": 1},		# number of time frames for induction (1=simple) [default = 1]
		# "-C": {"default": 1000, "min": 0},	# max number of conflicts at a node (0=inifinite) [default = 1000]
		# "-L": {"default": 0},		# max number of levels to consider (0=all) [default = 0]
		# "-N": {"default": 0},		# number of last POs treated as constraints (0=none) [default = 0] // ABC error: Multiple help explanations
		# "-S": {"default": 2}, 		# additional simulation frames for c-examples (0=none) [default = 2]
		# "-I": {"default": -1}, 		# iteration number to stop and output SR-model (-1=none) [default = -1]
		# "-V": {"default": 5000}, 	# min var num needed to recycle the SAT solver [default = 5000]
		# "-M": {"default": 250},		# min call num needed to recycle the SAT solver [default = 250]
		# "-X": {"default": 0}, 		# the number of iterations of little or no improvement [default = 0]
		# "-c": None,					# toggle using explicit constraints [default = no] // NO
		# "-m": None,					# toggle full merge if constraints are present [default = no] // NO
		# "-p": None,					# toggle aligning polarity of SAT variables [default = no]
		# "-l": None,					# toggle doing latch correspondence [default = no]
		# "-k": None,					# toggle doing constant correspondence [default = no]
		# "-o": None,					# toggle doing 'PO correspondence' [default = no]
		# "-d": None,					# toggle dynamic addition of constraints [default = no]		// NO
		# "-s": None,					# toggle local simulation in the cone of influence [default = no]		// NO
		# "-e": None,					# toggle dumping disproved internal equivalences [default = no]		// NO
		# "-f": None,					# toggle dumping proved internal equivalences [default = no]		// NO
		# "-q": None,					# toggle quitting when PO is not a constant candidate [default = no]
	}
}

# FROM_AIG = [
# 	("collapse", [		# collapses the network by constructing global BDDs
#   		"-r",			# toggles dynamic variable reordering -> off
# 		"-o",			# toggles reverse variable ordering
# 		"-d"			# toggles dual-rail collapsing mode
# 	]),
# 	("satclp", [		# performs SAT based collapsing
# 		"-C (0>)",		# the limit on the SOP size of one output
# 		"-c",			# toggles using canonical ISOP computation
# 		"-r",			# toggles using reverse veriable ordering
# 		"-s",			# toggles shared CNF computation
# 	]),
# 	("multi", [			# transforms an AIG into a logic network by creating larger nodes
# 		"-F (<20>)",	# the maximum fanin size after renoding
# 		"-T (<1>)",		# the threshold for AIG node duplication
# 		"-m",			# creates multi-input AND graph
# 		"-s",			# creates a simple AIG (no renoding) 
# 		"-f",			# creates a factor-cut network
# 		"-c",			# performs renoding to derive the CNF
# 	]),
# 	("renode", [		# transforms the AIG into a logic network with larger nodes 
# 						# while minimizing the number of FF literals of the node SOPs
# 		"-K (2-16)",	# the max cut size for renoding
# 		"-C (0-4096)",	# the max number of cuts used at a node
# 		"-F (0>)",		# the number of area flow recovery iterations
# 		"-A (0>)",		# the number of exact area recovery iterations
# 		"-s",			# toggles minimizing SOP cubes instead of FF lits
# 		"-b",			# toggles minimizing BDD nodes instead of FF lits
# 		"-c",			# toggles minimizing CNF clauses instead of FF lits
# 		"-i",			# toggles minimizing MV-SOP instead of FF lits
# 		"-a",			# toggles area-oriented mapping
# 	])
# ]


# TO_AIG = [
# 	("strash", [	# transforms combinational logic into an AIG
# 		"-a",		# toggles between using all nodes and DFS nodes
# 		"-c",		# toggles cleanup to remove the dagling AIG nodes
# 		"-r",		# toggles using the record of AIG subgraphs
# 		"-i"		# toggles complementing the POs of the AIG
# 	]),
# 	("fraig", [			# Transforms the current network into a functionally-reduced AIG
# 		"-R (127-32769)",	# number of random patterns (127 < num < 32769) [default = 2048]
# 		"-D (127-32769)",	# number of systematic patterns (127 < num < 32769) [default = 2048]
# 		"-C	(<100>)",	# number of backtracks for one SAT problem [default = 100]
# 		"-r",			# toggle functional reduction [default = yes]
# 		"-s",			# toggle considering sparse functions [default = yes]
# 		"-c",			# toggle accumulation of choices [default = no]
# 		"-p",			# toggle proving the miter outputs [default = no]
# 		"-e",			# toggle functional sweeping using EXDC [default = no]
# 		"-a",			# toggle between all nodes and DFS nodes [default = dfs]
# 		"-t",			# toggle using partitioned representation [default = no]
# 	]),	
# ]

# ON_NOT_AIG = [
# 	("eliminate", [		# traditional "eliminate -1", which collapses the node into its fanout
#                    		# if the node's variable appears in the fanout's factored form only once
# 		"-V (-1>)", 	# the "value" parameter used by "eliminate" in SIS [default = -1]
# 		"-N	(<12>)", 	# the maximum node support after collapsing [default = 12]
# 		"-I (0>)", 		# the maximum number of iterations [default = 1]
# 		"-g", 			# toggle using greedy eliminate (without "value") [default = no]
# 		"-r", 			# use the reverse topological order [default = no]
# 		"-s", 			# toggle eliminating similar nodes [default = no]
# 	]),
# 	("logicpush"),	# performs logic pushing to reduce structural bias NOT AIG
# ]

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
