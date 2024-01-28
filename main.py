from optimization_profiler import *

LAPTOP = False

if LAPTOP:
	fourth_try(3, 10, TestSize.Small)
else:
	fourth_try(40, 60, TestSize.Big)
