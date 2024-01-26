from optimization_profiler import *

LAPTOP = False

if LAPTOP:
	third_try(3, 10, TestSize.Small)
else:
	third_try(20, 60, TestSize.Big)
