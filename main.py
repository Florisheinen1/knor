from optimization_profiler import *

LAPTOP = False

if LAPTOP:
	test_8(3, 10, TestSize.Small)
else:
	test_8(20, 60, TestSize.Big)
