from compression_profiler import *

LAPTOP = False

if LAPTOP:
	do_tests(3, 20, 10, TestSize.Small)
else:
	do_tests(20, 120, 60, TestSize.Big)
