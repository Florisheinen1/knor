from optimization_profiler import *

LAPTOP = False
i = 0

if LAPTOP:
	fifth_try(3, 10, TestSize.Small, i)
else:
	fifth_try(40, 60, TestSize.Big, i)
