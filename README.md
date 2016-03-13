# galaxyz
Class assignment: Computing the two-point angular correlation function

This is my home assignment from class Advanced Computer Graphics and Graphics Hardware,
I'm currently taking at Abo Akademi. The idea is to run given example program on GPU,
which otherwise calculates 'something something' and that proves existence of dark matter.
I'm not really good at physics so for me it's just 'something something', but who is 
interested can read this: http://arxiv.org/abs/1208.3658

Info on kernel versions:
Kernel v1: This is basically merge of given example for performing calculations and OpenCL example
Kernel v2: Performing calculations only on bottom triange of matrix for DD and RR
Kernel v3: Using two dimensions for performing calculations
Kernel v4: Using local memory for storing results
Kernel v5: Modified version pf Kernel v2, now using same approach but in two dimensions
Kernel v6: Since now, every work item performed one calculation. This version introduce for loop 
			which perform more than one calculation, which lead to less necessary work items
Kernel v7: Updated version of Kernel v6 which use local memory

Since now Kernel v5/v6 showed best performance running on NVidia Tesla M2050 GPU.