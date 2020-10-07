# A simple set of test data for your Iterative Closest Points implementation
*J. Chazalon, E. Carlinet — 2020-09*

The files provided in the current directory (beside this `README.md` file) are good test cases for your implementation of the Iterative Closest Points algorithm in CUDA.
Of course, **you should start testing your algorithm with much smaller data (like 10 points or so) to check its correctness first.**

## Files
Here is a description of the files:
- `bun*.txt`:
   several variants of the Stanford bunny 3D model. 
   They do not have exactly the same number of points and it may be necessary to deal with noise to register those shapes correctly.
   **Keep those files for your last tests, not the first ones.**  
   *Contains between 30000 and 40300 points.*
- `cow_**.txt`:
   exact transformations of the same reference cloud point.
   This should be an easy set to test your algorithm on when your implementation is correct.  
   *Contains 2904 points.*
-  `horse_*.txt`:
   exact transformations of the same reference cloud point.
   This is a larger model which may but a bit more pressure on your implementation.  
  *Contains 48486 points.*


## More data
To have more data, you can:

- create transformations of the models we gave you;
- generate random point clouds;
- look for large models on the Internet (like areas captured using LIDAR);
- use your imagination.
