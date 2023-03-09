# Notes on dense vs. sparse assembly for the finite element method Laplacian problem

This program, given an input of a folder of STL, OFF, and OBJ models, outputs:
- number of vertices in each model,
- time to assemble a sparse Laplace-Beltrami cotangent matrix for the mesh in ms (computing cotangents not included),
- time to assemble a dense Laplace-Beltrami cotangent matrix in ms,
- time to assemble the dense matrix plus the time it takes to zero the matrix in ms, assuming it's already in cache,
- and the total time to load the matrix into cache, zero it, and assemble the dense matrix.

Early results that only measured assembly were promising; indeed, assembly is about 2x faster for small models (n < 10000) when done densely, and solving may also be faster, but this speed increase is negligible when compared to the time it takes to load the matrix into cache and set all values to zero in the first place. The code to reproduce these results is provided. This experiment was done on a subset of [Thingi10k](https://ten-thousand-models.appspot.com/), a database of 10,000 models from Thingiverse.

## Acknowledgements

I would like to acknowledge [Thingi10k](https://ten-thousand-models.appspot.com/)

## Dependencies

The only dependencies are STL, Eigen, [libigl](http://libigl.github.io/libigl/) and the dependencies
of the `igl::opengl::glfw::Viewer` (OpenGL, glad and GLFW).
The CMake build system will automatically download libigl and its dependencies using
[CMake FetchContent](https://cmake.org/cmake/help/latest/module/FetchContent.html),
thus requiring no setup on your part.

To use a local copy of libigl rather than downloading the repository via FetchContent, you can use
the CMake cache variable `FETCHCONTENT_SOURCE_DIR_LIBIGL` when configuring your CMake project for
the first time:
```
cmake -DFETCHCONTENT_SOURCE_DIR_LIBIGL=<path-to-libigl> ..
```
When changing this value, do not forget to clear your `CMakeCache.txt`, or to update the cache variable
via `cmake-gui` or `ccmake`.

## Compile

Compile this project using the standard cmake routine:

    mkdir build
    cd build
    cmake -DCMAKE_BUILD_TYPE=Release ..
    make

## Run

From within the `build` directory just issue:

    ./dense_timing [path to folder with models] > output.txt
