# Iterative closest point 
Iterative closest point : GPU (CUDA) and CPU implementations

## Installation
Install the project requirements:

```bash
pipenv install
pipenv shell
```

Create a build folder:

```bash
mkdir build
cd build
```

Install the project dependencies:

```bash
conan install ..
```

Build the project:

```bash
cmake ..
make
cd ..
```

Run the project:
```bash
./icp [...]
```
