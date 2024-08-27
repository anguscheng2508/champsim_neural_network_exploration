# Champsim Neural Network Exploration
## Introduction


## Getting started - Prerequisites
ChampSim must be installed locally - available on their [GitHub](https://github.com/ChampSim/ChampSim).

Libtorch will be used as the machine learning library as C++ is the language for ChampSim - available on [PyTorch](https://pytorch.org/get-started/locally/). 

A set of 30 sensitive SPEC CPU 2017 traces will be used as the datasets for training and testing. Available [here](https://gitlab.scss.tcd.ie/chenga/champsim-nn-traces-txt). (~19GB necessary for these 30 traces alone.)
(Optionally the full set can be found [here](https://dpc3.compas.cs.stonybrook.edu/champsim-traces/speccpu/). ~72GB of storage necessary for full set.)

## Usage
### First Steps - Data Preparation
The prefetcher algorithm "nn-outputs" must be compiled into a binary, using the "champsim_config_nn_outputs.json" file. This binary will later be used alongside a Python script (ThreadedBaselines) to generate the dataset.
```
cd [path]/[to]/ChampSim
./config.sh champsim_config_nn_outputs.json
make
```
Now that the binary is built, copy this binary into the ~/Threadedbaselines/bin/ directory. ThreadedBaselines.py will be used to run this binary on the set of 30 traces and store the resulting outputs into the ~/Threadedbaselines/results/ directory, as .txt's. Each benchmark will be stored into a their respective .txt files, with the file name being the corresponding benchmark. User will have to direct the path of traces inside the script to their own directories.
**Python 2.7 is recommended to run ThreadedBaselines.py script. Some functions/libraries (particularly threading/Queueing) are deprecated in new versions of Python.**
```
python ThreadedBaselines.py
or
Open ThreadedBaselines.py with IDLE 2.7 -> Then run through Python's default shell.
```
Wait for script to finish. 
After completion, the ~/Threadedbaselines/results/ directory will contain text files of the outputs from the prefetch algorithm, of each benchmark. These text files will be used as the dataset for training and testing neural networks. They will be parsed from inside the neural network source codes.

### Building
The CMakeLists.txt file must be changed for the user, to the path to where the libtorch library is located.
First time building the source codes:
```
cd ~/neural_nets/build
cmake -DCMAKE_PREFIX_PATH=/[path]/[to]/libtorch/ ..
cmake --build . --config Release
```
After initial set up, to build and compile code will just be with:
```
cd neural_net/build
cmake --build . --config Release
```

