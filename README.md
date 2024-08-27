# Champsim Neural Network Exploration
## Introduction
This repository has been set up as the fruits of our labour, exlporing the feasability of neural networks in prefetching algorithms with the use of ChampSim and LibTorch. What started as an undergraduate project and progressed through a summer research internship (and may be proceeding into an MAI project), my supervisor; Dr. Philippos Papaphilippou, and I; Angus Cheng, have been working on exploring integrating neural networks into prefetching algorithms. Neural networks are known for their versatility and adaptibility, which can sound enticing for prefetch algorithms, where predictive patterns is the name of the game. However, given their notoriety for expensive computing, this project was created to explore how feasible it was for neural networks to be integrated.

This project focuses largely on the design space exploration of neural networks, in simulated 'offline' environments as well as 'online'. **'Offline** refers to an environment where the dataset is already prepared, a neural network is trained and then accuracies are measured by testing on this same dataset. **'Online'** is when the dataset is prepared, but fed into the neural network as a stream for training, and accuracies are measured during this training - imitating an 'on-board' implementation. Recurrent neural networks are also explored.

An online and offline optimal accuracy measurement calculation, per trace, has been developed. These will be used as a sort of comparison for the neural network performances.

A genetic algorthm is used to find an adequate neural network architecture with a decent average accuracy, to then be implemented into ChampSim - which will then finally test the IPC and compare with state-of-the-art benchmarks.


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

### Command Line - Executing and Genetic Algorithm
Each (recurrent) neural network source code take in arguments. Make sure the arguments provided are in the correct positions.
**nn-offline.cpp:**
```
./nn-off <arg1: training trace file path> 
<arg2: nn -or 0> 
<arg3: inference trace file path> 
<arg4: #inputs> 
<arg5: #outputs> 
<arg6: #layers> 
<arg7: neurons per layer> 
<arg8: MSE limit> 
<arg9: learning rate> 
<arg10: time limit> 
<arg11: activation function> 
<arg12: batch size>
<arg13: parameters saved path>

Example usage: 
./nn-off /Users/anguscheng/ChampSim/Threadedbaselines/results/champsim_nn,410.bwaves-1963B.txt 0 /Users/anguscheng/ChampSim/Threadedbaselines/results/champsim_nn,603.bwaves_s-891B.txt 3 3 9 50 0.001 0.0001 600 relu 10 /Users/anguscheng/neural_nets/params_dir/
```

**nn-online.cpp**
```
./nn-on <arg1: training trace file path>
<arg2: nn -or 0> 
<arg3: #inputs> 
<arg4: #outputs> 
<arg5: #layers> 
<arg6: neurons per layer>
<arg7: learning rate> 
<arg8: time limit> 
<arg9: activation function> 
<arg10: batch size>

Example usage:
./nn-on /Users/anguscheng/ChampSim/Threadedbaselines/results/champsim_nn,410.bwaves-1963B.txt 0 5 2 5 35 0.0001 1200 selu 15
```

**rnn-offline.cpp**
```
./rnn-off <arg1: training trace file path> 
<arg2: nn -or 0> 
<arg3: inference trace file path> 
<arg4: #inputs> 
<arg5: #outputs> 
<arg6: #layers> 
<arg7: neurons per layer> 
<arg8: MSE limit> 
<arg9: learning rate> 
<arg10: time limit> 
<arg11: batch size>

Example usage:
./rnn-off /Users/anguscheng/ChampSim/Threadedbaselines/results/champsim_nn,410.bwaves-1963B.txt 0 /Users/anguscheng/ChampSim/Threadedbaselines/results/champsim_nn,603.bwaves_s-891B.txt 7 3 4 15 0.001 0.001 180 20
```

**rnn-online.cpp**
```
./rnn-on <arg1: training trace file path> 
<arg2: nn -or 0> 
<arg3: #inputs> 
<arg4: #outputs> 
<arg5: #layers> 
<arg6: neurons per layer> 
<arg7: learning rate> 
<arg8: time limit> 
<arg9: batch size>

Example usage:
./rnn-on /Users/anguscheng/ChampSim/Threadedbaselines/results/champsim_nn,410.bwaves-1963B.txt 0 8 5 7 40 0.01 360 20
```

### Licensing
All new code is licensed under the following license:

[GNU General Public License, version
2](https://www.gnu.org/licenses/old-licenses/gpl-2.0.html)

```
  Copyright 2024 Angus Cheng, Philippos Papaphilippou @ TCD, Ireland

  This program is free software; you can redistribute it and/or
  modify it under the terms of the GNU General Public License
  as published by the Free Software Foundation; either version 2
  of the License, or (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program; If not, see <http://www.gnu.org/licenses/>.
```

