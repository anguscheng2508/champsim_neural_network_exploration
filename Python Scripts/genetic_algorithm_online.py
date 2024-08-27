#!/bin/env python3 
# Fri  9 Aug 00:31:56 IST 2024 Philippos P
import pygad
import numpy
import random
import subprocess
import os

# Use a global threadpool (so each candidate can spawn multiple processes)
num_threads=20 # Change to number of threads available/willing to use on own machine.
import concurrent.futures 
threadpool = concurrent.futures.ThreadPoolExecutor(num_threads)
#https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ThreadPoolExecutor
#https://superfastpython.com/threadpoolexecutor-number-of-threads/
#https://www.digitalocean.com/community/tutorials/how-to-use-threadpoolexecutor-in-python-3
#https://docs.python.org/3/library/subprocess.html

benchmark_directory = '/[path]/[to]/[benchmarks]/[directory]/champsim-nn-traces-txt'

# Define the design space
learning_rate = [0.0001, 0.001]
batch_size = [40, 20]
num_inputs =  [6, 8, 10]
layers = [3, 4, 5, 6, 7]
hidden_size = [40, 50, 60]
activation_function = ['relu', 'selu', 'leaky_relu', 'hardtanh', 'elu', 'rrelu', 'celu', 'hardshrink', 'tanhshrink', 'softsign', 'softplus', 'softshrink', 'silu', 'mish']
time_limit = [1200]
mse_limit = ['NA']

benchmark_names = [f for f in os.listdir(benchmark_directory) if os.path.isfile(os.path.join(benchmark_directory, f))]
benchmark_accuracies = {benchmark_name: [] for benchmark_name in benchmark_names}

def fitness_worker(ga_instance, solution_i, solution_idx):
    solution = [parameter_space[i][int(solution_i[i])] for i in range(len(solution_i))]
    lr = solution[0]
    bs = solution[1]
    inp = solution[2]
    lyr = solution[3]
    hs = solution[4]
    act = solution[5]
    time = solution[6]

        
    # Fake wait occupying 1 thread
    # (can use this list to test multiple benchmarks)
    # cmds = ["stress -t 1 -c 1"]

    cmds = []
    for benchmark_name in benchmark_names:
        hyper_p_combo = f'./nn-on {benchmark_directory}/{benchmark_name} 0 {inp} 1 {lyr} {hs} {lr} {time} {act} {bs}' 
        command = f'{hyper_p_combo}'
        cmds.append(command)
        
        benchmark_accuracies = {benchmark_name: []}

    
    futures = []
    for cmd in cmds:
        futures.append(threadpool.submit(subprocess.getoutput, cmd))


    accuracies = []
    # (the output can be stored locally so you can average IPCs)
    for future in concurrent.futures.as_completed(futures):
        output = future.result()
        separated = output.split()
        benchmark_name = separated[1]
        try:
            accuracy = float(separated[-1])
            benchmark_accuracies[benchmark_name].append((solution, accuracy))
            accuracies.append(accuracy)
        except ValueError:
            print("Solution: ", solution)
            print(f"Error processing output: {output}")
            return 0
    

    if accuracies:
        fitness_value = sum(accuracies) / len(accuracies)
    else:
        fitness_value = 0.0
    
    #print ((ga_instance, solution, solution_idx))
    #return random.random() # maximising this value
    print("Fitness value:", fitness_value, "Solution:", solution)
    return fitness_value

# Genetic exploration using PyGAD    
# https://pygad.readthedocs.io/en/latest/pygad_more.html
exec(open("/opt/gen_common.py").read())


ga_instance = pygad.GA(num_generations=15,
                       sol_per_pop=16, # e.g. total 10*16 candidates 
                       num_parents_mating=5,
                       fitness_func=fitness_worker,                       
                       num_genes=len(gene_space),
                       gene_space = gene_space,
                       parent_selection_type="sss",
                       keep_parents=1,
                       crossover_type="scattered",
                       mutation_type="random",
                       mutation_percent_genes=20,
                       on_generation=on_generation,
                       parallel_processing=["thread",num_threads])
                                      
ga_instance.run()

solution_i, solution_fitness, solution_idx = ga_instance.best_solution()
solution = [parameter_space[i][int(solution_i[i])] for i in range(len(solution_i))]
print("Benchmark accuracies per configuration:",benchmark_accuracies)
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
