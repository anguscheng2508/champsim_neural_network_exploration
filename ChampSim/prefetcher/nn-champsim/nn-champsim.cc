//
//  Created by Angus Cheng on 08/02/2024.
//

#include <stdio.h>
#include <torch/torch.h>
#include <iostream>
#include "cache.h"
#include <algorithm>
#include <cstdio>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <ctype.h>
#include <torch/script.h>
#include <chrono>
#include <fstream>
#include <streambuf>

// Copyright 2020-present pytorch-cpp Authors
#include <iomanip>
#include "neural_net.h"
using namespace std;

void CACHE::prefetcher_initialize() {}

vector<int> prev_addrs(20, 0);
vector<int> delta_arr(20, 0);
vector<int> counts(127, 0); //range[-63, 63]
int delta_min = 64;
int delta;
int delta_same_page;
int max_freq = 0;
int max_freq_delta = 0;
int prevVector_reset_count = 0;
int first_iter = 0;
int delta_not_found = 0;
int prev_addr_immediate;
std::vector<float> deltas_parsed_train;
std::vector<std::pair<std::vector<float>, std::vector<float>>> sequence_and_label_pairs_train;

uint32_t CACHE::prefetcher_cache_operate(uint64_t addr, uint64_t ip, uint8_t cache_hit, bool useful_prefetch, uint8_t type, uint32_t metadata_in)
{

    
    printf("\naddress: %d\n",addr);
    // Declare delta_in_range here to reset it for each iteration
    int delta_in_range;
    int page = (addr >> 6) >> 6;
    
    //Starting at the end of the vector, then decrement - proper methodolgy. Allows for the edge case i=0 to be dealt with
    //This for loop is responsible for recording the previous 10 addresses and storing into an array
    for (int i = prev_addrs.size()-1; i >= 0; i--) {
        if (first_iter == 0) {
            break;
        }
        if (i == 0) {
            prev_addrs[i] = prev_addr_immediate;
        }
        else {
            prev_addrs[i] = prev_addrs[i-1];
        }
        
        delta = (addr - prev_addrs[i]) >> 6;    //Calc delta
        delta_arr[i] = delta;
        printf("Delta: %d\n", delta);
        int prev_page = (prev_addrs[i] >> 6) >> 6;
        
        if (delta_not_found == 0) {
            if (page == prev_page) {
                delta_not_found = 1;
                delta_in_range = delta;
                printf("Delta_in_range: %d\n", delta_in_range);
                deltas_parsed_train.push_back(delta_in_range);
            }
        }

        //We take absolute value here to get the delta closest to 0 i.e. closest to current address
        if (abs(delta_in_range) < abs(delta_min)) {    //If calculated abs(delta) is < than abs(minimum delta)
            delta_min = delta_in_range; //then assign delta_min = delta_in_range
            printf("Delta_min: %d\n", delta_min);
        }
    }
    
    delta_not_found = 0;
    
    
    //First instruction will have no need to store previous addresses - because theres none
    first_iter = 1;
    prev_addr_immediate = addr;
    
    //********** This section will read in deltas from a text file and store in variable ********
//    int num_inputs = atoi(argv[4]);
//    int num_outputs = atoi(argv[5]);
    int num_inputs = 3;
    int num_outputs = 1;
//    int layers = atoi(argv[6]);
    int layers = 3;
//    std::string activation_function = argv[11];
    std::string activation_function = "relu";
    long int_delta;
        


    for (int i = 0; i < deltas_parsed_train.size(); i++) {
        // Extract input sequence (three consecutive numbers)
        std::vector<float> input_sequence(deltas_parsed_train.begin() + i, deltas_parsed_train.begin() + i + num_inputs);

        // Extract label (next number after the input sequence)
        std::vector<float> labels(deltas_parsed_train.begin() + i + num_inputs, deltas_parsed_train.begin() + i + num_inputs + num_outputs);
        // Add the sample to the dataset
        sequence_and_label_pairs_train.push_back({input_sequence, labels});
    }
    
    for (auto x : sequence_and_label_pairs_train) {
        std::cout << x.first << " | " << x.second << std::endl;
    }
    

 
    //************************* Neural Network *******************************
//    std::cout << "Neural Network\n\n";

    // Device
    auto cuda_available = torch::cuda::is_available();
    torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
//    std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

    // Hyper parameters
    const int64_t input_size = num_inputs;
    const int64_t num_classes = num_outputs;
//    const int64_t batch_size = atoi(argv[12]);
    const int64_t batch_size = 30;
    const size_t num_epochs = 10000;
//    const int64_t hidden_size = atoi(argv[7]);
    const int64_t hidden_size = 10;
//    float mse_limit = atof(argv[8]);
    float mse_limit = 0.001;
    int divisor = 1;
//    const double learning_rate = atof(argv[9]);
    const double learning_rate = 0.0001;
//    int time_limit = atoi(argv[10]);
    int time_limit = 600;
    torch::Tensor loss;

        
    int sample_size = sequence_and_label_pairs_train.size()/divisor;
//        int sample_size = 200;
        

    // Neural Network model
    NeuralNet model(input_size, hidden_size, num_classes);
//    model->to(device);
//    // Optimizer
//    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(learning_rate));
//    // Set floating point output precision
//    std::cout << std::fixed << std::setprecision(4);
//
////    std::cout << "Training...\n";
////    std::cout << "Hidden Size: " << hidden_size << std::endl;
//
//    // Start time of training
//    auto start_time = std::chrono::high_resolution_clock::now();
//        
//    // Train the model
//    for (size_t epoch = 1; epoch != num_epochs; ++epoch) {
//        size_t num_correct = 0;
//        
//        auto current_time = std::chrono::high_resolution_clock::now();
//        auto duration = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();
//        std::cout << "Training has been running for " << duration << " seconds" << std::endl;
//        
//        // If duration is longer than 10 mins on this epoch, then don't do this epoch and break.
//        if (duration > time_limit) {
//            break;
//        }
//        
//        // Collect all inputs and targets into single tensors
//        std::vector<torch::Tensor> all_inputs_train, all_targets_train;
//        
////        for (size_t i = 0; i < sample_size; ++i) {
////            all_inputs_train.push_back(torch::tensor(sequence_and_label_pairs_train[i].first)/64);
////            all_targets_train.push_back(torch::tensor(sequence_and_label_pairs_train[i].second)/64);
////        }
//        for (size_t batch_start = 0; batch_start < sequence_and_label_pairs_train.size(); batch_start += batch_size) {
//            std::vector<torch::Tensor> batch_inputs, batch_targets;
//            for (size_t i = batch_start; i < std::min<int>(batch_start + batch_size, sequence_and_label_pairs_train.size()); ++i) {
//                duration = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();
//                if (duration > time_limit) {
//                    break;
//                }
//                batch_inputs.push_back(torch::tensor(sequence_and_label_pairs_train[i].first)/64);
//                batch_targets.push_back(torch::tensor(sequence_and_label_pairs_train[i].second)/64);
//            }
//            
//            duration = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();
//            if (duration > time_limit) {
//                break;
//            }
//            
//            // Concatenate tensors to create the full dataset
//            auto inputTensor_train = torch::stack(batch_inputs).unsqueeze(1);
//            auto targetTensor_train = torch::stack(batch_targets).unsqueeze(1);
//            
//            
//            // Forward pass
//            auto output = model->forward(inputTensor_train, activation_function, layers);
//
//            loss = torch::nn::functional::mse_loss(output.squeeze(2), targetTensor_train.squeeze(2));
//            
//            // Backward pass and optimize
//            optimizer.zero_grad();
//            loss.backward();
//            optimizer.step();
//        }
////        std::cout << "MSE: " << loss.item<float>() << " | Epoch: " << epoch << std::endl << std::endl;
//        if (loss.item<float>() < mse_limit) {
//            break;
//        }
//    }

    
        
    
    

    
    //------------This section has no relation to the vector and deltas of interest------------
    if (delta_min > -64 && delta_min < 64) {
        counts[delta_min+63]++;
    }
    
    if (counts[delta_min+63]>max_freq) {
        max_freq = counts[delta_min+63];
        max_freq_delta = delta_min;
    }
    delta_min = 64;
    //-----------------------------------------------------------------------------------------
    

    printf("Prev vector: \n");
    copy(prev_addrs.begin(), prev_addrs.end(), ostream_iterator<int>(cout, "\n"));
    printf("\n");
    
    // This prefetch is not 'used'. Only interested in the deltas printed (in the above sections).
    uint64_t pf_addr = addr + (max_freq_delta << LOG2_BLOCK_SIZE);
    prefetch_line(pf_addr, true, metadata_in);
    return metadata_in;
}

uint32_t CACHE::prefetcher_cache_fill(uint64_t addr, uint32_t set, uint32_t way, uint8_t prefetch, uint64_t evicted_addr, uint32_t metadata_in)
{
  return metadata_in;
}

void CACHE::prefetcher_cycle_operate() {}

void CACHE::prefetcher_final_stats() {}


