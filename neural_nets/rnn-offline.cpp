//
//  Created by Angus Cheng on 08/02/2024.
//

#include <cstdio>
#include <stdio.h>
#include <torch/torch.h>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <ctype.h>
#include <algorithm>
#include <torch/script.h>
#include <chrono>
#include <fstream>
#include <streambuf>

// Copyright 2020-present pytorch-cpp Authors
#include <iomanip>
#include "rnn.h"



int main(int argc, char *argv[]) {
//********** This section will read in deltas from a text file and store in variable ********
    const char* train_file = argv[1];
    int nn = atoi(argv[2]);
    const char* test_file = argv[3];
    int num_inputs = atoi(argv[4]);
    int num_outputs = atoi(argv[5]);
    int layers = atoi(argv[6]);
    char str_delta[60];
    long int_delta;
    char *stopstring;
    
    
    std::vector<float> deltas_parsed_train;
    std::vector<std::pair<std::vector<float>, std::vector<float>>> sequence_and_label_pairs_train;
    
    FILE *in_file_train = fopen(train_file, "r");
    if(in_file_train == NULL) {
        printf("Error file missing\n");
        return 1;
    }

    // Read and process the file
    char string_match[2000];
    char delta_str_match[] = "Delta_in_range:";
    
    while(fscanf(in_file_train,"%s", string_match) == 1) {
        if (strstr(string_match, delta_str_match) != 0) { // If match found
            fgets(str_delta, 60, in_file_train); // Get the delta value from the line (as string)
            int_delta = strtol(str_delta, &stopstring, 10); // Cast str to int for use
            deltas_parsed_train.push_back(int_delta);
        }
    }
    fclose(in_file_train);

    for (int i = 0; i < deltas_parsed_train.size(); i++) {
        // Extract input sequence (three consecutive numbers)
        std::vector<float> input_sequence(deltas_parsed_train.begin() + i, deltas_parsed_train.begin() + i + num_inputs);

        // Extract label (next number after the input sequence)
        std::vector<float> labels(deltas_parsed_train.begin() + i + num_inputs, deltas_parsed_train.begin() + i + num_inputs + num_outputs);
        // Add the sample to the dataset
        sequence_and_label_pairs_train.push_back({input_sequence, labels});
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
    const int64_t batch_size = atoi(argv[11]);
    const size_t num_epochs = 10000;
    const int64_t hidden_size = atoi(argv[7]);
    float mse_limit = atof(argv[8]);
    const double learning_rate = atof(argv[9]);
    int time_limit = atoi(argv[10]);
    int divisor = 1;
    
    torch::Tensor loss;
    std::vector<float> targetTest;
    torch::Tensor inputTest;

    int sample_size = sequence_and_label_pairs_train.size()/divisor;
//        int sample_size;
        
    // Neural Network model
    RNN model(input_size, hidden_size, layers, num_classes);
    model->to(device);
    // Optimizer
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(learning_rate));
    // Set floating point output precision
    std::cout << std::fixed << std::setprecision(4);

//    std::cout << "Training...\n";
//    std::cout << "Hidden Size: " << hidden_size << std::endl;

    // Start time of training
    auto start_time = std::chrono::high_resolution_clock::now();
        
    // Train the model
    // Train the model
    for (size_t epoch = 1; epoch != num_epochs; ++epoch) {
        
        auto current_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();
//        std::cout << "Training has been running for " << duration << " seconds" << std::endl;
        
        // If duration is longer than 10 mins on this epoch, then don't do this epoch and break.
        if (duration > time_limit) {
            break;
        }
        // Collect all inputs and targets into single tensors
        std::vector<torch::Tensor> all_inputs_train, all_targets_train;
        
        // Outer for loop is the 'marker' for the start of the sliding window
        for (size_t batch_start = 0; batch_start < sequence_and_label_pairs_train.size(); batch_start += batch_size) {
            std::vector<torch::Tensor> batch_inputs, batch_targets;
            for (size_t i = batch_start; i < std::min<int>(batch_start + batch_size, sequence_and_label_pairs_train.size()); ++i) {
                current_time = std::chrono::high_resolution_clock::now();
                
                duration = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();
                if (duration > time_limit) {
                    break;
                }
                batch_inputs.push_back(torch::tensor(sequence_and_label_pairs_train[i].first)/64);
                batch_targets.push_back(torch::tensor(sequence_and_label_pairs_train[i].second)/64);
            }
            current_time = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();


            if (duration > time_limit) {
                break;
            }
            
            // Concatenate tensors to create the full dataset
            auto inputTensor_train = torch::stack(batch_inputs).unsqueeze(1);
            auto targetTensor_train = torch::stack(batch_targets).unsqueeze(1);
            
            // Forward pass
            //        auto output = model->forward(inputTensor_train, activation_function, layers);
            auto output = model->forward(inputTensor_train);
            
            
            loss = torch::nn::functional::mse_loss(output, targetTensor_train.squeeze(2));
            
            // Backward pass and optimize
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
        }
        if (loss.item<float>() < mse_limit) {
            break;
        }
    }
    
    
        // Testing Section--------------------------------------------------------------
    std::vector<std::pair<std::vector<float>, std::vector<float>>> sequence_and_label_pairs_test;
    std::vector<float> deltas_parsed_test;
    
    FILE *in_file_test = fopen(test_file, "r");
    if(in_file_test == NULL) {
        printf("Error file missing\n");
        return 1;
    }
    
    while (fscanf(in_file_test, "%s", string_match) == 1) {
        // Add a for loop till strstr(string, delta) does not return null.
        if (strstr(string_match, delta_str_match) != 0) { // If match found
            fgets(str_delta, 60, in_file_test); // Get the delta value from the line (as string)
            int_delta = strtol(str_delta, &stopstring, 10); // Cast str to int for use
            deltas_parsed_test.push_back(int_delta);
        }
    }
    // Close the file
    fclose(in_file_test);
    
    for (int i = 0; i < deltas_parsed_test.size(); i++) {
        // Extract input sequence (three consecutive numbers)
        std::vector<float> input_sequence(deltas_parsed_test.begin() + i, deltas_parsed_test.begin() + i + num_inputs);

        // Extract label (next number after the input sequence)
//            float label = deltas_parsed_test[i + num_inputs];
        std::vector<float> labels(deltas_parsed_test.begin() + i + num_inputs, deltas_parsed_test.begin() + i + num_inputs + num_outputs);

        // Add the sample to the dataset
        sequence_and_label_pairs_test.push_back({input_sequence, labels});
    }
    
    
    
        model->eval();
        torch::NoGradGuard no_grad;
    
    std::vector<torch::Tensor> all_inputs_test;
    for (size_t i = 0; i < sample_size; ++i) {
        all_inputs_test.push_back(torch::tensor(sequence_and_label_pairs_test[i].first));
    }
    
    
    //    std::cout << "Testing..." << std::endl;
        std::vector<torch::Tensor> all_predictions;
        for (size_t i = 0; i < sample_size; ++i) {
            // Make sure the input tensor is in the correct form (normalized data)
            torch::Tensor inputTensor_test = all_inputs_test[i] / 64;

            // Perform forward pass
            torch::Tensor outputTensor_test = model->forward(torch::stack(inputTensor_test).unsqueeze(1));

            // Extract prediction from the output
            auto prediction = outputTensor_test * 64;
    //        std::cout << "PREDICTION: " << prediction << std::endl;
            
            // Collect all predictions
            all_predictions.push_back(prediction);
        }
    
        
    size_t num_correct = 0;
    for (size_t i = 0; i < sample_size; ++i) {
        for (int j = 0; j < num_outputs; j++) {
            int64_t pred_int = all_predictions[i].index({j}).item<int64_t>();
            int64_t target = sequence_and_label_pairs_test[i].second[j];
            if (round(pred_int) == static_cast<int64_t>(target)) {
                num_correct++;
//                std::cout << "Num correct: " << num_correct << std::endl;
//                std::cout << "PREDICTION: " << round(pred_int) << " | TARGET: " << static_cast<int64_t>(target) << std::endl << std::endl;
            }
        }
    }

//    std::cout << "Training/testing finished!\n\n";
//************************* End Section *******************************

//    std::cout << "Sample size: " << sample_size << std::endl;
//    std::cout << "Total number correct predictions: " << num_correct << std::endl;
//    std::cout << "Input sequence length: " << num_inputs << std::endl;
//    std::cout << "Number of outputs: " << num_outputs << std::endl;
//    std::cout << "Layers: " << layers << std::endl;
//    std::cout << "Hidden Size: " << hidden_size << std::endl;
//    std::cout << "Learning rate: " << learning_rate << std::endl;
//    std::cout << "Window size: " << batch_size << std::endl;
    
    float accuracy = (static_cast<float>(num_correct) / static_cast<float>(sample_size*num_outputs)) * 100;
//    std::cout << "Testing finished!\n\n";
    
    std::string trace_path = argv[1];
    size_t pos = trace_path.find_last_of("/\\");
    
    // Extracting trace name from the train_file path
    std::string trace;
    if (pos != std::string::npos) {
        trace = trace_path.substr(pos+1);
    } else {
        // If no separator is found, entire string is the trace name
        trace = trace_path;
    }
    
    // Removing the txt at the end of the trace name
//    std::string extension = ".txt";
//    if(trace.size() >= extension.size() &&
//        trace.compare(trace.size() - extension.size(), extension.size(), extension) == 0) {
//        trace.erase(trace.size() - extension.size(), extension.size());
//    }
    
    // Line to be printed then parsed:
    std::cout << "Trace: " << trace << " Num_inputs: " << argv[4] << " Num_layers: " << argv[6] << " Hidden_size: " << argv[7] << " MSE_limit: " << argv[8] << " Learning_rate: " << argv[9] << " Time_limit: " << argv[10] << " Batch_size: " << argv[11] << " Acc: " << accuracy << std::endl;
    
    
return 0;
}
