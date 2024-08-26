//
//  example-app.cpp
//
//
//  Created by Angus Cheng on 08/02/2024.
//

#include <stdio.h>
#include <torch/torch.h>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <ctype.h>
#include <algorithm>
#include <torch/script.h>
#include <unordered_map>
#include <sstream>


// Copyright 2020-present pytorch-cpp Authors
#include <iomanip>
#include "neural_net.h"



int main() {
    //********** This section will read in deltas from a text file and store in variable ********
    char word[2000];
    char string_match[50];
    char str_deltaValue[60], str_numInstr[60];
    long int_delta, int_numInstr;
    char *stopstring;
    std::string key;

    int num_delta = 0;
    int numDeltas_found = 0;
        
    std::vector<float> delta_vector;

    std::vector<std::pair<std::vector<float>, int>> data3;
    int num_inputs = 3;

    std::unordered_map<std::string, int> sequence_match;
    
    int max_correct;
    
    int prev_x = -1;
    int counter = 0;
    int most_freq_delta;
    std::unordered_map<std::string, std::vector<int>> label_array;
    std::vector<int> label_pushback_arr;

        
    // List of file paths
    std::vector<std::string> file_paths = {
//        "/Users/anguscheng/ChampSim/Threadedbaselines/results/champsim_nn,410.bwaves-1963B.txt"
//        "/Users/anguscheng/ChampSim/Threadedbaselines/results/champsim_nn,450.soplex-92B.txt"
//        "/Users/anguscheng/ChampSim/Threadedbaselines/results/champsim_nn,483.xalancbmk-127B.txt"
//        "/Users/anguscheng/ChampSim/Threadedbaselines/results/champsim_nn,602.gcc_s-2226B.txt"
//        "/Users/anguscheng/ChampSim/Threadedbaselines/results/champsim_nn,603.bwaves_s-2609B.txt"
//        "/Users/anguscheng/ChampSim/Threadedbaselines/results/champsim_nn,619.lbm_s-2677B.txt"
//        "/Users/anguscheng/ChampSim/Threadedbaselines/results/champsim_nn,654.roms_s-523B.txt"
//        "/Users/anguscheng/ChampSim/Threadedbaselines/results/champsim_nn,605.mcf_s-472B.txt"
//        "/Users/anguscheng/ChampSim/Threadedbaselines/results/champsim_nn,623.xalancbmk_s-700B.txt"
        "/Users/anguscheng/ChampSim/Threadedbaselines/results/champsim_nn,654.roms_s-1390B.txt"
        // Add more file paths as needed
    };

    // Array of FILE pointers
    std::vector<FILE*> file_pointers;

    // Open each file
    for (const auto& file_path : file_paths) {
        FILE* file_ptr = fopen(file_path.c_str(), "r");
        if (file_ptr == nullptr) {
            std::cout << "Error opening file: " << file_path << std::endl;
            // Close already opened files
            for (auto fp : file_pointers) {
                fclose(fp);
            }
            return 1;
        }
        // Add the file pointer to the array
        file_pointers.push_back(file_ptr);

        // Read and process the file
        char string_match[2000];
        char delta[] = "Delta_in_range:";
        char sim_instr[] = "Simulation";

        while (fscanf(file_ptr, "%s", string_match) == 1) {
            // If statement to get the number of simulation instructions on startup.
            // Do at the start of the file so we can declare an array for the size of the number of instructions, aka the number of deltas.
            if (numDeltas_found == 0) {
                if (strstr(string_match, sim_instr) != 0) {
                    fgets(str_numInstr, 60, file_ptr);
                    numDeltas_found = 1;
                    for (int i = 0; i < strlen(str_numInstr); i++) {
                        if (isdigit(str_numInstr[i])) {
                            char d[1];
                            d[0] = str_numInstr[i];
                            int digit = atoi(d);
                            num_delta = (num_delta * 10) + digit;
                        }
                    }
                }
            }
            
            // Add a for loop till strstr(string, delta) does not return null.
            if (strstr(string_match, delta) != 0) { // If match found
                fgets(str_deltaValue, 60, file_ptr); // Get the delta value from the line
                int_delta = strtol(str_deltaValue, &stopstring, 10); // Cast str to int for use
                delta_vector.push_back(int_delta);
            }
        }
        for (int i = 0; i < delta_vector.size(); i++) {
            // Extract input sequence (three consecutive numbers)
            std::vector<float> input_sequence(delta_vector.begin() + i, delta_vector.begin() + i + num_inputs);
           
            // Extract label (next number after the input sequence)
            int label = delta_vector[i + num_inputs];
            
            // Add the sample to the dataset
            data3.push_back({input_sequence, label});
            
            // Declare a string stream to prepare converting vector<float> into string
            std::stringstream ss;
            for(size_t i = 0; i < input_sequence.size(); ++i) {
                if(i != 0){
                    ss << " ";
                }
              ss << input_sequence[i];
            }
            std::string input_str = ss.str();
            
            key = input_str;
            
            if (label_array.find(key) != label_array.end()) {
                // If found, then get the corresponding label vector of given key.
                // Append new label into label vec
                // Put back this new label vec into key
                label_pushback_arr = label_array[key];
                label_pushback_arr.push_back(label);
                label_array[key] = label_pushback_arr;
                
                //This part is handling multiple values of a key.
                //sort the label vector, then iterate through the vector,
                //and 1 2 3 3 3 4 4 4 4 5 6
                // :  0 0 0 1 2 0 1 2 3 0 0 <- These are the counters for number appearances
                //or:(1 1 1 2 3 1 2 3 4 1 1)<- same here, counters for number..
                sort(label_pushback_arr.begin(), label_pushback_arr.end());
                std::cout << "Label pushback arr: ";
                for (auto x : label_pushback_arr) {
                    std::cout << x << " ";
                }
                std::cout << std::endl;
                
                int most_freq_counter = 0;
                for (auto x : label_pushback_arr) {
                    std::cout << "x: " << x << std::endl;
                    if (x == prev_x) {
                        counter++;
                        if (counter > most_freq_counter) {
                            // Most freq keeps track of the most frequently it appeared, not the actual delta.
                            most_freq_counter = counter;
                            most_freq_delta = x;
                            std::cout << "Most freq delta: " << most_freq_delta << std::endl;
                        }
                    }
                    else {
                        counter = 0;
                    }
                }
            }
            else {
                // If key is not in umap, then clear the label vector
                // Insert label into label vec
                // And assign key and label vec into umap
                label_pushback_arr.clear();
                label_pushback_arr.push_back(label);
                label_array[key] = label_pushback_arr;
            }
            
            if (sequence_match.find(key) != sequence_match.end()) {
                // If the current parsed label (prediction) is the same as the one currently inside the sequence match, then the prediction is correct. AKA max_correct++
                if (sequence_match[key] == most_freq_delta) {
                    max_correct++;
                }
            }
            sequence_match[key] = label;
            
        }
        // Close the file
        fclose(file_ptr);
    }

    // Close all the files when done
    for (auto fp : file_pointers) {
        fclose(fp);
    }
    
    // Calculate the number of unique predictions
    int unique_predictions = sequence_match.size();

    
    // Output the results
    std::cout << "Total entries: " << data3.size() << std::endl;
    std::cout << "Unique predictions: " << unique_predictions << std::endl;
    std::cout << "Max correct: " << max_correct << std::endl;
    
    float tot_entr = data3.size();
    float percentage = (max_correct / tot_entr) * 100;
    
    std::cout << "Percentage: " << percentage << std::endl;
    
        
//     Print the umap
//    for (auto x : label_array) {
//        std::cout << x.first << " |" <<
//                x.second << std::endl;
//    }

    return 0;

}


