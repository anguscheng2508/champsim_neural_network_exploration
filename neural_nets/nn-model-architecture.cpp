// Copyright 2020-present pytorch-cpp Authors
#include "neural_net.h"
#include <torch/torch.h>
#include <iostream>
#include <string.h>


NeuralNetImpl::NeuralNetImpl(int64_t input_size, int64_t hidden_size, int64_t num_classes)
    : fc1(input_size, hidden_size), fc3(hidden_size, hidden_size), fc4(hidden_size, hidden_size), fc5(hidden_size, hidden_size), fc6(hidden_size, hidden_size), fc7(hidden_size, hidden_size), fc8(hidden_size, hidden_size), fc9(hidden_size, hidden_size), fc10(hidden_size, hidden_size), fc11(hidden_size, hidden_size), fc12(hidden_size, hidden_size), fc2(hidden_size, num_classes) {
    register_module("fc1", fc1);
    register_module("fc3", fc3);
    register_module("fc4", fc4);
    register_module("fc5", fc5);
    register_module("fc6", fc6);
    register_module("fc7", fc7);
    register_module("fc8", fc8);
    register_module("fc9", fc9);
    register_module("fc10", fc10);
    register_module("fc2", fc2);
}


torch::Tensor activation_function(torch::Tensor x, torch::nn::Linear fc, std::string act_func) {
    if (act_func == "relu") {
        return torch::nn::functional::relu(fc->forward(x));
    }
    else if (act_func == "selu") {
        return torch::nn::functional::selu(fc->forward(x));
    }
    else if (act_func == "leaky_relu") {
        return torch::nn::functional::leaky_relu(fc->forward(x));
    }
    else if (act_func == "hardtanh") {
        return torch::nn::functional::hardtanh(fc->forward(x));
    }
    else if (act_func == "elu") {
        return torch::nn::functional::elu(fc->forward(x));
    }
    else if (act_func == "rrelu") {
        return torch::nn::functional::rrelu(fc->forward(x));
    }
    else if (act_func == "celu") {
        return torch::nn::functional::celu(fc->forward(x));
    }
    else if (act_func == "gelu") {
        return torch::nn::functional::gelu(fc->forward(x));
    }    
    else if (act_func == "hardshrink") {
        return torch::nn::functional::hardshrink(fc->forward(x));
    }
    else if (act_func == "tanhshrink") {
        return torch::nn::functional::tanhshrink(fc->forward(x));
    }
    else if (act_func == "softsign") {
        return torch::nn::functional::softsign(fc->forward(x));
    }
    else if (act_func == "softplus") {
        return torch::nn::functional::softplus(fc->forward(x));
    }
    else if (act_func == "softshrink") {
        return torch::nn::functional::softshrink(fc->forward(x));
    }
    else if (act_func == "silu") {
        return torch::nn::functional::silu(fc->forward(x));
    }
    else if (act_func == "mish") {
        return torch::nn::functional::mish(fc->forward(x));
    }
    else {
        std::cout << "Error! No valid activation function provided!" << std::endl;
        std::cout << "RETURN" << std::endl;
        return fc->forward(x);
    }
}


torch::Tensor NeuralNetImpl::forward(torch::Tensor x, std::string act_func, int layers) {
    // Input layer, always activate
    x = activation_function(x, fc1, act_func);
    
    switch(layers) {
        case 1:
            // Hard code the print of number of layers for validation
//            std::cout << "1 Hidden layer" << std::endl;
            x = activation_function(x, fc3, act_func);
            break;
        case 2:
//            std::cout << "2 Hidden layers" << std::endl;
            x = activation_function(x, fc3, act_func);
            x = activation_function(x, fc4, act_func);
            break;
        case 3:
//            std::cout << "3 Hidden layers" << std::endl;
            x = activation_function(x, fc3, act_func);
            x = activation_function(x, fc4, act_func);
            x = activation_function(x, fc5, act_func);
            break;
        case 4:
//            std::cout << "4 Hidden layers" << std::endl;
            x = activation_function(x, fc3, act_func);
            x = activation_function(x, fc4, act_func);
            x = activation_function(x, fc5, act_func);
            x = activation_function(x, fc6, act_func);
            break;
        case 5:
//            std::cout << "5 Hidden layers" << std::endl;
            x = activation_function(x, fc3, act_func);
            x = activation_function(x, fc4, act_func);
            x = activation_function(x, fc5, act_func);
            x = activation_function(x, fc6, act_func);
            x = activation_function(x, fc7, act_func);
            break;
        case 6:
//            std::cout << "6 Hidden layers" << std::endl;
            x = activation_function(x, fc3, act_func);
            x = activation_function(x, fc4, act_func);
            x = activation_function(x, fc5, act_func);
            x = activation_function(x, fc6, act_func);
            x = activation_function(x, fc7, act_func);
            x = activation_function(x, fc8, act_func);
            break;
        case 7:
//            std::cout << "7 Hidden layers" << std::endl;
            x = activation_function(x, fc3, act_func);
            x = activation_function(x, fc4, act_func);
            x = activation_function(x, fc5, act_func);
            x = activation_function(x, fc6, act_func);
            x = activation_function(x, fc7, act_func);
            x = activation_function(x, fc8, act_func);
            x = activation_function(x, fc9, act_func);
            break;
        case 8:
//            std::cout << "8 Hidden layers" << std::endl;
            x = activation_function(x, fc3, act_func);
            x = activation_function(x, fc4, act_func);
            x = activation_function(x, fc5, act_func);
            x = activation_function(x, fc6, act_func);
            x = activation_function(x, fc7, act_func);
            x = activation_function(x, fc8, act_func);
            x = activation_function(x, fc9, act_func);
            x = activation_function(x, fc10, act_func);
            break;
        case 9:
//            std::cout << "9 Hidden layers" << std::endl;
            x = activation_function(x, fc3, act_func);
            x = activation_function(x, fc4, act_func);
            x = activation_function(x, fc5, act_func);
            x = activation_function(x, fc6, act_func);
            x = activation_function(x, fc7, act_func);
            x = activation_function(x, fc8, act_func);
            x = activation_function(x, fc9, act_func);
            x = activation_function(x, fc10, act_func);
            x = activation_function(x, fc11, act_func);
            break;
        case 10:
//            std::cout << "10 Hidden layers" << std::endl;
            x = activation_function(x, fc3, act_func);
            x = activation_function(x, fc4, act_func);
            x = activation_function(x, fc5, act_func);
            x = activation_function(x, fc6, act_func);
            x = activation_function(x, fc7, act_func);
            x = activation_function(x, fc8, act_func);
            x = activation_function(x, fc9, act_func);
            x = activation_function(x, fc10, act_func);
            x = activation_function(x, fc11, act_func);
            x = activation_function(x, fc12, act_func);
            break;
    }
//    x = torch::nn::functional::relu(fc1->forward(x));
//    x = torch::nn::functional::relu(fc3->forward(x));
    // Return output layer
    return fc2->forward(x);
}


