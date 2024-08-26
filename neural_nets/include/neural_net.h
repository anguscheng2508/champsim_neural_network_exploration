// Copyright 2020-present pytorch-cpp Authors
#pragma once

#include <torch/torch.h>
#include <string.h>


class NeuralNetImpl : public torch::nn::Module {
 public:
    NeuralNetImpl(int64_t input_size, int64_t hidden_size, int64_t num_classes);

    torch::Tensor forward(torch::Tensor x, std::string act_func, int layers);

 private:
    torch::nn::Linear fc1;
    torch::nn::Linear fc3;
    torch::nn::Linear fc4;
    torch::nn::Linear fc5;
    torch::nn::Linear fc6;
    torch::nn::Linear fc7;
    torch::nn::Linear fc8;
    torch::nn::Linear fc9;
    torch::nn::Linear fc10;
    torch::nn::Linear fc11;
    torch::nn::Linear fc12;
    torch::nn::Linear fc2;
};

TORCH_MODULE(NeuralNet);
