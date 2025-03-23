#ifndef MLP_H
#define MLP_H

#include "ap_fixed.h"
#include "ap_int.h"

#define INPUT_SIZE 784
#define LAYER1_SIZE 256
#define LAYER2_SIZE 128
#define OUTPUT_SIZE 10

typedef ap_fixed<16,6> input_t;
typedef ap_fixed<16,6> output_t;
typedef ap_int<8> weight_t;

void mlp_fast(const input_t input[INPUT_SIZE], output_t output[OUTPUT_SIZE]);

#endif
