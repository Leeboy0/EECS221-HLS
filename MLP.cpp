#include "MLP.h"
#include "mnist.h"

// Pre-computed inverse scale factors (1/scale)
const ap_fixed<24,12> inv_scale_w0 = 1.0 / 288.0;
const ap_fixed<24,12> inv_scale_b0 = 1.0 / 1419.0;
const ap_fixed<24,12> inv_scale_w1 = 1.0 / 270.0;
const ap_fixed<24,12> inv_scale_b1 = 1.0 / 1405.0;
const ap_fixed<24,12> inv_scale_w2 = 1.0 / 245.0;
const ap_fixed<24,12> inv_scale_b2 = 1.0 / 2290.0;

// Top-level synthesis function
void mlp_fast(const input_t input[INPUT_SIZE], output_t output[OUTPUT_SIZE]) {
    #pragma HLS INTERFACE mode=s_axilite port=return bundle=CTRL_BUS
    #pragma HLS INTERFACE mode=s_axilite port=input bundle=CTRL_BUS
    #pragma HLS INTERFACE mode=s_axilite port=output bundle=CTRL_BUS
    #pragma HLS ARRAY_PARTITION variable=input cyclic factor=16
    #pragma HLS ARRAY_PARTITION variable=output cyclic factor=16

    // Hidden layer buffers
    ap_fixed<24,12> layer1[LAYER1_SIZE];
    ap_fixed<24,12> layer2[LAYER2_SIZE];
    #pragma HLS ARRAY_PARTITION variable=layer1 cyclic factor=16
    #pragma HLS ARRAY_PARTITION variable=layer2 cyclic factor=16

    // Layer 1 computation (fully fixed-point)
LAYER_1: 
    for(int i = 0; i < LAYER1_SIZE; i++){
        #pragma HLS PIPELINE
        ap_fixed<24,12> sum = bias_1[i] * inv_scale_b0;
        INPUT_LOOP_1: 
        for(int j = 0; j < INPUT_SIZE; j++){
            sum += input[j] * (weights_1[i * INPUT_SIZE + j] * inv_scale_w0);
        }
        layer1[i] = (sum > 0) ? sum : ap_fixed<24,12>(0);
    }

    // Layer 2 computation (fully fixed-point)
LAYER_2: 
    for(int i = 0; i < LAYER2_SIZE; i++){
        #pragma HLS PIPELINE
        ap_fixed<24,12> sum = bias_2[i] * inv_scale_b1;
        INPUT_LOOP_2: 
        for(int j = 0; j < LAYER1_SIZE; j++){
            sum += layer1[j] * (weights_2[i * LAYER1_SIZE + j] * inv_scale_w1);
        }
        layer2[i] = (sum > 0) ? sum : ap_fixed<24,12>(0);
    }

    // Output layer computation (fully fixed-point)
OUTPUT_LAYER: 
    for(int i = 0; i < OUTPUT_SIZE; i++){
        #pragma HLS PIPELINE
        ap_fixed<24,12> sum = bias_3[i] * inv_scale_b2;
        INPUT_LOOP_OUT: 
        for(int j = 0; j < LAYER2_SIZE; j++){
            sum += layer2[j] * (weights_3[i * LAYER2_SIZE + j] * inv_scale_w2);
        }
        output[i] = sum;
    }
}
