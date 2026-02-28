#include "MLP.h"
#include "mnist.h"

// Pre-computed inverse scale factors (1/scale)
const ap_fixed<24,12> inv_scale_w0 = 1.0 / 227.6;
const ap_fixed<24,12> inv_scale_b0 = 1.0 / 774.9;
const ap_fixed<24,12> inv_scale_w1 = 1.0 / 259.6;
const ap_fixed<24,12> inv_scale_b1 = 1.0 / 905.3;
const ap_fixed<24,12> inv_scale_w2 = 1.0 / 235.2;
const ap_fixed<24,12> inv_scale_b2 = 1.0 / 1230.8;

// Top-level synthesis function
//
// Interface fix: input/output are large arrays (784 and 10 elements).
// AXI-Lite (s_axilite) is a register interface for scalars and cannot
// transfer bulk data. These ports use m_axi so the PS can DMA data
// via physical addresses written to the AXI-Lite control registers.
// The PYNQ notebook already writes physical addresses at offsets 0x10/0x18,
// which is exactly what m_axi expects.
void mlp_fast(const input_t input[INPUT_SIZE], output_t output[OUTPUT_SIZE]) {
    #pragma HLS INTERFACE m_axi     port=input  bundle=GMEM   offset=slave
    #pragma HLS INTERFACE m_axi     port=output bundle=GMEM   offset=slave
    #pragma HLS INTERFACE s_axilite port=input  bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=output bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=return bundle=CTRL

    #pragma HLS ARRAY_PARTITION variable=input  cyclic factor=16
    #pragma HLS ARRAY_PARTITION variable=output cyclic factor=10

    // Hidden layer buffers
    ap_fixed<24,12> layer1[LAYER1_SIZE];
    ap_fixed<24,12> layer2[LAYER2_SIZE];
    #pragma HLS ARRAY_PARTITION variable=layer1 cyclic factor=16
    #pragma HLS ARRAY_PARTITION variable=layer2 cyclic factor=16

    // Layer 1: 784 -> 256, ReLU
LAYER_1:
    for (int i = 0; i < LAYER1_SIZE; i++) {
        #pragma HLS PIPELINE
        ap_fixed<24,12> sum = bias_1[i] * inv_scale_b0;
    INPUT_LOOP_1:
        for (int j = 0; j < INPUT_SIZE; j++) {
            sum += input[j] * (weights_1[i * INPUT_SIZE + j] * inv_scale_w0);
        }
        layer1[i] = (sum > 0) ? sum : ap_fixed<24,12>(0);
    }

    // Layer 2: 256 -> 128, ReLU
LAYER_2:
    for (int i = 0; i < LAYER2_SIZE; i++) {
        #pragma HLS PIPELINE
        ap_fixed<24,12> sum = bias_2[i] * inv_scale_b1;
    INPUT_LOOP_2:
        for (int j = 0; j < LAYER1_SIZE; j++) {
            sum += layer1[j] * (weights_2[i * LAYER1_SIZE + j] * inv_scale_w1);
        }
        layer2[i] = (sum > 0) ? sum : ap_fixed<24,12>(0);
    }

    // Output layer: 128 -> 10 (raw logits, argmax on PS side)
OUTPUT_LAYER:
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        #pragma HLS PIPELINE
        ap_fixed<24,12> sum = bias_3[i] * inv_scale_b2;
    INPUT_LOOP_OUT:
        for (int j = 0; j < LAYER2_SIZE; j++) {
            sum += layer2[j] * (weights_3[i * LAYER2_SIZE + j] * inv_scale_w2);
        }
        output[i] = sum;
    }
}