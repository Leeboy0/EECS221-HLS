/*
 * MLP_tb.cpp — Vitis HLS C Simulation Testbench
 *
 * Loads real MNIST test images and checks predicted vs actual labels.
 * Returns 0 (pass) if accuracy >= threshold, non-zero (fail) otherwise.
 *
 * MNIST binary files needed (download from http://yann.lecun.com/exdb/mnist/):
 *   t10k-images-idx3-ubyte
 *   t10k-labels-idx1-ubyte
 */

#include <iostream>
#include <fstream>
#include <cstdint>
#include "MLP.h"

#define NUM_TEST        100
#define ACCURACY_THRESH 0.90f

static uint32_t swap32(uint32_t x) {
    return ((x & 0xFF) << 24) | (((x >> 8) & 0xFF) << 16) |
           (((x >> 16) & 0xFF) << 8) | ((x >> 24) & 0xFF);
}

bool load_mnist_images(const char* path, float images[][INPUT_SIZE], int count) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "ERROR: Cannot open " << path << std::endl;
        return false;
    }
    uint32_t magic, num, rows, cols;
    f.read((char*)&magic, 4); magic = swap32(magic);
    f.read((char*)&num,   4); num   = swap32(num);
    f.read((char*)&rows,  4);
    f.read((char*)&cols,  4);

    for (int i = 0; i < count && i < (int)num; i++)
        for (int j = 0; j < INPUT_SIZE; j++) {
            uint8_t pixel; f.read((char*)&pixel, 1);
            images[i][j] = pixel / 255.0f;
        }
    return true;
}

bool load_mnist_labels(const char* path, int labels[], int count) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "ERROR: Cannot open " << path << std::endl;
        return false;
    }
    uint32_t magic, num;
    f.read((char*)&magic, 4);
    f.read((char*)&num,   4); num = swap32(num);
    for (int i = 0; i < count && i < (int)num; i++) {
        uint8_t label; f.read((char*)&label, 1);
        labels[i] = (int)label;
    }
    return true;
}

int main() {
    static float   images[NUM_TEST][INPUT_SIZE];
    static int     labels[NUM_TEST];
    input_t  hls_input[INPUT_SIZE];
    output_t hls_output[OUTPUT_SIZE];

    if (!load_mnist_images("t10k-images-idx3-ubyte", images, NUM_TEST)) return -1;
    if (!load_mnist_labels("t10k-labels-idx1-ubyte", labels, NUM_TEST)) return -1;

    std::cout << "Running C-sim on " << NUM_TEST << " MNIST images..." << std::endl;

    int correct = 0;
    for (int n = 0; n < NUM_TEST; n++) {
        for (int i = 0; i < INPUT_SIZE; i++)
            hls_input[i] = (input_t)images[n][i];

        mlp_fast(hls_input, hls_output);

        int predicted = 0;
        output_t max_val = hls_output[0];
        for (int i = 1; i < OUTPUT_SIZE; i++)
            if (hls_output[i] > max_val) { max_val = hls_output[i]; predicted = i; }

        if (predicted == labels[n]) correct++;
        if (n < 10)
            std::cout << "  [" << n << "] predicted=" << predicted
                      << " actual=" << labels[n]
                      << (predicted == labels[n] ? " PASS" : " FAIL") << std::endl;
    }

    float accuracy = (float)correct / NUM_TEST;
    std::cout << "\nAccuracy: " << correct << "/" << NUM_TEST
              << " = " << accuracy * 100.0f << "%" << std::endl;

    if (accuracy >= ACCURACY_THRESH) {
        std::cout << "PASS" << std::endl; return 0;
    } else {
        std::cout << "FAIL" << std::endl; return 1;
    }
}