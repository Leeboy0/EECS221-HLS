#include <iostream>
#include <cstdlib>
#include <ctime>
#include "MLP.h"

int main() {

    // Input/output initialization
    input_t test_input[INPUT_SIZE];
    output_t output[OUTPUT_SIZE];

    // Seed random number generator
    srand(time(0));

    // Initialize inputs with random normalized values
    for (int i = 0; i < INPUT_SIZE; i++) {
        test_input[i] = (input_t)((rand() % 256) / 255.0);
    }

    // Call the top-level function
    mlp_fast(test_input, output);

    // Display output
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        std::cout << "Digit " << i << ": " << float(output[i]) << std::endl;
    }

    return 0;
}
