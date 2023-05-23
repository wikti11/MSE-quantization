# nie działa

import numpy as np

def optimal_uniform_quantizer(samples, levels):
    # Initialization
    min_val = min(samples)
    max_val = max(samples)
    step_size = (max_val - min_val) / levels

    quantization_points = np.linspace(min_val + step_size/2, max_val - step_size/2, levels)

    while True:
        # Quantize samples based on current quantization points
        quantized_samples = np.round((samples - min_val) / step_size) * step_size + min_val

        # Update quantization points based on the mean values of the samples
        updated_points = np.zeros(levels)
        counts = np.zeros(levels)

        for i in range(len(samples)):
            index = int((samples[i] - min_val) // step_size)  # Corrected index calculation
            updated_points[index] += samples[i]
            counts[index] += 1

        for i in range(levels):
            if counts[i] > 0:
                quantization_points[i] = updated_points[i] / counts[i]

        if np.allclose(quantization_points, updated_points / counts):
            break

    return quantization_points

# Example data
samples = np.array([1.1, 3.2, -5.5, 4.2, -5.5, 0, -0.7])
levels = 10

quantizer = optimal_uniform_quantizer(samples, levels)

print("Quantization points:")
print(quantizer)

quantized_samples = np.round((samples - min(samples)) / ((max(samples) - min(samples)) / levels)) * ((max(samples) - min(samples)) / levels) + min(samples)
mse = np.mean((samples - quantized_samples) ** 2)
print("Quantization error (MSE):")
print(mse)