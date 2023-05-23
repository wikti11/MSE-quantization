import numpy as np

def optimal_uniform_quantizer(samples, levels):
    # Inicjalizacja punktów kwantyzacji
    min_val = min(samples)
    max_val = max(samples)
    quantization_points = np.linspace(min_val, max_val, levels)

    # Iteracyjne aktualizowanie punktów kwantyzacji
    while True:
        # Kwantyzacja próbek
        quantized_samples = np.zeros_like(samples)
        for i in range(len(samples)):
            quantized_samples[i] = quantization_points[np.argmin(np.abs(samples[i] - quantization_points))]

        # Aktualizacja punktów kwantyzacji
        updated_points = np.zeros(levels)
        counts = np.zeros(levels)
        for i in range(len(samples)):
            index = np.argmin(np.abs(samples[i] - quantization_points))
            updated_points[index] += samples[i]
            counts[index] += 1

        for i in range(levels):
            if counts[i] > 0:
                quantization_points[i] = updated_points[i] / counts[i]

        if np.allclose(quantized_samples, samples):
            break

    return quantization_points

# Dane wejściowe
samples = np.array([1.1, 3.2, -5.5, 4.2, -5.5, 0, -0.7])
levels = 10

# Znalezienie optymalnego kwantyzatora
quantizer = optimal_uniform_quantizer(samples, levels)

# Kwantyzacja próbek za pomocą optymalnego kwantyzatora
quantized_samples = np.zeros_like(samples)
for i in range(len(samples)):
    quantized_samples[i] = quantizer[np.argmin(np.abs(samples[i] - quantizer))]

# Obliczenie błędu kwantyzacji (MSE)
mse = np.mean((samples - quantized_samples) ** 2)

# Wyświetlenie wyników
print("Optymalny kwantyzator jednostajny:")
print("Próbki kwantyzowane:")
print(quantized_samples)
print("Punkty kwantyzacji:")
print(quantizer)
print("Błąd kwantyzacji (MSE):", mse)