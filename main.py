import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from heapq import heappop, heappush, heapify
from collections import defaultdict


def fft_compression(img, keep_fraction=0.9):
    # Compute the FFT of the img
    f = np.fft.fft2(img)

    # Compute the magnitude and keep only the top components
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    threshold = np.percentile(magnitude_spectrum, 100 * (1 - keep_fraction))
    print(f"threshold: {threshold}")

    compressed_fft = np.where(magnitude_spectrum >= threshold, f, 0)

    return compressed_fft


def quantize(compressed_fft, levels=256):
    max_val = np.max(np.abs(compressed_fft))
    quantized = np.round(compressed_fft * (levels - 1) / max_val)
    print(f"max_val: {max_val}")

    return quantized, max_val


from collections import Counter
from heapq import heappush, heappop


class Node:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq


def build_huffman_tree(data):
    frequency = Counter(data)
    priority_queue = [Node(char, freq) for char, freq in frequency.items()]
    while len(priority_queue) > 1:
        left = heappop(priority_queue)
        right = heappop(priority_queue)
        merged = Node(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        heappush(priority_queue, merged)
    return priority_queue[0]


def huffman_codes(tree, prefix="", mapping=None):
    if mapping is None:
        mapping = dict()
    if tree is not None:
        if tree.char is not None:
            mapping[tree.char] = prefix
        huffman_codes(tree.left, prefix + "0", mapping)
        huffman_codes(tree.right, prefix + "1", mapping)
    return mapping


def huffman_encode(data, mapping):
    return "".join([mapping[char] for char in data])


def huffman_decode(encoded, tree):
    decoded = []
    current = tree
    for bit in encoded:
        if bit == "0":
            current = current.left
        else:
            current = current.right
        if current.char is not None:
            decoded.append(current.char)
            current = tree
    return decoded


img = cv.imread("Dhoni_dive.jpg", cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"

# compression
compressed_fft = fft_compression(img)
print(f"compressed_fft: {compressed_fft}")
quantized, max_val = quantize(compressed_fft)
flattened_quantized = quantized.ravel()  # flatten the 2D numpy array into 1D
tree = build_huffman_tree(flattened_quantized)

mapping = huffman_codes(tree)
encoded = huffman_encode(flattened_quantized, mapping)

# decompression
decoded = huffman_decode(encoded, tree)

reconstructed_fft = np.array(decoded) * max_val / 255
print(f"reconstructed_fft: {reconstructed_fft}")
reconstructed_fft = reconstructed_fft.reshape(img.shape)
print(f"reconstructed_fft: {reconstructed_fft}")


reconstructed_signal = np.fft.ifft2(reconstructed_fft).real
print(f"reconstructed_signal: {reconstructed_signal}")

# Display original image
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Original Image")
plt.axis("off")

# Display reconstructed image
plt.subplot(1, 2, 2)
plt.imshow(reconstructed_signal, cmap="gray")
plt.title("Reconstructed Image")
plt.axis("off")

plt.tight_layout()
plt.show()
