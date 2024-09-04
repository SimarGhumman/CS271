import numpy as np
from scipy.signal import convolve2d

def max_pooling(img, filter_size=3, stride=3):
    h, w = img.shape
    output_height = (h - filter_size) // stride + 1
    output_width = (w - filter_size) // stride + 1
    output = np.zeros((output_height, output_width))

    for i in range(0, h-filter_size+1, stride):
        for j in range(0, w-filter_size+1, stride):
            output[i//stride, j//stride] = np.max(img[i:i+filter_size, j:j+filter_size])

    return output

img = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
)

anti_diagonal_filter = np.array(
    [
        [-1, -1, 2],
        [-1, 2, -1],
        [2, -1, -1]
    ]
)

diagonal_filter = np.array(
    [
        [2, -1, -1],
        [-1, 2, -1],
        [-1, -1, 2]
    ]
)

conv_result = convolve2d(img, anti_diagonal_filter, mode='valid')

new_conv_result = convolve2d(img, diagonal_filter, mode='valid')
padded_result = np.pad(new_conv_result, ((0,1),(0,1)), mode='constant', constant_values=0)
pooled_result = max_pooling(padded_result, 3, 3)

print(conv_result)
print(pooled_result)