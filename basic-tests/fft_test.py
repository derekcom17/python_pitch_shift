import numpy as np
import time, timeit

fftSize = 1024
samples = np.random.random(fftSize)

def big_fft():
	np.fft.fft(samples)

print('starting timeit...')
time_ms = timeit.timeit(big_fft,number=100000)/100
print(time_ms,'ms')