# Reads in audio, shifts the pitches, and plays output. 
# Based on: http://www.guitarpitchshifter.com/index.html
import pyaudio
import numpy as np
import struct

pitchShift = -12 # semitones to shift by
outputGain = 1 # volume control

# Audio parameters
winSize = 2**11 # samples per FFT window
hops_per_winsize = 4 # winSize must be divisible by this
WIDTH = 2 # 2 bytes per sample
CHANNELS = 1 # Mono
RATE = 44100 # Sampling frequency

# Shifting variables
hop = round(winSize / hops_per_winsize) # time hop between each input frame
assert winSize % hop == 0 # winSize must be divisible by hops_per_winsize
alpha = 2**(pitchShift / 12) # 12 semitones in octave
hopout = round(alpha*hop) # time hop between each output frame

# setup audio
p = pyaudio.PyAudio()
stream = p.open(format   = p.get_format_from_width(WIDTH),
				channels = CHANNELS,
				rate     = RATE,
				input    = True,
				output   = True,
				frames_per_buffer = 1024) # Should this be "hop" ?

# Hanning windows
wn = np.hanning(2*winSize+1)
wn = wn[1:(2*winSize+1):2]
wn_before = wn / np.sqrt(((winSize/hop)/2))
wn_after  = wn / np.sqrt(((winSize/hopout)/2))

# Circular buffer object to hold frame pieces
class Circ_Buff:
	def __init__(self, depth, init=False):
		data = []
		for i in range(depth):
			data.append(init)
		self.depth = depth # Number of items in buffer
		self._data = data  # Data stored in list
		self._lru  = 0     # Least reciently used pointer
	
	def push(self, item):
		self._data[self._lru] = item
		self._lru = (self._lru + 1) % self.depth
		
	def get(self, n): # '0' returns newest data
		assert n in range(self.depth) # index must be in [0,...,depth-1]
		return self._data[(self._lru-1-n)%self.depth] 

# Input and output buffers
buff_in  = Circ_Buff(hops_per_winsize,np.zeros(hop))
buff_out = Circ_Buff(int(np.ceil(winSize/hopout)),np.zeros(winSize))

# Phase data saved between windows
previousPhase   = np.zeros(winSize)
phaseCumulative = np.zeros(winSize)
circleVec = np.linspace(0, 2*np.pi*(winSize-1)/winSize, winSize)

# Called at beginning to initialize
def setup():
	print("* --- starting... ----------------")
	latency = stream.get_input_latency() + stream.get_output_latency()
	print('* Latency =', round(latency,3), 's')
	print('* Shift =',pitchShift)
	print('* Input hop =',hop,'Output hop =',hopout)
	print('* Output buffer depth =',int(np.ceil(winSize/hopout)))
	print('* Press ctrl-c to stop...')

# Called repeatidly 
def loop():	
	# read new audio data
	data_in_int = from_stream(hop)
	
	# Add to input buffer and assemble window
	buff_in.push(data_in_int)
	input_frame = data_in_int
	for i in range(1,buff_in.depth):
		input_frame = np.concatenate((buff_in.get(i), input_frame))
	
	# FFT
	input_frame = input_frame * wn_before
	freqs_in_rect = np.fft.fft(input_frame)
	
	# Perform computation on frequency data
	freqs_out_rect = compute_frame(freqs_in_rect)
	
	# IFFT 
	output_frame = np.fft.ifft(freqs_out_rect).real
	output_frame = output_frame * wn_after
	
	# Add to output buffer and fuse
	buff_out.push(output_frame)
	data_fused = np.zeros(hopout)
	for i in range(buff_out.depth):
		data_fused += pad_zeros(buff_out.get(i),(i+1)*hopout)[i*hopout:(i+1)*hopout]
	
	# Interploate back to original sampling frequency
	x     = np.linspace(0, hopout-1, hopout)
	new_x = np.linspace(0, hopout-1,    hop)
	y = data_fused
	data_out = np.interp(new_x, x, y)
	
	# Play data
	to_stream(data_out, outputGain)

# Frequency domain computation on each window of data
def compute_frame(freq_input_rect):
	global previousPhase, phaseCumulative
	magFrame = np.abs(freq_input_rect)     # Magnitude part
	phaseFrame = np.angle(freq_input_rect) # Angle part
	
	deltaPhi = phaseFrame - previousPhase
	previousPhase = phaseFrame
	deltaPhiPrime = deltaPhi - hop * circleVec
	deltaPhiPrimeMod = np.mod(deltaPhiPrime+np.pi, 2*np.pi) - np.pi
	trueFreq = circleVec + deltaPhiPrimeMod / hop
	phaseCumulative  = phaseCumulative + hopout * trueFreq
	
	return magFrame*np.exp(1j*phaseCumulative) # convert back to rectangular

# Called to close Audio Stream objects
def stop():
	stream.stop_stream()
	stream.close()
	p.terminate()
	print("* --- done -----------------------")

# Returns numpy array from audio input. Blocks until input is valid.
def from_stream(num):
	data_in_bytes = stream.read(num, exception_on_overflow=True)
	return struct.unpack(str(num)+'h', data_in_bytes)

# Sends numpy array to audio output. Blocks until output is ready.
def to_stream(data, gain=1):
	data = data * gain
	data_bytes = struct.pack(str(len(data))+'h', *data.astype(int)) 
	stream.write(data_bytes, len(data), exception_on_underflow=True) 

# If vector is smaller than length, zero extend until it is of size length.
def pad_zeros(a,length):
	if len(a) < length:
		a = np.concatenate((a,np.zeros(length - len(a))))
	return a 

if __name__ == '__main__':
	try:
		setup() # Call once at beginning
		while True: 
			loop() # Repeatidly call 
	except KeyboardInterrupt: # ctrl-C to stop script
		stop() # Call once at end
