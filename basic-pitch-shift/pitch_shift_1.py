# Reads in audio, shifts the pitches, and plays output. 
# Based on: http://www.guitarpitchshifter.com/index.html
import pyaudio
import numpy as np
import struct

pitchShift = 5 # [semitones]
outputGain = 1

# Audio parameters
winSize = 1024 # samples per FFT window
hop = round(winSize / 4) # Hardcoded for now
WIDTH = 2 # 2 bytes per sample
CHANNELS = 1 # Mono
RATE = 44100 # Sampling frequency

# setup audio
p = pyaudio.PyAudio()
stream = p.open(format   = p.get_format_from_width(WIDTH),
				channels = CHANNELS,
				rate     = RATE,
				input    = True,
				output   = True,
				frames_per_buffer = hop)

# Shifting constants
alpha = 2**(pitchShift / 12) # 12 semitones in octave
hopout = round(alpha*hop)

# Hanning window
wn = np.hanning(2*winSize+1)
wn = wn[1:(2*winSize+1):2]
wn_before = wn / np.sqrt(((winSize/hop)/2))
wn_after  = wn / np.sqrt(((winSize/hopout)/2))

# Phase data saved between windows
previousPhase = np.zeros(winSize)
phaseCumulative = np.zeros(winSize)
circleVec = np.linspace(0, 2*np.pi*(winSize-1)/winSize, winSize)

# Saved frame pieces
frame_in_A = np.zeros(hop) # A is newest
frame_in_B = np.zeros(hop)
frame_in_C = np.zeros(hop)
frame_in_D = np.zeros(hop) # D is oldest

frame_out_A = np.zeros(winSize) # A is newest
frame_out_B = np.zeros(winSize)
frame_out_C = np.zeros(winSize)
frame_out_D = np.zeros(winSize)
frame_out_E = np.zeros(winSize) # E is oldest

def setup():
	print("* --- starting... ----------------")
	latency = stream.get_input_latency() + stream.get_output_latency()
	print('* latency =', round(latency,3), 's')
	print('* Press ctrl-c to stop...')

# Called repeatidly. Blocks at audio buffer input and output. 
def loop():
	global frame_in_A,frame_in_B,frame_in_C,frame_in_D
	global frame_out_A,frame_out_B,frame_out_C,frame_out_D
	
	# shift input frames
	frame_in_D = frame_in_C
	frame_in_C = frame_in_B
	frame_in_B = frame_in_A
	
	# read audio data
	data_in_bytes = stream.read(hop) # BLOCKS unitl valid
	data_in_int = struct.unpack(str(hop)+'h',data_in_bytes)
	frame_in_A = data_in_int # save new data
	
	# Assemble frame
	input_frame = np.concatenate((frame_in_D,frame_in_C,frame_in_B,frame_in_A))
	
	# FFT
	input_frame = input_frame * wn_before
	freqs_in_rect = np.fft.fft(input_frame)
	
	# Perform computation on frequency data
	freqs_out_rect = compute_frame(freqs_in_rect)
	
	# IFFT 
	output_frame = np.fft.ifft(freqs_out_rect).real
	output_frame = output_frame * wn_after
	
	# Shift output frames 
	frame_out_E = frame_out_D
	frame_out_D = frame_out_C
	frame_out_C = frame_out_B
	frame_out_B = frame_out_A
	frame_out_A = output_frame
	
	# fuse frames 
	part_e = pad_zeros(frame_out_E,5*hopout)[4*hopout:5*hopout]
	part_d = pad_zeros(frame_out_D,4*hopout)[3*hopout:4*hopout]
	part_c = pad_zeros(frame_out_C,3*hopout)[2*hopout:3*hopout]
	part_b = pad_zeros(frame_out_B,2*hopout)[1*hopout:2*hopout]
	part_a = pad_zeros(frame_out_A,1*hopout)[0*hopout:1*hopout]
	data_fused = part_a + part_b + part_c + part_d + part_e 
	
	# Interploate
	x = np.linspace(0,hopout-1,hopout)
	y = data_fused
	new_x = np.linspace(0,hopout-1,hop)
	data_out = np.interp(new_x, x, y)
	
	#package data for playback
	data_out_scaled = data_out * outputGain
	data_out_bytes = struct.pack(str(hop)+'h', *data_out_scaled.astype(int))
	stream.write(data_out_bytes, hop, exception_on_underflow=True) # BLOCKS until ready

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

# If vector is smaller than length, zero extend until it is of size length.
def pad_zeros(a,length):
	if len(a) < length:
		a = np.concatenate((a,np.zeros(length - len(a))))
	return a 

# Called to close Audio Stream objects
def stop():
	print("* --- done -----------------------")
	stream.stop_stream()
	stream.close()
	p.terminate()

if __name__ == '__main__':
	try:
		setup() # Call once at beginning
		while True: 
			loop() # Repeatidly call 
	except KeyboardInterrupt: # ctrl-C to stop script
		stop() # Call once at end
