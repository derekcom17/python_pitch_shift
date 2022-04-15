# Test framing for pitch shift

import pyaudio
import numpy as np
import struct

# Audio parameters
winSize = 1024
hop = round(winSize / 4) # Hardcoded for now
WIDTH = 2
CHANNELS = 1
RATE = 44100

# setup audio
p = pyaudio.PyAudio()
stream = p.open(format=p.get_format_from_width(WIDTH),
	channels=CHANNELS,
	rate=RATE,
	input=True,
	output=True,
	frames_per_buffer=hop)

# Hanning window
wn = np.hanning(2*winSize+1)
wn = wn[1:(2*winSize+1):2]
wn = wn / np.sqrt(((winSize/hop)/2)) # What is this for???

def setup():
	print("* starting...")
	print('input  latency = ',stream.get_input_latency())
	print('output latency = ',stream.get_output_latency())

frame_in_A = np.zeros(hop) # A is newest
frame_in_B = np.zeros(hop)
frame_in_C = np.zeros(hop)
frame_in_D = np.zeros(hop) # D is oldest

frame_out_A = np.zeros(winSize).astype(int) # A is newest
frame_out_B = np.zeros(winSize).astype(int)
frame_out_C = np.zeros(winSize).astype(int)
frame_out_D = np.zeros(winSize).astype(int) # D is oldest

def loop():
	global frame_in_A,frame_in_B,frame_in_C,frame_in_D
	global frame_out_A,frame_out_B,frame_out_C,frame_out_D
	
	# shift input frames
	frame_in_D = frame_in_C
	frame_in_C = frame_in_B
	frame_in_B = frame_in_A
	
	# read audio data
	data_in_bytes = stream.read(hop)
	data_in_int = struct.unpack(str(hop)+'h',data_in_bytes)
	frame_in_A = data_in_int # save new data
	
	# Assemble frame
	input_frame = np.concatenate((frame_in_D,frame_in_C,frame_in_B,frame_in_A))
	
	# Hanning window
	input_frame = input_frame * wn
	
	# FFT
	freqs_in_rect = np.fft.fft(input_frame)
	
	# Perform computation/edit ---------------
	freqs_out_rect = np.roll(freqs_in_rect,0)
	# ----------------------------------------
	
	# IFFT 
	output_frame = np.fft.ifft(freqs_out_rect).real
	output_frame = output_frame * wn
	#print(output_frame)
	
	# Shift output frames
	frame_out_D = frame_out_C
	frame_out_C = frame_out_B
	frame_out_B = frame_out_A
	frame_out_A = output_frame
	
	# sum frames
	data_out_int = np.zeros(hop).astype(int)
	for i in range(hop):
		d = frame_out_D[3*hop+i]
		c = frame_out_C[2*hop+i]
		b = frame_out_B[1*hop+i]
		a = frame_out_A[0*hop+i]
		data_out_int[i] = round(a + b + c + d)
	
	#print(data_out_int)
	#package data for playback
	data_out_bytes = b''
	gain = 1
	for d in data_out_int:
		data_out_bytes += struct.pack('h',d*gain)
	stream.write(data_out_bytes, hop, exception_on_underflow=True)

def stop_script():
	print("* done.")
	stream.stop_stream()
	stream.close()
	p.terminate()

if __name__ == '__main__':
	try:
		setup()
		while True:
			loop()
	except KeyboardInterrupt:
		stop_script()
