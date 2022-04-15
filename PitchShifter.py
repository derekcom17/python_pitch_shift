# PitchShifter.py
# Reads in audio, shifts the pitches, and plays output. 
# Based on: http://www.guitarpitchshifter.com/index.html
import pyaudio
import numpy as np
import time, struct, threading

class PitchShifter:
	def __init__(self, shift=0, gain=1, winSize=2048, hops=8):
		self.param_lock = threading.Lock()
		self.shift = 0 # init to no shift
		self.set(shift, winSize, hops)
		
		self.AUDIO_WIDTH = 2 # 2 bytes per sample
		self.AUDIO_CHANNELS = 1 # Mono
		self.AUDIO_FREQUENCY = 44100 # Sampling frequency
		self.AUDIO_FRAME_NUM = 2048 # This is arbitrary?
		self.outputGain = gain
		
		# Initialize audio
		self.p = pyaudio.PyAudio() 
		
	def set(self, shift, winSize=-1, hops=-1):
		updateFrames = updatePitch = False
		if winSize != -1 and hops != -1:
			updateFrames = True
			self.winSize = winSize
			self.hop = round(winSize / hops)
		if shift != self.shift or updateFrames:
			self.shift = shift
			updatePitch = True
		
		if updatePitch:
			self.param_lock.acquire() # Enables safe writing to parameters
			
			alpha = 2**(shift / 12) # 12 semitones in octave
			self.hopout = int(round(alpha*self.hop)) # time hop between each output frame
			wn = np.hanning(2*self.winSize+1) # Hanning windows
			wn = wn[1:(2*self.winSize+1):2]
			self.wn_before = wn / np.sqrt(((self.winSize/self.hop)/2))
			self.wn_after  = wn / np.sqrt(((self.winSize/self.hopout)/2))
			
			if updateFrames:
				print('updating frames...')
				self.winSize = winSize
				self.hop = hop = round(winSize / hops) # time hop between each input frame		
				assert winSize % hop == 0 # winSize must be divisible by hops_per_winsize
				
				# Input and output buffers
				self.buff_in  = Circ_Buff(round(winSize/hop), np.zeros(hop))
				#self.buff_out = Circ_Buff(int(np.ceil(winSize/self.hopout)),np.zeros(winSize))
				self.buff_out = Circ_Buff(hops*2,np.zeros(winSize)) # Assume shift no lower than 1 octave

				# Phase data saved between windows
				self.previousPhase   = np.zeros(winSize)
				self.phaseCumulative = np.zeros(winSize)
				self.circleVec = np.linspace(0, 2*np.pi*(winSize-1)/winSize, winSize)
			
			self.param_lock.release() # Release lock
	
	def start(self):
		self.stream = self.p.open(format   = self.p.get_format_from_width(self.AUDIO_WIDTH),
								  channels = self.AUDIO_CHANNELS,
								  rate     = self.AUDIO_FREQUENCY,
								  input    = True,
								  output   = True,
								  frames_per_buffer = self.AUDIO_FRAME_NUM)
		
		# Start thread
		self.audio_thread = threading.Thread(target=self.run_stream, args=(), daemon=True)
		self.audio_thread.start()
		
	def stop(self):
		# Stop thread
		self.continue_run = False
		self.audio_thread.join()		
		
		# Stop stream
		self.stream.stop_stream()
		self.stream.close()
		#self.p.terminate()
	
	def run_stream(self):
		self.continue_run = True
		while self.continue_run:
			# Update params
			self.param_lock.acquire()
			(winSize, hop, hopout) = (self.winSize, self.hop, self.hopout)
			(wn_before, wn_after) = (self.wn_before, self.wn_after)
			(buff_in, buff_out) = (self.buff_in, self.buff_out)
			
			# read new audio data
			data_in_int = self.from_stream(hop)
			
			# Add to input buffer and assemble window
			buff_in.push(data_in_int)
			input_frame = data_in_int
			for i in range(1,buff_in.depth):
				input_frame = np.concatenate((buff_in.get(i), input_frame))
			
			# Perform computation on frequency data 
			input_frame  = input_frame * wn_before # Window
			freqs_in_rect = np.fft.fft(input_frame) # FFT
			freqs_out_rect = self.compute_frame(freqs_in_rect)
			output_frame  = np.fft.ifft(freqs_out_rect).real # IFFT
			output_frame = output_frame * wn_after # Window
			
			# Add to output buffer and fuse
			buff_out.push(output_frame)
			data_fused = np.zeros(hopout)
			for i in range(int(np.ceil(winSize/self.hopout))):
				data_fused += pad_zeros(buff_out.get(i),(i+1)*hopout)[i*hopout:(i+1)*hopout]
			
			# Interploate back to original sampling frequency
			x     = np.linspace(0, hopout-1, hopout)
			new_x = np.linspace(0, hopout-1,    hop)
			self.param_lock.release() # Done with params
			y = data_fused
			data_out = np.interp(new_x, x, y)
			
			# Play data
			self.to_stream(data_out, self.outputGain)
			
	# Frequency domain computation on each window of data
	def compute_frame(self, freq_input_rect):
		(hop, hopout, circleVec) = (self.hop, self.hopout, self.circleVec)
	
		magFrame = np.abs(freq_input_rect)     # Magnitude part
		phaseFrame = np.angle(freq_input_rect) # Angle part
		
		deltaPhi = phaseFrame - self.previousPhase
		self.previousPhase = phaseFrame
		deltaPhiPrime = deltaPhi - hop * circleVec
		deltaPhiPrimeMod = np.mod(deltaPhiPrime+np.pi, 2*np.pi) - np.pi
		trueFreq = circleVec + deltaPhiPrimeMod / hop
		self.phaseCumulative  = self.phaseCumulative + hopout * trueFreq
		
		return magFrame*np.exp(1j*self.phaseCumulative) # convert back to rectangular
	
	# Returns numpy array from audio input. Blocks until input is valid.
	def from_stream(self, num):
		data_in_bytes = self.stream.read(num, exception_on_overflow=True)
		return struct.unpack(str(num)+'h', data_in_bytes)
	
	# Sends numpy array to audio output. Blocks until output is ready.
	def to_stream(self, data, gain=1):
		data = data * gain
		data_bytes = struct.pack(str(len(data))+'h', *data.astype(int)) 
		self.stream.write(data_bytes, len(data), exception_on_underflow=True) 

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

# If vector is smaller than length, zero extend until it is of size length.
def pad_zeros(a,length):
	if len(a) < length:
		a = np.concatenate((a,np.zeros(length - len(a))))
	return a

# Demo the Pitch Shifter. Shifts pitch based on user input
def main():
	try:
		# Init Pitch Shift stream
		shifter = PitchShifter() # No shift to start
		
		# Start the stream
		shifter.start() # Start
		print('--- Stream started ---')
		
		# Get user input to change shift
		while True:
			shifter.set(float(input('Set shift: ')))
		
	except KeyboardInterrupt: # Press ctrl-c to stop demo
		# Stop the stream
		shifter.stop()
		print('\n--- Stream stopped ---')

if __name__ == '__main__':
	main()
