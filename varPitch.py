# Test varying pitches
from PitchShifter import *
import time
import numpy as np

# Init Pitch Shift stream
shifter = PitchShifter(winSize=2048, hops=4) # No shift to start

MV_PERIOD = .010
MV_MAG = 2.0 #0.5
MV_OFFSET = 4

def mover():
	for x in np.linspace(0,2*np.pi,75):
		time.sleep(MV_PERIOD)
		shifter.set(MV_OFFSET + MV_MAG * np.sin(x))
	

def main():
	try:
		# Start the stream
		shifter.start() # Start
		print('--- Stream started ---')
		
		while True:
			mover()
		
	except KeyboardInterrupt: # Press ctrl-c to stop demo
		# Stop the stream
		shifter.stop()
		print('\n--- Stream stopped ---')

if __name__ == '__main__':
	main()