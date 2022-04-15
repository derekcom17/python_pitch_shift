# Pitch selecting GUI
from appJar import gui
import PitchShifter

app = gui('Pitch Shift', '1000x200') # GUI object
shift = PitchShifter.PitchShifter() # Pitch Shifter opject

# Pitch select buttons
def change(button):
	shift.set(int(button))

# Quit button
def stop(button):
	app.stop()

# Two octaves of piano key placement
keys = [1,0,1,0,1,1,0,1,0,1,0,1, 1, 0,1,0,1,1,0,1,0,1,0,1]

# Makey piano keys for pitch select
app.setSticky('nsew')
for i in range(-12,13):
	app.addButton(str(i),change, keys[i],(i+12),1,2)
	if not keys[i]:
		app.addLabel(str(i), ' ', 2,(i+12))
	if keys[i]:
		app.setButtonBg(str(i),'white')
		app.setButtonFg(str(i),'black')
	else:
		app.setButtonBg(str(i),'black')
		app.setButtonFg(str(i),'white')

# Make quit button
app.setSticky('')
app.addButton('Quit', stop, 3,0,25,1)

shift.start()
app.go()
shift.stop()
