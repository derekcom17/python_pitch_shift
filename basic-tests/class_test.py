# class_test.py
import numpy as np

class SUBTEST:
	def __init__(self):
		self.data = [1,2,3,4]

class TEST:
	def __init__(self):
		self.sub = SUBTEST()
		self.arr = np.array([10,11,12])
	
	def test_set(self):
		sub = self.sub # <-- Referance semantics is used here!!!
		arr = self.arr # <-- Value semantics is used here!!!
		
		sub.data = [5,6,7,8]
		arr = np.array([13,14,15])
		self.other_met()
		
	
	def other_met(self):
		print('other_met ran!')