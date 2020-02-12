"""
	Kaggle Titanic Competition -- 2020-02-12
	kaggle/mimocha
	github/mimocha
	Chawit Leosrisook
"""

import copy
import csv
import math
import random

"""
	Function to read & extract data from csv file
	
	IO Format:
	list(list(...)) = ReadCSV (str)
"""
def ReadCSV (filename):
	output = []
	with open(filename, 'r') as csvfile:
		datastream = csv.reader(csvfile, delimiter=',')
		for row in datastream:
			output.append(row)
		return output

"""
	Function to save prediction to csv file
	
	IO Format:
	SaveCSV (str, list(int), list(int))
"""
def SaveCSV (filename, passengerID, prediction):
	with open(filename, 'w', newline='') as file:
		writer = csv.writer(file)
		writer.writerow(["PassengerId", "Survived"])
		for pid, p in zip(passengerID, prediction):
			writer.writerow([pid, p])

"""
	Function to split training-testing data
	Shuffles data order
	Default ratio set to 80% Training : 20% Testing
	First row of training set is always header
	Testing set has no header row
	
	IO Format:
	tuple(list(...), list(...)) = SplitData(list(list(...)), float)
"""
def SplitData (rawData, ratio=0.8):
	# Make copy of data to manipulate
	data = copy.deepcopy(rawData)

	# Remove header row from data set
	header = data.pop(0)

	# Shuffle & Split
	random.shuffle(data)
	dataNum = len(data)
	p = math.floor(dataNum * ratio) # index pivot

	# Header goes back into training set
	train = copy.deepcopy([header] + data[0:p])
	test = copy.deepcopy(data[p:-1])

	# Return Result
	return (train, test)
