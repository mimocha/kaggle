"""
	Kaggle Titanic Competition -- 2020-02-02
	kaggle/mimocha
	github/mimocha
	Chawit Leosrisook
"""

""" =========================================================================================== """
""" ================================== FUNCTION DEFINITION ==================================== """
""" =========================================================================================== """

"""
	Function to read & extract data from csv file
"""
def ExtractData (filename):
	output = []
	with open(filename, 'r') as csvfile:
		datastream = csv.reader(csvfile, delimiter=',')
		for row in datastream:
			# do something
			# output as what kind of data types?
			# options to select which variables to read?
			output.append(row)
		return output

"""
	Function to split training-testing data
	Shuffles data order
	Default ratio set to 80% Training : 20% Testing
	
	IO Format:
	tuple(list(...), list(...)) = SplitData(list(list(...)), float)
"""
def SplitData (data, ratio=0.8):
	# Shuffle Dataset
	random.shuffle(data)
	dataNum = len(data)
	p = math.floor(dataNum * ratio) # index pivot

	# Split Dataset
	train = copy.deepcopy(data[0:p])
	test = copy.deepcopy(data[p:-1])

	# Return Result
	return (train, test)

"""
	Survival chance, based on features
	If feature is not provided, shows raw survival chance
	
	IO Format:
	dict{key : tuple(...)} = SurvivalChance(list(list(...)), 'string')
"""
def SurvivalChance (data, feat=None):
	# Raw survival chance
	if feat == None:
		s = 0 # Survived
		t = 0 # Total
		for row in data[1:-1]:
			s += int(row[1])
			t += 1
		return {'raw' : (s, t, s/t)}

	# Survival chance based on arbitrary feature
	# Invalid type catch
	if feat in data[0] == False:
		print('Feature \'{0}\' not found.\n'.format(feat))
		return -1
	# Get column index
	else:
		col = data[0].index(feat)

	# Make dictionary := featureValue : (rawValue, survivalRatio)
	# Skip data[0] -> Header row
	sDict = {} # Survival
	tDict = {} # Total
	for row in data[1:-1]:
		sDict[row[col]] = sDict.get(row[col], 0) + int(row[1])
		tDict[row[col]] = tDict.get(row[col], 0) + 1

	rDict = {} # Result
	for s,t in zip(sDict.items(), tDict.values()):
		# featureValue : (rawSurvival, rawTotal, ratioSurvive)
		rDict[s[0]] = (s[1], t, s[1]/t)

	# Sort Dictionary
	rDict = {k:v for k,v in sorted(rDict.items())}
	return rDict



""" =========================================================================================== """
""" ======================================= MAIN SCRIPT ======================================= """
""" =========================================================================================== """

import csv
import random
import math
import copy
import numpy as np
import matplotlib.pyplot as plt

# Read Full Data
filename = './data/train.csv'
rawdata = ExtractData(filename)

print('Feat No. : Feat Name')
for i,feat in enumerate(rawdata[0]):
	print('{0:2d} : {1}'.format(i, feat))

# Raw Survival Chance
rawSC = SurvivalChance(rawdata)
print('\n\tRaw Survival:')
for key,val in rawSC.items():
	print('{0:s} :: {1:d} / {2:d} : {3:.2f}%'.format(key, val[0], val[1], val[2]*100))

# Class Survival Chance
classSC = SurvivalChance(rawdata, 'Pclass')
print('\n\tClass Survival:')
for key,val in classSC.items():
	print('{0:s} :: {1:d} / {2:d} : {3:.2f}%'.format(key, val[0], val[1], val[2]*100))

# Gender Survival Chance
genderSC = SurvivalChance(rawdata, 'Sex')
print('\n\tGender Survival:')
for key,val in genderSC.items():
	print('{0:s} :: {1:d} / {2:d} : {3:.2f}%'.format(key, val[0], val[1], val[2]*100))

# Age Survival Chance
# Age Groups: No Data, Baby, Children, Young Adult, Middle-Age Adult, Senior Adult
# Division: -1, 0-2, 3-12, 13-30, 31-59, 60+
ageSC = SurvivalChance(rawdata, 'Age')
print('\n\tAge Survival:')

ageGroup = {'No Data': [0,0,0],
			'Baby': [0,0,0],
			'Child': [0,0,0],
			'Y.Adult': [0,0,0],
			'M.Adult': [0,0,0],
			'Senior': [0,0,0]}
cutoff = [-1, 2, 12, 30, 59, 999]
for key,val in ageSC.items():
	try: age = float(key)
	except: age = -1
	for cut, keyiter in zip(cutoff, ageGroup.keys()):
		if age <= cut:
			ageGroup[keyiter][0] = ageGroup[keyiter][0] + val[0] # Survivor
			ageGroup[keyiter][1] = ageGroup[keyiter][1] + val[1] # Total
			ageGroup[keyiter][2] = ageGroup[keyiter][0] / ageGroup[keyiter][1] # Ratio
			break

for key, val in ageGroup.items():
	print('{0:s} :: {1:d} / {2:d} : {3:.2f}%'.format(key, val[0], val[1], val[2]*100))

plt.bar([i for i in range(6)], [i[2] for i in ageGroup.values()])
plt.show()

## Training Loop

# Split Training-Testing Data
# (train, test) = SplitData(rawdata, 0.8)

# Train Model


# Test Model

## Use Model on Final Test

# Submit Results

