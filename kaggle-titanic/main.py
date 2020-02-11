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
	
	IO Format:
	list(list(...)) = ExtractData (str)
"""
def ExtractData (filename):
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
def SaveCSV (filename, passid, prediction):
	with open(filename, 'w', newline='') as file:
		writer = csv.writer(file)
		writer.writerow(["PassengerId", "Survived"])

		for pid, hp in zip(passid, prediction):
			writer.writerow([pid, hp])

"""
	Function to split training-testing data
	Shuffles data order
	Default ratio set to 80% Training : 20% Testing
	
	IO Format:
	tuple(list(...), list(...)) = SplitData(list(list(...)), float)
"""
def SplitData (rawData, ratio=0.8):
	# Make copy of data to manipulate
	data = copy.deepcopy(rawData)

	# Get Data Header
	header = data.pop(0)

	# Shuffle Dataset
	random.shuffle(data)
	dataNum = len(data)
	p = math.floor(dataNum * ratio) # index pivot

	# Split Dataset
	train = copy.deepcopy([header] + data[0:p])
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

"""
	Preliminary Test with raw data
"""
def Preliminary (rawData):
	print('Feat No. : Feat Name')
	for i, feat in enumerate(rawData[0]):
		print('{0:2d} : {1}'.format(i, feat))

	# Raw Survival Chance
	rawSC = SurvivalChance(rawData)
	print('\n\tRaw Survival:')
	for key, val in rawSC.items():
		print('{0:s} :: {1:d} / {2:d} : {3:.2f}%'.format(key, val[0], val[1], val[2] * 100))

	# Class Survival Chance
	classSC = SurvivalChance(rawData, 'Pclass')
	print('\n\tClass Survival:')
	for key, val in classSC.items():
		print('{0:s} :: {1:d} / {2:d} : {3:.2f}%'.format(key, val[0], val[1], val[2] * 100))

	# Gender Survival Chance
	genderSC = SurvivalChance(rawData, 'Sex')
	print('\n\tGender Survival:')
	for key, val in genderSC.items():
		print('{0:s} :: {1:d} / {2:d} : {3:.2f}%'.format(key, val[0], val[1], val[2] * 100))

	# Age Survival Chance
	# Age Groups: No Data, Baby, Children, Young Adult, Middle-Age Adult, Senior Adult
	# Division: -1, 0-2, 3-12, 13-30, 31-59, 60+
	ageSC = SurvivalChance(rawData, 'Age')
	print('\n\tAge Survival:')

	ageGroup = {'No.Data': [0, 0, 0],
				'Baby': [0, 0, 0],
				'Child': [0, 0, 0],
				'Y.Adult': [0, 0, 0],
				'M.Adult': [0, 0, 0],
				'Senior': [0, 0, 0]}
	cutoff = [-1, 2, 12, 30, 59, 999]
	for key, val in ageSC.items():
		try:
			age = float(key)
		except:
			age = -1
		for cut, keyiter in zip(cutoff, ageGroup.keys()):
			if age <= cut:
				ageGroup[keyiter][0] = ageGroup[keyiter][0] + val[0]  # Survivor
				ageGroup[keyiter][1] = ageGroup[keyiter][1] + val[1]  # Total
				ageGroup[keyiter][2] = ageGroup[keyiter][0] / ageGroup[keyiter][1]  # Ratio
				break

	for key, val in ageGroup.items():
		print('{0:s} :: {1:d} / {2:d} : {3:.2f}%'.format(key, val[0], val[1], val[2] * 100))


	plt.bar([i for i in range(6)], [i[2] for i in ageGroup.values()])
	plt.show()

"""
	Crude Prediction Model
"""
def ModelPredict (model, data, decisionBoundary, offset=0):
	classKey = 'PClass_' + data[2-offset]
	genderKey = 'Sex_' + data[4-offset]

	try:
		age = float(data[5-offset])
	except:
		age = -1
	for cut, keyiter in zip(cutoff, ageGroup.keys()):
		if age <= cut:
			ageKey = 'Age_' + keyiter
			break

	# Make decision based on probability
	decision = model[classKey][2] * model[genderKey][2] * model[ageKey][2]

	if decision >= decisionBoundary:
		return 1
	else:
		return 0


""" =========================================================================================== """
""" ======================================= MAIN SCRIPT ======================================= """
""" =========================================================================================== """

import copy
import csv
import random

import math
import matplotlib.pyplot as plt
import numpy as np

import Evaluate

# Read Full Data
filename = './data/train.csv'
rawData = ExtractData(filename)

# Preliminary(rawData)

""" =========================================================================================== """

# Skip Training Loop
SKIP = False

# Loop Sample Size
sampleSize = 25

ageGroup = {'No.Data': [0,0,0],
			'Baby': [0,0,0],
			'Child': [0,0,0],
			'Y.Adult': [0,0,0],
			'M.Adult': [0,0,0],
			'Senior': [0,0,0]}
cutoff = [-1, 2, 12, 30, 59, 999]

accBest = 0
fBest = 0

## Decision Boundary Selection Loop
for dp in np.linspace(0.03, 0.06, 50):
	if SKIP: break
	decisionBoundary = dp

	## Training Loop
	print(f'Decision Boundary: {decisionBoundary:.4f} ||', end=' ')
	aveAccuracy = 0
	aveFscore = 0
	for i in range(sampleSize):
		# Split Training-Testing Data
		(trainData, testData) = SplitData(rawData, 0.8)

		# Train Model
		# Simplified Probabilistic Model
		classSC = SurvivalChance(trainData, 'Pclass')
		genderSC = SurvivalChance(trainData, 'Sex')
		ageSC = SurvivalChance(trainData, 'Age')
		for key,val in ageSC.items():
			try: age = float(key)
			except: age = -1
			for cut, keyiter in zip(cutoff, ageGroup.keys()):
				if age <= cut:
					ageGroup[keyiter][0] = ageGroup[keyiter][0] + val[0] # Survivor
					ageGroup[keyiter][1] = ageGroup[keyiter][1] + val[1] # Total
					ageGroup[keyiter][2] = ageGroup[keyiter][0] / ageGroup[keyiter][1] # Ratio
					break

		model = {}
		for k,v in classSC.items():
			key = 'PClass_' + k
			model[key] = v
		for k,v in genderSC.items():
			key = 'Sex_' + k
			model[key] = v
		for k,v in ageGroup.items():
			key = 'Age_' + k
			model[key] = v

		prediction = []
		goldstandard = []

		# Predict Data
		for row in testData:
			if float(row[1]) == 0: goldstandard.append(0)
			else: goldstandard.append(1)
			p = ModelPredict(model, row, decisionBoundary)
			prediction.append(p)

		# Score Result
		accuracy = Evaluate.accuracy(prediction, goldstandard)
		fscore = Evaluate.fmeasure(prediction, goldstandard, warn=False)
		aveAccuracy += accuracy
		aveFscore += fscore

	aveAccuracy /= sampleSize
	aveFscore /= sampleSize

	print(f'Accuracy: {aveAccuracy:.6f} | F1: {aveFscore:.6f}', end='')

	if aveAccuracy >= accBest:
		accBest = aveAccuracy
		print(' : Acc Best',end='')
	if aveFscore >= fBest:
		fBest = aveFscore
		print(' : F1 Best',end='')
		dpBest = decisionBoundary

	print('')


""" =========================================================================================== """

if SKIP: dpBest = 0.04
decisionBoundary = dpBest

classSC = SurvivalChance(rawData, 'Pclass')
genderSC = SurvivalChance(rawData, 'Sex')
ageSC = SurvivalChance(rawData, 'Age')
for key,val in ageSC.items():
	try: age = float(key)
	except: age = -1
	for cut, keyiter in zip(cutoff, ageGroup.keys()):
		if age <= cut:
			ageGroup[keyiter][0] = ageGroup[keyiter][0] + val[0] # Survivor
			ageGroup[keyiter][1] = ageGroup[keyiter][1] + val[1] # Total
			ageGroup[keyiter][2] = ageGroup[keyiter][0] / ageGroup[keyiter][1] # Ratio
			break

model = {}
for k,v in classSC.items():
	key = 'PClass_' + k
	model[key] = v
for k,v in genderSC.items():
	key = 'Sex_' + k
	model[key] = v
for k,v in ageGroup.items():
	key = 'Age_' + k
	model[key] = v

filename = './data/test.csv'
testData = ExtractData(filename)
testData.pop(0)

passid = []
prediction = []

# Predict Data
for row in testData:
	passid.append(row[0])
	p = ModelPredict(model, row, decisionBoundary, offset=1)
	prediction.append(p)

SaveCSV('./output.csv', passid, prediction)

