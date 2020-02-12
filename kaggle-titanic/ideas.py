"""
	Kaggle Titanic Competition -- 2020-02-12
	kaggle/mimocha
	github/mimocha
	Chawit Leosrisook
"""

""" File for containing random code scraps """

import DataHandler
import Evaluate

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

# Loop Sample Size
sampleSize = 10

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
for dp in np.linspace(0.03, 0.06, 100):
	decisionBoundary = dp

	## Training Loop
	print(f'Decision Boundary: {decisionBoundary:.4f} ||', end=' ')
	aveAccuracy = 0
	aveFscore = 0
	for i in range(sampleSize):
		# Split Training-Testing Data
		(trainData, testData) = DataHandler.SplitData(rawData, 0.8)

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

DataHandler.SaveCSV('./output.csv', passid, prediction)
