"""
	Kaggle Titanic Competition -- 2020-02-12
	kaggle/mimocha
	github/mimocha
	Chawit Leosrisook
"""

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

	# Embarkation Survival Chance
	embarkSC = SurvivalChance(rawData, 'Embarked')
	print('\n\tEmbarkation Survival:')
	for key, val in embarkSC.items():
		print('{0:s} :: {1:d} / {2:d} : {3:.2f}%'.format(key, val[0], val[1], val[2] * 100))

	# plt.bar([i for i in range(6)], [i[2] for i in ageGroup.values()])
	# plt.show()