"""
	Kaggle Titanic Competition -- 2020-02-11
	kaggle/mimocha
	github/mimocha
	Chawit Leosrisook
"""

import Evaluate

"""
	Naive Bayes Classifier Class Definition
	Creates a Naive Bayes Classifier Object
"""
class NaiveBayes:

	def __init__ (self):
		self.prediction = []

		self.model = {}

		# Dictionary of classes & probability
		self.prior = {}

	def Train (self, data):
		# Insert Training Montage
		# for feat in featList:
		return


	def Test (self, data, goldstandard):
		self.testData = data
		self.goldstandard = goldstandard

		# Test with labelled Data
		prediction = self.Predict(data)

		score = Evaluate.accuracy(prediction, goldstandard)

	"""
		Generates prediction based on data
		Returns list of 1s and 0s
	"""
	def Predict (self, data):
		prediction = []
		return prediction



""" =========================================================================================== """
""" ==================================== HELPER FUNCTION ====================================== """
""" =========================================================================================== """

"""
	Survival chance, based on features
	If feature is not provided, shows raw survival chance

	IO Format:
	dict{key : tuple(...)} = SurvivalChance(list(list(...)), 'string')
"""

def SurvivalChance(data, feat=None):
	# Raw survival chance
	if feat == None:
		s = 0  # Survived
		t = 0  # Total
		for row in data[1:-1]:
			s += int(row[1])
			t += 1
		return {'raw': (s, t, s / t)}

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
	sDict = {}  # Survival
	tDict = {}  # Total
	for row in data[1:-1]:
		sDict[row[col]] = sDict.get(row[col], 0) + int(row[1])
		tDict[row[col]] = tDict.get(row[col], 0) + 1

	rDict = {}  # Result
	for s, t in zip(sDict.items(), tDict.values()):
		# featureValue : (rawSurvival, rawTotal, ratioSurvive)
		rDict[s[0]] = (s[1], t, s[1] / t)

	# Sort Dictionary
	rDict = {k: v for k, v in sorted(rDict.items())}
	return rDict