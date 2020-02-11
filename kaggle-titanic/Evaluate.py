
def accuracy (prediction, goldstandard):
	if len(prediction) != len(goldstandard):
		print('Error: Accuracy length mismatch!')
		return -1

	score = 0
	max = len(goldstandard)

	for p, gs in zip(prediction, goldstandard):
		if p == gs:
			score += 1

	return score / max

def fmeasure (prediction, goldstandard, warn=True):
	if len(prediction) != len(goldstandard):
		print('Error: F1 Measure length mismatch!')
		return -1

	TP = 0; FP = 0; TN = 0; FN = 0

	for p, gs in zip(prediction, goldstandard):
		if (p == True) & (p == gs): TP += 1
		if (p == True) & (p != gs): FP += 1
		if (p == False) & (p == gs): TN += 1
		if (p == False) & (p != gs): FN += 1

	try:
		precision = TP / (TP + FP)
		recall = TP / (TP + FN)
		Fscore = 2 * precision * recall / (precision + recall)
	except:
		if warn: print('Warning: F1 Division by zero')
		Fscore = 0

	return Fscore