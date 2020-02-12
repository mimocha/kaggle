"""
	Kaggle Titanic Competition -- 2020-02-02
	kaggle/mimocha
	github/mimocha
	Chawit Leosrisook
"""

""" =========================================================================================== """
""" ======================================= MAIN SCRIPT ======================================= """
""" =========================================================================================== """

import DataAnalysis
import DataHandler

filename = './data/train.csv'
rawData = DataHandler.ReadCSV(filename)

# Preliminary Data Analysis
DataAnalysis.Preliminary(rawData)


