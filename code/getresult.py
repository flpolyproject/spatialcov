import os,sys,glob
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from scipy import stats

import seaborn as sns
sns.set_theme(style="whitegrid")

class T_test(object):

	def __init__(self, data1, data2):
		self.data1 = data1
		self.data2 = data2
		self.t_value, self.p_value = stats.ttest_ind(data1, data2, equal_var=True)
		self.get_cl()
	def __repr__(self):
		return f"{self.p_value}"

	def __str__(self):
		return f"T:{self.t_value} P:{self.p_value} mean: {(np.mean(self.data1), np.mean(self.data2))} sd:{(np.std(self.data1), np.std(self.data2))} cl: {self.MoE} dm:{self.diff_mean} interval: {self.interval}"

	def get_cl(self):
		self.diff_mean = abs(np.mean(self.data1) - np.mean(self.data2))
		self.df = len(self.data1) + len(self.data2) - 2
		t_val = stats.t.ppf([0.975], self.df) # this is for 95% cl
		std_avg = np.sqrt(((len(self.data1) - 1)*(np.std(self.data1))**2 + (len(self.data2) - 1)*(np.std(self.data2))**2) / self.df)
		last_comp = np.sqrt(1/len(self.data1) + 1/len(self.data2))
		self.MoE = abs(t_val *std_avg * last_comp) #margin of error this is +- from diff mean to get range of 95% conf interval
		self.interval = [self.diff_mean - self.MoE, self.diff_mean + self.MoE]


if __name__ == "__main__":
	data = pd.read_csv('./savedImages/factor750/coverage.csv')
	#data = pd.read_csv('./coverage.csv')
	#data = data.set_index('sim')

	#fig, axs = plt.subplots(2)

	basecov = data["baseCoverage"]
	algocov = data["ALGOCoverage"]
	greedycov = data["greedyCoverage"]

	data["dffalgo"] = basecov - algocov
	data["dffgreedy"] = basecov - greedycov

	result = T_test(greedycov, algocov)
	result1 = T_test(data["dffalgo"], data["dffgreedy"])

	print(result1)

	#plt.plot(data["sim"], basecov.cumsum(), "o", label="base", markersize=2)
	#plt.plot(data["sim"], algocov.cumsum(), "o", label="ALGO", markersize=2)
	#plt.plot(data["sim"], greedycov.cumsum(), "o", label="greedy", markersize=2)

	#plt.plot(data["sim"], data["dffalgo"].cumsum(), "o", label="ALGO", markersize=2)
	#plt.plot(data["sim"], data["dffgreedy"].cumsum(), "o", label="greedy", markersize=2)

	melt_df = pd.melt(data, id_vars=['sim'], value_vars=['ALGOCoverage', 'baseCoverage', "greedyCoverage"])
	print(melt_df)
	print(data['greedyCoverage'])

	ax = sns.boxplot(x="variable", y="value", data=melt_df, palette="Set3") #boxplot for 1 factor value
	ax= sns.swarmplot(x="variable", y="value", data=melt_df, color=".25")

	#plt.plot(data["sim"], data["dffalgo"], "-o", label="algo") #lineplot fo each sim number
	#plt.plot(data["sim"], data["dffgreedy"], "-o", label="greedy")

	#plt.plot(data["variable_value"], data["dffalgo"], "-o", label="algo")
	#plt.plot(data["variable_value"], data["dffgreedy"], "-o", label="greedy")

	plt.legend()
	plt.show()
