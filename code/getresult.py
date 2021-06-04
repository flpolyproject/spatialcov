import os,sys,glob
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl

from scipy import stats

import seaborn as sns
#sns.set_theme(style="whitegrid")


#sns.set(font='serif', rc={'figure.figsize':(20,8.27)})

sns.set_context("paper")

# Make the background white, and specify the
# specific font family
sns.set_style("white", {
	"font.family": "serif",
	"font.serif": ["Times", "Palatino", "serif"]
})

sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})

#mpl.rcParams['axes.color_cycle'] = [ "332288", "88CCEE", "44AA99", "117733", "999933", "DDCC77", "CC6677", "882255", "AA4499"]
#mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=[ "332288", "88CCEE", "44AA99", "117733", "999933", "DDCC77", "CC6677", "882255", "AA4499"])

def roadutil(data):
	

	mean_cov = data.groupby(['variable_value']).mean().reset_index()

	baseutil = mean_cov["baseRU"]
	algoutil = mean_cov["algoRU"]
	greedyutil = mean_cov["greedyRU"]


	mean_cov["dffalgo"] = algoutil - baseutil
	mean_cov["dffgreedy"] = greedyutil - baseutil

	print(mean_cov)


	#plt.plot(mean_cov["variable_value"],mean_cov["dffalgo"], "-", label="ours")
	#plt.plot(mean_cov["variable_value"], mean_cov["dffgreedy"], "-", label="naive")

	plt.plot(mean_cov["variable_value"],mean_cov["algoRU"], "-", label="LDD")
	plt.plot(mean_cov["variable_value"], mean_cov["greedyRU"], "-", label="uniform")
	plt.plot(mean_cov["variable_value"], mean_cov["baseRU"], "-", label="baseline")

	plt.gca().spines['top'].set_visible(False)
	plt.gca().spines['right'].set_visible(False)

	plt.xlabel('Reward values')
	plt.ylabel('Road utilization')

	plt.legend(loc="upper left", fontsize=15)
	plt.savefig(os.path.join(base_save, f'{"resultcomp"}.eps'), dpi=300)
	plt.savefig(os.path.join(base_save, f'{"resultcomp"}.png'))
	plt.show()


	

def spatialcov(data, average=True):
	if average:
		basecov = data["baseCoverage"]
		algocov = data["ALGOCoverage"]
		greedycov = data["greedyCoverage"]

		data["dffalgo"] = basecov - algocov
		data["dffgreedy"] = basecov - greedycov

		data["ALGOCoverage"] = 1/ data["ALGOCoverage"] 
		data["baseCoverage"] = 1/ data["baseCoverage"]
		data["greedyCoverage"] = 1/ data["greedyCoverage"]
		melt_df = pd.melt(data, id_vars=['sim'], value_vars=['ALGOCoverage', 'baseCoverage', "greedyCoverage"])
		print(melt_df)

		#ax = sns.boxplot(x="variable", y="value", data=melt_df, palette="Set3") #boxplot for 1 factor value
		#ax= sns.swarmplot(x="variable", y="value", data=melt_df, color=".25")

		#plt.plot(data["sim"], data["dffalgo"], "-o", label="algo") #lineplot fo each sim number
		#plt.plot(data["sim"], data["dffgreedy"], "-o", label="greedy")

		#plt.plot(data["variable_value"], data["dffalgo"], "-o", label="algo")
		#plt.plot(data["variable_value"], data["dffgreedy"], "-o", label="greedy")

		'''
		plt.plot(data["variable_value"], data["ALGOCoverage"], "-o", label="algo")
		plt.plot(data["variable_value"], data["greedyCoverage"], "-o", label="greedy")
		plt.plot(data["variable_value"], data["baseCoverage"], "-o", label="base")
		'''




		mean_cov = data.groupby(['variable_value']).mean().reset_index()
		print(mean_cov)



		basecov = mean_cov["baseCoverage"]
		algocov = mean_cov["ALGOCoverage"]
		greedycov = mean_cov["greedyCoverage"]

		mean_cov["dffalgo"] = basecov - algocov
		mean_cov["dffgreedy"] = basecov - greedycov

		#plt.plot(mean_cov["variable_value"],mean_cov["dffalgo"], "-", label="ours")
		#plt.plot(mean_cov["variable_value"], mean_cov["dffgreedy"], "-", label="naive")
		plt.plot(mean_cov["variable_value"],mean_cov["ALGOCoverage"], "-", label="LDD", linewidth=3.0)
		plt.plot(mean_cov["variable_value"], mean_cov["greedyCoverage"], "-", label="uniform", linewidth=3.0)
		plt.plot(mean_cov["variable_value"], mean_cov["baseCoverage"], "-", label="baseline", linewidth=3.0)
		plt.gca().spines['top'].set_visible(False)
		plt.gca().spines['right'].set_visible(False)

		plt.xlabel('Participants')
		plt.ylabel('Spatial Coverage')


		

		plt.legend(loc="lower right", fontsize=13)
		plt.savefig(os.path.join(base_save, f'{"resultcomp"}.eps'), dpi=300)
		plt.savefig(os.path.join(base_save, f'{"resultcomp"}.png'))
		plt.show()






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
	base_save = './savedImages/inc_player'

	data = pd.read_csv(os.path.join(base_save,'coverage.csv'))
	

	#result = T_test(greedycov, algocov)
	#result1 = T_test(data["dffalgo"], data["dffgreedy"])

	spatialcov(data)
	#roadutil(data)

