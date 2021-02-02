import numpy as np 
import similaritymeasures as sm 
import matplotlib.pyplot as plt 
import os, glob, sys
import pandas as pd
import seaborn as sns

#creating 2D Brownian walks 


def find_fdist(paths, save=False, show=False):
	 

	N = len(paths)
	fd = np.zeros((N,N))
	for i in range(N-1):
		for j in range(i+1,N):
			fd[i,j] = sm.frechet_dist(paths[i], paths[j])
			
			
	#compute diversities 
	diversities = np.zeros(N)
	#alpha = 0.5
	alpha = 5 / np.max(fd[:])
	print("alpha value is ", alpha)
	for i in range(N):
		diversity = 0
		for j in range(N):
			if j>=i: diversity = diversity + np.exp(-alpha*fd[i,j])
			else: diversity = diversity + np.exp(-alpha*fd[j,i])
		diversities[i] = 1/diversity

		
	print("Diversity  Array is ", diversities)



	for i in range(len(paths)):
		plt.plot(paths[i][:,0], paths[i][:,1],label=str(i))
		#ax = sns.heatmap(df)
	plt.legend()
	if show:
		plt.show()
	if save:
		plt.savefig(os.path.join("./", f'{"heatmap"}.eps'), dpi=300)
		plt.savefig(os.path.join("./", f'{"heatmap"}.png'))



if __name__ =="__main__":

	l = 100

	paths = []
	for i in range(5):
		s1 = 0.05*np.array([2,1])+ 0.1*np.random.randn(l,2)
		p1 = np.cumsum(s1,axis=0) 
		paths.append(p1)

	s2 = 0.05*np.array([1,2])+ 0.1*np.random.randn(l,2)
	p2 = np.cumsum(s2,axis=0)
	paths.append(p2) 



	find_fdist(paths)











