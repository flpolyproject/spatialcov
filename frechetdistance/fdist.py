import numpy as np 
import similaritymeasures as sm 
import matplotlib.pyplot as plt 
import os, glob, sys
import pandas as pd
import seaborn as sns
from math import exp
import logging

#creating 2D Brownian walks


class RouteObj(object):
    def __init__(self, id, pathlist, div, start=None, end=None,reward=100, cost = 1):
        self.id = id
        self.pathlist = pathlist
        self.div = div
        self.reward = reward
        self.cost = cost/div# * div #cost time the diverity value will yield Si
        self.start = start
        self.end=end
        self.sp = 0
        self.dev_value_before = 0
        self.dev_value_after = 0
        self.devpathlist = None 
        self.nonedivlist = None
        self.utility = [] #base, ours, naive
        print(f"VEHICLE {self.id} START {start} {end}")

class TempPlayer(object):
	def __init__(self, id, cost, reward, div=1):
		self.id = id
		self.cost=cost
		self.reward = reward
		self.div = div
		self.sp=0




def NormalizeData(data,start=0, end=1):
	return (end-start)*((data - np.min(data)) / (np.max(data) - np.min(data))) + start

def compute_utility(sp, totalsp, reward, sigma, cost):
	print()

	totalsp_value = sum([x/sigma for x in totalsp]) 

	print(f"{sp} R:{reward}  sig:{sigma} cost:{cost} {totalsp_value} first {(sp/(totalsp_value*sigma)) * reward} second {(cost *(sp/sigma))}")
	print(((sp/(totalsp_value*sigma)) * reward) - (cost *(sp/sigma)))
	#exit()

	return ((sp/(totalsp_value*sigma)) * reward) - (cost *(sp/sigma))


def neqfunction(route_dict, factor_change = None, cust_cost = None, cust_reward=None):
	#should be a sensing plan for each individual player
	#given each players cost, and their diversity value, we can calcualte their utility 
	#return a dict filled with 
	#i can compute utility every step
	if cust_cost:
		cust_cost = itertools.cycle(cust_cost)
		for value in route_dict:
			prev_cost = value.cost
			value.cost = next(cust_cost) * value.cost
			print(f"changing cost {value.id} from {prev_cost} to {value.cost}")

	if cust_reward:
		cust_reward = itertools.cycle(cust_reward)
		for value in route_dict:
			value.reward = next(cust_reward)

	for value in route_dict:
		print(f"{value.id} cost:{value.cost} reward:{value.reward}")


	number_players = len(route_dict)


	player_sp_dict = {}
	player_collection = []
	#player_utility_dict = {}

	total_sensing_plan = 0

	H = [0, 1]
	i = 2

	#constant_sp = 0.02 #this value need to be constantly adjusted
	constant_sp = 0

	try:
		sorted_player_list = sorted(route_dict, key=lambda x: x.cost) #this would only cause error
		player_value_dict = {intind:value for intind, value in enumerate(sorted_player_list)}
		player_sp_dict = {intind:constant_sp for intind, value in enumerate(sorted_player_list)}

		#print("im here again ", len(sorted_player_list))
		#sorted_player_list = route_dict
		
		for index_value, value in enumerate(sorted_player_list):
			 #doesnt print all vehicles

			
			
			if index_value >= i: #making sure we start at the i value, and not 0
				

				compare_function = (sum([sorted_player_list[index].cost for index in H]) + value.cost)/len(H)
				if value.cost < compare_function:
					H.append(index_value)
				else:
					break
	except KeyError:
		print("Error computing sensing plan")
		#temprandomcost location failed due to no reward
		sorted_player_list = player_consider_list # if we get key error due to trying to sort list
		#assert not return_value, f"no reward at location {location} shouldve skipped" 
	print(f"H INDEX is : {H}")

	for index_value, player in enumerate(sorted_player_list):

		temp_sensing_plan = constant_sp



		if number_players == 1:
			temp_sensing_plan = 1
		else:
			if index_value in H:
				
				denom = sum([sorted_player_list[index].cost for index in H])
				temp_sensing_plan = (((len(H) -1) * player.reward)/denom)*(1 - (((len(H) -1) * player.cost)/denom))
			else:
				temp_sensing_plan = constant_sp

		#if (player.div != 0) and (temp_sensing_plan > 0): #if the sensing plan or div is 0 it will lead to infinity value which thus resulting in inf softmax sum
		player_sp_dict[index_value] = (temp_sensing_plan / player.div) #because dividing makes more sense


		print(f"{player.id} index: {index_value} cost: {player.cost} reward: {player.reward} 1/sigma: {player.div} si:{temp_sensing_plan} ti:{player_sp_dict[index_value]}")
		#print(f"index {index_value} temp_sensing_plan : {temp_sensing_plan} player_sp : {player_sp_dict[player.id]}")

		#player_value_dict[player.id].sp = 1/(temp_sensing_plan * player.div)
		

	#softmax function make sure sp stays between 0 and 1
	counter = 0
	
	total_sp_value = sum(player_sp_dict.values())

	after_norm_list_y = []
	for key, value in player_value_dict.items():
		if key in player_sp_dict:
			value.sp = player_sp_dict[key]
			print(f"key {key} vehicle {value.id} sensing plan is {value.sp}")
			after_norm_list_y.append(factor_change*value.sp)
	mean_sp = np.mean(after_norm_list_y)



	temp_dev_list = {}

	#print(f"NQ {after_norm_list_y} mean_sp {mean_sp}")

	for key, value in player_value_dict.items():
		#print(f"{counter} {after_norm_list_y}")

		

		value.dev_value_before = after_norm_list_y[counter]

		value.utility.append(compute_utility(value.dev_value_before, after_norm_list_y, value.reward, 1/value.div, value.cost))

		temp_dev_list[value.id] = value.dev_value_before

		value.utility.append(compute_utility(mean_sp, after_norm_list_y, value.reward, 1/value.div, value.cost))

		value.dev_value_after = mean_sp
		counter += 1
	print(f"dev list: {temp_dev_list} average: {mean_sp}")


	return player_value_dict.values() 


		
		

	


def find_fdist(paths, save=False, show=False, player_list=None): #route sample contains all the sumo routes the vehicle pass

	#print(paths)
	N = len(paths)
	print(f"Generating fdist path for {N} routes....")
	fd = np.zeros((N,N))
	for i in range(N-1):
		for j in range(i+1,N):
			fd[i,j] = sm.frechet_dist(paths[i], paths[j])

	#print("fdmatrix is ",fd)
			
			
	#compute diversities 
	diversities = np.zeros(N)
	#alpha = 0.5
	alpha = 5 / np.max(fd[:])
	print("alpha value is ", alpha)
	for i in range(N):
		#print(f'shape {i} is {paths[i].shape}')
		diversity = 0
		for j in range(N):
			if j>=i: diversity = diversity + np.exp(-alpha*fd[i,j])
			else: diversity = diversity + np.exp(-alpha*fd[j,i])
		diversities[i] = 1/diversity

		
	print("Diversity  Array is ", diversities)
	

	#diversity = 1/singma


	#for i in range(len(paths)):
	#	plt.plot(paths[i][:,0], paths[i][:,1],label=str(i))
		#ax = sns.heatmap(df)
	plt.legend()
	if show:
		plt.show()
	if save:
		plt.savefig(os.path.join("./", f'{"heatmap"}.eps'), dpi=300)
		plt.savefig(os.path.join("./", f'{"heatmap"}.png'))


	allrouteobj = []

	for i, path in enumerate(paths):
		#allrouteobj.append(RouteObj(i, path, diversities[i], route_sample[i], node_hit[i]))
		if player_list:
			allrouteobj.append(RouteObj(i, path, diversities[i], player_list[i].start, player_list[i].destination))
		else:
			allrouteobj.append(RouteObj(i, path, diversities[i], path[0], path[-1]))


	return allrouteobj




if __name__ =="__main__":
	route_dict = []
	for i in range(10):
		route_dict.append(TempPlayer(i, 1, 3, 1))
	#res = neqfunction(route_dict, factor_change=1)

	x = []
	y = []

	id_value = 0
	for x_val in range(0, 1000):
		route_dict[0].cost =  x_val/100
		res = neqfunction(route_dict, factor_change=1)
		x.append(route_dict[0].cost)
		final = [x.sp for x in res if x.id==0]
		y.append(final[0])

	print(f"{x}\n{y}")	
	plt.plot(x,y,"-o")
	plt.show()
	



	'''

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

	'''










