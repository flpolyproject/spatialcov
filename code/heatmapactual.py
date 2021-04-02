import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde as kde
from matplotlib.colors import Normalize
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os, sys, glob


import seaborn as sns; #sns.set_theme()
from traci_env import EnvironmentListener, BaseEnv
import traci
import traci.constants as tc
from settings import Settings, GraphSetting
from postprocess import *
import cv2
from math import exp, pi
from scipy.spatial import distance
from fdist import *
from postprocess import *
import operator
import itertools
from copy import deepcopy
import heapq as hq
from collections import defaultdict
import copy
import logging

import traceback
import networkx




dt = datetime.datetime.utcnow().timestamp()
dir_name = os.path.join('./savedImages', str(dt))
os.mkdir(dir_name)
os.mkdir(os.path.join(dir_name, "base_routes"))

print("making dir ", dir_name)
logging.basicConfig(filename=os.path.join(dir_name, 'output.log'), filemode='w', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)

fig, axs = plt.subplots(3)
factor = 2000


traci.start(["sumo", "-c", Settings.sumo_config])
env = BaseEnv(sim_number=1, _seed=3, init=False)
junctions = np.array([[obj.x, obj.y] for obj in env.sim_env.map_data.junctions.values()])

'''
x = np.max(junctions.T[0])
y = np.max(junctions.T[1])
map_matrix = np.zeros((int(x) + 1, int(y)+ 1))
junctions_int = junctions.T.astype(np.int)
map_matrix[junctions_int[0], junctions_int[1]] += 1
with open("./savedImages/map_matrix/map.npy", 'wb') as f:
	np.save(f, map_matrix)
'''

def angle_rowwise_v2(A, B):
    p1 = np.einsum('ij,ij->i',A,B)
    p2 = np.einsum('ij,ij->i',A,A)
    p3 = np.einsum('ij,ij->i',B,B)
    p4 = p1 / np.sqrt(p2*p3)
    return np.arccos(np.clip(p4,-1.0,1.0))

def find_junction_heatheap(heatheap, potential_dict, spatial_matrix, global_heatheap):
	#heatheap is the local 
	top_dist = hq.nlargest(1, heatheap)
	if top_dist:
		max_spatial_point = top_dist[0][1]
	else:
		max_spatial_point = global_heatheap


	key_list = list(potential_dict.keys())
	potential_xy = np.array([[env.sim_env.map_data.junctions[x].x, env.sim_env.map_data.junctions[x].y] for x in key_list])
	top_dist_matrix = np.tile(np.array(list(max_spatial_point)), (len(key_list),1))
	angles = (2*pi) - angle_rowwise_v2(potential_xy, top_dist_matrix)
	distances = closest_node(max_spatial_point, potential_xy, entire_array=True).flatten()

	point_array = np.column_stack((distances, angles))
	zeros = np.tile(np.array([0,0]), (distances.shape[0],1))
	length_vector = np.linalg.norm(point_array- zeros, axis=1)

	#angle_min = np.where(length_vector==np.min(length_vector[np.nonzero(angles)]))[0][0]
	angle_min = np.argmin(length_vector)
	junction_id = key_list[angle_min]


	return junction_id


def changeRouteTest(route_dict, simnumber, vis = False, to_csv=False,heatmap_dict=None, spatial_matrix=None, global_heatheap=None,save_dir=None):
	#factor multiply by the sp value result in the final distance deviation its allowed to make
	#takes in the existing routes
	#i need the time it takes to the specific junction
	#the time it takes from the junction to destination

	#print(route_dict[0].node_hit)
	#spvalue too small, only works when taking the inverse
	#choose the route with shortest l2 distance
	#how to choose starting point for deviation(at what point of the route?)
	#how to choose the next road to deviation to?
	#how to use the sp values since they are very small
	#do we want the vehicle to eventually come back to the orginal route or potentially take the completely different route
	#would the spatial coverage also can be affected by the road network of the map we are working on? some areas are more densly connected and some areas are not
	#this can affect where of the route can be best used for deviation, the location weight for deviation should be based on the road network and other vehicles

	#using conv sliding box, get the spatial coverage value in each box

	#find the midpoint from a to b
	logging.debug(f"Starting div route gneration... ")

	dev_junction = {}

	#iterate each route and get the potential junctions

	if save_dir:
		x = np.max(junctions.T[0])
		y = np.max(junctions.T[1])

		base_matrix = np.zeros((int(x), int(y)))
		starting_matrix = np.zeros((int(x), int(y)))
		ending_matrix = np.zeros((int(x), int(y)))
		derived_matrix = np.zeros((int(x), int(y)))


	potential_junct_dict = {} #id:{junction:distance, junct:dist}

	for routeobj in route_dict:
		#generate the nx dict from the start set the factor*sp*constant/2

		from_node = env.sim_env.map_data.edges[routeobj.start]._from
		to_node = env.sim_env.map_data.edges[routeobj.end]._to

		base_distance = env.sim_env.map_data.sumonet.find_shortest_length(from_node, to_node) #get base distance

		#if not routeobj.sp: #if the sp is none
		#	routeobj.sp = 0.0

		#print(f"wtfff {routeobj.sp} {not routeobj.sp} {np.isnan(routeobj.sp)} {type(routeobj.sp)}")

		#dev_value = base_distance + (factor*routeobj.sp) #calculate the predicted dev value
		dev_value = base_distance + routeobj.dev_value_before

		#print(f"{routeobj.id} base {base_distance} diff {routeobj.dev_value_before} target {dev_value}")

		if (dev_value == base_distance) or (np.isnan(routeobj.sp)): #no adjustment need to be made
			#dev_junction[routeobj.id] = None
			continue


		start_node_dict = env.sim_env.map_data.sumonet.sub_new(from_node, dev_value*1) #make two circles find the locations where its half of the deviation value
		end_node_dict = env.sim_env.map_data.sumonet.sub_new(to_node, dev_value*1)

		#print(max(start_node_dict, key=start_node_dict.get))
		#print(start_node_dict[max(start_node_dict, key=start_node_dict.get)])

		#print(start_node_dict)
		rdpSet = set(start_node_dict)
		namesSet = set(end_node_dict)

		potential_dict = {}

		for name in rdpSet.intersection(namesSet):
			#print(name, end_node_dict[name])
			if (end_node_dict[name] + start_node_dict[name]) <= dev_value:
				potential_dict[name] = dev_value - (end_node_dict[name] + start_node_dict[name])


		#we want the weight each potential dict. the prob is the diff_value*spatial_value of nearest*
		"""
		we want to achieve max deviation to the deviation value smaller value with less diff
		the route

		"""
		prob = np.array(list(potential_dict.values()))
		prob = 1/prob
		sum_value = np.sum(prob)
		prob /= sum_value
		
		key = list(potential_dict.keys())

		#choice_junction = np.random.choice(key, 1, replace=True, p=prob)[0]
		if not potential_dict:
			print(f"cant find junction for {routeobj.id} {routeobj.sp} {base_distance} {dev_value}")

		choice_junction = min(potential_dict, key=potential_dict.get)
		#if heatmap_dict:
		#	choice_junction = find_junction_heatheap(heatmap_dict[routeobj.id], potential_dict, spatial_matrix, global_heatheap)

		

		if vis:
			axs[0].scatter(env.sim_env.map_data.junctions[choice_junction].x, env.sim_env.map_data.junctions[choice_junction].y, label=routeobj.id, s=100)

		print(f"Route {routeobj.id} found {len(potential_dict)} potential routes...")
		if len(potential_dict) <= 1:
			#only the dest junct is added
			continue
		print(f"Route {routeobj.id} from {from_node} to {to_node} toedge:{routeobj.end} dist is {base_distance} expected dev of {dev_value} sp:{routeobj.sp} to {choice_junction} actual dev {dev_value - potential_dict[choice_junction]}")
		#print("distance to max cost node.....")
		#print(env.sim_env.map_data.sumonet.find_shortest_length(from_node, max(start_node_dict, key=start_node_dict.get)))
		potential_junct_dict[routeobj.id] = potential_dict
		dev_junction[routeobj.id] = choice_junction



	for routeobj in route_dict:
		if not routeobj.id in dev_junction:
			routeobj.devpathlist = routeobj.pathlist
			print(f"No route change for {routeobj.id}...")
			#print("here in not ", save_dir)
			#exit()
			if save_dir:
				#saving the matrixes in the folder
				int_path = routeobj.pathlist.T.astype(np.int)
				int_devpath = routeobj.devpathlist.T.astype(np.int)
				base_matrix[int_path[0], int_path[1]] += 1
				derived_matrix[int_devpath[0], int_devpath[1]] += 1

				from_node = env.sim_env.map_data.edges[routeobj.start]._from
				to_node = env.sim_env.map_data.edges[routeobj.end]._to
				x_start_node, y_start_node = env.sim_env.map_data.junctions[from_node].x, env.sim_env.map_data.junctions[from_node].y
				x_end_node, y_end_node = env.sim_env.map_data.junctions[to_node].x, env.sim_env.map_data.junctions[to_node].y

				#print(f"{from_node}-{to_node} {x_start_node} {y_start_node}: {x_end_node} {y_end_node}")

				starting_matrix[int(x_start_node), int(y_start_node)] += 1
				ending_matrix[int(x_end_node),int(y_end_node)] +=1
			continue

		from_node = env.sim_env.map_data.edges[routeobj.start]._from
		to_node = env.sim_env.map_data.edges[routeobj.end]._to

		new_path_start = []
		new_path_end = []

		while True: #need to make sure the junction chosen can be arrived and go to dest from there

			try:
				route_to_target = env.sim_env.map_data.sumonet.find_shortest_path(from_node, dev_junction[routeobj.id])
				print(f"Testing.... {routeobj.id} {dev_junction} {to_node} endedge: {routeobj.end} supposed junction {env.sim_env.map_data.edges[routeobj.end]._to}")
				route_from_target = env.sim_env.map_data.sumonet.find_shortest_path(dev_junction[routeobj.id], to_node)[1:]
				break
			except networkx.exception.NetworkXNoPath:
				del potential_junct_dict[routeobj.id][dev_junction[routeobj.id]]
				logging.debug(f"no path from {dev_junction[routeobj.id]} to {to_node}, removing {dev_junction[routeobj.id]}")
				dev_junction[routeobj.id] = min(potential_junct_dict[routeobj.id], key=potential_junct_dict[routeobj.id].get)




		assert route_from_target, f"no routes to target for id {routeobj.id} {route_from_target}"

		for i in itertools.zip_longest(route_to_target, route_from_target):
			if i[0]:
				new_path_start.append([env.sim_env.map_data.junctions[i[0]].x, env.sim_env.map_data.junctions[i[0]].y])
			if i[1]:
				new_path_end.append([env.sim_env.map_data.junctions[i[1]].x, env.sim_env.map_data.junctions[i[1]].y])

		routeobj.devpathlist = np.array(new_path_start + new_path_end)

		if save_dir:
			int_path = routeobj.pathlist.T.astype(np.int)
			int_devpath = routeobj.devpathlist.T.astype(np.int)
			base_matrix[int_path[0], int_path[1]] += 1
			derived_matrix[int_devpath[0], int_devpath[1]] += 1

			from_node = env.sim_env.map_data.edges[routeobj.start]._from
			to_node = env.sim_env.map_data.edges[routeobj.end]._to
			x_start_node, y_start_node = env.sim_env.map_data.junctions[from_node].x, env.sim_env.map_data.junctions[from_node].y
			x_end_node, y_end_node = env.sim_env.map_data.junctions[to_node].x, env.sim_env.map_data.junctions[to_node].y

			starting_matrix[int(x_start_node), int(y_start_node)] += 1
			ending_matrix[int(x_end_node),int(y_end_node)] +=1
	if save_dir:
		base_none_zeros = np.where(base_matrix!=0)

		base_save = np.array([base_none_zeros[0], base_none_zeros[1], base_matrix[base_none_zeros[0], base_none_zeros[1]]])

		derived_none_zeros = np.where(derived_matrix!=0)

		derived_save = np.array([derived_none_zeros[0], derived_none_zeros[1], derived_matrix[derived_none_zeros[0], derived_none_zeros[1]]])


		starting_none_zeros = np.where(starting_matrix!=0)

		starting_save = np.array([starting_none_zeros[0], starting_none_zeros[1], starting_matrix[starting_none_zeros[0], starting_none_zeros[1]]])


		ending_none_zeros = np.where(ending_matrix!=0)

		ending_save = np.array([ending_none_zeros[0], ending_none_zeros[1], ending_matrix[ending_none_zeros[0], ending_none_zeros[1]]])

		with open(os.path.join(save_dir, f"base_routes/{simnumber}.npy"), 'wb') as f:
			np.save(f, base_save)
			np.save(f, derived_save)
			np.save(f, starting_save)
			np.save(f, ending_save)





	if to_csv:
		original_path = []

		for routeobj in route_dict:
			for i, x in enumerate(routeobj.pathlist):
				original_path.append([i, routeobj.id, x[0], x[1], 0]) #0 is original
			for j, y in enumerate(routeobj.devpathlist):
				original_path.append([j, routeobj.id, y[0], y[1], 1])


		if save_dir:
			pd.DataFrame(original_path,columns=["timestamp","veh_id","x", "y","orginal_route"]).to_csv(os.path.join(save_dir, f"vehicledata{simnumber}.csv"))
		else:
			pd.DataFrame(original_path,columns=["timestamp","veh_id","x", "y","orginal_route"]).to_csv("vehicledata.csv")




	if vis:

		


		axs[0].scatter(junctions.T[0], junctions.T[1], alpha=0.2, s=0.8)
		axs[1].scatter(junctions.T[0], junctions.T[1], alpha=0.2, s=0.8)



		for i,obj in enumerate(route_dict):
			axs[0].plot(obj.pathlist.T[0], obj.pathlist.T[1], label=obj.id)
			axs[1].plot(obj.devpathlist.T[0], obj.devpathlist.T[1], label=f"{obj.id}D")





		axs[0].legend(ncol=2)
		axs[1].legend(ncol=2)
	



	logging.debug(f"Ending div route gneration... ")

	return route_dict




def NormalizeData(data):
	return (data - np.min(data)) / (np.max(data) - np.min(data))

def closest_node(node, nodes, slice_indexs=None, entire_array=False):
	if not slice_indexs:
		slice_indexs = [0] * len(nodes)

	dist_array = distance.cdist([node], nodes)
	if entire_array:
		return dist_array
	closest_index = dist_array.argmin()
	#return the point, the 
	return nodes[closest_index], dist_array[0][closest_index], np.argmax(slice_indexs>closest_index), closest_index

def diverse_calculation(route_dict, sampling=10,vis=False, div_routes="pathlist"):
	#find the distance
	#a = random.randint(1000, size=(50000, 2))
	#some_pt = (1, 2)
	#formulate a list of points combining all the points in the routes
	

	x = np.max(junctions.T[0])
	y = np.max(junctions.T[1])

	logging.debug(f"computing spatial coverage for {div_routes}... ")

	sampling_x = int(x/sampling)
	sampling_y = int(y/sampling)

	dist_value = []

	total_spatial_cov = 0
	new_routes = []
	counter_value = 0
	slice_indexs = [0] #indexes for the nodes list

	

	heatmap_dict = defaultdict(list) #dictionary map the id to 

	for routeobj in route_dict:
		#for route in routeobj.pathlist:
		for route in getattr(routeobj, div_routes):
			new_routes.append(route)
			counter_value += 1
		slice_indexs.append(counter_value)


	new_routes_np = np.array(new_routes)


	spatial_cov_matrix = [] #store spatial cov value distance to cloest route in here
	max_point = None
	max_dist = 0

	for i, row in enumerate(range(int(x))):
		row_list = []
		
		for j, col in enumerate(range(int(y))):
			if ((i%sampling_x ==0) and (j%sampling_y==0)):
				point, dist, path_id, index = closest_node((row,col), new_routes, slice_indexs)
				if (i==0) and (j==0):
					hq.heappush(heatmap_dict[path_id], (dist, (1, 1)))
				else:
					hq.heappush(heatmap_dict[path_id], (dist, (i, j)))
				#print(f"{row} {col} distance to shortest is {dist} at point {point}")
				total_spatial_cov += dist
				if dist > max_dist:
					if (i==0) and (j==0):
						pass
					else:
						max_dist = dist
						max_point = (i, j)

				if vis:
					dist_value.append(dist)

				
				

			
			row_list.append(dist)

		spatial_cov_matrix.append(row_list)

	spatial_cov_matrix_df = np.array(spatial_cov_matrix) #this contains the matrix showing the distance result
	#fartest_point_to_route = hq.nlargest(1, heatmap_dict[routeobj.id])

	#print(f"Matrix populate complete.. {routeobj.id} farthest point is {fartest_point_to_route}")




	#exit()


	if vis:
		constant = 0.5

		dist_value = NormalizeData(np.array(dist_value))
		softmax_total = sum([exp(constant * x_i) for x_i in dist_value])
		dist_list = [exp(constant * x_i)/softmax_total for x_i in dist_value]


		scatter_size = [x*10000 for x in dist_value]

	
		colours = makeColours(np.array(dist_list))
		counter = 0
		for i, row in enumerate(range(int(x))):
			for j, col in enumerate(range(int(y))):
				if ((i%sampling_x ==0) and (j%sampling_y==0)):
					plt.scatter(i, j,alpha=0.4, s=scatter_size[counter], color=colours[counter])
					counter+=1

		for i,obj in enumerate(route_dict):
			plt.plot(getattr(obj, div_routes).T[0], getattr(obj, div_routes).T[1], label=obj.id)
		m = cm.ScalarMappable(cmap=cm.jet)
		m.set_array([])
		plt.colorbar(m)
		plt.legend(ncol=2)
		plt.show()

	if max_point[0] == 0 and max_point[1] == 0:
		print(f"Route {routeobj.id} maxpoint is {max_point}, failed ")
		exit()

	logging.debug(f"Finished computing spatial coverage for {div_routes}... ")

	return total_spatial_cov, heatmap_dict, spatial_cov_matrix_df, max_point
	#return int, dictionary {id:heap}, matrix 

def makeColours( vals ):

	print(vals)
	print(len(vals))

	colours = np.zeros( (len(vals),3) )
	norm = Normalize( vmin=vals.min(), vmax=vals.max() )
	#exit()

	#Can put any colormap you like here.
	#coloursb = [cm.ScalarMappable( norm=norm, cmap='jet') for val in vals]
	colours = [cm.ScalarMappable( norm=norm, cmap='jet').to_rgba( val ) for val in vals]
	#divider = make_axes_locatable()
	#cax = divider.append_axes('right', size='5%', pad=0.05)

	#plt.colorbar(vals, orientation='vertical')
	#exit()

	return colours


def genheatmap(route_dict):
	#the input of the function takes in the route dict {}
	#print(route_dict)
	print(f'len of route dict {len(route_dict)}')
	

	#perform softmax maxmize the difference between them
	softmax_total = sum([exp(x.div) for x in route_dict])
	div_list = [exp(x.div)/softmax_total for x in route_dict]
	colors = makeColours(np.array(div_list))



	plt.scatter(junctions.T[0], junctions.T[1],color=(0,0,0), alpha=0.3, s=0.8)

	for i,obj in enumerate(route_dict):
		plt.plot(obj.pathlist.T[0], obj.pathlist.T[1],'o', label=obj.id, color=colors[i])



	m = cm.ScalarMappable(cmap=cm.jet)
	m.set_array([])
	plt.colorbar(m)
	#plt.legend()
	plt.savefig(os.path.join("./", f'{"heatmap_mod"}.eps'), dpi=300)
	plt.savefig(os.path.join("./", f'{"heatmap_mod"}.png'))
	plt.show()




def find_df():

	folder = os.path.join(Settings.sim_save_path_graph, "spatialtest")

	files = glob.glob(os.path.join(folder, r'*.sim'))
	obj = MultiCapture('test').pickle_load(files[0], directory=False)
	for i, sim_number in enumerate(obj.simulation_list):
		algo = sim_number.setting.current_running_algo
		algo = algo.lower()
		if algo == "atne":
			algo = "ours"
		route_dict = sim_number.compute_spatial_coverage(heatmap=True)
		return route_dict



def find_non_div_route(route_dict, vis=False):
	#implement the greedy approach find the total deviation made by vehicles and randomly find
	#total_deviation = sum([factor*x.sp for x in route_dict])
	#change_dev = total_deviation/len(route_dict)
	logging.debug(f"Starting greedy route gneration... ")

	dev_junction = {}
	potential_junct_dict={}

	for routeobj in route_dict:
		#generate the nx dict from the start set the factor*sp*constant/2



		from_node = env.sim_env.map_data.edges[routeobj.start]._from
		to_node = env.sim_env.map_data.edges[routeobj.end]._to

		base_distance = env.sim_env.map_data.sumonet.find_shortest_length(from_node, to_node)

		#dev_value = base_distance + change_dev
		dev_value = base_distance + routeobj.dev_value_after

		start_node_dict = env.sim_env.map_data.sumonet.sub_new(from_node, dev_value*1) #make two circles find the locations where its half of the deviation value
		end_node_dict = env.sim_env.map_data.sumonet.sub_new(to_node, dev_value*1)

		rdpSet = set(start_node_dict)
		namesSet = set(end_node_dict)

		potential_dict = {}

		for name in rdpSet.intersection(namesSet):
			#print(name, end_node_dict[name])
			if (end_node_dict[name] + start_node_dict[name]) <= dev_value:
				potential_dict[name] = dev_value - (end_node_dict[name] + start_node_dict[name])


		#we want the weight each potential dict. the prob is the diff_value*spatial_value of nearest*
		"""
		we want to achieve max deviation to the deviation value smaller value with less diff
		the route

		"""
		prob = np.array(list(potential_dict.values()))
		prob = 1/prob
		sum_value = np.sum(prob)
		prob /= sum_value
		
		key = list(potential_dict.keys())

		#choice_junction = np.random.choice(key, 1, replace=True, p=prob)[0]
		choice_junction = min(potential_dict, key=potential_dict.get)
		if len(potential_dict) == 1:
			continue

		potential_junct_dict[routeobj.id] = potential_dict
		dev_junction[routeobj.id] = choice_junction


	for routeobj in route_dict:
		if not routeobj.id in dev_junction:
			routeobj.devpathlist = routeobj.pathlist
			continue

		from_node = env.sim_env.map_data.edges[routeobj.start]._from
		to_node = env.sim_env.map_data.edges[routeobj.end]._to

		new_path_start = []
		new_path_end = []

		while True: #need to make sure the junction chosen 

			try:
				route_to_target = env.sim_env.map_data.sumonet.find_shortest_path(from_node, dev_junction[routeobj.id])
				print(f"Greedy Testing.... {routeobj.id} {dev_junction} {to_node} endedge: {routeobj.end} supposed junction {env.sim_env.map_data.edges[routeobj.end]._to} FROM: {dev_junction[routeobj.id]} start: {from_node}")
				route_from_target = env.sim_env.map_data.sumonet.find_shortest_path(dev_junction[routeobj.id], to_node)[1:]
				break
			except networkx.exception.NetworkXNoPath:
				del potential_junct_dict[routeobj.id][dev_junction[routeobj.id]]
				logging.debug(f"no path from {dev_junction[routeobj.id]} to {to_node}, removing {dev_junction[routeobj.id]}")
				dev_junction[routeobj.id] = min(potential_junct_dict[routeobj.id], key=potential_junct_dict[routeobj.id].get)

		

		for i in itertools.zip_longest(route_to_target, route_from_target):
			if i[0]:
				new_path_start.append([env.sim_env.map_data.junctions[i[0]].x, env.sim_env.map_data.junctions[i[0]].y])
			if i[1]:
				new_path_end.append([env.sim_env.map_data.junctions[i[1]].x, env.sim_env.map_data.junctions[i[1]].y])

		routeobj.devpathlist = np.array(new_path_start + new_path_end)


	if vis:

		


		axs[2].scatter(junctions.T[0], junctions.T[1], alpha=0.2, s=0.8)


		for i,obj in enumerate(route_dict):
			axs[2].plot(obj.devpathlist.T[0], obj.devpathlist.T[1], label=f"{obj.id}D")





		axs[2].legend(ncol=2)

	logging.debug(f"Ending greedy route gneration... ")

	return route_dict


class TempPlayer(object):
	def __init__(self, id, start, end):
		self.id = id
		self.start=start
		self.destination=end	

def get_route_dict(initial_routes, factor_change=None):
	points_total = []
	player_obj = {}
	for i, edges in initial_routes.items():

		#edges = route.edges
		points_player = []
		for edge in edges:
			junction_from = env.sim_env.map_data.edges[edge]._from
			points_player.append([env.sim_env.map_data.junctions[junction_from].x, env.sim_env.map_data.junctions[junction_from].y])
		junction_to = env.sim_env.map_data.edges[edge]._to
		points_player.append([env.sim_env.map_data.junctions[junction_to].x, env.sim_env.map_data.junctions[junction_to].y])
		points_total.append(np.array(points_player))

		#print(f"{i} {edges}")
		player_obj[i] = TempPlayer(i, edges[0], edges[-1])

	route_dict = find_fdist(points_total, player_list=player_obj)


	return neqfunction(route_dict, factor_change=factor_change)


def main():
	pd_list = []
	count=5
	for i in range(2):
		try:
			initial_routes = env.initial_route_random(count, return_dicts=True)
			route_dict = get_route_dict(initial_routes)

			coverage_value, heatmap_dict_heap, spatial_matrix, global_heatheap = diverse_calculation(route_dict, vis=False)


			target_junction_dict = changeRouteTest(route_dict, i, vis=True, to_csv=True,heatmap_dict=heatmap_dict_heap, spatial_matrix=spatial_matrix, global_heatheap=global_heatheap, save_dir="./savedImages")
			coverage_value_after, heatmap_dict_heap_after, spatial_matrix, global_heatheap = diverse_calculation(target_junction_dict, div_routes="devpathlist")

			new_greedy_dict = find_non_div_route(target_junction_dict, vis=True)
			coverage_value_after_after, heatmap_dict_heap_after_after, spatial_matrix_after, global_heatheap_after = diverse_calculation(new_greedy_dict, div_routes="devpathlist")

			pd_list.append([i,count,coverage_value,coverage_value_after,coverage_value_after_after])

			plt.savefig(os.path.join('./savedImages', f"{i}.png"))
			axs[0].cla()
			axs[1].cla()
			axs[2].cla()
		except Exception as e:
			print("ERROR ", e)
			continue


	pd.DataFrame(pd_list,columns=["sim","playerNumber","baseCoverage", "ALGOCoverage","greedyCoverage"]).to_csv("coverage.csv")

def main_factor_test(path_given):

	global_routes = None

	j = 0
	simulation_number = 200
	pd_list = []
	count=10
	#factor_values = [x for x in range(50, 3050, 50)]
	#factor_values = [x for x in range(20, 1000, 10)]
	

	factor_values = [750]
	for i in factor_values:
		while j < simulation_number:
			try:
				'''
				if not global_routes:
					initial_routes = env.initial_route_random(count, return_dicts=True, min_dist=True)
					
					global_routes = copy.deepcopy(initial_routes)

					print("Generating Routes.....")
				else:
					initial_routes = copy.deepcopy(global_routes)
					print("Loading Routes.....")
				'''

				initial_routes = env.initial_route_random(count, return_dicts=True, min_dist=True)
				route_dict = get_route_dict(initial_routes, factor_change=i)

				#genheatmap(route_dict)

				coverage_value, heatmap_dict_heap, spatial_matrix, global_heatheap = diverse_calculation(route_dict, vis=False)


				target_junction_dict = changeRouteTest(route_dict, i, vis=True, to_csv=True,heatmap_dict=heatmap_dict_heap, spatial_matrix=spatial_matrix, global_heatheap=global_heatheap, save_dir=path_given)
				coverage_value_after, heatmap_dict_heap_after, spatial_matrix, global_heatheap = diverse_calculation(target_junction_dict, div_routes="devpathlist")

				new_greedy_dict = find_non_div_route(target_junction_dict, vis=True)
				coverage_value_after_after, heatmap_dict_heap_after_after, spatial_matrix_after, global_heatheap_after = diverse_calculation(new_greedy_dict, div_routes="devpathlist")

				pd_list.append([j, i,count,coverage_value,coverage_value_after,coverage_value_after_after])


				logging.info(f"Variable {i} simulation {j} base_cov:{coverage_value} algo_cov:{coverage_value_after} greedy_cov:{coverage_value_after_after}")

				plt.savefig(os.path.join(path_given, f"{i}.png"))
				logging.info(f"Image {i} saved...")
				j += 1
				
			except Exception as e:
				print("ERROR ", e)
				pd.DataFrame(pd_list,columns=["sim", "variable_value","playerNumber","baseCoverage", "ALGOCoverage","greedyCoverage"]).to_csv(os.path.join(path_given ,"coverageError.csv"))
				logging.error(f"{traceback.format_exc()} {str(e)}")
				exit()
			finally:
				axs[0].cla()
				axs[1].cla()
				axs[2].cla()
			

		j = 0

	pd.DataFrame(pd_list,columns=["sim", "variable_value","playerNumber","baseCoverage", "ALGOCoverage","greedyCoverage"]).to_csv(os.path.join(path_given ,"coverage.csv"))


def main_test():
	R = 5
	cost_list = [1,2,3,4,5]

	#before calc nash modify
	''' 
	for i, route in enumerate(route_dict):
		route.reward = R
		route.cost = cost_list[i]
	'''
	#genheatmap(route_dict)

	route_dict = find_df()

	coverage_value, heatmap_dict_heap, spatial_matrix, global_heatheap = diverse_calculation(route_dict, vis=False)

	route_dict = neqfunction(route_dict)

	target_junction_dict = changeRouteTest(route_dict, 1, vis=True, to_csv=True,heatmap_dict=heatmap_dict_heap, spatial_matrix=spatial_matrix, global_heatheap=global_heatheap)

	coverage_value_after, heatmap_dict_heap_after, spatial_matrix, global_heatheap = diverse_calculation(target_junction_dict, div_routes="devpathlist")

	print(f"Coverage Before:{coverage_value}, Coverage After:{coverage_value_after}, change:{-1 *((coverage_value_after- coverage_value)/coverage_value)*100}%")

	new_greedy_dict = find_non_div_route(target_junction_dict, vis=True)

	coverage_value_after, heatmap_dict_heap_after, spatial_matrix, global_heatheap = diverse_calculation(new_greedy_dict, div_routes="devpathlist")

	print(f'Greedy coverage value is {coverage_value_after}, change:{abs((coverage_value_after- coverage_value)/coverage_value)*100}%')

	plt.show()

def test_junctions(from_node,to_node):

	#route_to_end = env.sim_env.map_data.sumonet.find_shortest_length(from_node, to_node)
	route_to_end = env.sim_env.map_data.find_best_route(from_node, to_node)

	#route_to_target = env.sim_env.map_data.sumonet.find_shortest_path(from_node, middle)
	#route_from_target = env.sim_env.map_data.sumonet.find_shortest_path(middle, to_node)
	print("entire", route_to_end)
	#print("startmiddle ", route_to_target)
	#print("middleend ", route_from_target)

if __name__ == "__main__":



	#main()
	#main_test()
	main_factor_test(path_given=dir_name)
	#test_junctions("104301649", "2584020709")
	traci.close()
	


#30805471