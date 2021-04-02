

import os, sys, glob
import traci
import traci.constants as tc
from settings import Settings, GraphSetting
from random import choice
sys.path.append("./../")
from traci_env import EnvironmentListener, BaseEnv, GreedyEnv, RandomEnv
from visualize import Visualize
from postprocess import PostGraph, MultiCaptureGraph
from uav import UAV
import datetime

import logging
import pandas as pd
import copy

import traceback


def start_simulation(Env, sim_number, gui=False, _seed=None, setting_obj=None, dir_name=None, main_env=None, new_players=False):

	#testpath = "./../map/london-seg4/data/london-seg4.sumocfg")

	if gui:
		traci.start(["sumo-gui", "-c", Settings.sumo_config])
	else:
		traci.start(["sumo", "-c", Settings.sumo_config])


	
	env = Env(sim_number=sim_number, _seed=_seed, setting_obj=setting_obj, main_env=main_env, new_players=new_players)
	my_vis = Visualize(env)

	while True:

		traci.simulationStep()
		traci.addStepListener(env)
		if gui:
			my_vis.show()  #this is for visualization of the path
		if env.break_condition:
			break


	#env.reward_to_json(os.path.join(dir_name, f"{sim_number}"))

	print("veh succesffuly arrived ", env.sim_env.success_veh)
	traci.close()



	#env.post_process.to_csv()
	return env.post_process, env

	#return env.sim_env


def run(gui=False, number=1, Env=EnvironmentListener, setting_obj=None, file_title=None, dir_name=None, main_env=None, new_players=False):  #handle all the simulations with one simulation parameter
	_seed=3

	post_process = PostGraph(file_title, columns=["sim_number", "sim_step", "veh_id", "edge_id", "speed", "capacity", "budget", "prev_poi", "algo"], dir_name=dir_name)

	#Here is the global multi capture

	multi_cap = MultiCaptureGraph("capture")

	sim_number = 1
	while sim_number <= number:

		try:
			

			temp_process, env = start_simulation(Env, sim_number, gui=gui, _seed=_seed, setting_obj=setting_obj, dir_name=dir_name, main_env=main_env, new_players=new_players)

			post_process.df_list = post_process.df_list + temp_process.df_list
			logging.info(f'SUCCESS FINISHED simulation {sim_number}')

			sim_number+=1

			env.sim_env.post_process_graph.setting = env.sim_env.GraphSetting
			multi_cap.simulation_list.append(env.sim_env.post_process_graph)

			

		except traci.exceptions.TraCIException as e:
			#print("Restarting simulation failed at number ", sim_number, e)
			logging.info(f"Failed simulation {sim_number} {str(e)}")

			traci.close()


	multi_cap.pickle_save(os.path.join(dir_name, f'{file_title}.sim'))
	post_process.to_csv()

	return env



def runbase(dir_name):
	gui = False
	main_env = None
	mySetting = GraphSetting()
	run(gui=gui, number=1, Env=BaseEnv, setting_obj = mySetting, file_title=f"{mySetting.car_numbers}_BASE", dir_name = dir_name, main_env=main_env)



if __name__ == '__main__':
	print(traci.__file__)


	dt = datetime.datetime.utcnow().timestamp()
	dir_name = os.path.join(Settings.sim_save_path_graph, str(dt))
	os.mkdir(dir_name)

	print("making dir ", dir_name)


	logging.basicConfig(filename=os.path.join(dir_name, 'output.log'), filemode='w', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

	
	runbase(dir_name)


	
