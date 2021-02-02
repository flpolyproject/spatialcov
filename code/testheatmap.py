import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as sp
import os, glob, sys
import numpy as np; np.random.seed(0)
import seaborn as sns; #sns.set_theme()
from traci_env import EnvironmentListener, BaseEnv
import traci
import traci.constants as tc
from settings import Settings, GraphSetting
from postprocess import *


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde as kde
from matplotlib.colors import Normalize
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

#add all the points of the line into the array
#


















N = 10000
mean = [0,0]
cov = [[2,2],[0,2]]

samples = np.random.multivariate_normal(mean,cov,N).T

#plt.scatter(samples[:, 0], samples[:, 1])
#plt.show()
#exit()

print(samples)
print(samples.shape)
#exit()
densObj = kde( samples)



def makeColours( vals ):

	print(vals)
	print(len(vals))

	colours = np.zeros( (len(vals),3) )
	norm = Normalize( vmin=vals.min(), vmax=vals.max() )

	print("after")
	print(norm)
	#exit()

	#Can put any colormap you like here.
	#coloursb = [cm.ScalarMappable( norm=norm, cmap='jet') for val in vals]
	colours = [cm.ScalarMappable( norm=norm, cmap='jet').to_rgba( val ) for val in vals]
	print(colours)

	#divider = make_axes_locatable()
	#cax = divider.append_axes('right', size='5%', pad=0.05)

	#plt.colorbar(vals, orientation='vertical')
	#exit()

	return colours

colours = makeColours( densObj.evaluate( samples ) )

#plt.scatter( samples[0], samples[1], color=colours )
#plt.show()





def find_df():

	folder = os.path.join(Settings.sim_save_path_graph, "spatialtest")

	files = glob.glob(os.path.join(folder, r'*.sim'))
	obj = MultiCapture('test').pickle_load(files[0], directory=False)
	for i, sim_number in enumerate(obj.simulation_list):
		algo = sim_number.setting.current_running_algo
		algo = algo.lower()
		if algo == "atne":
			algo = "ours"
		sim_number.compute_spatial_coverage(heatmap=True)
find_df()


traci.start(["sumo", "-c", Settings.sumo_config])
env = BaseEnv(sim_number=1, _seed=3, init=False)
paths = np.array([[obj.x, obj.y] for obj in env.sim_env.map_data.junctions.values()])

pt = paths.T
densObjnew = kde(pt)
colours = makeColours( densObjnew.evaluate( pt ) )

#plt.scatter( pt[0], pt[1], color=colours , s=0.8, alpha=1)

'''
m = cm.ScalarMappable(cmap=cm.jet)
m.set_array([])
plt.colorbar(m)
'''

plt.plot(paths[:,0], paths[:,1], 'o', alpha=0.4, linestyle="", markersize=0.6)
plt.show()


