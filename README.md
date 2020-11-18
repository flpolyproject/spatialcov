# spatialcov
spatial coverage

The goal of the spatial coverage project is to take a set of routes for a series of autonomous vehicles with on board sensors, and create a new set of routes for those vehicles with an increased diversity of routes taken while still having the same starting and ending points. This will increase the time spent making measurements with the sensors, and also increase the diversity of the sensing data obtained.

The first requirement is to create a random series of vehicle routes that adhere to real world expectations. For instance, back roads should have little traffic, while highways and intersections should have a greater density of vehicles traveling on them.

After creating the inital routes, the fréchet distances between each of the routes will need to be calculated, and this distance can be used to determine the diversity of the routes being taken. This diversity can in turn be used to determine the utility that each vehicle has on its original route.

With the utility of each vehicle calculated, a nash equilibrium can then be created that maximizes the utility of each vehicle by altering the paths taken for each vehicle's route.
