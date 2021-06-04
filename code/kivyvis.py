from random import random, choice
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics import Color, Ellipse, Line
import heatmapactual
from fdist import *
import numpy as np
from kivy.core.window import Window
from itertools import chain
from kivy.uix.slider import Slider

def ellipse_points(x1,y1,x2,y2,c, show=False):
	a1 = x1#1
	b1 = y1#2
	a2 = x2#5
	b2 = y2#7
	#c = 100

	# Compute ellipse parameters
	a = c / 2                                # Semimajor axis
	x0 = (a1 + a2) / 2                       # Center x-value
	y0 = (b1 + b2) / 2                       # Center y-value
	f = np.sqrt((a1 - x0)**2 + (b1 - y0)**2) # Distance from center to focus
	b = np.sqrt(a**2 - f**2)                 # Semiminor axis
	#print(f"{f} {b} {a} {f**2} {a**2} {a**2 - f**2}")
	phi = np.arctan2((b2 - b1), (a2 - a1))   # Angle betw major axis and x-axis

	# Parametric plot in t
	resolution = 1000
	t = np.linspace(0, 2*np.pi, resolution)
	x = x0 + a * np.cos(t) * np.cos(phi) - b * np.sin(t) * np.sin(phi)
	y = y0 + a * np.cos(t) * np.sin(phi) + b * np.sin(t) * np.cos(phi)
	if show:
		#Plot ellipse
		plt.plot(x, y)

		# Show focii
		plt.plot(a1, b1, 'bo')
		plt.plot(a2, b2, 'bo')

		plt.axis('equal')
		plt.show()

	return x, y

	



class MyPaintWidget(Widget):

	global_dict = []
	temp_array = []
	color_dict = []

	def on_touch_down(self, touch):
		self.color = (random(), 1, 1)
		self.temp_array = []
		with self.canvas:
			Color(*self.color, mode='hsv')
			d = 30.
			Ellipse(pos=(touch.x - d / 2, touch.y - d / 2), size=(d, d))
			touch.ud['line'] = Line(points=(touch.x, touch.y))
		self.temp_array.append([touch.x, touch.y])


	def on_touch_move(self, touch):
		touch.ud['line'].points += [touch.x, touch.y]
		self.temp_array.append([touch.x, touch.y])

	def on_touch_up(self, touch):
		print("adding a path")
		self.global_dict.append(np.array(self.temp_array))
		self.color_dict.append(self.color)

	def draw_new_path(self, id, tuples):
		with self.canvas:
			#print(f"processing... {id} {self.color_dict}")
			Color(*self.color_dict[id], mode='hsv')
			d = 10.
			for value in tuples:
				Ellipse(pos=(value[0] - d / 2,  value[1]- d / 2), size=(d, d))

			flatten_list = list(chain.from_iterable(tuples))
			Line(points=flatten_list)



class MyPaintApp(App):

	def build(self):
		self.route_dict= None
		layout = BoxLayout()
		self.painter = MyPaintWidget()
		#self.s = Slider(min=50, max=1000, value=750, size_hint=(0.8, 0.1))
		self.clearbtn = Button(text='Clear', size_hint=(0.1, 0.1))
		self.clearbtn.bind(on_release=self.clear_canvas)


		compute_cov_button = Button(text="sp", size_hint=(0.1, 0.1))
		compute_cov_button.bind(on_release=self.compute_sp)


		spButton = Button(text="new_path", size_hint=(0.1, 0.1))
		spButton.bind(on_release=self.new_path)

		greedyButton = Button(text="greedy_path", size_hint=(0.1, 0.1))
		greedyButton.bind(on_release=self.greedy_path)

	   
		layout.add_widget(self.painter)
		layout.add_widget(self.clearbtn)
		layout.add_widget(compute_cov_button)
		layout.add_widget(spButton)
		layout.add_widget(greedyButton)
		#layout.add_widget(self.s)

		return layout

	def compute_sp(self, obj):
		self.route_dict = find_fdist(self.painter.global_dict[:-1])
		self.route_dict = neqfunction(self.route_dict, factor_change=800)
		coverage_value_after, heatmap_dict_heap_after, spatial_matrix, global_heatheap = heatmapactual.diverse_calculation(self.route_dict, Window.size[0], Window.size[1])

		print(f"Total spatial coverage is ", coverage_value_after)
	def greedy_path(self, obj):
		if not self.route_dict:
			self.compute_sp(obj)
		self.clear_canvas(obj)
		for value in self.route_dict:
			#print(f"{value.id} start {value.start} to {value.end} c: {value.dev_value_after}")
			x,y = ellipse_points(value.start[0], value.start[1], value.end[0], value.end[1], value.dev_value_after)
			index, xvalue = choice(list(enumerate(x)))
			print(f"{value.id} point chosen is {xvalue} {y[index]} allowed to dev {value.dev_value_after}")
			self.painter.draw_new_path(value.id, (value.start, [xvalue, y[index]], value.end))



	def clear_canvas(self, obj, color=False):
		self.painter.canvas.clear()
		self.painter.global_dict = []
		self.painter.temp_array = []
		if not color:
			self.color_dict = []


	def new_path(self, obj):
		if not self.route_dict:
			self.compute_sp(obj)
		self.clear_canvas(obj, color=True) #keep the color
		for value in self.route_dict:
			#print(f"{value.id} start {value.start} to {value.end} c: {value.dev_value_after}")
			x,y = ellipse_points(value.start[0], value.start[1], value.end[0], value.end[1], value.dev_value_before)
			index, xvalue = choice(list(enumerate(x)))
			print(f"{value.id} point chosen is {xvalue} {y[index]} allowed to dev {value.dev_value_before}")
			self.painter.draw_new_path(value.id, (value.start, [xvalue, y[index]], value.end))



		


if __name__ == '__main__':
	MyPaintApp().run()
	#ellipse_points(257,212,577,29.9,7.5, show=True)