from random import random
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics import Color, Ellipse, Line
import heatmapactual
from fdist import *
import numpy as np
from kivy.core.window import Window

def ellipse_points():
	a1 = 1
	b1 = 2
	a2 = 5
	b2 = 7
	c = 9

	# Compute ellipse parameters
	a = c / 2                                # Semimajor axis
	x0 = (a1 + a2) / 2                       # Center x-value
	y0 = (b1 + b2) / 2                       # Center y-value
	f = np.sqrt((a1 - x0)**2 + (b1 - y0)**2) # Distance from center to focus
	b = np.sqrt(a**2 - f**2)                 # Semiminor axis
	phi = np.arctan2((b2 - b1), (a2 - a1))   # Angle betw major axis and x-axis

	# Parametric plot in t
	resolution = 1000
	t = np.linspace(0, 2*np.pi, resolution)
	x = x0 + a * np.cos(t) * np.cos(phi) - b * np.sin(t) * np.sin(phi)
	y = y0 + a * np.cos(t) * np.sin(phi) + b * np.sin(t) * np.cos(phi)

	# Plot ellipse
	#plt.plot(x, y)

	# Show focii
	#plt.plot(a1, b1, 'bo')
	#plt.plot(a2, b2, 'bo')

	#plt.axis('equal')
	#plt.show()

class MybuttonWidget(Widget):

    pass


class MyPaintWidget(Widget):

    global_dict = []
    temp_array = []

    def on_touch_down(self, touch):
        color = (random(), 1, 1)
        self.temp_array = []
        with self.canvas:
            Color(*color, mode='hsv')
            d = 30.
            Ellipse(pos=(touch.x - d / 2, touch.y - d / 2), size=(d, d))
            touch.ud['line'] = Line(points=(touch.x, touch.y))


    def on_touch_move(self, touch):
        touch.ud['line'].points += [touch.x, touch.y]
        self.temp_array.append([touch.x, touch.y])

    def on_touch_up(self, touch):
        print("adding a path")
        self.global_dict.append(np.array(self.temp_array))


class MyPaintApp(App):

    def build(self):
        layout = BoxLayout()
        self.painter = MyPaintWidget()
        self.clearbtn = Button(text='Clear', size_hint=(0.5, 0.5))
        self.clearbtn.bind(on_release=self.clear_canvas)


        compute_cov_button = Button(text="sp", size_hint=(0.1, 0.1))
        compute_cov_button.bind(on_release=self.compute_sp)


        spButton = Button(text="SP", size_hint=(0.1, 0.1))
        spButton.bind(on_release=self.compute_sp)
       
        layout.add_widget(self.painter)
        layout.add_widget(self.clearbtn)
        layout.add_widget(compute_cov_button)
        return layout

    def compute_sp(self, obj):
        route_dict = find_fdist(self.painter.global_dict[:-1])
        neqfunction(route_dict, factor_change=50)

    def clear_canvas(self, obj):
        self.painter.canvas.clear()
        self.painter.global_dict = []
        self.painter.temp_array = []
    def compute_spatial(self, obj):
        print(len(self.painter.global_dict))
        route_dict = find_fdist(self.painter.global_dict[:-1])
        #print(Window.size)

        total_cov = heatmapactual.diverse_calculation(route_dict, Window.size[0], Window.size[1])

        print(f"Total spatial coverage is ", total_cov)



if __name__ == '__main__':
    MyPaintApp().run()
    #ellipse_points()