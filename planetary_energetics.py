import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import numpy as np

c = ['#fbb4ae','#b3cde3','#ccebc5','#decbe4','#fed9a6','#ffffcc','#e5d8bd','#fddaec','#f2f2f2']



class Layer:

    def __init__( self, inner_radius, outer_radius ):
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.thickness = outer_radius-inner_radius

        assert( self.thickness > 0.0 )

        self.inner_surface_area = 4.0 * np.pi * self.inner_radius * self.inner_radius
        self.outer_surface_area = 4.0 * np.pi * self.outer_radius * self.outer_radius

        self.volume = 4.0/3.0 * np.pi * \
                      ( np.power( self.outer_radius, 3.0 ) - \
                        np.power( self.inner_radius, 3.0 ) )

    def lower_heat_flux (self):
        raise NotImplementedError("Need to define a heat flux function")

    def upper_heat_flux (self):
        raise NotImplementedError("Need to define a heat flux function")
   
    def update( self, lower_temperature, upper_temperature, time ):
        raise NotImplementedError("Need to define a physics yo")


class Planet:

    def __init__( self, layers, T0 ):
       self.layers = layers
       self.temperatures = np.zeros( len(layers) )
       self.temperatures[-1] = T0
      
       
    def draw(self):
        fig = plt.figure()
        axes = fig.add_subplot(111)

        wedges = []
        for i,layer in enumerate(self.layers):
           wedges.append( patches.Wedge( (0.0,0.0), layer.outer_radius, 70.0, 110.0, width=layer.thickness, color=c[i]) )
        p = PatchCollection( wedges, match_original = True )
        axes.add_collection( p )
        r = max( [l.outer_radius for l in self.layers ] ) * 1.1

        axes.set_ylim( 0, r)
        axes.set_xlim( -r/2.0 , r/2.0 )
        plt.axis('off')
        plt.show()

        


mercury = Planet( [Layer(0.0,2020.0e3), Layer(2020.e3, 2440.0e3)], 1000.0 )
       
mercury.draw()

    
