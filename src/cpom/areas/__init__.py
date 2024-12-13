"""cpom.areas

# CPOM Area Definitions and Polarplot class

## definitions  

contain standard area definition files used by the Polarplot class and related tools 
such as plot_map.py. Each area definition is stored in a separate <area_name>.py file.

## areas.py

contains code to handle the area definitions

## area_plot.py

contains the Polarplot class. The main external function of the Polarplot class
is:

`cpom.areas.area_plot.Polarplot.plot_points`

## Examples of using Polarplot().plot_points()

The purpose of Polarplot('some_area_name').plot_points() is to plot latitude, longitude, 
and values on predefined area maps. plot_points() takes one or more dataset dictionaries as input, 
containing the lat,lon,values and associated parameters.

The following example plots 2 data sets on a basic map of Antarctica:

```
import numpy as np
from cpom.areas.area_plot import Polarplot

dataset1={
    "lats": np.linspace(-80,-70,100),
    "lons": np.linspace(0,5,100),
    "vals": np.linspace(0,10,100),
    "units": 'm',
    "name": 'test_param',
    "valid_range": [0,5],
}

dataset2={
    "lats": np.linspace(-80,-70,100),
    "lons": np.linspace(20,25,100),
    "vals": np.linspace(0,10,100),
    "name": 'test_param',
    "cmap_name": 'viridis',
}

Polarplot('antarctica').plot_points(dataset1,dataset2)

```

![my image](/cpom_software2/images/plot_points_example1.jpg "my image")


"""
