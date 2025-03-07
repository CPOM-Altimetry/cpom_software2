"""cpom.areas

# CPOM Area Definitions and Polarplot class

## definitions  

contain standard area definition files used by the Polarplot class and related tools 
such as plot_map.py. Each area definition is stored in a separate <area_name>.py file.

## areas.py

contains code to handle the area definitions

## area_plot.py

contains the Polarplot class. The main external function of the Polarplot class
is **plot_points()**:

`cpom.areas.area_plot.Polarplot.plot_points`

The purpose of Polarplot('some_area_name').plot_points() is to plot latitude, longitude, 
and values on predefined area maps. plot_points() takes one or more dataset dictionaries as input, 
containing the lat,lon,values and associated parameters. For a full list of the data set options
see `cpom.areas.area_plot.Polarplot.plot_points`.

### Example 1 : plot two data sets on a basic map of Antarctica

The following example plots 2 data sets on a basic map of Antarctica. 

* The first dataset has a valid range set which will only plots vals within this range.
* The second dataset is plotted using the viridis colormap 
      (the first uses the default colormap).

**Note** that the ancillary plots (histograms, etc) only apply to the first data set plotted.
The top right map shows the locations of the invalid points (since we set a valid_range there
are some points that have invalid values between 5 and 10).

You can scale the size of the points plotted in the main and ancillary maps using
dataset parameters. For example to make the point size larger you might use:
 ("plot_size_scale_factor":10, and "minimap_val_scalefactor": 10,). Default point size
 is 1.0. Note that it is not a linear scale, so some experimentation is required.

```
import numpy as np
from cpom.areas.area_plot import Polarplot

dataset1={
    "lats": np.linspace(-80,-70,100), # example 100 latitude values between -80N and -70N
    "lons": np.linspace(0,5,100),     # example 100 longitude values between 0 and 5 degs E.
    "vals": np.linspace(0,10,100),    # example 100 data values between 0 and 10.
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

### Example 2: Map only (no histograms), hill shaded Antarctic map

In this example we plot a single dataset, add the map_only=True option to plot_points() 
to remove the default histograms,
and we change the map area definition name to antarctica_hs. antarctica_hs is
an area definition that applies a hill shaded DEM background and adds some
bathymetry to the ocean.

```
import numpy as np
from cpom.areas.area_plot import Polarplot

dataset1={
    "lats": np.linspace(-80,-70,100), # example 100 latitude values between -80N and -70N
    "lons": np.linspace(0,5,100),     # example 100 longitude values between 0 and 5 degs E.
    "vals": np.linspace(0,10,100),    # example 100 data values between 0 and 10.
    "name": 'test_param',
    "valid_range": [0,5],
}

Polarplot('antarctica_hs').plot_points(dataset1,map_only=True)
```

![my image](/cpom_software2/images/plot_points_example2.jpg "my image")

### Example 3: Save the plot to a file /tmp/greenland.png

In this case the plot of Greenland will not be displayed but will be saved to a png file.

```
import numpy as np
from cpom.areas.area_plot import Polarplot

dataset1={
    "lats": np.linspace(80,50,100), # example 100 latitude values between -80N and -70N
    "lons": np.linspace(300,340,100),     # example 100 longitude values between 0 and 5 degs E.
    "vals": np.linspace(0,10,100),    # example 100 data values between 0 and 10.
}

Polarplot('greenland_hs').plot_points(dataset1,output_file='/tmp/greenland.png')
```

![my image](/cpom_software2/images/plot_points_example3.jpg "my image")


"""
