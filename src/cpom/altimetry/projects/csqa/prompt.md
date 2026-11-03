This task is to create a new CryoSat-2 Monitoring service
This will have two components:
a) a set of one or more QCV processing tools written in python 3.13. The purpose of these tools is to generate content (plots and stats) for the portal, processed from an archive of CryoSat-2 L2i or L2 ESA products.
b) a public website which shows the QCV outputs

Both components will run on a production server which hosts the full archive of CryoSat-2 L2 and L2i products and an apache web server, however we will develop the components locally using a subset of products and a local web server, and sync the production server using git.

The python tools will be developed within the CPOM v2 python repository so that we can benefit from existing plotting classes (ie as in Polarplot in /Users/alanmuir/software/cpom_software2/src/cpom/areas/area_plot.py) for polar and global areas. 
Locally the QCV tools will be developed within the folder
/Users/alanmuir/software/cpom_software2/src/cpom/altimetry/projects/csqa

DATA INPUTS:
Lets have a global configuration file which sets these paths for the tools.
Cryosat-2 L2i:
/raid6/cpdata/SATS/RA/CRY/L2I/<LRM,SAR-A,SIN>/<YYYY>/<MM>
Locally we have example input files in 
/raid6/cpdata/SATS/RA/CRY/L2I/<LRM,SAR-A,SIN>/2026/08/
/raid6/cpdata/SATS/RA/CRY/L2I/<LRM,SAR-A,SIN>/2011/01/

For SIN (in 2026/08) we have a full month, whereas for all other months and modes we only a single days worth of files to save disk space.
Initially we will restrict the Data Inputs to L2i but we may expand to L2 if necessary for other parameters later.

QA Data Takes:

Importantly we will organize the QA plots and stats not by month and year but by 30-day sub-cycles of CryoSat-2 from a start date at the beginning of the mission. The initial value (which should be set in the global config file) will be 18/10/2010, and we shall number the cycles from 1 (ie cycle 1 is from 00:00 on 18/10/2010 for 30-days)

QA Processing Tool Requirements:

For each L2i parameter configured we should produce 30-day cycle plots of this parameter for the following areas: global, south polar (measurements located < -60 degs latitude), north polar (measurements located > 60 degs latitude). 
For the south polar plots we can use the Polarplot('south_polar_csqa').plot_points()
For the north polar plots we can use the Polarplot('north_polar_csqa').plot_points()
For the global plots we can use Polarplot('global_csqa').plot_points()
