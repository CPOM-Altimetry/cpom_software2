"""
Sea ice parameter information
"""

import sys

all_si_params = [
    "Thickness",
    "FloeChordLength",
    "IceConcentration",
    "IceType",
    "LeadFraction",
    "FloeFraction",
    "UnkFraction",
    "RadarFreeboard",
    "SeaLevelAnomaly",
    "WarrenSnowDepth",
]


class SIParams:  # pylint: disable=too-few-public-methods,too-many-instance-attributes
    """Class to store sea ice parameter information."""

    def __init__(self, param=None):

        self.iscompoundvariable = (
            False  # True if parameter is calculated from other parameters or
            # external data
        )
        self.plot_range_low = None  # (m) fixed low plot range
        self.plot_range_high = None  # (m) fixed low plot range
        self.cmap_blocks = None  # colormap: Continuous=None, n othersize
        self.isflag = False  # Parameter is a flag
        self.units = ""  # parameter units
        self.param = param  # parameter name
        self.long_name = ""  # longer parameter name
        self.show_fv_plot = False  # show fill value sub-plot

        # ---------------------------------------------------------------------
        # Parameters
        # ---------------------------------------------------------------------
        if param == "Thickness":
            self.plot_range_low = 0.0
            self.plot_range_high = 3.0
            self.units = "m"
            self.long_name = "Sea Ice Thickness"

        elif param == "FloeChordLength":
            self.plot_range_low = 0.0
            self.plot_range_high = 3.0
            self.units = "m"
            self.long_name = "Floe Chord Length"

        elif param == "IceConcentration":
            self.plot_range_low = 60.0
            self.plot_range_high = 100.0
            self.units = "%"
            self.long_name = "Sea Ice Concentration"

        elif param == "IceType":
            self.long_name = "Multi-year Ice Fraction"
            self.units = ""
            self.plot_range_low = 0.0
            self.plot_range_high = 1.0

        elif param == "LeadFraction":
            self.plot_range_low = 0.0
            self.plot_range_high = 1.0
            self.units = ""
            self.long_name = "Lead Fraction"

        elif param == "FloeFraction":
            self.plot_range_low = 0.0
            self.plot_range_high = 1.0
            self.units = ""
            self.long_name = "Floe Fraction"

        elif param == "RadarFreeboard":
            self.plot_range_low = 0.0
            self.plot_range_high = 0.3
            self.units = "m"
            self.long_name = "Radar Freeboard"

        elif param == "SeaLevelAnomaly":
            self.plot_range_low = -0.25
            self.plot_range_high = 0.25
            self.units = "m"
            self.long_name = "Sea Level Anomaly"

        elif param == "WarrenSnowDepth":
            self.plot_range_low = 0.10
            self.plot_range_high = 0.3
            self.units = "m"
            self.long_name = "Warren Snow Depth"

        elif param == "UnkFraction":
            self.plot_range_low = 0.0
            self.plot_range_high = 1.0
            self.units = ""
            self.long_name = "Unknown Fraction"
        else:
            sys.exit(f"Unrecognised sea ice parameter name {param}")
