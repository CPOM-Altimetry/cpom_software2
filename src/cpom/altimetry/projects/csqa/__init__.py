"""
# CryoSat-2 Performance Monitoring (CSQA) QCV processing tools

Generate the content (map plots and statistics) of the CryoSat-2 performance monitoring portal
from an archive of CryoSat-2 L2 (GDR-A) and L2i ESA products.

Plots and statistics are organised by product baseline and 30-day data take (cycle), numbered
from 1 starting at 00:00 UTC on 18-Oct-2010.

## Configuration

- `config/csqa_config.yaml` : input archive paths, cycle definition, baselines, areas, output
  directory and plot settings
- `config/csqa_parameters.yaml` : the monitored parameters

## Tools

- `process_cycles.py` : process cycles (statistics + plots), then update the portal index
- `build_portal_index.py` : rebuild the portal manifest and statistics timeseries

Example:

```
python process_cycles.py --cycles 193 --baselines F
python process_cycles.py --latest 3 --update --workers 4     # ie from cron
```
"""

__version__ = "1.0.0"
