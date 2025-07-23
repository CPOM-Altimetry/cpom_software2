from datetime import datetime

dataset_definition = {
    "mission": "s3a",
    "long_name": "Sentinel-3A",
    "launch_datetime": datetime(2016, 2, 16, 17, 57),
    "operational_phases": ["A"],
    "operational_phase_start_datetimes": [
        datetime(2016, 12, 7, 22, 22)
    ],  # list of the start datetimes of each operational phase
    "operational_phase_repeat_period": [27],  # list of repeat cycle lengths in days for each phase
    "operational_phase_start_cycle_number": [12],  # list of cycle numbers of start of phase
    "operational_phase_end_cycle_number": [
        None
    ],  # list of cycle numbers of end of phase, None if ongoing current phase
    "operational_phase_pass_numbering_low": [
        0
    ],  # lowest pass number (half-orbit relative to cycle start) in a cycle for each phase
    "operational_phase_pass_numbering_high": [771],  # highest pass number in a cycle for each phase
}
