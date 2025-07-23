from datetime import datetime

dataset_definition = {
    "mission": "e2",
    "long_name": "ERS-2",
    "launch_datetime": datetime(1995, 4, 21, 0, 0),
    "operational_phases": ["A"],  # ['A','B',...], A=1st phase of operational phase, B=2nd phase
    "operational_phase_start_datetimes": [
        datetime(1995, 5, 15, 22, 36)
    ],  # list of the start datetimes of each operational phase
    "operational_phase_repeat_period": [35],  # list of repeat cycle lengths in days for each phase
    "operational_phase_start_cycle_number": [1],  # list of cycle numbers of start of phase
    "operational_phase_end_cycle_number": [85],
    "operational_phase_pass_numbering_low": [
        1
    ],  # lowest pass number (half-orbit relative to cycle start) in a cycle for each phase
    "operational_phase_pass_numbering_high": [
        1002
    ],  # highest pass number in a cycle for each phase
}
