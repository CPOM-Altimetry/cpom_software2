from datetime import datetime

mission_definition = {
    "mission": "cs2",
    "long_name": "Cryosat-2",
    "launch_datetime": datetime(2010, 4, 8, 13, 57, 4),  # (y,m,d,hour,min), launch datetime in UTC
    "operational_phases": ["A"],  # ['A','B',...], A=1st phase of operational phase, B=2nd phase
    "operational_phase_start_datetimes": [
        datetime(2010, 10, 18, 0, 0)
    ],  # list of the start datetimes of each operational phase
    "operational_phase_repeat_period": [
        30
    ],  # NOTE (for CS2 we (in this case) use the subcycle length not the repeat period which is 369 days)
    "operational_phase_start_cycle_number": [1],  # list of cycle numbers of start of phase
    "operational_phase_pass_numbering_low": [
        0
    ],  # lowest pass number (half-orbit relative to cycle start) in a cycle for each phase
    "operational_phase_pass_numbering_high": [0],  # highest pass number in a cycle for each phase
    # Mission Phases (using true repeat period of 369 days)
    "true_operational_phases": [
        "A"
    ],  # ['A','B',...], A=1st phase of operational phase, B=2nd phase
    "true_operational_phase_start_datetimes": [
        datetime(2010, 10, 18, 0, 0)
    ],  # list of the start datetimes of each operational phase
    "true_operational_phase_repeat_period": [369],  # list of repeat periods for each phase
    "true_operational_phase_start_cycle_number": [1],  # list of cycle numbers of start of phase
    "true_operational_phase_end_cycle_number": [
        None
    ],  # list of cycle numbers of end of phase (None if ongoing phase)
}
