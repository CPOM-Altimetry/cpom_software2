from datetime import datetime

mission_definition = {
    "mission": "is2",
    "long_name": "ICESat-2",
    "reference_ground_track_low": 1,  # lowest Ref. Ground Track in a cycle
    "reference_ground_track_high": 1387,  # highest Ref. Ground Track in a cycle
    "launch_datetime": datetime(2010, 4, 8, 13, 57, 4),  # (y,m,d,hour,min), launch datetime in UTC
    "operational_phases": ["A"],  # ['A','B',...], A=1st phase of operational phase, B=2nd phase
    "operational_phase_start_datetimes": [
        datetime(2018, 9, 28, 12, 15)
    ],  # list of the start datetimes of each operational phase
    "operational_phase_repeat_period": [
        91
    ],  # NOTE (for CS2 we (in this case) use the subcycle length not the repeat period which is 369 days)
    "operational_phase_start_cycle_number": [1],  # list of cycle numbers of start of phase
    "operational_phase_pass_numbering_low": [
        0
    ],  # lowest pass number (half-orbit relative to cycle start) in a cycle for each phase
    "operational_phase_pass_numbering_high": [0],  # highest pass number in a cycle for each phase
}
