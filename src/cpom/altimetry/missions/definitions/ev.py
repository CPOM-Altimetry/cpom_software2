from datetime import datetime

dataset_definition = {
    "mission": "ev",
    "long_name": "ENVISAT",
    "launch_datetime": datetime(2002, 3, 1, 1, 7),
    "operational_phases": ["A", "B", "C", "D"],
    "operational_phase_start_datetimes": [
        datetime(2002, 9, 30, 21, 34),  # phase A (35-day repeat)
        datetime(2010, 10, 19, 0, 0),  # phase B (transition, orbit lowering)
        datetime(2010, 10, 24, 0, 0),  # phase C (transition, orbit lowering)
        datetime(2010, 10, 28, 0, 0),  # phase D (30 day repeats)
    ],
    # list of the start datetimes of each operational phase
    "operational_phase_repeat_period": [
        35,
        5,
        4,
        30,
    ],  # list of repeat cycle lengths in days for each phase
    "operational_phase_start_cycle_number": [
        10,
        94,
        95,
        96,
    ],  # list of cycle numbers of start of phase
    "operational_phase_end_cycle_number": [
        93,
        94,
        95,
        113,
    ],
    "operational_phase_pass_numbering_low": [
        1,
        1,
        1,
        1,
    ],  # lowest pass number (half-orbit relative to cycle start) in a cycle for each phase
    "operational_phase_pass_numbering_high": [
        1002,
        1002,
        1002,
        862,
    ],
}
