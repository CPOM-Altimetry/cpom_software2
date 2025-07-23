from datetime import datetime

dataset_definition = {
    "mission": "e1",
    "long_name": "ERS-1",
    "launch_datetime": datetime(1991, 7, 17, 0, 0),  # (y,m,d,hour,min), launch datetime in UTC
    "operational_phases": [
        "A",
        "B",
        "C",
        "D",
        "E1",
        "E2",
        "F1",
        "F2",
        "F3",
        "G1",
        "G2",
    ],  # ['A','B',...], A=1st phase of operational phase, B=2nd phase
    "operational_phase_start_datetimes": [
        datetime(1991, 8, 3, 0, 0),  # phase A  (start time), 3 day repeat
        datetime(1991, 12, 14, 0, 0),  # phase B  (start time), 3 day repeat
        datetime(1992, 3, 31, 0, 0),  # phase C  (start time), 35 day repeat
        datetime(1993, 12, 23, 0, 0),  # phase D  (start time), 3 day repeat
        datetime(1994, 4, 10, 0, 0),  # phase E1  (start time), 43 days length, 168 day repeat
        datetime(1994, 5, 23, 0, 0),  # phase E2  (start time), 128 days length, 168 day repeat
        datetime(1994, 9, 28, 0, 0),  # phase F1  (start time), 3 days length
        datetime(1994, 10, 1, 0, 0),  # phase F2  (start time), 168 days
        datetime(1995, 3, 18, 0, 0),  # phase F3  (start time), 3 days
        datetime(1995, 3, 21, 0, 0),  # phase G1  (start time), 19 days
        datetime(1995, 4, 9, 0, 0),  # phase G2  (start time), 35 days
    ],
    "operational_phase_repeat_period": [
        3,
        3,
        35,
        3,
        43,
        128,
        3,
        168,
        3,
        19,
        35,
    ],  # list of repeat cycle lengths in days for each phase
    "operational_phase_start_cycle_number": [
        2,
        47,
        83,
        102,
        139,
        140,
        141,
        142,
        143,
        144,
        145,
    ],  # list of cycle numbers of start of phase
    "operational_phase_end_cycle_number": [
        46,
        82,
        101,
        138,
        139,
        140,
        141,
        142,
        143,
        144,
        156,
    ],
    "operational_phase_pass_numbering_low": [
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    ],  # lowest pass number (half-orbit relative to cycle start) in a cycle for each phase
    "operational_phase_pass_numbering_high": [
        86,
        86,
        1002,
        86,
        1002,
        5000,
        5000,
        5000,
        86,
        1002,
        1002,
    ],  # highest pass number in a cycle for each phase
}
