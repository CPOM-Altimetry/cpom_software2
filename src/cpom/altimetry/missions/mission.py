"""cpom.altimetry.missions.mission

Mission class with mission dependent parameters and functions
for all supported Altimetry missions : S3A, S3B, CS2, EV, IS2, E1, E2
"""

mission_list = [
    "cs2", 
    "s3a", 
    "s3b",
    "e1",
    "e2",
    "ev",
    "is2"
]

class Mission:
    """Class to support Altimetry missions."""

    def __init__(self, name: str):
        """
        Class initialization.

        Args:
            name (str): Mission name.
        """
        self.name = name

        if name not in mission_list:
            raise ValueError(f"{name} is not a supported mission in Mission class")
        
    
    def get_cycle_date(): 
        pass 


    def cycle_dates(): 
        pass 

    def get_phase_for_cycle_number(self, cycle_number: int):
        pass

    def get_oribit_number_of_cycle(): 
        pass

    
