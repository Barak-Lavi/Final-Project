import SSP_Enum


class SurgeryType(object):

    def __init__(self, st_id, name, urgency, complexity, duration, utility):
        self.st_id = st_id
        self.name = name
        self.urgency = urgency
        self.complexity = complexity
        self.duration = duration
        self.utility = utility

