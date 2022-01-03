import Room, datetime, SSP_Enum, Surgery_Request, Constraint
from Surgery_Variable import SurgeryVariable
from datetime import date, datetime, time, timedelta
from copy import deepcopy, copy

#Neighbours: SurgeryVariable-SurgeryRequest with same index, SurgeonVariables in the same day

#in the furture we would like arguments like approved by ward/patient notified


class SurgeryVariable_Surgeon(SurgeryVariable):
    def __init__(self, room, day, order, surgeon_domain, constraints, start_time=time(hour=0, minute=0),
                 end_time=time(hour=0, minute=0), surgeon=None):
        # self.with_surgery_Team = {'Nurses': False, 'Anesthetists': False, 'Equipment': False}
        super(SurgeryVariable_Surgeon, self).__init__(room, day, order, start_time, end_time, constraints, surgeon_domain, surgeon)

    def initialize_domain(self, surgeon_domain):
        
        """
        initializes the domain of the specific variable according to unary hard constraints
        :param surgeon_domain:set of all the surgeons of the ward
        :return: set of surgeons of the ward with shift on the day of the variable and that can preform surgery in the
        room
        """
        
        new_domain = surgeon_domain.copy()
        if len(new_domain) > 0:
            # Surgeon must have surgical shift  on the surgery date
            for surgeon in surgeon_domain:
                if self.day not in surgeon.surgical_shifts:
                    new_domain.discard(surgeon)
                    continue
                # checks if surgeon can operate in room
                surgeon_st = surgeon.get_surgery_types_id()  # set of surgery type id that surgeon holds
                room_st = self.room.get_surgery_types()   # set of surgery type id that room holds
                if len(surgeon_st.intersection(room_st)) < 1:
                    new_domain.discard(surgeon)
        return new_domain

    def __str__(self):
        return "SV: room-" + str(self.room) + " day- " + str(self.day) + " order- " + str(self.order)

    '''def get_init_d_key(self):
        d_key = []
        for v in self.domain:
            if v is None:
                continue
            d_key.append(str(self.day) + '_' + str(v.id))
        return d_key'''




