import Room, SSP_Enum, Constraint
from datetime import date, datetime, timedelta, time
from Surgery_Variable import SurgeryVariable
from Surgery_Request import Surgery_Request
from copy import deepcopy, copy
import sys
# Neighbours : SurgeryVariable_Surgeon with same index, SurgeryRequest Varialbe in the same day,
# in principle all diff - all surgeryRequest Variables

# in the furture we would like arguments like approved by ward/patient notified


class SurgeryVariable_SurgeryRequest(SurgeryVariable):
    def __init__(self, room, day, order, sr_domain, constraints, start_time=time(hour=0, minute=0),
                 end_time=time(hour=0, minute=0), surgery_request=None):
        self.with_surgery_team = {'Nurse': False, 'Anesthetist': False, 'Equipment': False}
        self.surgery_team_in_update = {'Nurse': False, 'Anesthetist': False, 'Equipment': False}
        self.value_in_update = None
        super(SurgeryVariable_SurgeryRequest, self).__init__(room, day, order, start_time, end_time, constraints,
                                                             sr_domain, surgery_request)


    def initialize_domain(self, sr_domain):
        """initializes the domain of the specific variable according to unary hard constraints
        :param sr_domain:list [set of all the RTG of the ward, list of surgery types objects that can be performed in
        this day in dependency of the surgeons on shifts this day, ward object]
        :return: set of RTG of the ward according to constraints
        """
        surgeon_st = sr_domain[1]
        newDomain = sr_domain[0].copy()
        ward = sr_domain[2]
        if len(newDomain) > 0:
            for sr in sr_domain[0]:
                # specific senior must be on shift
                if sr.specific_senior is not None:
                    ss = ward.find_surgeon_by_id(sr.specific_senior)
                    if self.day not in ss.surgical_shifts:
                        newDomain.discard(sr)
                # Room must hold the surgery type needed by the patient
                if sr.surgery_type.st_id not in self.room.surgery_types:
                    newDomain.discard(sr)
                    continue
                # Surgery date can't be after patients pre operation due date
                if sr.preOp_exp_date is not None:
                    if sr.preOp_exp_date < self.day:
                        newDomain.discard(sr)
                        continue
                else:
                    newDomain.discard(sr)
                # Surgery date can't be before pre op date
                if sr.preOp_date is not None:
                    if sr.preOp_date > self.day:
                        newDomain.discard(sr)
                        continue
                else:
                    newDomain.discard(sr)
                # status is open
                if sr.status != 1.1 and sr.status != 1.3 and sr.status != 1.2 and sr.status != 4:
                # if sr.status != 4:
                    newDomain.discard(sr)
                    continue
                # surgery date can't be before entrance date
                if sr.entrance_date > self.day:
                    newDomain.discard(sr)
                    continue
                # surgery type can only be one that the surgeons on shift are capable do preform
                if sr.surgery_type not in surgeon_st:
                    newDomain.discard(sr)
                    continue
                # surgery date can't be after deadline
                if sr.schedule_deadline is not None:
                    if sr.schedule_deadline < self.day:
                        newDomain.discard(sr)
                        continue
                # surgery date can't be before schedule_from
                if sr.schedule_from is not None:
                    if sr.schedule_from > self.day:
                        newDomain.discard(sr)
                        continue

        return newDomain

    def __str__(self):
        return "SRV: room-" + str(self.room) + " day- " + str(self.day) + " order- " + str(self.order)













