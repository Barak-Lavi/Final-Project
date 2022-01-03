from Surgery_Variable import SurgeryVariable
from datetime import time


class SurgeryVariable_Anesthetist(SurgeryVariable):
    def __init__(self, room, day, order, domain, constraints, start_time=time(hour=0, minute=0),
                 end_time=time(hour=0, minute=0), anesthetist=None):
        self.surgery_request = None
        self.value_in_update = None
        self.need_stable = False
        super(SurgeryVariable_Anesthetist, self).__init__(room, day, order, start_time, end_time, constraints, domain,
                                                          anesthetist)

    def initialize_domain(self, domain):
        """
        Initializes the domain of the variable according to unary hard constraints
        :param domain: list: domain[0] = ward_id of ward which received the room in room allocation
        domain[1] = set of all anesthetists of the hospital
        :return: set of anesthetists available for surgery on surgery date and which can operate for this ward.
        """
        w_id = domain[0]
        new_domain = domain[1].copy()
        if len(new_domain) > 0:
            for a in domain[1]:
                # anesthetist must be available to surgery on day
                if self.day not in a.surgical_days:
                    new_domain.discard(a)
                    continue
                # rank of operating anesthetist can only be Stagiaire if the ward given the room in room allocation
                # is his current rotation or if he already was in its rotation.
                if a.rank == 'Stagiaire':
                    if a.speciality != w_id:
                        if w_id not in a.rotation:
                            new_domain.discard(a)
                            continue
        new_domain.add(None)
        return new_domain

    '''def get_init_d_key(self):
        d_key = []
        for v in self.domain:
            if v is None:
                continue
            d_key.append(str(self.day) + '_' + str(v.id))
        return d_key'''


