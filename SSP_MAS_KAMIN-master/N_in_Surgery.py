from Surgery_Variable import SurgeryVariable
from datetime import time


class SurgeryVariable_Nurse(SurgeryVariable):
    def __init__(self, room, day, order, domain, constraints, start_time=time(hour=0, minute=0),
                 end_time=time(hour=0, minute=0), nurse=None):
        self.surgery_request = None
        self.n_type = domain[0]
        self.value_in_update = None
        self.need_stable = False
        super(SurgeryVariable_Nurse, self).__init__(room, day, order, start_time, end_time, constraints,
                                                    domain, nurse)


    def initialize_domain(self, nurse_domain):
        """
        initializes the domain of the specific variable according to unary hard constraints
        :param nurse_domain:list: nurse_domain[0] - String representing the type of nurse CN/SN
                                  nurse_domain[1] - int ward id of the specific variable
                                  nurse_domain[2] set of all the nurses of the hospital
        :return: list of tuples (nurse, grade) available for surgery on surgery date and qualified for surgery
        in the ward"""

        nurse_type = nurse_domain[0]
        ward_id = nurse_domain[1]
        new_domain = nurse_domain[2].copy()
        if len(new_domain) > 0:
            for n in nurse_domain[2]:
                # nurse must be available to surgery on day
                if self.day not in n.surgical_days:
                    new_domain.discard(n)
                    continue
                # nurse must be able to perform at least a single surgery type that is done in the room and ward
                if ward_id not in n.skills[nurse_type]:
                    new_domain.discard(n)
                    continue
                else:
                    if not len(n.skills[nurse_type][ward_id].intersection(self.room.surgery_types)) > 0:
                        new_domain.discard(n)
        new_domain = self.domain_with_grades(new_domain, ward_id, self.room, nurse_type)
        new_domain.append(tuple([None, 0]))
        return new_domain

    def get_init_d_key(self):
        d_key = []
        for nurse in self.domain:
            if nurse[0] is None:
                continue
            d_key.append(str(self.day) + '_' + str(nurse[0].id))
        return d_key

    def get_constraint_dro_key(self):
        dro_key = str(self.day) + '_' + str(self.room.num) + '_' + str(self.order) + '_' + str(self.n_type)
        # date_room_order
        return dro_key

    def domain_with_grades(self, domain, w_id, r, n_type):
        """
        converts the domain to a list of tuples where the tuple is composed of a nurse and her grade depending on the
        type of nurse, ward, and room
        :param n_type: String representing the nurse type CN or SN
        :param domain: set of nurses object
        :param w_id: int ward id
        :param r: int room num
        :return: list of tuples (nurse, grade)
        """
        domain_with_grades = []
        for n in domain:
            grade = n.ward_grades[self.day][n_type][w_id][r.num]
            domain_with_grades.append((n, grade))
        return domain_with_grades







