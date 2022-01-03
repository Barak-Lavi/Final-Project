from datetime import datetime


class Nurse (object):

    def __init__(self, n_id, surgical_days, skills, hospital):
        self.id = n_id
        self.surgical_days = surgical_days  # list of dates
        self.skills = skills
        # dictionary {'CN' : {ward_id: (st1_id..stn_id), w2: (st1..stn)...},
        # 'SN': {ward_id: (st1_id..stn_id), w2: (st1_id..stn_id)...}
        # each collection of surgery type is a set
        self.ward_grades = self.init_ward_grades(hospital)

    def init_ward_grades(self, hospital):
        """
        each ward and room will have a numeric grade that represents the match of the nurse to the ward and room -
        a different grade will be given to CN and SN. the grade will be calculated as the ratio of the number of
        surgery types the nurse is certified to do in a certain room and ward from the total number of surgery
        types that can be done in a certain room of a certain ward in a certain day
        (taking into account the surgeons on shift)
        :param hospital: hospital object
        :return: dictionary {day: {'CN': {ward_id:{r1: grade, r2 :grade...}, 'SN' : {ward_id:{r1: grade, r2 :grade...}}
        """
        ward_d_grades = {}
        for w in hospital.wards:
            for d in w.room_allocation:
                d1 = datetime.strptime(d, '%Y-%m-%d').date()
                if d1 not in ward_d_grades:
                    ward_d_grades[d1] = {'CN': {}, 'SN': {}}
                w_st = w.st_surgeons_day(d)  # set of surgery types objects that can be preformed in day d - depending
                # on surgeons shifts
                for n_type in ward_d_grades[d1]:
                    ward_d_grades[d1][n_type][w.w_id] = {}
                    for r in hospital.rooms:
                        if w.w_id in self.skills[n_type]:
                            ward_d_grades[d1][n_type][w.w_id][r.num] = self.calc_ward_room_grade(n_type, w.w_id, w_st, r)
        return ward_d_grades

    def calc_ward_room_grade(self, n_type, ward_id, w_st, room):
        """
        each ward and room will receive a grade representing the ratio match to the ward - higher grade means better
        match. The grade will be calculated as the ratio of the number of surgery types the nurse is certified to do in
        a certain room and ward from the total number of surgery types that can be done in a certain room of a
        certain ward in a certain day
        :param room: room object
        :param n_type: nurse type String 'CN' or 'SN' - for circulating nurse and scrubbing nurse
        :param ward_id: int
        :param w_st: set of surgery types objects that can be preformed in day d - depending on surgeons shifts
        :return: float between 0-1
        """
        w_st_id = set()
        for st in w_st:
            w_st_id.add(st.st_id)
        # all surgery type id that can be done on a certain day and room of a certain ward
        w_d_r_st = room.surgery_types.intersection(w_st_id)  # ward day room surgery type
        if len(w_d_r_st) == 0:
            return 0
        n_w_r_st = w_d_r_st.intersection(self.skills[n_type][ward_id])  # nurse ward room surgery type
        return len(n_w_r_st)/len(w_d_r_st)






