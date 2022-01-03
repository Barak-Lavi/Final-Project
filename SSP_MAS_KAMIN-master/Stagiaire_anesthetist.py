from Anesthetist import Anesthetist


class Stag_Anesth(Anesthetist):

    def __init__(self, a_id, rank, speciality, surgical_days, rotation, ward_id_list, st_id_dict):
        self.rotation = rotation
        self.speciality = speciality
        self.skills = st_id_dict  # {wid: [st1_id, st2_id..],...}
        super(Stag_Anesth, self).__init__(a_id, rank, surgical_days, self.update_ward_id_list(ward_id_list))

    def update_ward_id_list(self, ward_id_list):
        """
        updates the list to only include wards in which the stagiaire was in rotation
        :param ward_id_list: list of all w_id of all wards in the hospital
        :return: list of two lists - the first will only include the ward of the current rotation of the stagiaire, the
        second will include all the wards the stagiaire was already in rotation
        """
        w_id_list = [[], []]
        for w_id in ward_id_list:
            if w_id in self.rotation:
                w_id_list[1].append(w_id)
                continue
            if w_id == self.speciality:
                w_id_list[0].append(w_id)
        return w_id_list

