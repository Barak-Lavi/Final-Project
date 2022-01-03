import SSP_Enum, datetime, Surgery_Type


class Surgeon(object):

    def __init__(self, s_id, surgical_grades, surgical_shifts):
        self.id = s_id
        self.surgical_grades = surgical_grades  # dictionary of {surgery Type object : grade}
        self.surgical_shifts = surgical_shifts

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.id == other.id

    def get_surgery_types_id(self):
        st_id_set = set()
        for st in self.surgical_grades:
            st_id_set.add(st.st_id)
        return st_id_set

    def __hash__(self):
        return self.id
