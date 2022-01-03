from abc import ABC, abstractmethod
import SSP_Enum
from datetime import datetime


class Patient(object):

    def __init__(self, p_id, birth_date, gender):
        self.p_id = p_id
        self.birth_date = birth_date
        self.gender = gender

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.p_id == other.p_id

    def __hash__(self):
        return self.p_id

    def get_age(self, d):
        """
        calculates age in a given date
        :param d: given date
        :return: age of patient
        """
        # date_d = datetime.strptime(d, '%Y-%m-%d').date()  # converts string to date object
        date_b_date = datetime.strptime(self.birth_date, '%Y-%m-%d').date()
        age = d.year - date_b_date.year - ((d.month, d.day) < (date_b_date.month, date_b_date.day))
        return age
