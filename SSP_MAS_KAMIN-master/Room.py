import SSP_Enum, datetime


class Room(object):

    def __init__(self, num, surgery_types):
        self.num = num
        self.surgery_types = surgery_types  # set of id of the s.t.

    def get_surgery_types(self):
        return self.surgery_types

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.num == other.num

    def __hash__(self):
        return self.num

    def __str__(self):
        return str(self.num)


