import datetime
from byRoom_Variable import Room_Variable


def calc_end_time(start_time, duration_min):
    """
    help function to calculate time objects
    :param start_time: time object including hour and minutes
    :param duration_min: duration of a process in minutes
    :return: time object of the time after the duration process
    """
    end_time_min = start_time.hour * 60 + start_time.minute + duration_min
    hour = int(end_time_min / 60)
    minute = end_time_min % 60
    if hour > 23:
        hour = 23
        minute = 59
    end_time = datetime.time(hour=hour, minute=minute)
    return end_time


class SurgeryVariable(Room_Variable):
    def __init__(self, room, day, order, start_time, end_time, constraints, domain, value):
        self.order = order
        self.start_time = start_time
        self.end_time = end_time
        super(SurgeryVariable, self).__init__(day, room, domain, constraints, value)

    def get_constraint_dro_key(self):
        dro_key = str(self.day) + '_' + str(self.room.num) + '_' + str(self.order)  # date_room_order
        return dro_key

    def init_constraints_in_variables(self, c_dict):
        """
        adds the required keys to each dictionary constraint and initializes the price value to 0. via the keys the price of
        the concerning variable will be updated the keys refer to the index of the variable.
        :param dro: key referring to date room order
        :param dr: key referring to date room
        :param d: key referring to date
        :param c_dict: dictionary of constraints of a variable type
        """
        if 'dro' in c_dict:
            dro_cons = c_dict['dro']
            for cons in dro_cons:
                dro_cons[cons].prices[self.get_constraint_dro_key()] = 0

        if 'dr' in c_dict:
            dr_cons = c_dict['dr']
            for cons in dr_cons:
                dr_cons[cons].prices[self.get_constraint_dr_key()] = 0

        if 'd' in c_dict:
            d_cons = c_dict['d']
            for cons in d_cons:
                d_key_list = self.get_init_d_key()
                for d_key in d_key_list:
                    d_cons[cons].prices[d_key] = 0

        return c_dict

    def set_surgery_time(self, start_time, duration):
        """
        sets the start time and end time of a surgery - calculates and manipulates time to define end time.
        :param duration: int - resembles the length of the surgery in minutes
        :param start_time: time object -
        """
        self.start_time = start_time
        end_time = calc_end_time(start_time, duration)
        self.end_time = end_time

    def nullify_surgery_time(self):
        """
        sets a surgery start and end time to 00:00
        """
        self.start_time = datetime.time(hour=0, minute=0)
        self.end_time = datetime.time(hour=0, minute=0)

    def initialize_domain(self, domain):
        return domain

    def get_init_d_key(self):
        d_key = []
        for v in self.domain:
            if v is None:
                continue
            d_key.append(str(self.day) + '_' + str(v.id))
        return d_key







