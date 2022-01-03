import Surgery_Type, SSP_Enum
from datetime import datetime

class Surgery_Request(object):

    def __init__(self, surgery_type, urgency, complexity, duration, status, entrance_date, num_cancellations, patient,
                 ward_name, preOp_exp_date, preOp_date, id1, schedule_from, schedule_deadline, schedule_date,
                 specific_senior, equipments):
        self.request_num = id1
        self.surgery_type = surgery_type
        self.urgency = urgency
        self.complexity = complexity
        self.duration = duration
        self.status = status
        self.entrance_date = entrance_date
        self.entrance_date_cut = None
        self.num_cancellations = num_cancellations
        self.patient = patient
        self.ward_name = ward_name
        self.preOp_exp_date = preOp_exp_date
        self.preOp_date = preOp_date
        self.assigned = set()  # number of variables the algorithm assigned this surgery - list of dro key
        self.schedule_from = schedule_from
        self.schedule_deadline = schedule_deadline
        self.schedule_date = schedule_date
        self.specific_senior = specific_senior
        self.equipments = equipments  # list of e_id

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.patient == other.patient and (
               self.request_num == other.request_num)

    def __hash__(self):
        return int(self.request_num)

    def __str__(self):
        return self.request_num

    def age_cut(self, d):
        """
        categorization on age
        :param d: a certain date which the age of the patient will be determined by it
        :return: age cut i.e int between 1-12
        """
        age = self.patient.get_age(d)
        if age <= 3 or age > 80:
            return int(12)
        if (age > 3 or age <= 5) or (age > 70 or age <= 80):
            return int(10)
        if (age > 5 or age <= 10) or (age > 60 or age <= 70):
            return int(5)
        if (age > 10 or age <= 15) or (age > 50 or age <= 60):
            return int(3)
        if (age > 15 or age <= 20) or (age > 40 or age <= 50):
            return int(2)
        if age > 20 or age <= 40:
            return int(1)

    def calc_waiting_days(self, d):
        """
        calculates the waiting time of the surgery request to a certain date
        :param d: the date to which the waiting time is calculated
        :return: the difference in days
        """
        # date_d = datetime.strptime(d, '%Y-%m-%d').date()
        # date_entrance_date = datetime.strptime(self.entrance_date, '%Y-%m-%d').date()
        if d > self.entrance_date:
            return (d-self.entrance_date).days
        else:
            return 0

