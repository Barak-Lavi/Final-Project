import random


def init_ward_grades(ward_id_list):
    """
    calculates a grade for each ward. each anesthetist holds a sorted list of ward's id defining the match of the ward
    to the anesthetist. Expert and Senior Anesthetist will hold a list of all ward_id while stagiaire will hold only
    the ward's they've been to rotation including their current rotation. the grade to each ward will be calculated :
    (1 - position of ward/number of wards). When Experts and sceniors grade will be devided by two inorder to encourage
    stagiaire operating. And each grade for ward in past rotations of stagiaire will be divided by three - we want this
    option to only be used if needed.
    :param ward_id_list: list of all ward id
    :return: dictionary {ward_id : grade}
    """
    ward_grades = {}
    if isinstance(ward_id_list[0], list):  # Stagiaire
        speciality = ward_id_list[0][0]  # the stagiaire current rotation
        ward_grades[speciality] = 1
        past_rotations = ward_id_list[1]
        random.shuffle(past_rotations)  # defining the match of ward and anesthetist
        if len(past_rotations) > 0:
            for j in range(len(past_rotations)):  # the stagiaire past rotations
                grade = (1 - (j / len(past_rotations))) / 3
                ward_grades[past_rotations[j]] = grade

    else:  # Expert/Senior
        random.shuffle(ward_id_list)  # to determine the match of the wards to the anesthetist
        for i in range(len(ward_id_list)):
            grade = (1 - (i / len(ward_id_list))) / 2
            ward_grades[ward_id_list[i]] = grade
    return ward_grades


class Anesthetist(object):

    def __init__(self, a_id, rank, surgical_days, ward_id_list):
        self.id = a_id
        self.rank = rank  # Senior/Expert/Stagiaire
        # self.speciality = speciality
        self.surgical_days = surgical_days  # list of surgical days - datetime objects
        self.ward_grades = init_ward_grades(ward_id_list)
        self.assigned = {'FM': [], 'RM': [], 'OA': []}  # Floor Manager, Room Manager, Operating Anesthetist

    def get_d_key(self, schedule_date):
        """
        key for d constraints
        :param schedule_date: string of date
        :return:
        """
        return schedule_date + '_' + str(self.id)
