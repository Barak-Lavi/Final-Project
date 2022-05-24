from Surgery_Request import Surgery_Request
import math
from Surgery_Type import SurgeryType
from Surgeon import Surgeon
from Unit import Unit
import pandas as pd
import requests
import datetime
import random

path = r'C:\Users\User\Desktop\Final-Project\Final-Project\ex3'


def _init_surgery_types(get_st_db):
    """
    initiates surgery type instants like those in  the DB
    :param get_st_db : list of dictionaries of surgery_type data
    :return: set of surgery types of the ward
    """
    init_st = get_st_db  # SQL command to recieve from DB returns a list
    surgery_types = set()
    for st in init_st:
        st_id = st['surgery_type_id']
        name = st['name']
        urgency = st['urgency']
        complexity = st['complexity']
        # duration = st['duration']
        duration = 30
        utility = st['utility']
        surgery_types.add(SurgeryType(st_id, name, urgency, complexity, duration, utility))
    return surgery_types


def _getShiftsDB(surgeon_id, DB):
    """
    get https request
    :param DB:  boolean - True if read from DB False if read from csv
    :param surgeon_id:
    :return: set of all the shifts ie dates of a specific surgeon
    """
    if DB:
        url = "https://www.api.orm-bgu-soroka.com/shifts_seniors/get_by_senior_id"
        params = {'senior_id': surgeon_id}
        r = requests.get(url=url, json=params)
        data = r.json()
    else:
        df = pd.read_csv(path + r'\shift_seniors.csv')
        data = df.loc[lambda x: x['senior_id'] == surgeon_id].to_dict('records')

    shifts = set()
    for s in data:
        shifts.add(datetime.datetime.strptime(s['st_time'][:10], '%Y-%m-%d').date())
    return shifts


def _getRTG_equipment(DB):
    """
    get https request
    :param DB: boolean if to read from DB or from csv
    :return: pd data frame of all the equipments of surgery requests
    """
    url = "https://www.api.orm-bgu-soroka.com/equipment_for_surgery_requests/get_all"
    if DB:
        r = requests.get(url)
        data = r.json()
        df = pd.DataFrame(data)  # todo verify this is transformed correctly
    else:
        df = pd.read_csv(path + r'\equipment_for_surgery_requests.csv')
    return df


class Ward(object):

    def __init__(self, name, w_id, hospital, day_duration, start_d_hour, preop_exp_days, DB):

        # every time RTG changes or gets updated entrance_date_cut and max attributes change and need to be computed
        # again

        self.w_id = w_id
        self.name = name
        self.d_duration = day_duration  # surgical minutes in a single day
        self.start_d_hour = start_d_hour
        self.preop_exp_days = preop_exp_days
        # self.surgery_types = self._init_surgery_types()  # set of surgery types of the ward
        self.units = self._init_units(DB)
        self.room_allocation = self._init_room_allocation(hospital, DB)
        # dictionary of wards room allocation {date: set of rooms}
        self.ward_surgeons = self._init_surgeons(DB)
        self.RTG = self._init_RTG(hospital, DB)  # need to update to the correct status - include NRTG and ect..
        self._init_entrance_date_cut()
        self.schedule_gaps = {0: 0, 7: 0.25, 30: 0.75}  # init_schedule_gaps - key - gap value - cost factor
        # weights between the constraints sum = 1
        # self.constraints_w = {"surgery_order": 1 / 6, "surgery_date": 1 / 6, "surgeon_patient": 1 / 6,
        #                       "homo_hetero": 1 / 6,
        #                       "efficiency": 1 / 6, "schedule_gap": 1 / 6}
        # # weights between the parameters of every constraint for each constraint sum =1
        # self.parameter_w = {
        #     "surgery_order": {"num_cancellations": 0.25, "complexity": 0.25, "duration": 0.25, "age": 0.25},
        #     "surgery_date": {"num_cancellations": 1 / 3, "urgency": 1 / 3, "entrance_date": 1 / 3},
        #     "surgeon_patient": {"complexity": 0.5, "skill": 0.5}}
        self.constraints_w = {"surgery_order": 0.1, "surgery_date": 0.6, "surgeon_patient": 0.1,
                              "homo_hetero": 0.1,
                              "efficiency": 0.1}
        self.parameter_w = {
            "surgery_order": {"num_cancellations": 1, "complexity": 1, "urgency": 1, "age": 1},
            "surgery_date": {"num_cancellations": 1, "urgency": 1, "entrance_date": 1},
            "surgeon_patient": {"complexity": 1, "skill": 1}}
        self.max_attributes = self._init_max_attributes(DB)
        self.min_attributes = self._init_min_attributes(DB)

        #Reinforcement Learning Agent features
        self.reward= None
        self.max_reward = 0
        self.curr_state = [1,1,1]
        self.visited_states=[[1,1,1]]
        self.available_actions_list =[]
        self.best_state = None


    def valid_action(self,action):
        valid = True
        if action in self.visited_states:
            valid= False
        for i in action:
            if i<=0 or i>=11:
                valid= False
        return valid

    def available_actions(self,curr_state):
        optional_action = curr_state.copy()
        available_actions=[]
        for i in range(len(curr_state)):
            optional_action[i] += 1
            if self.valid_action(optional_action):
                available_actions.append(optional_action.copy())
            optional_action[i] -= 2
            if self.valid_action(optional_action):
                available_actions.append(optional_action.copy())
            optional_action = curr_state.copy()
        self.available_actions_list = available_actions
        return available_actions

    def choice_random_action(self):
        random_action=[]
        while True:
            cancellations = random.randint(1,10)
            urgency = random.randint(1,10)
            entrance_date = random.randint(1,10)
            random_action =[cancellations,urgency,entrance_date]
            if self.valid_action(random_action):
                self.curr_state = random_action
                return

    def update_action(self,action):
        self.parameter_w['surgery_date']['num_cancellations'] = action[0]
        self.parameter_w['surgery_date']['urgency'] = action[1]
        self.parameter_w['surgery_date']['entrance_date'] = action[2]
        self.curr_state = action
        self.visited_states.append(action)
        if len(self.available_actions_list)>0:
            self.available_actions_list.remove(action)
        print(self.parameter_w['surgery_date'])

    def update_best_state(self,action):
        self.parameter_w['surgery_date']['num_cancellations'] = action[0]
        self.parameter_w['surgery_date']['urgency'] = action[1]
        self.parameter_w['surgery_date']['entrance_date'] = action[2]
        self.curr_state = action
        print(self.parameter_w['surgery_date'])

    '''def init_max_slots(self):
        """
        calculates the maximum number of surgeries that can be preformed in a single day in the ward - in accordance
        to the duration of the surgery types of the ward
        :return: int max_slots
        """
        min_duration_st = min(self.surgery_types, key=self._by_duration)
        max_slots = int(self.d_duration / min_duration_st.duration)
        return max_slots'''

    def _init_entrance_date_cut(self):
        """
        good for the static problem when the problem will be dynamic will need to change - according to the given RTG
        each surgery request will receive a field of entrance date cut in a way that a categorization will be made
        the cuts will be determined by percentage of 10, i.e. the first 10% of referral dates will receive the value 1
        the next 10% the value 2 and so on.  The earlier the referral date is, the larger the cut.

        """
        sorted_RTG = sorted(self.RTG, key=self._by_entrance_d, reverse=True)
        if len(sorted_RTG) < 10:
            for i in range(1, len(sorted_RTG) + 1):
                sorted_RTG[i - 1].entrance_date_cut = i
        else:
            ten_p = self.round_down(len(sorted_RTG) / 10)  # ten_p - 10%
            j = 1
            for i in range(len(sorted_RTG)):
                if i < j * ten_p:
                    sorted_RTG[i].entrance_date_cut = j
                else:
                    j += 1
                    sorted_RTG[i].entrance_date_cut = j

    @staticmethod
    def round_down(n, decimals=0):
        multiplier = 10 ** decimals
        return math.floor(n * multiplier) / multiplier

    def _init_max_attributes(self, DB):
        """
        creates dictionary of the current max parameters concerning current RTG - for parameter normalization in cost
        functions
        :param DB: boolean - True if read from DB False if read from csv
        :return: dictionary - key parameter name - value its max or min value
        """

        if len(self.RTG) > 0:
            max_attributes = {
                "cancellations": max(self.RTG, key=self._by_cancellations_num).num_cancellations,
                # the max cancellations in current RTG
                "complexity": max(self.RTG, key=self._by_importance).complexity,
                # the max complexity in current RTG
                "duration": max(self.RTG, key=self._by_duration).duration,
                # the max duration in current RTG
                "min_birth_d": min(self.RTG, key=self._by_birth_d),
                # surgery request with the youngest patient
                "max_birth_d": max(self.RTG, key=self._by_birth_d),
                #  surgery request with the eldest patient
                "urgency": max(self.RTG, key=self._by_urgency).urgency,
                # the max urgency in current RTG
                "entrance_d": min(self.RTG, key=self._by_entrance_d),
                # surgery request with earliest entrance date to ward
                "entrance_d_cut": max(self.RTG, key=self._by_entrance_d_cut).entrance_date_cut,
                # max entrance date cut in current RTG - the edc of the surgery request with earliest entrance date
                "skill": self.get_max_skills(DB)}  # dictionary {key - surgery type id : value - max skill grade}

            return max_attributes
        else:
            return {}
    def _init_min_attributes(self, DB):
        """
        creates dictionary of the current min parameters concerning current RTG - for parameter normalization in cost
        functions
        :param DB: boolean - True if read from DB False if read from csv
        :return: dictionary - key parameter name - value its max or min value
        """

        if len(self.RTG) > 0:
            min_attributes = {
                "cancellations": min(self.RTG, key=self._by_cancellations_num).num_cancellations,
                # the max cancellations in current RTG
                "complexity": min(self.RTG, key=self._by_importance).complexity,
                # the max complexity in current RTG
                "duration": min(self.RTG, key=self._by_duration).duration,
                # the max duration in current RTG
                "min_birth_d": min(self.RTG, key=self._by_birth_d),
                # surgery request with the youngest patient
                "max_birth_d": max(self.RTG, key=self._by_birth_d),
                #  surgery request with the eldest patient
                "urgency": min(self.RTG, key=self._by_urgency).urgency,
                # the max urgency in current RTG
                "entrance_d": max(self.RTG, key=self._by_entrance_d),
                # surgery request with earliest entrance date to ward
                "entrance_d_cut": min(self.RTG, key=self._by_entrance_d_cut).entrance_date_cut,
                # max entrance date cut in current RTG - the edc of the surgery request with earliest entrance date
                "skill": self.get_min_skills(DB)}  # dictionary {key - surgery type id : value - max skill grade}

            return min_attributes
        else:
            return {}

    @staticmethod
    def _by_entrance_d_cut(sr):
        return sr.entrance_date_cut

    @staticmethod
    def _by_cancellations_num(sr):
        """
        parameter key function - to determine max return method
        :param sr: surgery request
        :return: by num of cancellations
        """
        return sr.num_cancellations

    @staticmethod
    def _by_importance(sr):
        return sr.complexity

    @staticmethod
    def _by_duration(sr):
        return sr.duration

    @staticmethod
    def _by_birth_d(sr):
        return sr.patient.birth_date

    @staticmethod
    def _by_urgency(sr):
        return sr.urgency

    @staticmethod
    def _by_entrance_d(sr):
        return sr.entrance_date

    @staticmethod
    def _by_schedule_d(sr):
        return sr.schedule_date

    def get_max_skills(self, DB):
        """
        defines the max skill i.e. surgeon grade for each kind of surgery type
        :param DB: boolean - True if read from DB False if read from csv
        :return: dictionary : {key - st.id : value - max grade}
        """
        max_skills = {}
        for u in self.units:
            for st in u.surgery_types:
                st_id = st.st_id
                max_skills[st_id] = self.get_max_skill_by_st_id(st_id, DB)
        return max_skills

    def get_min_skills(self, DB):
        """
        defines the max skill i.e. surgeon grade for each kind of surgery type
        :param DB: boolean - True if read from DB False if read from csv
        :return: dictionary : {key - st.id : value - max grade}
        """
        min_skills = {}
        for u in self.units:
            for st in u.surgery_types:
                st_id = st.st_id
                min_skills[st_id] = self.get_min_skill_by_st_id(st_id, DB)
        return min_skills

    def get_max_skill_by_st_id(self, st_id, DB):
        """
        gets from DB all skills i.e. grades of certain surgery type 
        :param DB: boolean - True if read from DB False if read from csv
        :param st_id: surgery type id
        :return: max grade of skill in the ward for this surgery type
        """""
        if DB:
            url = "https://www.api.orm-bgu-soroka.com/skills_seniors/get_by_surgery_type_id"
            params = {'surgery_type_id': st_id}
            r = requests.post(url=url, json=params)
            skills = r.json()
        else:
            df = pd.read_csv(path + r'\skill_seniors.csv')
            skills = df.loc[lambda x: x['surgery_type_id'] == st_id].to_dict('records')
        if len(skills) > 0:  # only because DB is not finished delete afterwards
            max_skill = max(skills, key=self.by_skill)['skill']
        else:
            max_skill = 0
        return int(max_skill)

    def get_min_skill_by_st_id(self, st_id, DB):
        """
        gets from DB all skills i.e. grades of certain surgery type 
        :param DB: boolean - True if read from DB False if read from csv
        :param st_id: surgery type id
        :return: max grade of skill in the ward for this surgery type
        """""
        if DB:
            url = "https://www.api.orm-bgu-soroka.com/skills_seniors/get_by_surgery_type_id"
            params = {'surgery_type_id': st_id}
            r = requests.post(url=url, json=params)
            skills = r.json()
        else:
            df = pd.read_csv(path + r'\skill_seniors.csv')
            skills = df.loc[lambda x: x['surgery_type_id'] == st_id].to_dict('records')
        if len(skills) > 0:  # only because DB is not finished delete afterwards
            max_skill = min(skills, key=self.by_skill)['skill']
        else:
            min_skill = 0
        return int(max_skill)
    @staticmethod
    def by_skill(surgeon_skill):
        """
        key used in max function
        :param surgeon_skill: dictionary as read from DB {'id': _ , 'surgeon_id': _ , 'surgeryT_id' : _ , 'skill':_}
        :return: the skill of the surgeon
        """
        return int(surgeon_skill['skill'])

    def _init_room_allocation(self, hospital, DB):
        """
        initiates the room_allocation variable
        :param DB:  boolean - True if read from DB False if read from csv
        :return: dictionary of wards room allocation {date: set of rooms}
        """
        init_ra = self._getRoomAllocastionDB(DB)
        # list of dictionaries containing information of the room allocations {'id':, 'ward_id': ,'date': , 'room_number':}
        room_allocation = {}
        for ra in init_ra:
            if ra['date'] in room_allocation:
                room_allocation[ra['date']].add(hospital.findRoom(ra['room_id']))
            else:
                room_allocation[ra['date']] = {hospital.findRoom(ra['room_id'])}
        return room_allocation

    def _init_units(self, DB):
        """
        initiates units instances each unit has id , name and set of surgery_types objects
        :param DB: boolean - True if read from DB False if read from csv
        :return: set of units of ward
        """
        init_units = self._get_units_db(DB)
        units = set()
        for u in init_units:
            unit_id = u['unit_id']
            name = u['name']
            surgery_types = _init_surgery_types(self._get_st_by_unit_db(unit_id, DB))
            units.add(Unit(unit_id, name, surgery_types))
        return units

    def _init_surgeons(self, DB):
        """
        initiates the surgeons objects of the ward and aggregates them to a set
        :param DB:  boolean - True if read from DB False if read from csv
        :return: set of the ward's surgeons
        """
        init_s = self._getSurgeonsDB(DB)  # return list of surgeon id
        surgeons = set()
        for s in init_s:  # s=surgeon id
            s_shifts = _getShiftsDB(s, DB)  # return set of dates of all the shifts of s
            s_skills = self._getSkillsDB(s, DB)  # return a Dictionary {Surgery Type : grade}
            surgeons.add(Surgeon(int(s), s_skills, s_shifts))
        return surgeons

    def _init_RTG(self, hospital, DB):
        """
        initiates all the surgery requests that are rtg of the ward i.e. ready to go patients
        :param DB:  boolean - True if read from DB False if read from csv
        :return: set of surgery requests
        """

        init_RTG = self._getRTG_DB(DB)  # return list of all surgery request
        df_RTG_equipment = _getRTG_equipment(DB)
        RTG = set()
        for rtg in init_RTG:
            patient = hospital.findPatientByID(rtg['patient_id'])
            surgery_type = self.findSTbyId(rtg['surgery_type_fk'])  # surgery type object
            urgency = rtg['urgency']
            complexity = rtg['complexity']
            # duration = rtg['duration'] # TODO barak - check if need to change
            duration = 30
            if 'request_status' in rtg:
                status = float(rtg['request_status'])
            else:
                status = float(rtg['status'])
            # status = float(rtg['request_status']) when reading from DB
            request_num = rtg['request_id']
            entrance_date = datetime.datetime.strptime(rtg['entrance_date'], '%Y-%m-%d').date()
            cancellations = rtg['cancellations']
            if rtg['pre_op_date'] is not None:
                preOp_exp_date = datetime.datetime.strptime(rtg['pre_op_date'], '%Y-%m-%d').date() + \
                                 datetime.timedelta(days=self.preop_exp_days)
                preOp_date = datetime.datetime.strptime(rtg['pre_op_date'], '%Y-%m-%d').date()
            else:
                preOp_exp_date = rtg['pre_op_date']
                preOp_date = None

            if rtg['schedule_from'] is not None:
                schedule_from = datetime.datetime.strptime(rtg['schedule_from'], '%Y-%m-%d').date()
            else:
                schedule_from = rtg['schedule_from']
            if rtg['schedule_deadline'] is not None:
                schedule_deadline = datetime.datetime.strptime(rtg['schedule_deadline'], '%Y-%m-%d').date()
            else:
                schedule_deadline = rtg['schedule_deadline']
            #if rtg['specific_senior'] is not None:
             #   specific_senior = int(float(rtg['specific_senior']))
            #else:
            specific_senior = None
            if rtg['schedule_date'] is not None:
                schedule_date = datetime.datetime.strptime(rtg['schedule_date'], '%Y-%m-%d').date()
            else:
                schedule_date = rtg['schedule_date']
            equipment = df_RTG_equipment.loc[lambda x: x['surgery_request_id'] == request_num]['equipment_id'].tolist()
            RTG.add(
                Surgery_Request(surgery_type=surgery_type, urgency=urgency, complexity=complexity, duration=duration,
                                status=status, entrance_date=entrance_date, num_cancellations=cancellations,
                                patient=patient, ward_name=self.name, preOp_exp_date=preOp_exp_date,
                                preOp_date=preOp_date, id1=request_num, schedule_from=schedule_from,
                                schedule_deadline=schedule_deadline, schedule_date=schedule_date,
                                specific_senior=specific_senior, equipments=equipment))
        return RTG

    def findSTbyId(self, st_id):
        """

        :param st_id: surgery type id
        :return: object of surgery type
        """
        for u in self.units:
            for st in u.surgery_types:
                if st.st_id == st_id:
                    return st
        return None

    def _getRoomAllocastionDB(self, DB):
        """
        get http request - receives room allocation of certain ward
        :param DB:  boolean - True if read from DB False if read from csv
        :return: list of dictionaries containing information of the room allocations {'id':, 'ward_id': ,'date': , 'room_number':}
        """
        if DB:
            url = "https://www.api.orm-bgu-soroka.com/rooms_allocations/get_by_ward_id"
            params = {'ward_id': self.w_id}
            r = requests.post(url=url, json=params)
            room_allocation = r.json()
        else:
            df = pd.read_csv(path + r'\rooms_allocations.csv')
            room_allocation = df.loc[lambda x: x['ward_id'] == self.w_id].to_dict('records')
        return room_allocation

    def _get_units_db(self, DB):
        """
        get http request - units of a certain ward
        :param DB: boolean - True if read from DB False if read from csv
        :return: list of dictionaries containing information of the units {id,ward_id, name}
        """
        if DB:
            url = "https://www.api.orm-bgu-soroka.com/units/get_by_ward_id"
            params = {'ward_id': self.w_id}
            r = requests.get(url=url, json=params)
            units = r.json()
        else:
            df = pd.read_csv(path + r'\units.csv')
            units = df.loc[lambda x: x['ward_id'] == self.w_id].to_dict('records')
        return units

    def _get_st_by_unit_db(self, u_id, DB):
        """
        get http request - recieves surgery types of a certain unit
        :param DB:  boolean - True if read from DB False if read from csv
        :return:  list of dictionaries containing information of the surgery types {id,ward_id, name, urgency, complexity
        , duration}
        """
        if DB:
            url = "https://www.api.orm-bgu-soroka.com/surgeries_types/get_by_unit_id"
            params = {"unit_id": u_id}
            r = requests.post(url=url, json=params)
            surgery_types = r.json()
        else:
            df = pd.read_csv(path + r'\surgery_types.csv')
            surgery_types = df.loc[lambda x: x['unit_id'] == u_id].to_dict('records')
        return surgery_types

    def _getSurgeryTypesDB(self):
        """
        get http request - receives surgery types of a certain ward
        :return: list of dictionaries containing information of the surgery types {id,ward_id, name, urgency, complexity
        , duration}
        """
        url = "https://www.api.orm-bgu-soroka.com/surgeries_types/get_by_ward_id"
        params = {"ward_id": self.w_id}
        r = requests.get(url=url, json=params)
        surgery_types = r.json()
        return surgery_types

    def _getSurgeonsDB(self, DB):
        """
        get http request
        :param DB:  boolean - True if read from DB False if read from csv
        :return: list of ward's surgeon's id
        """
        if DB:
            url = "https://www.api.orm-bgu-soroka.com/surgeons_seniors/get_by_ward_id"
            params = {"ward_id": self.w_id}
            r = requests.get(url=url, json=params)
            surgeons_j = r.json()
        else:
            df = pd.read_csv(path + r'\surgeon_seniors.csv')
            surgeons_j = df.loc[lambda x: x['ward_id'] == self.w_id].to_dict('records')
        surgeons = []
        for s in surgeons_j:
            surgeons.append(s['senior_id'])
        return surgeons

    def _getSkillsDB(self, surgeon_id, DB):
        """
        get http request
        :param DB:  boolean - True if read from DB False if read from csv
        :param surgeon_id:
        :return: dictionary of {surgery Type object : grade} skills of specific surgeon
        """
        if DB:
            url = "https://www.api.orm-bgu-soroka.com/skills_seniors/get_by_senior_id"
            params = {'senior_id': surgeon_id}
            r = requests.get(url=url, json=params)
            data = r.json()
        else:
            df = pd.read_csv(path + r'\skill_seniors.csv')
            data = df.loc[lambda x: x['senior_id'] == surgeon_id].to_dict('records')
        skills = {}
        for s in data:
            st_id = s['surgery_type_id']
            st = self.findSTbyId(st_id)
            skills[st] = int(s['skill'])
        return skills

    def _getRTG_DB(self, DB):
        """
        :param DB  boolean - True if read from DB False if read from csv
        :return:
        """
        url = "https://www.api.orm-bgu-soroka.com/surgery_requests/get_by_surgery_type"
        sr = []
        for u in self.units:
            for st in u.surgery_types:
                if DB:
                    params = {'surgery_type': st.st_id}
                    r = requests.get(url=url, json=params)
                    data = r.json()
                else:
                    df = pd.read_csv(path + r'\surgery_requests.csv')
                    df = df.where(pd.notnull(df), None)
                    data = df.loc[lambda x: x['surgery_type_fk'] == st.st_id].to_dict('records')
                sr.extend(data)
        return sr

    def st_surgeons_day(self, day):
        """
        creates a set of the surgery types that can be performed in a certain day depending on the surgeons
        on shift of this day
        :param day: string in format 'YYYY-MM-DD'
        :return: set of surgery types objects
        """
        st = set()
        d = datetime.datetime.strptime(day, '%Y-%m-%d').date()
        for surgeon in self.ward_surgeons:
            if d in surgeon.surgical_shifts:
                s_st = surgeon.surgical_grades.keys()
                st.update(s_st)
        return st

    def st_by_id(self, st_id):
        """
        :param st_id: recieves a surgery type id
        :return: the surgery type with this id
        """
        for u in self.units:
            for st in u.surgery_types:
                if st.st_id == st_id:
                    return st

    def find_surgeon_by_id(self, surgeon_id):
        """
        locates the surgeon with the given id
        :param surgeon_id: int id
        :return: surgeon object
        """
        for s in self.ward_surgeons:
            # if s.id == int(surgeon_id):
            if s.id == surgeon_id:
                return s

    def get_unit_st(self, st):
        """
        finds the unit of a surgery type
        :param st: surgery type object
        :return: unit object
        """

        for u in self.units:
            if st in u.surgery_types:
                return u
        return None

    def max_slots_room_day(self, day, room):
        """
        calculates the max slots of surgery that can be done in a certain day by looking at the min duration of
        the surgery types in the intersection of surgery types that can be done in a certain room and the surgery
        types that can be done by the surgeons on shift on a certain day
        :param day: string representation of day 'YYYY-MM-DD'
        :param room: room object
        :return: int max slots
        """
        surgeon_st = self.st_surgeons_day(day)
        st_id_set = room.surgery_types.intersection(set(st.st_id for st in surgeon_st))
        st_set = (self.st_by_id(st_id) for st_id in st_id_set)
        min_surgery_duration_day = min(st_set, key=lambda st: st.duration).duration
        max_slots = self.d_duration // min_surgery_duration_day
        return max_slots, min_surgery_duration_day

    def get_num_surgery_types(self):
        """
        counts the number of surgery types in the ward
        :return: int
        """
        num_st = 0
        for u in self.units:
            num_st += len(u.surgery_types)
        return num_st

    def find_surgery_request(self, request_num):
        """
        searches for surgery request of ward with specific request num
        :param request_num: int
        :return: surgery request object with request num
        """
        for sr in self.RTG:
            if sr.request_num == request_num:
                return sr

