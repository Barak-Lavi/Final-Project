from Ward import Ward
from Patient import Patient
from Room import Room
from Nurse import Nurse
from Stagiaire_anesthetist import Stag_Anesth
from Anesthetist import Anesthetist
from Equipment import Equipment
import requests
import datetime
import pandas as pd
import random
path = r'C:\Users\User\Desktop\final project\ex2'
# random.seed(4)


def _get_nurse_cn_skills_DB(DB, n_id):
    """
    creates dictionary of all the surgery type nurse n_id can preform as circulating nurse
    :param DB: boolean - True if read from DB False if read from csv
    :param n_id: int nurse id
    :return: cn_skills dictionary - {ward_id: (st1..stn), w2: (st1..stn)...} each surgery type collection is a set
    """
    if DB:
        url = 'https://www.api.orm-bgu-soroka.com/nurses_cn_skills/get_by_nurse_id'
        params = {"nurse_id": n_id}
        r = requests.get(url=url, json=params)
        data = r.json()
    else:
        df = pd.read_csv(path + r'\nurses_cn_skills.csv')
        data = df.loc[lambda x: x['nurse_id'] == n_id].to_dict('records')

    cn_skills = {}
    for d in data:
        if d['ward_id'] not in cn_skills:
            cn_skills[d['ward_id']] = set()
        cn_skills[d['ward_id']].add(d['surgery_type_id'])
    return cn_skills


def _getWards_DB(DB):
    """

    :param DB: boolean - True if read from DB False if read from csv
    :return:
    """
    if DB:
        URL = " https://www.api.orm-bgu-soroka.com/wards/get_all"
        r = requests.get(url=URL)
        wards = r.json()
    else:
        df = pd.read_csv(path + r'\wards.csv')
        wards = df.to_dict('records')
    return wards


def _get_nurses_DB(DB):
    """
    :param DB: boolean - True if read from DB False if read from csv
    :return: n_id - nurse id - list of all nurses id
    """

    if DB:
        n_id = []
        url = 'https://www.api.orm-bgu-soroka.com/nurses/get_all'
        r = requests.get(url=url)
        nurses = r.json()
        for n in nurses:
            n_id.append(n['nurse_id'])
    else:
        df = pd.read_csv(path + r'\nurses.csv')
        n_id = df['nurse_id'].tolist()
    return n_id


def _getRoomsDB(DB):
    '''
    http get request
    :param DB boolean - True if read from DB False if read from csv
    :return: dictionary {room num: set(surgery type id)}
    '''
    if DB:
        url = " https://www.api.orm-bgu-soroka.com/rooms/get_all"
        r = requests.get(url=url)
        data = r.json()
    else:
        df = pd.read_csv(path + r'\rooms.csv')
        data = df.to_dict('records')
    rooms = {}
    for r in data:
        if r['room_id'] in rooms:
            rooms[r['room_id']].add(r['surgery_type_id'])
        else:
            rooms[r['room_id']] = {r['surgery_type_id']}
    return rooms


def _getPatients_DB(DB):
    """
    :param DB boolean - True if read from DB False if read from csv
    :return: dictionary of patients data
    """
    if DB:
        URL = "https://www.api.orm-bgu-soroka.com/patients/get_all"
        r = requests.get(url=URL)
        patients = r.json()
    else:
        df = pd.read_csv(path + r'\patients.csv')
        patients = df.to_dict('records')
    return patients


def _get_nurse_surgical_days_DB(DB, n_id):
    """
    :param DB: boolean - True if read from DB False if read from csv
    :param n_id: int nurse id
    :return: sd list of the surgical days of nurse n_id
    """
    if DB:
        url = 'https://www.api.orm-bgu-soroka.com/nurses_surgical_days/get_by_nurse_id'
        params = {"nurse_id": n_id}
        r = requests.get(url=url, json=params)
        data = r.json()

    else:
        df = pd.read_csv(path + r'\nurses_surgical_days.csv')
        data = df.loc[lambda x: x['nurse_id'] == n_id].to_dict('records')
    sd = []
    for d in data:
        sd.append(datetime.datetime.strptime(d['date'], '%Y-%m-%d').date())
    return sd


def _get_nurse_sn_skills_DB(DB, n_id):
    """
    creates dictionary of all the surgery type nurse n_id can preform as scrubbing nurse
    :param DB: boolean - True if read from DB False if read from csv
    :param n_id: int nurse id
    :return: sn_skills dictionary - {ward_id: [st1..stn], w2: [st1..stn]...}
    """
    if DB:
        url = 'https://www.api.orm-bgu-soroka.com/nurses_sn_skills/get_by_nurse_id'
        params = {"nurse_id": n_id}
        r = requests.get(url=url, json=params)
        data = r.json()
    else:
        df = pd.read_csv(path + r'\nurses_sn_skills.csv')
        data = df.loc[lambda x: x['nurse_id'] == n_id].to_dict('records')

    sn_skills = {}
    for d in data:
        if d['ward_id'] not in sn_skills:
            sn_skills[d['ward_id']] = set()
        sn_skills[d['ward_id']].add(d['surgery_type_id'])
    return sn_skills


def _init_rooms(DB):
    """
    receives the rooms data from DB transform to objects
    :param DB: boolean - True if read from DB False if read from csv
    :return: set of rooms
    """
    init_rooms = _getRoomsDB(DB)  # dictionary {room num: set(surgery type id)}
    rooms = set()
    for r in init_rooms:
        room_num = r
        surgery_types = init_rooms[r]
        rooms.add(Room(num=room_num, surgery_types=surgery_types))
    return rooms


def _init_patients(DB):
    """
    init patients objects and returns them as field of class
    :param DB: boolean - True if read from DB False if read from csv
    :return: set of patients object
    """
    init_patients = _getPatients_DB(DB)
    patients = set()
    for patient in init_patients:
        p_id = int(patient['patient_id'])
        birth_date = patient['date_of_birth']
        gender = patient['gender']
        patients.add(Patient(p_id, birth_date, gender))
    return patients


def  _get_anesthetist_DB(DB):
    """
    :param DB: boolean - True if read from DB False if read from csv
    :return: dictionary with all the records data
    """
    if DB:
        url = 'https://www.api.orm-bgu-soroka.com/anesthetists/get_all'
        r = requests.get(url)
        anesthetists = r.json()
    else:
        df = pd.read_csv(path + r'\anesthetists.csv')
        df = df.where(pd.notnull(df), None)
        anesthetists = df.to_dict('records')
    return anesthetists


def _get_anesthetist_surgical_days_DB(DB, a_id):
    """
    :param DB: boolean - True if read from DB False if read from csv
    :param a_id: int aneshtetist id
    :return: list of surgical days
    """
    if DB:
        url = 'https://www.api.orm-bgu-soroka.com/anesthetists_surgical_days/get_by_anesthetist_id'
        params = {"anesthetist_id": a_id}
        r = requests.get(url=url, json=params)
        data = r.json()

    else:
        df = pd.read_csv(path + r'\anesthetists_surgical_days.csv')
        data = df.loc[lambda x: x['anesthetist_id'] == a_id].to_dict('records')
    surgical_days = []
    for d in data:
        surgical_days.append(datetime.datetime.strptime(d['date'], '%Y-%m-%d').date())
    return surgical_days


def _get_anesthetist_stagiaire_rotation_DB(DB, a_id):
    """
    :param DB: boolean - True if read from DB False if read from csv
    :param a_id: int anesthetist id
    :return: list of wards the stagiaire was already in rotation
    """
    if DB:
        url = 'https://www.api.orm-bgu-soroka.com/stagiaire_rotations/get_by_anesthetist_id'
        params = {"anesthetist_id": a_id}
        r = requests.get(url=url, json=params)
        data = r.json()
        rotations = []
        for d in data:
            rotations.append(d['ward_id'])
    else:
        df = pd.read_csv(path + r'\stagiaire_rotations.csv')
        rotations = df.loc[lambda x: x['anesthetist_id'] == a_id]['ward_id'].tolist()
    return rotations


def _get_equipment_DB(DB):
    """
    :param DB: boolean - True if read from DB False if read from csv
    :return:
    """
    if DB:
        url = 'https://www.api.orm-bgu-soroka.com/equipments/get_all'
        r = requests.get(url)
        data = r.json()
    else:
        df = pd.read_csv(path + r'\equipments.csv')
        data = df.to_dict('records')
    return data


def _init_equipment(DB):
    """
    :param DB:boolean - True if read from DB False if read from csv
    :return: set of all the equipment types in the hospital
    """
    data = _get_equipment_DB(DB)
    equipment = set()
    for d in data:
        e_id = d['equipment_id']
        units = d['max_in_hospital']
        equipment.add(Equipment(e_id, units))
    return equipment


def by_day_duration(ward):
    return ward.d_duration


def by_start_d_hour(ward):
    return ward.start_d_hour


def _get_anesthetist_stagiaire_skills(DB, a_id):
    data = {}
    if DB:
        pass
    else:
        df = pd.read_csv(path + r'\stag_skills.csv')
        if 'anesthetist_id' in df.columns:
            data = df.loc[lambda x: x['anesthetist_id'] == a_id].to_dict('records')
        else:
            data = df.loc[lambda  x: x['stag_id'] == a_id].to_dict('records')





    stag_skills = {}
    for d in data:
        if d['ward_id'] not in stag_skills:
            stag_skills[d['ward_id']] = set()
        stag_skills[d['ward_id']].add(d['surgery_type_id'])
    return stag_skills


class Hospital(object):

    def __init__(self, h_id, name, DB):
        """
        :param DB: boolean - true if records are read from DB false if read from csv as experiment
        """
        self.h_id = h_id
        self.name = name
        self.rooms = _init_rooms(DB)  # set
        self.patients = _init_patients(DB)
        self.wards = self._init_wards(DB)
        self.nurses = self._init_nurses(DB)  # set
        self.anesthetists = self._init_anesthetists(DB)  # set
        self.equipment = _init_equipment(DB)
        # self.ward_strategy_grades = {0: 0.1, 1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.1, 6: 0.1, 7: 0.1, 8: 0.1, 9: 0.1}
        #self.ward_strategy_grades = {0: 0.5, 1: 0.3, 2: 0.2}
        self.ward_strategy_grades = {0: 0.5, 1: 0.5}

    def _init_wards(self, DB):
        """
        :param DB: boolean - True if read from DB False if read from csv
        :return:
        """

        initWards = _getWards_DB(DB)  # list of <id. name> of the wards as in DB.
        wards = set()
        for ward in initWards:
            w_id = ward['ward_id']
            name = ward['name']
            d_duration = ward['day_duration']
            # start_d_hour = datetime.strptime(ward['day_start_time'][:5], '%H%M').time()
            start_d_hour = datetime.time(*map(int, ward['day_start_time'][:5].split(':')))
            preop_exp_days = ward['pre_op_expiration_date']
            wards.add(Ward(name, w_id, self, d_duration, start_d_hour, preop_exp_days, DB))
        return wards

    def findPatientByID(self, p_id):
        for p in self.patients:
            if p.p_id == int(p_id):
                return p
        return None

    def findRoom(self, room_num):
        for r in self.rooms:
            if r.num == room_num:
                return r
        return None

    def find_ward(self, w_id):
        for w in self.wards:
            if w.w_id == w_id:
                return w

    def _init_nurses(self, DB):
        """
        :param DB: boolean - True if read from DB False if read from csv
        :return:
        """
        nurses = set()
        nurse_id = _get_nurses_DB(DB)  # list of nurses id
        for n_id in nurse_id:
            surgical_days = _get_nurse_surgical_days_DB(DB, n_id)  # list of dates
            skills = {}
            skills['CN'] = _get_nurse_cn_skills_DB(DB, n_id)  # {ward_id: [st1..stn], w2: [st1..stn]...}
            skills['SN'] = _get_nurse_sn_skills_DB(DB, n_id)  # {ward_id: [st1..stn], w2: [st1..stn]...}
            nurses.add(Nurse(n_id, surgical_days, skills, self))
        return nurses

    def ward_id_list(self):
        """
        creates a list of all the ward_id in a hospital
        :return: list int of ward_id
        """
        w_id_list = []
        for w in self.wards:
            w_id_list.append(w.w_id)
        return w_id_list

    def _init_anesthetists(self, DB):
        """
        :param DB: boolean - True if read from DB False if read from csv
        :return:
        """
        anesthetists = set()
        data = _get_anesthetist_DB(DB)  # dictionary with records data
        for a in data:
            a_id = a['anesthetist_id']
            rank = a['rank']

            surgical_days = _get_anesthetist_surgical_days_DB(DB, a_id)
            if rank == 'Stagiaire':
                s_rotation = _get_anesthetist_stagiaire_rotation_DB(DB, a_id)  # list of ward id
                speciality = int(a['speciality'])
                s_st_dict = _get_anesthetist_stagiaire_skills(DB, a_id)
                anesthetists.add(Stag_Anesth(a_id, rank, speciality, surgical_days, s_rotation, self.ward_id_list(),
                                             s_st_dict))
            else:
                anesthetists.add(Anesthetist(a_id, rank, surgical_days, self.ward_id_list()))
        return anesthetists

    def max_d_duration(self):

        return max(self.wards, key=by_day_duration).d_duration

    def get_earliest_start_hour(self):

        return min(self.wards, key=by_start_d_hour).start_d_hour










