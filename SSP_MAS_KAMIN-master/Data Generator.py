import pandas as pd
# from Data_updates import random_date
from datetime import datetime, timedelta
import random
import string
import math
import numpy as np
import requests
from scipy.stats import truncnorm

path = r'C:\Users\User\Desktop\Final-Project\Final-Project\ex4'

# if i want to write to xlsx and update the same file on different sheets
'''from openpyxl import load_workbook
book = load_workbook(path)
writer = pd.ExcelWriter(path, engine='openpyxl')
writer.book = book
writer.save()
writer.close()
'''
day_duration = 420

def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

def random_date(start, end):
    """
    This function will return a random datetime between two datetime
    objects.
    """
    delta = end - start
    int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
    random_second = random.randrange(int_delta)
    return start + timedelta(seconds=random_second)

def patients_generator():
    """
    :return: list of patient_id
    """
    gender = ['f', 'm']
    start_date = datetime.strptime('1925-01-01', '%Y-%m-%d').date()
    end_date = datetime.strptime('2020-01-01', '%Y-%m-%d').date()
    num_patients = input('how many patients?')
    data = []
    for i in range(int(num_patients)):
        g = random.choice(gender)
        b_date = str(random_date(start_date, end_date))
        data.append([i, g, b_date])
    df = pd.DataFrame(data, columns=['patient_id', 'gender', 'date_of_birth'])
    df.to_csv(path + r'\patients.csv', date_format='%Y-%m-%d')
    return df['patient_id'].tolist()


def ward_generator():
    """
    :return: list of all ward_id
    """
    # names of wards are random strings if needed in the future can choose from a group of strings
    dd = day_duration  # day duration
    day_start_time = '08:00:00'
    pre_op_exp_date = 30
    num_wards = input('how many wards?')
    data = []
    for i in range(int(num_wards)):
        name = ''.join(random.choice(string.ascii_lowercase) for i in range(8))  # random string
        data.append([i, name, dd, day_start_time, pre_op_exp_date])
    df = pd.DataFrame(data, columns=['ward_id', 'name', 'day_duration', 'day_start_time', 'pre_op_expiration_date'])
    df.to_csv(path + r'\wards.csv', date_format='%Y-%m-%d')
    return df['ward_id'].tolist()


def unit_generator(w_id):
    """
    :param w_id: list of ward_id
    :return dictionary {w_id : [unit_ids...]}
    """
    data = []
    random_generator = input('would you like division of units to wards to be generated randomly? 1 - yes 0 - No')
    if int(random_generator):
        num_units = int(input('how many units overall? (>= # ' + str(len(w_id) * 2) + ')'))
        # each ward has at least one unit and the rest are splited randomly between the wards
        num_wards = len(w_id)
        min_units_for_single_ward = int(num_units * (2 / 3) / num_wards)
        num_units_unequally = num_units - min_units_for_single_ward * num_wards
        u = 0
        for w in range(len(w_id)):
            if w == len(w_id) - 1:
                num_extra_units_for_ward = num_units_unequally  # all that are left
            else:
                num_extra_units_for_ward = random.randint(0, num_units_unequally)
            num_units_for_w = min_units_for_single_ward + num_extra_units_for_ward
            for j in range(num_units_for_w):
                name = ''.join(random.choice(string.ascii_lowercase) for i in range(8))  # random string
                data.append([u, w, name])
                u += 1
            num_units_unequally = num_units_unequally - num_extra_units_for_ward

    else:
        u = 0
        for w in w_id:
            num_units_for_ward = int(input('how many units for ward ' + str(w) + '?'))
            for i in range(u, u + num_units_for_ward):
                name = ''.join(random.choice(string.ascii_lowercase) for i in range(8))  # random string
                data.append([u, w, name])
                u += 1
    df = pd.DataFrame(data, columns=['unit_id', 'ward_id', 'name'])
    df.to_csv(path + r'\units.csv')

    unit_dict_by_ward = {}
    for w in w_id:
        unit_dict_by_ward[w] = df.loc[lambda x: x['ward_id'] == w]['unit_id'].tolist()
    return unit_dict_by_ward


def surgery_type_generator(unit_ward_dict):
    """
    :param unit_ward_dict: {w_id : [unit_ids...]}
    :return: list of surgery type id,
            st_dict : {w_id : (st_1, st_2...), w2: (st1..stn)...} each value of the different
            dict keys is a set,
            u_st_dict: {w_id: {u_id:[st_1..st_n], u_2:[st1..stn]}...}
            st_duration_dict: {surgery type id: duration}
    """
    data = []
    random_generator = input('would you like division of st to wards to be generated randomly? 1 - yes, 0 - no')
    if int(random_generator):
        num_units = len([item for sub_l in unit_ward_dict.values() for item in sub_l])
        # at least two surgery types for unit
        num_st = int(input('how many st overall? (# >= ' + str(num_units * 3) + ')'))
        min_num_st_for_unit = int((num_st * (2 / 3)) / num_units)
        unequal_num_st = num_st - min_num_st_for_unit * num_units
        st = 0
        num_w = 0
        for w in unit_ward_dict:
            num_w += 1
            num_u = 0
            for u in unit_ward_dict[w]:
                num_u += 1
                if num_w == len(unit_ward_dict) and num_u == len(unit_ward_dict[w]):  # last unit to receive st
                    extra_num_st_for_u = unequal_num_st
                else:
                    extra_num_st_for_u = random.randint(0, unequal_num_st)
                num_st_for_u = min_num_st_for_unit + extra_num_st_for_u
                for i in range(num_st_for_u):
                    if i == num_st_for_u - 1:
                        # for every unit at least a single surgery type with duration of half an hour
                        st_record = single_record_st(st, w, u)
                        st_record[6] = 30
                        data.append(st_record)
                    else:
                        data.append(single_record_st(st, w, u))
                    st += 1
                unequal_num_st = unequal_num_st - extra_num_st_for_u

    else:
        # st to units randomly - not big issue - each unit has at least one st even if st input was less than the number of units
        st = 0
        for w in unit_ward_dict:
            # at least two surgery types for unit
            num_st_for_ward = int(
                input('how many st for ward ' + str(w) + '? (# >=' + str(len(unit_ward_dict[w]) * 3) + ')'))
            min_num_of_st_for_unit = int((num_st_for_ward * (2 / 3)) / len(unit_ward_dict[w]))
            unequal_num_st = num_st_for_ward - min_num_of_st_for_unit * len(unit_ward_dict[w])
            for u in range(len(unit_ward_dict[w])):
                if u == len(unit_ward_dict[w]) - 1:
                    num_st_for_unit = min_num_of_st_for_unit + unequal_num_st
                else:
                    extra_st_for_u = random.randint(0, unequal_num_st)
                    num_st_for_unit = min_num_of_st_for_unit + extra_st_for_u
                    unequal_num_st = unequal_num_st - extra_st_for_u
                for i in range(st, st + num_st_for_unit):
                    data.append(single_record_st(st, w, unit_ward_dict[w][u]))
                    st += 1

    df = pd.DataFrame(data, columns=['surgery_type_id', 'ward_id', 'unit_id', 'name', 'urgency', 'complexity',
                                     'duration', 'utility'])
    df.to_csv(path + r'\surgery_types.csv')
    st_dict = {}
    u_st_dict = {}
    st_list = df['surgery_type_id'].tolist()
    st_duration_dict = {}
    for w in unit_ward_dict:
        st_dict[w] = set(df.loc[lambda x: x['ward_id'] == w]['surgery_type_id'].tolist())
        u_st_dict[w] = {}
        for u in unit_ward_dict[w]:
            u_st_dict[w][u] = df.loc[lambda x: (x['ward_id'] == w) & (x['unit_id'] == u)]['surgery_type_id'].tolist()
    for st in st_list:
        st_duration_dict[st] = pd.to_numeric(df.loc[lambda x: x['surgery_type_id'] == st, 'duration']).to_numpy()[0]
    return st_list, st_dict, u_st_dict, st_duration_dict


def check_for_not_allocated_st(st_id, allocated_st, num_rooms, rr_id):
    """
    compares two sets the st_id set of all surgery types id in database and allocated_st which is the set of st_id of
    the s.t. that were allocated in atleast one room - if there are st not allocated the function allocates them in
    random rooms. The function chooses for each non allocated st a group of rooms to be allocated in.
    :param st_id: list of all st id in db
    :param allocated_st: set of st id that have already been allocated to different rooms.
    :param num_rooms: the number of rooms in this problem
    :param rr_id:  the present room record id
    :return: list of lists of records [[rr_id, r_id, st_id], [rr_id, r_id, st_id]...]
    """
    data = []
    st_id_set = set(st_id)
    non_allocated_st = st_id_set.difference(allocated_st)
    if len(non_allocated_st) > 0:
        for st in non_allocated_st:
            num_rooms_st = random.randint(1, num_rooms)
            rooms = random.sample(range(num_rooms), k=num_rooms_st)
            for r in rooms:
                data.append([rr_id, r, st])
                rr_id += 1
    return data


def room_generator(st_id):
    """
    :param st_id: list of st_id
    :return: dictionary {r_id : (st_1..st_n), r2 : (st1..stn)...} the values of the keys are of set kind
    """
    data = []
    num_rooms = input('how many rooms?')
    random_generator = input('would you like allocation of st to rooms to be random? 1 - yes, 0 - no')
    actual_allocated_st = set()
    if int(random_generator):
        rr_id = 0  # room record id
        for r in range(int(num_rooms)):
            num_st_r = random.randint(1, len(st_id))  # number of surgery types in room r
            st_r = random.sample(st_id, k=num_st_r)  # set of random surgery types id for room
            actual_allocated_st.update(st_r)
            for st in st_r:
                data.append([rr_id, r, st])  # id, room_id, surgery_type_id - room record
                rr_id += 1
        new_data = check_for_not_allocated_st(st_id, actual_allocated_st, int(num_rooms), rr_id)  # list of new records
        if len(new_data) > 0:
            data.extend(new_data)
    else:
        rr_id = 0
        for r in range(int(num_rooms)):
            num_st_r = input('how many st would you like there to be in room ' + str(r) + '?')
            random_st = input('would you like the st to be chosen randomly? 1- yes 0 - no')
            # now it is programmed in a way that if a st wasn't chosen randomly then st will be added to random rooms
            # in a way that all s.t. will be allocated to at least one room. (check_for_not_allocated_st) can
            # be changed in the future if we will see it is necessary.
            if int(random_st):
                st_r = random.sample(st_id, k=int(num_st_r))  # set of random surgery types id for room
                actual_allocated_st.update(st_r)
                for st in st_r:
                    data.append([rr_id, r, st])  # id, room_id, surgery_type_id - room record
                    rr_id += 1
            else:
                for i in range(1, int(num_st_r) + 1):
                    st = input('please insert number ' + str(i) + ' st_id for room ' + str(r) +
                               ': (make sure each st_id is entered once for each room) if not all s_t id will be'
                               'entered than they will be allocated to rooms randomly')
                    actual_allocated_st.add(int(st))
                    data.append([rr_id, r, st])  # id, room_id, surgery_type_id - room record
                    rr_id += 1
        new_data = check_for_not_allocated_st(st_id, actual_allocated_st, int(num_rooms), rr_id)
        if len(new_data) > 0:
            data.extend(new_data)
    df = pd.DataFrame(data, columns=['id', 'room_id', 'surgery_type_id'])
    df.to_csv(path + r'\rooms.csv')
    r_dict = {}
    for r in range(int(num_rooms)):
        r_dict[r] = set(df.loc[lambda x: x['room_id'] == r]['surgery_type_id'].tolist())
    return r_dict


def rooms_allocations_generator(r_dict, w_id):
    """
    :param w_id: list of w_id
    :param r_dict: dictionary - {r_id : [w1 , w2 ... wn], r_2 : [w1...]...}
    :return dictionary {w1 : {date : [room_ids]...}, date object: first allocation date, int - number of days
    allocation was preformed
    """
    data = []
    wa = set()  # ward in allocation
    a_date = datetime.strptime \
        (input('First date of allocation: please enter in format of YYYY-MM-DD'), '%Y-%m-%d').date()
    days = input('For how many days to generate room_allocation:')
    random_generator = input('would you like the allocations to be done randomly? 1- yes , 0 - no')
    a_id = 0  # allocation id
    for i in range(int(days)):
        for r in r_dict:
            if int(random_generator):
                if len(set(r_dict[r]).difference(wa)) > 0:
                    w = random.choice(list(set(r_dict[r]).difference(wa)))
                else:
                    w = random.choice(r_dict[r])
            else:
                w = int(input(
                    'Allocation Date: ' + str(a_date) + ': Choose Ward for room ' + str(r) + ':' + str(r_dict[r])))
            data.append([a_id, w, str(a_date), r])
            wa.add(w)
            a_id += 1
        a_date = a_date + timedelta(days=1)
    df = pd.DataFrame(data, columns=['allocation_id', 'ward_id', 'date', 'room_id'])
    w_ra_dict = {}
    for w in w_id:
        a_date = a_date - timedelta(days=int(days))
        w_ra_dict[w] = {}
        for i in range(int(days)):
            w_ra_dict[w][str(a_date)] = df.loc[lambda x: (x['ward_id'] == w) & (x['date'] == str(a_date))][
                'room_id'].tolist()
            a_date = a_date + timedelta(days=1)
    df.to_csv(path + r'\rooms_allocations.csv', date_format='%Y-%m-%d')
    return w_ra_dict, a_date + timedelta(days=int(days)), int(days)


def single_record_st(st, w, u):
    name = ''.join(random.choice(string.ascii_lowercase) for i in range(8))  # random string
    # urgency = random.randint(1, 6)
    # urgency = random.choices([1, 2, 3, 4, 5, 6], weights=[10, 20, 30, 30, 20, 10], k=1)[0]
    urgency_dist = get_truncated_normal(mean=3.5, sd=1, low=1, upp=6)
    urgency = urgency_dist.rvs()
    # complexity = random.randint(1, 6)
    complexity = random.choices([1, 2, 3, 4, 5, 6], weights=[10, 20, 30, 30, 20, 10], k=1)[0]
    # durations only multiple of 30
    duration_range = list(range(30, day_duration + 30, 30))
    duration = random.choice(duration_range)
    # duration = random.randint(30, day_duration)
    utility = random.randint(150, 800)
    return [st, w, u, name, urgency, complexity, duration, utility]


def create_room_ward_dict(r_st_dict, w_st_dict):
    """
    :param r_st_dict: dictionary {r_id : (st_1..st_n), r2 : (st1..stn)...} the values of the keys are of set kind
    :param w_st_dict: dictionary {w_id : (st_1, st_2...), w2: (st1..stn)...} each value of the different
    dict keys is a set
    :return: dictionary {r_id: [w1, w2..wn], r2: [w2,w4..wn]...}
    """
    rw_dict = {}
    for r in r_st_dict:
        rw_dict[r] = []
        for w in w_st_dict:
            if len(r_st_dict[r].intersection(w_st_dict[w])):
                rw_dict[r].append(w)
    return rw_dict


def surgeon_generator(w_st_dict, w_ra_dict):
    """
    :param w_ra_dict: dictionary {w1 : {date : [room_ids]...}...}
    :param w_st_dict: dictionary {w_id : (st_1, st_2...), w2: (st1..stn)...} each value of the different
    dict keys is a set
    :return: dictionary {w_id: [s_id...], ...}
    """
    data = []
    s_ids = []
    random_generator = input('number of surgeons for ward decided automatically? 1 - yes, 0 - no')

    for w in w_st_dict:
        if int(random_generator):
            # number of surgeons for ward between the largest number of rooms given in a day in allocation and number
            # of st in ward * 3
            num_s_w = random.randint(len(max(w_ra_dict[w].values(), key=by_length)), len(w_st_dict[w]) * 3)
        else:
            num_s_w = int(input('how many surgeons for ward: ' + str(w) +
                                '? (At least: ' + str(len(max(w_ra_dict[w].values(), key=by_length))) + ')'))
        for s in range(num_s_w):
            while True:
                s_id = random.randint(100_000_000, 999_999_999)
                if s_id not in s_ids:
                    break
            s_ids.append(s_id)
            data.append([s_id, w])
    df = pd.DataFrame(data, columns=['senior_id', 'ward_id'])
    df.to_csv(path + r'\surgeon_seniors.csv')

    surgeons = {}
    for w in w_st_dict:
        surgeons[w] = df.loc[lambda x: x['ward_id'] == w]['senior_id'].tolist()
    return surgeons


def by_length(li):
    return len(li)


def skill_seniors_generator(w_s_dict, w_st_dict):
    """
    :param w_s_dict: dictionary {w_id: [s_id...], ...}
    :param w_st_dict: dictionary {w_id : (st_1, st_2...), w2: (st1..stn)...} each value of the different
    dict keys is a set
    :return: dictionary {w_id: {s_id: {st:skill, st:skill},...}}, dictionary: {w_id: {st1 : [surgeon1, surgeon2...]...}
    """
    data = []
    skill_id = 0
    # for each st at least one surgeon with skill 6
    ss = {}  # {w_id: {s_id: {st:skill, st:skill},...}}
    for w in w_st_dict:
        ss[w] = {}
        for st in w_st_dict[w]:
            s_id = random.choice(w_s_dict[w])
            data.append([s_id, st, 6, skill_id])
            skill_id += 1
            if s_id not in ss[w]:
                ss[w][s_id] = {}
                ss[w][s_id][st] = 6
            else:
                ss[w][s_id][st] = 6
    for w in w_s_dict:
        for s in w_s_dict[w]:
            # group of surgery types for each surgeon between 1 and number of st
            s_st = random.sample(w_st_dict[w], k=random.randint(1, len(w_st_dict[w])))
            for st in s_st:
                if (s in ss[w] and st not in ss[w][s]) or (s not in ss[w]):
                    skill = random.randint(1, 6)
                    data.append([s, st, skill, skill_id])
                    skill_id += 1
                    if s not in ss[w]:
                        ss[w][s] = {}
                        ss[w][s][st] = skill
                    else:
                        ss[w][s][st] = skill
    df = pd.DataFrame(data, columns=['senior_id', 'surgery_type_id', 'skill', 'skill_id'])
    df.to_csv(path + r'\skill_seniors.csv')
    st_surgeon_dict = {}
    for w in w_st_dict:
        st_surgeon_dict[w] = {}
        for st in w_st_dict[w]:
            st_surgeon_dict[w][st] = df.loc[lambda x: x['surgery_type_id'] == st]['senior_id'].tolist()
    return ss, st_surgeon_dict


def shift_senior_generator(ss_dict, w_ra_dict, r_st_dict, w_st_dict, st_s_dict):
    """
    :param ss_dict: dictionary {ward_id: {surgeon_id: {surgery type:skill<1-6>, st:skill},...}}
    :param w_ra_dict: dictionary {w1 : {date : [room_ids]...}
    :param r_st_dict: dictionary {r_id : (st_1..st_n), r2 : (st1..stn)...} the values of the keys are of set kind
    :param w_st_dict: dictionary {w_id : (st_1, st_2...), w2: (st1..stn)...} each value of the different
    dict keys is a set
    :param st_s_dict: dictionary: {w_id: {st1 : [surgeon1, surgeon2...]...}
    :return: dictionary {w_id : {day: [surgeon_id....], day_2 : [s1...sn]}...}
    """

    data = []
    # at least one high skilled surgeon for every st that can be preformed in rooms received in room allocation
    shift_id = 0
    shift_dict = {}
    for w in w_ra_dict:
        shift_dict[w] = {}
        for d in w_ra_dict[w]:
            shift_dict[w][d] = []
            for r in w_ra_dict[w][d]:
                for st in r_st_dict[r].intersection(w_st_dict[w]):
                    s_id = high_skilled_senior(ss_dict, w, st)
                    if s_id not in shift_dict[w][d]:
                        data.append([shift_id, s_id, d])
                        shift_id += 1
                        shift_dict[w][d].append(s_id)
    # add an addition of 1 surgeon per st for room received in room allocation
    for w in w_ra_dict:
        for d in w_ra_dict[w]:
            for r in w_ra_dict[w][d]:
                for st in r_st_dict[r].intersection(w_st_dict[w]):
                    while True:
                        s_id = random.choice(st_s_dict[w][st])
                        if s_id not in shift_dict[w][d]:
                            data.append([shift_id, s_id, d])
                            shift_id += 1
                            shift_dict[w][d].append(s_id)
                            break
                        if set(st_s_dict[w][st]).issubset(set(shift_dict[w][d])):
                            break
    df = pd.DataFrame(data, columns=['shift_id', 'senior_id', 'st_time'])
    df.to_csv(path + r'\shift_seniors.csv', date_format='%Y-%m-%d')
    return shift_dict


def high_skilled_senior(ss_dict, w, st):
    """
    finds a high skilled i.e. skill = 6 senior for a certain surgery type
    :param ss_dict: dictionary {w_id: {s_id: {st:skill, st:skill},...}}
    :param w: ward id of the st ward
    :param st: surgery type id
    :return: senior id
    """
    for s in ss_dict[w]:
        if st in ss_dict[w][s]:
            if ss_dict[w][s][st] == 6:
                return s


def surgery_request_generator(p_id, w_ra_dict, r_st_dict, w_st_dict, sh_dict, ss_dict, u_st_dict, st_d_dict, a_date,
                              a_days, w_st_cn_dict, n_sh_dict):
    """
    :param a_days: int number of days allocation was performed
    :param a_date: date object first allocation date
    :param st_d_dict: dictionary : {surgery_type_id : duration}
    :param p_id: list of patient id
    :param w_ra_dict: dictionary {w1 : {date : [room_ids]...} ward room allocation
    :param r_st_dict: dictionary {r_id : (st_1..st_n), r2 : (st1..stn)...} the values of the keys are of set kind
    room surgery type dictionary
    :param w_st_dict: dictionary {w_id : (st_1, st_2...), w2: (st1..stn)...} each value of the different
    dict keys is a set, ward surgery type dictionary
    :param sh_dict: dictionary {w_id : {day: [surgeon_id....], day_2 : [s1...sn]}...} shift dictionary
    :param ss_dict: dictionary {ward_id: {surgeon_id: {surgery type:skill, st:skill},...}}} surgeon skill dictionary
    :param u_st_dict: dictionary: {ward_id: {unit_id:[surgery type_1..st_n], u_2:[st1..stn]}...} unit surgery type
    dictionary
    :return: list of surgery request id
    """

    data = []
    ss_st_dict = choose_st_for_specific_seniors(w_st_dict)  # specific senior dict {ward id : surgery type id}
    r_id = 0
    # creating surgery requests that should fill 0.8 of the room occupancy for all room allocations
    for w in w_ra_dict:
        for d in w_ra_dict[w]:
            for r in w_ra_dict[w][d]:
                # u = choose_schedule_unit(r_st_dict[r], sh_dict[w][d], ss_dict[w], u_st_dict[w])
                n_w_st_dict = convert_nurse_st_dict(
                    w_st_cn_dict)  # nurse_ward_surgery_type : {nurse_id: { ward_id : [st1..stn], ward_id: [..]}...}
                st_options = possible_st_list(r_st_dict[r], sh_dict[w][d], ss_dict[w], n_w_st_dict, w, n_sh_dict[d])
                dd = 0  # day duration
                # at least a single surgery request with duration of 0.5 hour
                st_min_d_options = [st for st in st_options if st_d_dict[st] == 30]
                sr_record, sr_d = single_record_sr(p_id, r_id, st_min_d_options, st_d_dict, d, ss_st_dict[w], ss_dict[w]
                                                   , dd)
                if sr_record:
                    data.append(sr_record)
                    dd += sr_d
                    r_id += 1
                while dd < 0.8 * day_duration:
                    sr_record, sr_d = single_record_sr(p_id, r_id, st_options, st_d_dict, d, ss_st_dict[w], ss_dict[w],
                                                       dd)
                    if sr_record:
                        data.append(sr_record)
                        dd += sr_d
                        r_id += 1
                    else:
                        break
    '''# creating an amount of half of the above random surgery requests
    for i in range(r_id // 2):'''
    for i in range(r_id * 5):
        data.append(random_single_record_sr(r_id, p_id, st_d_dict, a_date, a_days, ss_st_dict, ss_dict))
        r_id += 1

    df = pd.DataFrame(data, columns=['request_id', 'patient_id', 'surgery_type_fk', 'urgency', 'complexity', 'duration',
                                     'request_status', 'cancellations', 'entrance_date', 'specific_senior', 'pre_op_date',
                                     'schedule_date', 'schedule_from', 'schedule_deadline'])
    df.to_csv(path + r'\surgery_requests.csv', date_format='%Y-%m-%d')
    sr_id = df['request_id'].tolist()
    return sr_id


def random_single_record_sr(r_id, p_id, st_d_dict, a_date, a_days, ss_st_dict, ss_dict):
    """

    :param r_id: int request id
    :param p_id:  list of patient id
    :param st_d_dict: dictionary : {surgery_type_id : duration}
    :param a_days: int number of days allocation was performed
    :param a_date: date object first allocation date
    :param ss_dict: surgeon skill dictionary {ward_id: {surgeon_id: {surgery type:skill, st:skill},...}}}
    :param ss_st_dict: specific surgeon surgery type dictionary {ward id : surgery type id}
    :return:
    """

    patient_id = random.choice(p_id)
    surgery_type = random.choice(list(st_d_dict.keys()))
    # urgency = random.randint(1, 6)
    # urgency = random.choices([1, 2, 3, 4, 5, 6], weights=[10, 20, 30, 30, 20, 10], k=1)[0]
    urgency_dist = get_truncated_normal(mean=3.5, sd=1, low=1, upp=6)
    urgency = urgency_dist.rvs()
    # complexity = random.randint(1, 6)
    complexity = random.choices([1, 2, 3, 4, 5, 6], weights=[10, 20, 30, 30, 20, 10], k=1)[0]
    duration = st_d_dict[surgery_type]
    status = 1.1
    #TODO change cancellation to (0,10)
    # cancellations = random.randint(1, 10)
    cancellations_dist = get_truncated_normal(mean=5, sd=2, low=0, upp=10)
    cancellations = round(cancellations_dist.rvs())

    entrance_date = random_date(a_date - timedelta(weeks=52), a_date + timedelta(days=a_days))
    specific_senior = ''
    if surgery_type in ss_st_dict.values():
        for w in ss_dict:
            surgeons_shuffled = list(ss_dict[w].keys())
            random.shuffle(surgeons_shuffled)
            for s in surgeons_shuffled:
                if surgery_type in ss_dict[w][s]:
                    if ss_dict[w][s][surgery_type] >= 5:
                        specific_senior = s
                        break
    # pre_op_date = random_date(a_date - timedelta(weeks=4), a_date + timedelta(days=a_days)) - realistic scenario
    pre_op_date = random_date(a_date - timedelta(weeks=4), a_date - timedelta(days=1))  # scenario for experiments
    schedule_date = ''
    schedule_from = ''
    schedule_deadline = ''
    schedule_flag = random.randint(2, 4)
    if schedule_flag == 2:
        # schedule_from = random_date(pre_op_date, pre_op_date + timedelta(days=30))  # realistic scenario
        schedule_from = random_date(pre_op_date, a_date)  # scenario for experiments
    if schedule_flag == 3:
        # realistic scenario
        # schedule_deadline = random_date(pre_op_date + timedelta(weeks=1), pre_op_date + timedelta(days=30))
        schedule_deadline = random_date(a_date + timedelta(days=1), pre_op_date + timedelta(days=30))  # scenario for experiments
    if schedule_flag == 4:
        # schedule_from = random_date(pre_op_date, pre_op_date + timedelta(days=30))
        schedule_from = random_date(pre_op_date, a_date)
        # schedule_deadline = random_date(schedule_from + timedelta(weeks=1), schedule_from + timedelta(weeks=3))
        schedule_deadline = random_date(a_date + timedelta(days=1), pre_op_date + timedelta(days=30))
        # scenario for experiments
    return [r_id, patient_id, surgery_type, urgency, complexity, duration, status, cancellations, str(entrance_date),
            specific_senior, str(pre_op_date), str(schedule_date), str(schedule_from), str(schedule_deadline)]


def single_record_sr(p_id, r_id, u_st, st_d_dict, day, ss_st, ss_dict, dd):
    """
    generates a record for a surgery request given a schedule date - the record is generated in a way that 80% of a
    room in room allocation will be scheduled
    :param dd: int day duration until now
    :param ss_dict: surgeon skill dictionary {surgeon_id: {surgery type:skill, st:skill},...}}
    :param p_id: list of patient id
    :param r_id: int request id
    :param u_st: list of surgery type id to choose from - (all of the same unit and ward)
    :param st_d_dict: dictionary : {surgery_type_id : duration}
    :param day: date object - schedule date (taken from room allocation)
    :param ss_st: surgery type id - surgery type witch require specific senior
    :return: list with all the fields of a surgery request, duration of the surgery request
    """

    st = list(u_st.copy())
    request_id = r_id
    while True:
        if len(st) > 0:
            surgery_type = random.choice(st)
            st.remove(surgery_type)
            duration = st_d_dict[surgery_type]
            if (dd + duration) < day_duration:
                break
        else:
            surgery_type = None
            duration = 0
            break
    if surgery_type is None:
        return [], 0
    else:
        patient_id = random.choice(p_id)
        # urgency = random.randint(1, 6)
        urgency = random.choices([1, 2, 3, 4, 5, 6], weights=[10, 20, 30, 30, 20, 10], k=1)[0]
        # complexity = random.randint(1, 6)
        complexity = random.choices([1, 2, 3, 4, 5, 6], weights=[10, 20, 30, 30, 20, 10], k=1)[0]
        status = 1.1
        cancellations = random.randint(1, 10)
        schedule_date = datetime.strptime(day, '%Y-%m-%d').date()
        pre_op_date = random_date(schedule_date - timedelta(days=30), schedule_date - timedelta(days=1))
        entrance_date = random_date(schedule_date - timedelta(weeks=52), pre_op_date - timedelta(days=1))
        specific_senior = ''
        if surgery_type == ss_st:
            for s in ss_dict:
                if surgery_type in ss_dict[s]:
                    if ss_dict[s][surgery_type] >= 5:
                        specific_senior = s
                        break
        schedule_from = ''
        schedule_deadline = ''
        schedule_flag = random.randint(2, 4)
        if schedule_flag == 2:
            schedule_from = random_date(pre_op_date, schedule_date)
        if schedule_flag == 3:
            schedule_deadline = random_date(schedule_date + timedelta(days=1), schedule_date + timedelta(weeks=2))
        if schedule_flag == 4:
            schedule_from = random_date(pre_op_date, schedule_date)
            schedule_deadline = random_date(schedule_date + timedelta(days=1), schedule_date + timedelta(weeks=2))

        return [request_id, patient_id, surgery_type, urgency, complexity, duration, status, cancellations,
                str(entrance_date),
                specific_senior, str(pre_op_date), str(schedule_date), str(schedule_from),
                str(schedule_deadline)], duration


def possible_st_list(r_st, shift_s, s_st_dict, n_w_st_dict, w, shift_n):
    """
    :param shift_n: list of nurse_id on shift in a certain date
    :param r_st: set of surgery types id - concerning the st that can be done in a certain room
    :param shift_s: list of surgeon id on shift in a certain date
    :param s_st_dict: dictionary {surgeon_id: {surgery type:skill, st:skill},...}} of a certain ward
    :param n_st_dict: nurse surgery type dict - {nurse_id:{w1: [st1, st2..stn], w2:...} of a certain ward of her cn skills
    (more constrained)
    :return:
    """
    # n_w_st_dict = convert_nurse_st_dict(w_st_cn_dict)  # nurse_ward_surgery_type : {nurse_id: { ward_id : [st1..stn], ward_id: [..]}...}
    s_st = set()
    n_st = set()
    for s in shift_s:
        s_st.update(s_st_dict[s].keys())
    for n in shift_n:
        if w in n_w_st_dict[n]:
            n_st.update(n_w_st_dict[n][w])
    st_set = s_st.intersection(r_st).intersection(n_st)
    return st_set


def choose_st_for_specific_seniors(w_st_dict):
    """
    for each ward a random surgery type is chosen - the surgery requests of this kind of s.t. will need a specific_senior
    :param w_st_dict: dictionary {w_id : (st_1, st_2...), w2: (st1..stn)...} each value of the different
    dict keys is a set,
    :return: dictionary {ward id : sugrgery type id}
    """
    ss_st_dict = {}
    for w in w_st_dict:
        ss_st_dict[w] = random.choice(list(w_st_dict[w]))
    return ss_st_dict


def nurses_generator(r_dict):
    """
    :param r_dict: room dictionary {r_id : (st_1..st_n), r2 : (st1..stn)...} the values of the keys are of set kind
    :return: list of nurses id
    """
    data = []
    num_nurses = input('how many nurses? (minimum - ' + str(len(r_dict) * 2) + ')')
    for i in range(int(num_nurses)):
        data.append([i])
    df = pd.DataFrame(data, columns=['nurse_id'])
    df.to_csv(path + r'\nurses.csv')
    return range(int(num_nurses))


def nurses_sn_skills_generator(n_id, w_st_dict, r_dict):
    """
    scrub nurse - promises full solution
    :param r_dict: room dictionary {r_id : (st_1..st_n), r2 : (st1..stn)...} the values of the keys are of set kind
    :param w_st_dict: ward surgery type dictionary - {w_id : (st_1, st_2...), w2: (st1..stn)...} each value of the
    different dict keys is a set
    :param n_id: list of nurses id
    :return:  w_st_n_dict ward surgery type nurse dictionary : {ward id: surgery type id : [nurse id1..nurse id n]...}..}
    """
    data = []
    s_id = 0
    random_generator = input('number of scrub nurses for each surgery type decided randomly? 1- yes 0 - no')
    w_st_n_dict = {}
    for w in w_st_dict:
        w_st_n_dict[w] = {}
        for st in w_st_dict[w]:
            w_st_n_dict[w][st] = []
            if int(random_generator):
                # minimum number of nurses is the number of surgical rooms and the max is all of the nurses
                num_nurses = random.randint(len(r_dict), len(n_id))
            else:
                num_nurses = int(input('how many scrub nurses for surgery type: ' + str(st) + ' of ward: ' + str(w) +
                                       '? - minimum- ' + str(len(r_dict)) + ' maximum- ' + str(len(n_id))))
            nurses = random.sample(n_id, k=num_nurses)
            for n in nurses:
                data.append([s_id, n, w, st])
                s_id += 1
                w_st_n_dict[w][st].append(n)
    df = pd.DataFrame(data, columns=['skill_id', 'nurse_id', 'ward_id', 'surgery_type_id'])
    df.to_csv(path + r'\nurses_sn_skills.csv')
    return w_st_n_dict


def nurses_sn_skills_generator1(n_id, w_st_dict, r_dict):
    """
    scrub nurse - randomized
    :param r_dict: room dictionary {r_id : (st_1..st_n), r2 : (st1..stn)...} the values of the keys are of set kind
    :param w_st_dict: ward surgery type dictionary - {w_id : (st_1, st_2...), w2: (st1..stn)...} each value of the
    different dict keys is a set
    :param n_id: list of nurses id
    :return:  w_st_n_dict ward surgery type nurse dictionary : {ward id: surgery type id : [nurse id1..nurse id n]...}..}
    """
    data = []
    s_id = 0
    random_generator = input('number of scrub nurses for each surgery type decided randomly? 1- yes 0 - no')
    w_st_n_dict = {}
    for w in w_st_dict:
        w_st_n_dict[w] = {}
        for st in w_st_dict[w]:
            w_st_n_dict[w][st] = []
            if int(random_generator):
                # minimum number of nurses is 1 and the max is all of the nurses
                num_nurses = random.randint(1, len(n_id))
            else:
                num_nurses = int(input('how many scrub nurses for surgery type: ' + str(st) + ' of ward: ' + str(w) +
                                       '? - minimum- ' + str(len(r_dict)) + ' maximum- ' + str(len(n_id))))
            nurses = random.sample(n_id, k=num_nurses)
            for n in nurses:
                data.append([s_id, n, w, st])
                s_id += 1
                w_st_n_dict[w][st].append(n)
    df = pd.DataFrame(data, columns=['skill_id', 'nurse_id', 'ward_id', 'surgery_type_id'])
    df.to_csv(path + r'\nurses_sn_skills.csv')
    return w_st_n_dict


def nurses_cn_skills_generator(w_st_n_dict, r_dict):
    """
    circulating nurse - promises full solution
    :param r_dict: room dictionary {r_id : (st_1..st_n), r2 : (st1..stn)...} the values of the keys are of set kind
    :param w_st_n_dict: ward surgery type scrubbing nurse dictionary :
    {ward id: surgery type id : [nurse id1..nurse id n]...}..}
    :return: w_st_cn_dict ward surgery type nurse dictionary : {ward id: surgery type id : [nurse id1..nurse id n]...}.}
    """
    data = []
    s_id = 0
    w_st_cn_dict = {}
    random_generator = input('number of circulating nurses for each surgery type decided randomly? 1- yes 0 - no')
    for w in w_st_n_dict:
        w_st_cn_dict[w] = {}
        for st in w_st_n_dict[w]:
            w_st_cn_dict[w][st] = []
            if int(random_generator):
                # minimum number of nurses is the number of surgical rooms and the max is the number of scrubbing nurses
                # for a certain surgery type - nurses are first qualified as sn and only then cn for a certain st
                num_nurses = random.randint(len(r_dict), len(w_st_n_dict[w][st]))
            else:
                num_nurses = int(
                    input('how many circulating nurses for surgery type: ' + str(st) + ' of ward: ' + str(w) +
                          '? - minimum- ' + str(len(r_dict)) + ' maximum- ' + str(len(w_st_n_dict[w][st]))))
            nurses = random.sample(w_st_n_dict[w][st], k=num_nurses)
            for n in nurses:
                data.append([s_id, n, w, st])
                s_id += 1
                w_st_cn_dict[w][st].append(n)
    df = pd.DataFrame(data, columns=['skill_id', 'nurse_id', 'ward_id', 'surgery_type_id'])
    df.to_csv(path + r'\nurses_cn_skills.csv')
    return w_st_cn_dict


def nurses_cn_skills_generator1(w_st_n_dict, r_dict):
    """
    circulating nurse
    :param r_dict: room dictionary {r_id : (st_1..st_n), r2 : (st1..stn)...} the values of the keys are of set kind
    :param w_st_n_dict: ward surgery type scrubbing nurse dictionary :
    {ward id: surgery type id : [nurse id1..nurse id n]...}..}
    :return: w_st_cn_dict ward surgery type nurse dictionary : {ward id: surgery type id : [nurse id1..nurse id n]...}.}
    """
    data = []
    s_id = 0
    w_st_cn_dict = {}
    random_generator = input('number of circulating nurses for each surgery type decided randomly? 1- yes 0 - no')
    for w in w_st_n_dict:
        w_st_cn_dict[w] = {}
        for st in w_st_n_dict[w]:
            w_st_cn_dict[w][st] = []
            if int(random_generator):
                # minimum number of nurses is 1 and the max is the number of scrubbing nurses
                # for a certain surgery type - nurses are first qualified as sn and only then cn for a certain st
                num_nurses = random.randint(1, len(w_st_n_dict[w][st]))
            else:
                num_nurses = int(
                    input('how many circulating nurses for surgery type: ' + str(st) + ' of ward: ' + str(w) +
                          '? - minimum- ' + str(len(r_dict)) + ' maximum- ' + str(len(w_st_n_dict[w][st]))))
            nurses = random.sample(w_st_n_dict[w][st], k=num_nurses)
            for n in nurses:
                data.append([s_id, n, w, st])
                s_id += 1
                w_st_cn_dict[w][st].append(n)
    df = pd.DataFrame(data, columns=['skill_id', 'nurse_id', 'ward_id', 'surgery_type_id'])
    df.to_csv(path + r'\nurses_cn_skills.csv')
    return w_st_cn_dict


def nurses_surgical_days_generator(w_ra_dict, r_st_dict, w_st_sn_dict, w_st_cn_dict, w_st_dict):
    """
    Takes into consideration the ward in the room allocation and chooses the nurses in a consistent way with their
    skills and the surgery types of the ward. This function generates for sure an amount of nurses that can deal
    with any scenario for each day - the problem will always be solved
    :param w_st_dict: wards_surgery_type dictionary {w_id : (st_1, st_2...), w2: (st1..stn)...}
    each value of the different dict keys is a set
    :param w_ra_dict: ward room allocation dictionary {w1 : {date : [room_ids]...}
    :param r_st_dict: room surgery type dictionary {r_id : (st_1..st_n), r2 : (st1..stn)...}
    the values of the keys are of set kind
    :param w_st_sn_dict: ward surgery type scrubbing nurse dictionary :
    {ward id: surgery type id : [nurse id1..nurse id n]...}..}
    :param w_st_cn_dict:ward surgery type circulating nurse dictionary :
    {ward id: surgery type id : [nurse id1..nurse id n]...}..}
    :return:
    """
    d_r_w_dict = create_date_room_allocation_dict(w_ra_dict)
    sn_st_dict = convert_nurse_st_dict(w_st_sn_dict)
    cn_st_dict = convert_nurse_st_dict(w_st_cn_dict)
    data = []
    sd_id = 0
    for d in d_r_w_dict:
        d_n_set = set()
        for r in d_r_w_dict[d]:
            w = d_r_w_dict[d][r]
            # surgery types that can be done in room and ward
            sn_st_set = r_st_dict[r].intersection(w_st_dict[w])
            cn_st_set = sn_st_set.copy()
            n_set = set()
            # for every s.t. at least one nurse with sn skill and a different one with cn skill
            for st in sn_st_set.union(cn_st_set):
                if st in sn_st_set and (not set(w_st_sn_dict[w][st]).issubset(d_n_set)):
                    while True:
                        sn = random.choice(w_st_sn_dict[w][st])
                        if sn not in n_set:
                            if sn not in d_n_set:
                                n_set.add(sn)
                                d_n_set.add(sn)
                                sn_st_set.difference_update(sn_st_dict[sn][w])  # update the surgery types left 
                                data.append([sd_id, sn, d])
                                sd_id += 1
                            break
                if ((st in cn_st_set) and (not set(w_st_cn_dict[w][st]).issubset(d_n_set))) or \
                        (len(d_n_set) < len(d_r_w_dict[d]) * 2):
                    while True:
                        cn = random.choice(w_st_cn_dict[w][st])
                        if cn not in n_set:
                            # if i see there are not enough nurses in each surgical day then turn into only one if of
                            # d_n_set and no double if also in sn..
                            if cn not in d_n_set:
                                n_set.add(cn)
                                d_n_set.add(cn)
                                cn_st_set.difference_update(cn_st_dict[cn][w])
                                data.append([sd_id, cn, d])
                                sd_id += 1
                            break

    df = pd.DataFrame(data, columns=['surgical_day_id', 'nurse_id', 'date'])
    df.to_csv(path + r'\nurses_surgical_days.csv', date_format='%Y-%m-%d')


def nurses_surgical_days_generator1(w_ra_dict, r_st_dict, w_st_sn_dict, w_st_cn_dict, w_st_dict):
    """
    constrained solution - will not bring to a full solution but a partial one
    :param w_st_dict: wards_surgery_type dictionary {w_id : (st_1, st_2...), w2: (st1..stn)...}
    each value of the different dict keys is a set
    :param w_ra_dict: ward room allocation dictionary {w1 : {date : [room_ids]...}
    :param r_st_dict: room surgery type dictionary {r_id : (st_1..st_n), r2 : (st1..stn)...}
    the values of the keys are of set kind
    :param w_st_sn_dict: ward surgery type scrubbing nurse dictionary :
    {ward id: surgery type id : [nurse id1..nurse id n]...}..}
    :param w_st_cn_dict:ward surgery type circulating nurse dictionary :
    {ward id: surgery type id : [nurse id1..nurse id n]...}..}
    :return:
    """
    d_r_w_dict = create_date_room_allocation_dict(w_ra_dict)  # d1: {r1 : w2, r3: w1...}...}
    sn_st_dict = convert_nurse_st_dict(w_st_sn_dict)  # {nurse_id: { ward_id : [st1..stn], ward_id: [..]}...}
    cn_st_dict = convert_nurse_st_dict(w_st_cn_dict)
    data = []
    sd_id = 0
    shift_dict = dict()
    for d in d_r_w_dict:
        shift_dict[d] = []
        num_nurses_needed = len(d_r_w_dict[d].values()) * 2
        n_list = list(sn_st_dict.keys())
        # for i in range(random.randint(2, num_nurses_needed - 1)):
        for i in range(num_nurses_needed):
            if len(n_list) > 0:
                n = random.choice(n_list)
                n_list.remove(n)
                data.append([sd_id, n, d])
                shift_dict[d].append(n)
                sd_id += 1
        '''for r in d_r_w_dict[d]:
            w = d_r_w_dict[d][r]
            # surgery types that can be done in room and ward
            sn_st_set = r_st_dict[r].intersection(w_st_dict[w])
            cn_st_set = sn_st_set.copy()
            n_set = set()
            # for every s.t. at least one nurse with sn skill and a different one with cn skill
            for st in sn_st_set.union(cn_st_set):  # this is the original code for having enough nures
                # the code includes both while True.. and the union part above
                if st in sn_st_set and (not set(w_st_sn_dict[w][st]).issubset(d_n_set)):
                    while True:
                        sn = random.choice(w_st_sn_dict[w][st])
                        if sn not in n_set:
                            if sn not in d_n_set:
                                n_set.add(sn)
                                d_n_set.add(sn)
                                sn_st_set.difference_update(sn_st_dict[sn][w])  # update the surgery types left 
                                data.append([sd_id, sn, d])
                                sd_id += 1
                            break
                if ((st in cn_st_set) and (not set(w_st_cn_dict[w][st]).issubset(d_n_set))) or\
                        (len(d_n_set) < len(d_r_w_dict[d]) * 2):
                    while True:
                        cn = random.choice(w_st_cn_dict[w][st])
                        if cn not in n_set:
                            # if i see there are not enough nurses in each surgical day then turn into only one if of
                            # d_n_set and no double if also in sn..
                            if cn not in d_n_set:
                                n_set.add(cn)
                                d_n_set.add(cn)
                                cn_st_set.difference_update(cn_st_dict[cn][w])
                                data.append([sd_id, cn, d])
                                sd_id += 1
                            break'''

    df = pd.DataFrame(data, columns=['surgical_day_id', 'nurse_id', 'date'])
    df.to_csv(path + r'\nurses_surgical_days.csv', date_format='%Y-%m-%d')
    return shift_dict


def convert_nurse_st_dict(w_st_n_dict):
    """
    converts dictionary of surgery type keys to be dictionary of nurses key.
    :param w_st_n_dict:  ward surgery type nurse dictionary :
    {ward id: surgery type id : [nurse id1..nurse id n]...}..}
    :return: dictionary nurse_ward_surgery_type : {nurse_id: { ward_id : [st1..stn], ward_id: [..]}...}
    """

    n_w_st_dict = dict()
    for w in w_st_n_dict:
        for st in w_st_n_dict[w]:
            for n in w_st_n_dict[w][st]:
                if n not in n_w_st_dict:
                    n_w_st_dict[n] = dict()
                if w not in n_w_st_dict[n]:
                    n_w_st_dict[n][w] = set()
                n_w_st_dict[n][w].add(st)
    return n_w_st_dict


def create_date_room_allocation_dict(w_ra_dict):
    """
    transforms the dictionary with the keys being the dates instead of the wards
    :param w_ra_dict: ward room allocation dictionary {w1 : {date : [room_ids]...}
    :return: day_room_allocation_ward dictionary {d1: {r1 : w2, r3: w1...}...}
    """
    d_ra_w_dict = {}

    for w in w_ra_dict:
        for d in w_ra_dict[w]:
            if d not in d_ra_w_dict:
                d_ra_w_dict[d] = {}
            for r in w_ra_dict[w][d]:
                d_ra_w_dict[d][r] = w
    return d_ra_w_dict


def anesthetist_generator(r_dict, w_id):
    """
    :param w_id: ward id - list of all ward ids
    :param r_dict: room dictionary {r_id : (st_1..st_n), r2 : (st1..stn)...} the values of the keys are of set kind
    :return: a_dict anesthetist dictionary
    {'Expert' : [a_id1..a_idn], 'Senior' : [a_id1..a_idn], 'Stagiaire' : {w_id1: [aid1..aidn], w_id2: ...}},
    a_list - list of all anesthetists is

    """

    data = []
    rank = ['Stagiaire', 'Expert', 'Senior']
    num_a = input('how many anesthetists? (minimum- ' +
                  str(max(len(r_dict) + math.ceil(len(r_dict) / 2) + 1, len(w_id) * 2 + 1)) + ')')
    a_id = 0
    # minimum number of Anesthetists (One Anesthetists for every room , one for every two rooms and a senior, or ,
    # Stagiaire for every ward, expert for every ward, and at least one Senior)
    data.append([a_id, 'Senior', None])
    a_id += 1
    for w in w_id:
        for j in ['Stagiaire', 'Expert']:
            data.append([a_id, j, w])
            a_id += 1
    if a_id + 1 < int(num_a):
        for i in range(int(num_a) - a_id - 1):
            r = np.random.choice(a=rank, size=1, replace=False, p=[0.45, 0.4, 0.15])
            if r[0] == 'Stagiaire':
                data.append([a_id, r[0], random.choice(w_id)])
                a_id += 1
            else:
                data.append([a_id, r[0], None])
                a_id += 1
    df = pd.DataFrame(data, columns=['anesthetist_id', 'rank', 'speciality'])
    df.to_csv(path + r'\anesthetists.csv')
    a_dict = {}
    for r in rank:
        if r != 'Stagiaire':
            a_dict[r] = set(df.loc[lambda x: x['rank'] == r]['anesthetist_id'].tolist())
        else:
            a_dict[r] = {}
            for w in w_id:
                a_dict[r][w] = set(
                    df.loc[lambda x: (x['rank'] == r) & (x['speciality'] == w)]['anesthetist_id'].tolist())
    a_list = df['anesthetist_id'].tolist()
    # stag_list = df.loc[lambda x: x['rank'] == 'Stagiaire']['anesthetist_id'].tolist()
    return a_dict, a_list


def anesthetists_surgical_days_generator(w_ra_dict, a_dict, a_list):
    """
    garantees enough anesthetist for surgical day
    :param a_list: anesthetists list - list of all anesthetists id
    :param w_ra_dict: ward room allocation dictionary {w1 : {date : [room_ids]...}
    :param a_dict: anesthetist dictionary:
     {'Expert' : (a_id1..a_idn), 'Senior' : (a_id1..a_idn), 'Stagiaire' : {w_id1: (aid1..aidn), w_id2: ...}}
     values are of set kind
    :return:
    """
    data = []
    senior_expert_set = a_dict['Senior'].union(a_dict['Expert'])
    sd_id = 0
    d_ra_w_dict = create_date_room_allocation_dict(w_ra_dict)  # {d1: {r1 : w2, r3: w1...}...}
    for d in d_ra_w_dict:
        a_set = set()
        senior = random.choice(list(a_dict['Senior']))
        a_set.add(senior)
        data.append([sd_id, senior, d])
        sd_id += 1
        for w in d_ra_w_dict[d].values():
            num_not_stag = 2
            # choose a stagiaire of the ward given the room if exists
            if not a_dict['Stagiaire'][w].issubset(a_set):
                stag = random.choice(list(a_dict['Stagiaire'][w] - a_set))
                data.append([sd_id, stag, d])
                sd_id += 1
                a_set.add(stag)
                num_not_stag -= 1
            # if no stag choose a regular anesthetist
            for i in range(num_not_stag):
                if not senior_expert_set.issubset(a_set):
                    a = random.choice(list(senior_expert_set - a_set))
                    data.append([sd_id, a, d])
                    sd_id += 1
                    a_set.add(a)
        # some extra anesthetists to make it interesting - half of the number of rooms
        if not set(a_list).issubset(a_set):
            extra_set = set(a_list) - a_set
            for i in range(len(d_ra_w_dict[d]) // 2):
                if len(extra_set) > 0:
                    a = random.choice(list(extra_set))
                    data.append([sd_id, a, d])
                    sd_id += 1
                    extra_set.discard(a)
                else:
                    break

    df = pd.DataFrame(data, columns=['surgical_day_id', 'anesthetist_id', 'date'])
    df.to_csv(path + r'\anesthetists_surgical_days.csv', date_format='%Y-%m-%d')


def anesthetists_surgical_days_generator1(w_ra_dict, a_dict, a_list):
    """
    not necessarily enough anesthetist for surgical day
    :param a_list: anesthetists list - list of all anesthetists id
    :param w_ra_dict: ward room allocation dictionary {w1 : {date : [room_ids]...}
    :param a_dict: anesthetist dictionary:
     {'Expert' : (a_id1..a_idn), 'Senior' : (a_id1..a_idn), 'Stagiaire' : {w_id1: (aid1..aidn), w_id2: ...}}
     values are of set kind
    :return:
    """
    data = []
    senior_expert_set = a_dict['Senior'].union(a_dict['Expert'])
    sd_id = 0
    d_ra_w_dict = create_date_room_allocation_dict(w_ra_dict)  # {d1: {r1 : w2, r3: w1...}...}
    for d in d_ra_w_dict:
        a_set = set()
        senior = random.choice(list(a_dict['Senior']))
        a_set.add(senior)
        data.append([sd_id, senior, d])
        sd_id += 1
        expert = random.choice(list(a_dict['Expert']))
        a_set.add(expert)
        sd_id += 1
        data.append([sd_id, expert, d])
        num_anesthetist_needed = math.ceil(len(d_ra_w_dict[d].values()) / 2) + len(d_ra_w_dict[d].values()) - 1
        num_anesthetists_for_day = random.randint(1, num_anesthetist_needed - 1)
        for i in range(num_anesthetists_for_day):  # enough room managers for all rooms
            aa_list = list(set(a_list) - a_set)
            if aa_list:
                a = random.choice(aa_list)
                data.append([sd_id, a, d])
                sd_id += 1
                a_set.add(a)
    df = pd.DataFrame(data, columns=['surgical_day_id', 'anesthetist_id', 'date'])
    df.to_csv(path + r'\anesthetists_surgical_days.csv', date_format='%Y-%m-%d')


def anesthetists_surgical_days_generator2(w_ra_dict, a_dict, a_list):
    """
    Exact the number of Anesthetist needed - with enough experts and seniors for room managers
    :param a_list: anesthetists list - list of all anesthetists id
    :param w_ra_dict: ward room allocation dictionary {w1 : {date : [room_ids]...}
    :param a_dict: anesthetist dictionary:
     {'Expert' : (a_id1..a_idn), 'Senior' : (a_id1..a_idn), 'Stagiaire' : {w_id1: (aid1..aidn), w_id2: ...}}
     values are of set kind
    :return:
    """
    data = []
    senior_expert_set = a_dict['Senior'].union(a_dict['Expert'])
    sd_id = 0
    d_ra_w_dict = create_date_room_allocation_dict(w_ra_dict)  # {d1: {r1 : w2, r3: w1...}...}
    for d in d_ra_w_dict:
        a_set = set()
        senior = random.choice(list(a_dict['Senior']))  # floor manager
        a_set.add(senior)
        data.append([sd_id, senior, d])
        sd_id += 1
        num_of_room_managers = math.ceil(len(d_ra_w_dict[d].values()) / 2)
        for i in range(num_of_room_managers):
            se_list = list(senior_expert_set - a_set)
            if se_list:
                expert = random.choice(se_list)
                a_set.add(expert)
                sd_id += 1
                data.append([sd_id, expert, d])
        num_of_operation_anesthetist = len(d_ra_w_dict[d].values())
        for i in range(num_of_operation_anesthetist):
            aa_list = list(set(a_list) - a_set)
            if aa_list:
                a = random.choice(aa_list)
                data.append([sd_id, a, d])
                sd_id += 1
                a_set.add(a)
    df = pd.DataFrame(data, columns=['surgical_day_id', 'anesthetist_id', 'date'])
    df.to_csv(path + r'\anesthetists_surgical_days.csv', date_format='%Y-%m-%d')


def stagiaire_rotations_generator(a_dict, w_id):
    """
    :param w_id: list of all ward id
    :param a_dict:  anesthetist dictionary:
     {'Expert' : (a_id1..a_idn), 'Senior' : (a_id1..a_idn), 'Stagiaire' : {w_id1: (aid1..aidn), w_id2: ...}}
    :return:
    """
    data = []
    r_id = 0
    rot_dict = {}
    for w in a_dict['Stagiaire']:
        for a in a_dict['Stagiaire'][w]:
            num_rotations = random.randint(0, len(w_id))
            ward_rotations = random.sample(w_id, k=num_rotations)
            for wr in ward_rotations:
                if wr != w:
                    if wr not in rot_dict:
                        rot_dict[wr] = set()
                    data.append([a, wr, r_id])
                    rot_dict[wr].add(a)
                    r_id += 1
    df = pd.DataFrame(data, columns=['anesthetist_id', 'ward_id', 'id'])
    df.to_csv(path + r'\stagiaire_rotations.csv')
    return rot_dict


def stagiaire_skills_generator(a_dict, w_st_dict, rotation_dict):
    """
    scrub nurse - randomized
    :param rotation_dict: dictionary {wid:(stag_id, stag_id1, stag_id2...),..}
    :param w_st_dict: ward surgery type dictionary - {w_id : (st_1, st_2...), w2: (st1..stn)...} each value of the
    different dict keys is a set
    :param a_dict:  anesthetist dictionary:
    {'Expert' : (a_id1..a_idn), 'Senior' : (a_id1..a_idn), 'Stagiaire' : {w_id1: (aid1..aidn), w_id2: ...}}
    :return:  w_st_n_dict ward surgery type nurse dictionary : {ward id: surgery type id : [nurse id1..nurse id n]...}..}
    """
    data = []
    s_id = 0
    random_generator = input('number of stagiaire for each surgery type decided randomly? 1- yes 0 - no')
    w_st_n_dict = {}
    for w in w_st_dict:
        w_st_n_dict[w] = {}
        if w in rotation_dict:
            stag_set = a_dict['Stagiaire'][w].union(rotation_dict[w])
        else:
            stag_set = a_dict['Stagiaire'][w]
        for st in w_st_dict[w]:
            w_st_n_dict[w][st] = []
            if int(random_generator):
                # minimum number of stag is 1 and the max is all of the stagiaire who are in rotation or were in
                # rotation
                num_stag = random.randint(1, len(stag_set))
            else:
                num_stag = int(input('how many stagiaire for surgery type: ' + str(st) + ' of ward: ' + str(w) +
                                     '? - minimum- ' + str(1) + ' maximum- ' + str(len(stag_set))))
            stag = random.sample(stag_set, k=num_stag)
            for n in stag:
                data.append([s_id, n, w, st])
                s_id += 1
                w_st_n_dict[w][st].append(n)
    df = pd.DataFrame(data, columns=['skill_id', 'anesthetist_id', 'ward_id', 'surgery_type_id'])
    df.to_csv(path + r'\stag_skills.csv')
    return w_st_n_dict


def equipments_generator(r_dict):
    """
    :param r_dict: {r_id : (st_1..st_n), r2 : (st1..stn)...} the values of the keys are of set kind
    :return: list of all equipment id
    """
    data = []
    num_equipment = int(input('how many equipment types?'))
    random_generator = input('amount of each equipment decided randomly? 1- yes, 0 - no')
    for e in range(num_equipment):
        if int(random_generator):
            # minimum - 1 maximum - the number of rooms
            num_e = random.randint(1, len(r_dict))
        else:
            num_e = int(input('how many units of equipment: ' + str(e) + '?'))
        data.append([e, num_e])
    df = pd.DataFrame(data, columns=['equipment_id', 'max_in_hospital'])
    df.to_csv(path + r'\equipments.csv')
    e_id = df['equipment_id'].tolist()
    return e_id


def equipment_for_surgery_requests_generator(e_id, sr_id):
    """

    :param e_id: list of equipment id
    :param sr_id:  list of surgery request id
    :return:
    """
    data = []
    sr_e_id = 0
    for sr in sr_id:
        num_equipment = random.randint(0, len(e_id))
        sr_e = random.sample(e_id, k=num_equipment)
        for e in sr_e:
            data.append([sr_e_id, sr, e])
            sr_e_id += 1
    df = pd.DataFrame(data, columns=['id', 'surgery_request_id', 'equipment_id'])
    df.to_csv(path + r'\equipment_for_surgery_requests.csv')

# random Problem
print('wtf')
ward_id = ward_generator()  # returnd list of ward id
patient_id = patients_generator()
u_d = unit_generator(ward_id)  # returns dictionary {w_id : [unit_ids...]}
surgery_type_id, ward_st_dict, unit_st_dict, st_duration_dict = surgery_type_generator(u_d)  # returns list of all st_id ,
# dictionary {w_id : (st_1, st_2...), w2: (st1..stn)...} each value of the different dict keys is a set,
# u_st_dict: {w_id: {u_id:[st_1..st_n], u_2:[st1..stn]}...}
room_st_dict = room_generator(surgery_type_id)  # returns dictionary
# {r_id : (st_1..st_n), r2 : (st1..stn)...} the values of the keys are of set kind
room_ward_dict = create_room_ward_dict(room_st_dict, ward_st_dict)  # returns dictionary
# {r_id:[w1, w2..wn], r2: [w2,w4..wn]...}
ward_room_a_dict, first_allocation_date, num_days_for_allocation = rooms_allocations_generator(room_ward_dict, ward_id)
ward_surgeon_dict = surgeon_generator(ward_st_dict, ward_room_a_dict)
surgeon_skill_dict, st_surgeon_dict = skill_seniors_generator(ward_surgeon_dict,
                                                              ward_st_dict)  # {w_id: {s_id: {st:skill, st:skill},...}}, dictionary: {w_id: {st1 : [surgeon1, surgeon2...]...}
s_shift_dict = shift_senior_generator(surgeon_skill_dict, ward_room_a_dict, room_st_dict, ward_st_dict, st_surgeon_dict)
nurse_id = nurses_generator(room_st_dict)  # list of nurses id
ward_st_s_nurse_dict = nurses_sn_skills_generator1(nurse_id, ward_st_dict, room_st_dict)
# {ward id: surgery type id : [nurse id1..nurse id n]...}..}
ward_st_cn_dict = nurses_cn_skills_generator1(ward_st_s_nurse_dict, room_st_dict)
n_shift_dict = nurses_surgical_days_generator1(ward_room_a_dict, room_st_dict, ward_st_s_nurse_dict, ward_st_cn_dict,
                                               ward_st_dict)
surgery_request_id = surgery_request_generator(patient_id, ward_room_a_dict, room_st_dict, ward_st_dict, s_shift_dict,
                                               surgeon_skill_dict,
                                               unit_st_dict, st_duration_dict, first_allocation_date,
                                               num_days_for_allocation, ward_st_cn_dict, n_shift_dict)
anesthetist_dict, anes_list = anesthetist_generator(room_st_dict, ward_id)
anesthetists_surgical_days_generator2(ward_room_a_dict, anesthetist_dict, anes_list)
rotation_dict = stagiaire_rotations_generator(anesthetist_dict, ward_id)
stagiaire_skills_generator(anesthetist_dict, ward_st_dict, rotation_dict)
equipment_id = equipments_generator(room_st_dict)
equipment_for_surgery_requests_generator(equipment_id, surgery_request_id)

# Demo
'''ward_st_dict = {1: set(), 2: set()}
ward_room_a_dict = {1: {}, 2: {}}
ward_id = list(ward_st_dict.keys())
st_url = 'https://www.api.orm-bgu-soroka.com/surgeries_types/get_by_ward_id'
ra_url = 'https://www.api.orm-bgu-soroka.com/rooms_allocations/get_by_ward_id'
r_url = 'https://www.api.orm-bgu-soroka.com/rooms/get_all'
sr_url = 'https://www.api.orm-bgu-soroka.com/surgery_requests/get_all'
for w_id in ward_st_dict:
    st_r = requests.get(url=st_url, json={"ward_id": w_id})
    st_data = st_r.json()
    for d in st_data:
        ward_st_dict[w_id].add(d['surgery_type_id'])
    ra_r = requests.post(url=ra_url, json={"ward_id": w_id})
    ra_data = ra_r.json()
    for d in ra_data:
        if d['date'] not in ward_room_a_dict[w_id]:
            ward_room_a_dict[w_id][d['date']] = []
        ward_room_a_dict[w_id][d['date']].append(d['room_id'])
r_r = requests.get(url=r_url)
r_data = r_r.json()
room_st_dict = {}
for d in r_data:
    if d['room_id'] not in room_st_dict:
        room_st_dict[d['room_id']] = set()
    room_st_dict[d['room_id']].add(d['surgery_type_id'])
surgery_request_id = []
sr_r = requests.get(url=sr_url)
sr_data = sr_r.json()
for d in sr_data:
    surgery_request_id.append(d['request_id'])
ward_surgeon_dict = surgeon_generator(ward_st_dict, ward_room_a_dict)
surgeon_skill_dict, st_surgeon_dict = skill_seniors_generator(ward_surgeon_dict, ward_st_dict)
s_shift_dict = shift_senior_generator(surgeon_skill_dict, ward_room_a_dict, room_st_dict, ward_st_dict, st_surgeon_dict)
nurse_id = nurses_generator(room_st_dict)
ward_st_s_nurse_dict = nurses_sn_skills_generator(nurse_id, ward_st_dict, room_st_dict)
ward_st_cn_dict = nurses_cn_skills_generator(ward_st_s_nurse_dict, room_st_dict)
n_shift_dict = nurses_surgical_days_generator(ward_room_a_dict, room_st_dict, ward_st_s_nurse_dict, ward_st_cn_dict,
                                               ward_st_dict)
anesthetist_dict, anes_list = anesthetist_generator(room_st_dict, ward_id)
anesthetists_surgical_days_generator(ward_room_a_dict, anesthetist_dict, anes_list)
rotation_dict = stagiaire_rotations_generator(anesthetist_dict, ward_id)
stagiaire_skills_generator(anesthetist_dict, ward_st_dict, rotation_dict)
equipment_id = equipments_generator(room_st_dict)
equipment_for_surgery_requests_generator(equipment_id, surgery_request_id)
print('let see')'''