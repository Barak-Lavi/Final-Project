from random import randrange
from datetime import timedelta, datetime
import requests
import pandas as pd
import math
import os
import random
import string

data_path = r'C:\Users\User\Desktop\thesis\system modeling\experiments\data_input\demo'


def random_date(start, end):
    """
    This function will return a random datetime between two datetime
    objects.
    """
    delta = end - start
    int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
    random_second = randrange(int_delta)
    return start + timedelta(seconds=random_second)


def patient_birthdate_update():
    """
    updates patients birthdate to a random date in DB
    :return:
    """
    URL = "https://www.api.orm-bgu-soroka.com/patients/get_all"
    r = requests.get(url=URL)
    patients = r.json()

    start_date = datetime.strptime('1925-01-01', '%Y-%m-%d').date()
    end_date = datetime.strptime('2020-01-01', '%Y-%m-%d').date()
    p_url = "https://www.api.orm-bgu-soroka.com/patients/update_date_of_birth"

    for p in patients:
        p_id = p['patient_id']
        b_date = str(random_date(start_date, end_date))
        params = {'patient_id': p_id, 'date_of_birth': b_date}
        r = requests.post(url=p_url, json=params)
        print(r)


def anesthetist_upload():
    pass


def surgery_request_upload():
    df = pd.read_csv(data_path + r'\surgery_requests1.csv')
    surgery_requests = df.to_dict('records')
    for sr in surgery_requests:
        delete = []
        for k in sr:
            if type(sr[k]) is float:
                if not math.isnan(sr[k]):
                    sr[k] = int(sr[k])
                else:
                    sr[k] = ''
            if type(sr[k]) is pd.Timestamp:
                sr[k] = str(sr[k].to_pydatetime().date())
            if isinstance(sr[k], pd._libs.tslibs.nattype.NaTType):
                delete.append(k)
            else:
                sr[k] = str(sr[k])
        for key in delete:
            del sr[key]
    print(surgery_requests)
    url = 'https://www.api.orm-bgu-soroka.com/surgery_requests/add_surgery_request'

    for sr in surgery_requests:
        params = {}
        for k in sr:
            if k == 'unit' or k == 'surgery_type_name':
                continue
            params[k] = sr[k]
        print(params)
        r = requests.post(url=url, json=params)
        print(r)


def DB_excel_patients_bakcup():
    url = 'https://www.api.orm-bgu-soroka.com/patients/get_all'
    r = requests.get(url=url)
    pd.DataFrame(r.json()).to_csv(data_path + r'\patients.csv',
                                  index=False)


def DB_excel_rooms_backup():
    url = 'https://www.api.orm-bgu-soroka.com/rooms/get_all'
    r = requests.get(url=url)
    pd.DataFrame(r.json()).to_csv(data_path + r'\rooms.csv',
                                  index=False)


def DB_excel_rooms_allocation_backup():
    url = 'https://www.api.orm-bgu-soroka.com/rooms_allocations/get_all'
    r = requests.get(url=url)
    pd.DataFrame(r.json()).to_csv(data_path + r'\rooms_allocations.csv',
                                  index=False)


def DB_excel_room_allocation_by_ward_backup(ward_id):
    url = 'https://www.api.orm-bgu-soroka.com/rooms_allocations/get_by_ward_id'
    params = {'ward_id': ward_id}
    r = requests.post(url=url, json=params)
    pd.DataFrame(r.json()).to_csv(data_path + r'\room_allocation_ward_' +
                                  str(ward_id) + '.csv', index=False)


def DB_excel_shift_seniors_by_ward_backup(ward_id):
    url = 'https://www.api.orm-bgu-soroka.com/shifts_seniors/get_by_ward_id'
    params = {'ward_id': ward_id}
    r = requests.get(url=url, json=params)
    r = r.json()
    for shift in r:
        del shift['surgeon_senior']
    pd.DataFrame(r).to_csv(data_path + r'C:\\Users\\User\\Desktop\\thesis\\system modeling\\DATA\\shift_seniors_ward_' +
                           str(ward_id) + '.csv', index=False)


def DB_excel_shift_seniors_backup():
    url = 'https://server-soroka-demo.herokuapp.com/shifts_seniors/get_all'
    r = requests.get(url=url)
    pd.DataFrame(r.json()).to_csv(data_path + r'\shift_seniors.csv',
                                  index=False)


def DB_excel_skill_seniors_backup():
    url = ' https://www.api.orm-bgu-soroka.com/skills_seniors/get_all'
    r = requests.get(url=url)
    pd.DataFrame(r.json()).to_csv(data_path + r'\skill_seniors.csv',
                                  index=False)


def DB_excel_skill_seniors_by_ward_backup(ward_id):
    url = 'https://www.api.orm-bgu-soroka.com/skills_seniors/get_by_ward_id'
    params = {'ward_id': ward_id}
    r = requests.get(url=url, json=params)
    r = r.json()
    for skill in r:
        del skill['surgeon_senior']
    pd.DataFrame(r).to_csv(data_path + r'\skill_seniors_ward_' +
                           str(ward_id) + '.csv', index=False)


def DB_excel_surgeon_seniors_by_ward_backup(ward_id):
    url = 'https://www.api.orm-bgu-soroka.com/surgeons_seniors/get_by_ward_id_full_info'
    params = {'ward_id': ward_id}
    r = requests.get(url=url, json=params)
    pd.DataFrame(r.json()).to_csv(data_path + r'\surgeon_seniors_ward_' +
                                  str(ward_id) + '.csv',
                                  index=False)


def DB_excel_surgery_type_by_ward_backup(ward_id):
    url = 'https://www.api.orm-bgu-soroka.com/surgeries_types/get_by_ward_id'
    params = {'ward_id': ward_id}
    r = requests.get(url=url, json=params)
    pd.DataFrame(r.json()).to_csv(data_path + r'\surgery_type_ward_' +
                                  str(ward_id) + '.csv',
                                  index=False)


def DB_excel_units_by_ward_backup(ward_id):
    url = 'https://www.api.orm-bgu-soroka.com/units/get_by_ward_id'
    params = {'ward_id': ward_id}
    r = requests.get(url=url, json=params)
    pd.DataFrame(r.json()).to_csv(data_path + r'\units_ward_' +
                                  str(ward_id) + '.csv', index=False)


def DB_excel_surgery_request_backup():
    url = 'https://www.api.orm-bgu-soroka.com/surgery_requests/get_all'
    r = requests.get(url=url)
    pd.DataFrame(r.json()).to_csv(data_path + r'\surgery_requests.csv',
                                  index=False)


def DB_excel_surgery_type_backup():
    url = 'https://www.api.orm-bgu-soroka.com/surgeries_types/get_all'
    r = requests.get(url=url)
    pd.DataFrame(r.json()).to_csv(data_path + r'\surgery_types.csv',
                                  index=False)


def DB_excel_unit_backup():
    url = 'https://www.api.orm-bgu-soroka.com/units/get_all'
    r = requests.get(url=url)
    pd.DataFrame(r.json()).to_csv(data_path + r'\units.csv',
                                  index=False)


def DB_excel_ward_backup():
    url = ' https://www.api.orm-bgu-soroka.com/wards/get_all'
    r = requests.get(url=url)
    pd.DataFrame(r.json()).to_csv(data_path + r'\wards.csv',
                                  index=False)


def DB_surgery_request_duration_update():
    url = r'https://www.api.orm-bgu-soroka.com/surgery_requests/get_all'
    r = requests.get(url=url)
    surgery_requests = r.json()
    post_url = r'https://www.api.orm-bgu-soroka.com/surgery_requests/update'

    for sr in surgery_requests:
        sr_id = sr['request_id']
        if int(sr['duration']) % 30 > 0.5:
            duration = math.ceil(int(sr['duration'] / 30)) * 30
        else:
            duration = math.floor(int(sr['duration'] / 30)) * 30
        r = requests.post(url=post_url, json={'request_id': sr_id, 'duration': duration})


def DB_surgery_type_duration_update():
    # todo
    pass


def DB_surgery_request_status_update():
    update_sr_url = "https://www.api.orm-bgu-soroka.com/surgery_requests/update"
    df = pd.read_csv(data_path + r'\surgery_requests.csv')
    surgery_requests = df.to_dict('records')
    for sr in surgery_requests:
        if type(sr['schedule_date']) == float:
            sr['schedule_date'] = None

        params = {'request_id': sr['request_id'], 'request_status': str(sr['request_status']),
                  'cancellations': sr['cancellations'], 'schedule_date': sr['schedule_date']}
        r = requests.post(url=update_sr_url, json=params)


def DB_upload_surgeon_seniors():
    post_url = r'https://www.api.orm-bgu-soroka.com/surgeons_seniors/add_surgeon_senior'
    df = pd.read_csv(data_path + r'\surgeon_seniors.csv')
    surgeons_j = df.to_dict('records')
    for s in surgeons_j:
        first_name = ''.join(random.choice(string.ascii_lowercase) for i in range(8))
        last_name = ''.join(random.choice(string.ascii_lowercase) for i in range(8))
        email = ''.join(random.choice(string.ascii_lowercase) for i in range(4)) + '@clalit.org.il'
        address = ''.join(random.choice(string.ascii_lowercase) for i in range(8))
        phone_number = random.sample(['054', '052', '050'], k=1)[0] + str(random.randint(10 ** 6, 10 ** 7 - 1))
        senior_id = s['senior_id']
        ward_id = s['ward_id']
        r = requests.post(url=post_url, json={'senior_id': senior_id, 'ward_id': ward_id, 'first_name': first_name,
                                              'last_name': last_name, 'address': address, 'email': email,
                                              'phone_number': phone_number})


def DB_upload_shift_seniors():
    post_url = r'https://www.api.orm-bgu-soroka.com/shifts_seniors/add_shift_senior'
    df = pd.read_csv(data_path + r'\shift_seniors.csv')
    shift_seniors = df.to_dict('records')
    for ss in shift_seniors:
        senior_id = ss['senior_id']
        st_time = ss['st_time'] + ' 08:00:00'
        end_time = ss['st_time'] + ' 16:00:00'
        r = requests.post(url=post_url, json={'senior_id': senior_id, 'st_time': st_time, 'end_time': end_time})


def DB_upload_skill_seniors():
    post_url = r'https://www.api.orm-bgu-soroka.com/skills_seniors/add_skill'
    df = pd.read_csv(data_path + r'\skill_seniors.csv')
    skill_seniors = df.to_dict('records')
    for ss in skill_seniors:
        senior_id = ss['senior_id']
        surgery_type_id = ss['surgery_type_id']
        skill = ss['skill']
        skill_id = ss['skill_id']
        r = requests.post(url=post_url,
                          json={'senior_id': senior_id, 'surgery_type_id': surgery_type_id, 'skill': skill,
                                'skill_id': skill_id})


def DB_upload_nurses():
    post_url = r' https://www.api.orm-bgu-soroka.com/nurses/add_nurse'
    df = pd.read_csv(data_path + r'\nurses.csv')
    nurses = df.to_dict('records')
    for n in nurses:
        nurse_id = n['nurse_id']
        first_name = ''.join(random.choice(string.ascii_lowercase) for i in range(8))
        last_name = ''.join(random.choice(string.ascii_lowercase) for i in range(8))
        email = first_name + '@clalit.org.il'
        address = ''.join(random.choice(string.ascii_lowercase) for i in range(8))
        phone_number = random.sample(['054', '052', '050'], k=1)[0] + str(random.randint(10 ** 6, 10 ** 7 - 1))
        r = requests.post(url=post_url, json={'nurse_id': nurse_id, 'first_name': first_name, 'last_name': last_name,
                                              'email': email, 'address': address, 'phone_number': phone_number})


def DB_upload_nurse_cn_skills():
    post_url = r'https://www.api.orm-bgu-soroka.com/nurses_cn_skills/add_nurse_cn_skill'
    df = pd.read_csv(data_path + r'\nurses_cn_skills.csv')
    cn_skills = df.to_dict('records')
    for cn in cn_skills:
        nurse_id = cn['nurse_id']
        ward_id = cn['ward_id']
        surgery_type_id = cn['surgery_type_id']
        r = requests.post(url=post_url, json={'nurse_id': nurse_id, 'ward_id': ward_id,
                                              'surgery_type_id': surgery_type_id})


def DB_upload_nurse_sn_skills():
    post_url = r'https://www.api.orm-bgu-soroka.com/nurses_sn_skills/add_nurse_sn_skill'
    df = pd.read_csv(data_path + r'\nurses_sn_skills.csv')
    sn_skills = df.to_dict('records')
    for sn in sn_skills:
        nurse_id = sn['nurse_id']
        ward_id = sn['ward_id']
        surgery_type_id = sn['surgery_type_id']
        r = requests.post(url=post_url, json={'nurse_id': nurse_id, 'ward_id': ward_id,
                                              'surgery_type_id': surgery_type_id})


def DB_upload_nurse_surgical_days():
    post_url = r'https://www.api.orm-bgu-soroka.com/nurses_surgical_days/add_nurse_surgical_day'
    df = pd.read_csv(data_path + r'\nurses_surgical_days.csv')
    nurse_surgical_days = df.to_dict('records')
    for nsd in nurse_surgical_days:
        nurse_id = nsd['nurse_id']
        date = nsd['date']
        r = requests.post(url=post_url, json={'nurse_id': nurse_id, 'date': date})


def DB_upload_anesthetist():
    post_url = r'https://www.api.orm-bgu-soroka.com/anesthetists/add_anesthetist'
    df = pd.read_csv(data_path + r'\anesthetists.csv')
    anesthetists = df.to_dict('records')
    for a in anesthetists:
        anesthetist_id = a['anesthetist_id']
        rank = a['rank']
        if math.isnan(a['speciality']):
            speciality = None
        else:
            speciality = int(a['speciality'])
        # speciality = a['speciality']
        first_name = ''.join(random.choice(string.ascii_lowercase) for i in range(8))
        last_name = ''.join(random.choice(string.ascii_lowercase) for i in range(8))
        email = first_name + '@clalit.org.il'
        address = ''.join(random.choice(string.ascii_lowercase) for i in range(8))
        phone_number = random.sample(['054', '052', '050'], k=1)[0] + str(random.randint(10 ** 6, 10 ** 7 - 1))
        r = requests.post(url=post_url, json={'anesthetist_id': anesthetist_id, 'first_name': first_name,
                                              'last_name': last_name, 'address': address, 'email': email,
                                              'phone_number': phone_number, 'rank': rank, 'speciality': speciality})


def DB_upload_anesthetist_surgical_days():
    post_url = r'https://www.api.orm-bgu-soroka.com/anesthetists_surgical_days/add_anesthetist_surgical_day'
    df = pd.read_csv(data_path + r'\anesthetists_surgical_days.csv')
    anes_surgical_days = df.to_dict('records')
    for asd in anes_surgical_days:
        anesthetist_id = asd['anesthetist_id']
        date = asd['date']
        r = requests.post(url=post_url, json={'anesthetist_id': anesthetist_id, 'date': date})


def DB_upload_stagiaire_rotations():
    post_url = r' https://www.api.orm-bgu-soroka.com/stagiaire_rotations/add_stagiaire_rotation'
    df = pd.read_csv(data_path + r'\stagiaire_rotations.csv')
    stagiaire_rotations = df.to_dict('records')
    for st_r in stagiaire_rotations:
        anesthetist_id = st_r['anesthetist_id']
        ward_id = st_r['ward_id']
        start_date = '2021-01-01'
        end_date = '2021-12-31'
        r = requests.post(url=post_url, json={'anesthetist_id': anesthetist_id, 'ward_id': ward_id,
                                              'start_date': start_date, 'end_date': end_date})


def DB_upload_equipments():
    post_url = r'https://www.api.orm-bgu-soroka.com/equipments/add_equipment'
    df = pd.read_csv(data_path + r'\equipments.csv')
    equipments = df.to_dict('records')
    for e in equipments:
        equipment_id = e['equipment_id']
        max_in_hospital = e['max_in_hospital']
        r = requests.post(url=post_url, json={'equipment_id': equipment_id, 'max_in_hospital': max_in_hospital})


def DB_upload_equipment_for_surgery_request():
    post_url = r'https://www.api.orm-bgu-soroka.com/equipment_for_surgery_requests/add_equipment_for_surgery_request'
    df = pd.read_csv(data_path + r'\equipment_for_surgery_requests.csv')
    surgery_request_equipment = df.to_dict('records')
    for sr_e in surgery_request_equipment:
        surgery_request_id = sr_e['surgery_request_id']
        equipment_id = sr_e['equipment_id']
        r = requests.post(url=post_url, json={'surgery_request_id': surgery_request_id, 'equipment_id': equipment_id})


# DB_excel_ward_backup()
# DB_excel_rooms_backup()
# DB_excel_patients_bakcup()
# DB_excel_rooms_allocation_backup()

# DB_excel_surgery_type_backup()
# DB_excel_unit_backup()
# DB_excel_surgery_request_backup()
# DB_surgery_request_duration_update()

# DB_upload_surgeon_seniors()
# DB_upload_shift_seniors()
# DB_upload_skill_seniors()
# DB_upload_nurses()
# DB_upload_nurse_cn_skills()
# DB_upload_nurse_sn_skills()
# DB_upload_nurse_surgical_days()
# DB_upload_anesthetist()
# DB_upload_anesthetist_surgical_days()
# DB_upload_stagiaire_rotations()
# DB_upload_equipments()
# DB_upload_equipment_for_surgery_request()

DB_surgery_request_status_update()

