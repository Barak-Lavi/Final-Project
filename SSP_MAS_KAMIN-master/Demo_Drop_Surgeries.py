import pandas as pd
import requests
from datetime import datetime


def drop_surgeries_DB(s_date):
    get_surgeries_url = r'https://www.api.orm-bgu-soroka.com/surgeries_preop_partials/get_by_date'
    update_surgery_request_url = r'https://www.api.orm-bgu-soroka.com/surgery_requests/update'
    delete_surgery_url = r'https://www.api.orm-bgu-soroka.com/surgeries_preop_partials/remove_by_date'
    get_surgery_request_url = r'https://www.api.orm-bgu-soroka.com/surgery_requests/get_by_request_id'
    get_not_scheduled_sr_url = r'https://www.api.orm-bgu-soroka.com/not_scheduled_surgery_requests/get_all'
    remove_not_scheduled_sr_url = r'https://www.api.orm-bgu-soroka.com/not_scheduled_surgery_requests/remove_by_date'
    drop_floor_managers_url = "https://www.api.orm-bgu-soroka.com/floor_managers/remove_by_date"
    drop_room_managers_url = "https://www.api.orm-bgu-soroka.com/room_managers/remove_by_date"

    r = requests.post(url=get_surgeries_url, json={'date': s_date})
    surgeries = r.json()
    for s in surgeries:  # updates the surgery request records of the scheduled surgeries
        request_id = s['request_id']
        r = requests.get(url=get_surgery_request_url, json={'request_id': request_id})
        sr = r.json()
        schedule_date = sr[0]['schedule_date']
        status = s['prior_status']
        if schedule_date is None:
            r = requests.post(url=update_surgery_request_url, json={'request_id': request_id,
                                                                    'request_status': str(1.1)})
        else:
            r = requests.post(url=update_surgery_request_url, json={'request_id': request_id,
                                                                    'request_status': str(status)})
    r = requests.post(url=delete_surgery_url, json={'date': s_date})
    r1 = requests.get(url=get_not_scheduled_sr_url, json={'schedule_date': s_date})
    not_scheduled_surgery_requests = r1.json()
    for sr in not_scheduled_surgery_requests:  # updates/returns the surgery request records of the not scheduled
        # surgeries - (scheduled by pre_operations but not by me)
        r = requests.post(url=update_surgery_request_url, json=sr)

    r2 = requests.post(url=drop_floor_managers_url, json={'date': s_date})
    r3 = requests.post(url=drop_room_managers_url, json={'date': s_date})
    r4 = requests.post(url=remove_not_scheduled_sr_url, json={'schedule_date': s_date})


# option 2
'''def drop_surgeries_DB(path, schedule_date):
    get_surgeries_url = r'https://www.api.orm-bgu-soroka.com/surgeries_preop_partials/get_all'  # todo change to get by date
    update_sr_url = "https://www.api.orm-bgu-soroka.com/surgery_requests/update"
    df = pd.read_csv(path)
    surgery_requests = df.to_dict('records')
    for sr in surgery_requests:
        if type(sr['schedule_date']) == float:
            sr['schedule_date'] = None
        r = requests.post(url=update_sr_url, json={'request_id': sr['request_id'], 'request_status': str(sr['request_status']),
                                                   'cancellations': sr['cancellations'], 
                                                   'schedule_date': sr['schedule_date']})
    r1 = requests.get(url=get_surgeries_url)
    surgeries = r1.json()
    for s in surgeries:
        request_id = s['request_id']
        r = requests.post(url=delete_surgery_url, json={'request_id': request_id})  # todo change to drop by date
    
    # todo drop floor managers by date
    # todo dorp room managers by date'''



# Main
drop_surgeries_DB(s_date='2021-04-04')
