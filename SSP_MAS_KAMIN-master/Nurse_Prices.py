import Prices
from E_in_Surgery import SurgeryVariable_Equipment

inf_price = 1_000_000  # for hard constraints
cost_factor = 1_000


def set_stable_schedule_price(n_v, with_sr):
    cost = 0
    if with_sr:
        # No change of nurse at all - same nurse
        ''' if n_v.value != n_v.value_in_update:
                    if n_v.need_stable:
                        cost += cost_factor  # todo this - is the parameter that will define the stability of the schedule
                        if isinstance(n_v, SurgeryVariable_Equipment):
                            cost = cost * len(n_v.surgery_request.equipments)'''
        # only don't change to from having a nurse to not having one
        if n_v.need_stable:
            if n_v.value != n_v.value_in_update:
                if isinstance(n_v, SurgeryVariable_Equipment):
                    # cost = cost * len(n_v.surgery_request.equipments)
                    cost += cost_factor
                else:
                    if (n_v.value is None) or (n_v.value_in_update is None):
                        cost += cost_factor
        else:
            if n_v.value == n_v.value_in_update:
                cost += cost_factor
    n_v.constraints['dro']['stable_schedule'].prices[n_v.get_constraint_dro_key()] = cost
    return cost


def set_dro_prices(n_v, ward_id, with_sr, stable_schedule_flag=True):
    """
    calculates and updates all the dro <date room order> constraints with cost
    :param n_v: nurse variable object = the variable with index d, r, o
    :param ward_id: int the id of the ward which received the room in room allocation
    :param with_sr: boolean True is schedule is already with surgeries from this ward
    :param stable_schedule_flag: boolean - if to take into acount stable schedule costs
    :return:
    """
    dro_cost = 0
    dro_cost += _set_nurse_surgery_price(n_v, ward_id, with_sr)
    if stable_schedule_flag:
        dro_cost += set_stable_schedule_price(n_v, with_sr)
    return dro_cost


def set_dr_prices(var_list):
    """
    calculates and updated all the dr date room constraints wit cost
    :param var_list: list of variables tuple : (sn_v,cn_v) with same indexes day and room.
    the list will be of the tuples concerning a certain day and room
    """
    dr_cost = 0
    n_v_list = []
    for t in var_list:
        for i in range(len(t)):
            n_v_list.append(t[i])
    dr_cost += _set_min_nurse_per_room(n_v_list)
    return dr_cost


def set_d_prices(v_dict, n_v, next=False):
    """
    calculates and updated all the d date constraints wit cost
    :param with_sr:
    :param next: defines if an update was done in the variable value or prior price is calculated
    :param v_dict: dictionary {ward : {room_num : [(CN_V, SN_V),(CN_v,SN_V)..], r2: [(cn, sn)..]}, w2 : ...}
    :param n_v: nurse variable or None if nurse allocation is refreshed after recieving surgeries
    """
    d_cost = 0
    n_v_list = []  # list of all the nurse variables in a certain day
    max_order = 0
    for w in v_dict:
        for r in v_dict[w]:
            if len(v_dict[w][r]) > max_order:
                max_order = len(v_dict[w][r])
            for t in v_dict[w][r]:
                n_v_list.append(t[0])
                n_v_list.append(t[1])

    if n_v is None:  # for calculating init day cost after surgeries update
        nurse_set = set(
            n_v.value for n_v in n_v_list)  # set of the different nurses assigned to surgery this day
        for n in nurse_set:
            if n is not None:
                nurse_v_list = n_v_list_by_nurse(n_v_list, n)
                # list of nurse variables of a certain nurse in a day
                if len(nurse_v_list) > 0:
                    d_cost += Prices.set_overlapping_prices(nurse_v_list, None)
    else:  # when costs need to be updated by a single change
        if n_v.value is not None or n_v.prior_value is not None:
            d_cost += Prices.surgeon_update_overlapping(n_v, n_v_list, next)
    '''else: (with_sr)
        d_cost += set_same_order_price(n_v, max_order, n_v_list)'''

    return d_cost


def _set_nurse_surgery_price(n_v, ward_id, with_sr):
    """
    if the surgery requests of the schedule are already known - checks if the nurse is certified to do the surgery
    type of surgery request
    :param n_v: nurse variable
    :param ward: ward object which received room in room allocation
    :param with_sr: boolean - if surgery requests are known
    :return: int cost
    """
    cost = 0
    if with_sr:
        if (n_v.value is not None) and (n_v.surgery_request is not None):
            if n_v.surgery_request.surgery_type.st_id in n_v.value.skills[n_v.n_type][ward_id]:
                n_v.constraints['dro']['nurse_sr'].prices[n_v.get_constraint_dro_key()] = 0
            else:
                n_v.constraints['dro']['nurse_sr'].prices[n_v.get_constraint_dro_key()] = inf_price
                cost += inf_price
        else:
            n_v.constraints['dro']['nurse_sr'].prices[n_v.get_constraint_dro_key()] = 0
    else:
        n_v.constraints['dro']['nurse_sr'].prices[n_v.get_constraint_dro_key()] = 0
    return cost


def set_same_order_price(n_v, max_order, n_v_list):
    """
    when the surgery schedule is yet not done a nurse will not be scheduled to two surgeries with the same order
    :param n_v: nurse variable object or None if init day value
    :param max_order: the maximum number of surgeries in the rooms today
    :param n_v_list:  list of all the nurse variables in a certain day
    :return: cost
    """
    cost = 0
    if n_v is None:
        for o in range(1, max_order + 1):
            n_list_by_o = [nv.value for nv in n_v_list if (nv.order == o and nv.value is not None)]
            if len(n_list_by_o) > 1:
                duplicates = set([x for x in n_list_by_o if n_list_by_o.count(x) > 1])
                for n in duplicates:
                    n_v_list[0].constraints['d']['overlapping'].prices[n_v_list[0].get_constraint_d_key(n.id)] = \
                        inf_price
                    cost += inf_price
    else:
        n_list_by_o = [nv.value for nv in n_v_list if nv.order == n_v.order]
        if len(n_list_by_o) > 1:
            # current value
            if n_v.value is not None:
                duplicates = n_list_by_o.count(n_v.value)
                if duplicates > 1:
                    n_v.constraints['d']['overlapping'].prices[n_v.get_constraint_d_key(n_v.value.id)] = inf_price
                    cost += inf_price
                else:
                    n_v.constraints['d']['overlapping'].prices[n_v.get_constraint_d_key(n_v.value.id)] = 0
            # prior value
            if n_v.prior_value is not None:
                duplicates = n_list_by_o.count(n_v.prior_value)
                if duplicates > 1:
                    n_v.constraints['d']['overlapping'].prices[
                        n_v.get_constraint_d_key(n_v.prior_value.id)] = inf_price
                    cost += inf_price
                else:
                    n_v.constraints['d']['overlapping'].prices[n_v.get_constraint_d_key(n_v.prior_value.id)] = 0
    return cost


def n_v_list_by_nurse(n_v_list, nurse):
    """
    formats a list of surgeon variables of a certain surgeon in a day
    :param n_v_list: list of nurse variables
    :param nurse: nurse object
    :return: list of nurse variables of a specific nurse
    """
    nurse_v_list = []
    for n_v in n_v_list:
        if n_v.value == nurse:
            nurse_v_list.append(n_v)
    return nurse_v_list


def num_unique_nurses_room(n_v_list):
    """
    counts the number of unique nurses from a list
    :param n_v_list: list of nurse variable objects
    :return number of unique surgeries in the room, num of surgeries in the room
        """
    num_nurses = 0
    nurse_list = []
    for n_v in n_v_list:
        if n_v.value is None:
            continue  # to enable partial days of surgery
        else:
            num_nurses += 1
            nurse_list.append(n_v.value)
    num_unique_nurse = len(set(nurse_list))
    return num_unique_nurse, num_nurses


def _set_min_nurse_per_room(n_v_list):
    """
    calculates the global cost of the min_nurse constraint  concerning the current nurses in a certain room,
    inserts once to constraint price table - the cost is normalized between 0 to 1 - hence duplicated by cost factor
    :param n_v_list: list of nurse variable objects - all the variables in a certain day and room
    """
    num_unique_nurses, num_nurses = num_unique_nurses_room(n_v_list)
    if num_unique_nurses == 2:
        cost = 0
    else:
        if num_nurses != 0:
            cost = (num_unique_nurses / num_nurses)
        else:
            cost = 0
    n_v_list[0].constraints['dr']['min_nurse'].prices[n_v_list[0].get_constraint_dr_key()] = cost * cost_factor
    return cost * cost_factor / 2




