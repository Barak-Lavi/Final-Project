import Nurse_Prices
import Prices
import math
from A_in_Surgery import SurgeryVariable_Anesthetist
from byRoom_Variable import Room_Variable
from Stagiaire_anesthetist import Stag_Anesth

cost_factor = 1_000
inf_price = 1_000_000


def set_stag_surgery_price(a_v, ward_id, with_sr):
    cost = 0
    if with_sr:
        if(a_v.value is not None) and (a_v.surgery_request is not None):
            if isinstance(a_v.value, Stag_Anesth):
                if a_v.surgery_request.surgery_type.st_id in a_v.value.skills[ward_id]:
                    a_v.constraints['dro']['stag_sr'].prices[a_v.get_constraint_dro_key()] = 0
                else:
                    a_v.constraints['dro']['stag_sr'].prices[a_v.get_constraint_dro_key()] = inf_price
                    cost += inf_price
            else:
                a_v.constraints['dro']['stag_sr'].prices[a_v.get_constraint_dro_key()] = 0
        else:
            a_v.constraints['dro']['stag_sr'].prices[a_v.get_constraint_dro_key()] = 0
    else:
        a_v.constraints['dro']['stag_sr'].prices[a_v.get_constraint_dro_key()] = 0
    return cost


def set_dro_prices(a_v, with_sr, ward_id, stable_schedule_flag=True):
    dro_cost = 0
    dro_cost += set_stag_surgery_price(a_v, ward_id, with_sr)
    if stable_schedule_flag:
        dro_cost += Nurse_Prices.set_stable_schedule_price(a_v, with_sr)
    return dro_cost


def set_dr_prices(var_list):
    """
    calculates and updates all the dr date_room constraints with cost
    :param var_list: list of operating anesthetists variables with same indices of day and room.
    """
    dr_cost = 0
    dr_cost += _set_min_anesthetist_per_room(var_list)
    return dr_cost


def set_d_prices(v_tuple, a_v, num_rooms_RM, next=False):
    """
    calculates and updates all the d day constraints with cost (overlapping, min_rm, all_diff)
    :param v_tuple:  tuple(floor manager variable, v_dict dictionary )
    v_dict - {ward: room: (room manager variable, [operating anes variable]), r2:(rm_v, [oav1, oav2..]}, w2...}
    :param a_v: anesthetist variable for single update or None for updating whole price
    :param num_rooms_RM: max number of rooms each room manager can be responsible for
    :param next: defines if an update was done in the variable value or prior price is calculated
    :return: daily cost
    """

    d_cost = 0
    oa_var_list = []  # list of all operating anesthetist variables in a certain day
    fm_v = v_tuple[0]
    v_dict = v_tuple[1]
    rm_var_list = []  # list of all room manager anesthetist variables in a certain day

    for w in v_dict:
        for r in v_dict[w]:
            rm_var_list.append(v_dict[w][r][0])
            for v in v_dict[w][r][1]:
                oa_var_list.append(v)

    if a_v is None:  # for calculating init day cost after surgeries update
        # set of the different operating anesthetists assigned to surgery this day
        oa_set = set(oa_v.value for oa_v in oa_var_list)
        for oa in oa_set:
            if oa is not None:
                # list of operating anesthetist variables of a certain anesthetist in a day
                oa_v_list = Nurse_Prices.n_v_list_by_nurse(oa_var_list, oa)
                if len(oa_v_list) > 0:
                    d_cost += Prices.set_overlapping_prices(oa_v_list, None)
        d_cost += set_all_diff_price(fm_v, num_rooms_RM)
        for rm_v in rm_var_list:
            d_cost += set_all_diff_price(rm_v, num_rooms_RM)
        d_cost += _set_min_rm_per_day(rm_var_list, num_rooms_RM)

    else:  # when costs need to be updated by a single change
        d_cost += set_all_diff_price(a_v, num_rooms_RM)
        if isinstance(a_v, SurgeryVariable_Anesthetist):
            if (a_v.value is not None) or (a_v.prior_value is not None):
                d_cost += Prices.surgeon_update_overlapping(a_v, oa_var_list, next)
        elif isinstance(a_v, Room_Variable):
            d_cost += _set_min_rm_per_day(rm_var_list, num_rooms_RM)
    return d_cost


def _set_min_anesthetist_per_room(var_list):
    """
    calculates the global cost of the min_anesthetist constraint concerning the current operating anesthetists in a
    certain room, inserts once to constraint price table - the cost is normalized between 0 to 1 - hence duplicated
    by cost factor
    :param var_list: list of operating anesthetists variable objects - all the variables of a certain day and room
    :return: cost
    """
    num_unique_anesthetists, num_anesthetists = Nurse_Prices.num_unique_nurses_room(var_list)

    if num_unique_anesthetists == 1:
        cost = 0
    else:
        if num_anesthetists != 0:
            cost = (num_unique_anesthetists/num_anesthetists)
        else:
            cost = 0
    var_list[0].constraints['dr']['min_anesthetist'].prices[var_list[0].get_constraint_dr_key()] = cost * cost_factor/2
    return cost * cost_factor/2


def _set_min_rm_per_day(rm_var_list, num_rooms_RM):
    """

    :param rm_var_list:
    :param num_rooms_RM:
    :return:
    """

    num_unique_managers, num_managers = Nurse_Prices.num_unique_nurses_room(rm_var_list)
    min_num_rm = math.ceil(num_managers/num_rooms_RM)
    if num_unique_managers > min_num_rm:
        cost = (num_unique_managers - min_num_rm) / num_unique_managers
    else:
        cost = 0
    rm_var_list[0].constraints['d']['min_rm'].prices[rm_var_list[0].get_constraint_d_key()] = cost * cost_factor
    return cost * cost_factor


def set_all_diff_price(a_v, num_rooms_RM):
    """
    verifies the legitimacy of the value. Floor Manager can only be Floor Manager. Room Manager can only by
    manager in a certain number of rooms and can't operate. So their are three scenarios of legit values for a certain
    Anesthetist (len(Fm) = 1, RM - Empty, OA - Empty)/(Fm empty, len(rm) <= num_rooms_RM, OA - Empty)/
    (FM empty, RM empty, OA - doesn't matter). each anesthetist holds a dictionary with keys referring to variables
    which their value is him.  the all diff constraint keys are of d_key for the fm - dr keys for all rm and dro keys
    for all oa variables
    :param a_v: anesthetist variable
    :param num_rooms_RM: limit of rooms each rm can manage
    :return:cost
    """
    cost = 0
    if a_v.value is not None:
        fm_assigned = a_v.value.assigned['FM']
        rm_assigned = a_v.value.assigned['RM']
        oa_assigned = a_v.value.assigned['OA']
        assigned = [fm_assigned, rm_assigned, oa_assigned]

        if isinstance(a_v, SurgeryVariable_Anesthetist):
            con_key = 'dro'
        elif isinstance(a_v, Room_Variable):
            con_key = 'dr'
        else:
            con_key = 'd'

        if not fm_assigned:  # fm_assigned is empty
            if not rm_assigned:  # rm assigned is empty
                # we are clear
                for key in oa_assigned:
                    a_v.constraints[con_key]['all_diff'].prices[key] = 0
                return 0
            else:
                if (len(rm_assigned) <= num_rooms_RM) and (not oa_assigned):
                    for key in rm_assigned:
                        a_v.constraints[con_key]['all_diff'].prices[key] = 0
                    return 0
                else:
                    for key in rm_assigned:
                        a_v.constraints[con_key]['all_diff'].prices[key] = inf_price
                        cost += inf_price
                    for key in oa_assigned:
                        a_v.constraints[con_key]['all_diff'].prices[key] = inf_price
                        cost += inf_price
                    return cost
        else:
            if (not rm_assigned) and (not oa_assigned):
                for key in fm_assigned:
                    a_v.constraints[con_key]['all_diff'].prices[key] = 0
                    return 0
            else:
                for assigned_list in assigned:
                    for key in assigned_list:
                        a_v.constraints[con_key]['all_diff'].prices[key] = inf_price
                        cost += inf_price
                return cost
    else:
        if isinstance(a_v, SurgeryVariable_Anesthetist):
            a_v.constraints['dro']['all_diff'].prices[a_v.get_constraint_dro_key()] = 0
        else:
            a_v.constraints['dr']['all_diff'].prices[a_v.get_constraint_dr_key()] = 0
        return cost




