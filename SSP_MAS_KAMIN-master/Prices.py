import sys
from math import factorial
from datetime import time

from R_in_Surgery import SurgeryVariable_SurgeryRequest

inf_price = 1_000_000_000  # for hard constraints
cost_factor = 1000

# stability factor:


def _set_stable_schedule_price(sr_v, ward_with_surgery_Team, no_good_sr, current_NCLO):
    cost = 0
    wst = [key for key in ward_with_surgery_Team if ward_with_surgery_Team[key]]  # defines the teams of whom the ward received
    need_stable = True
    # if all the teams are already allocated and good for the surgery we want it stable and want to fine a change
    # all the teams refer to all the teams the ward received until now
    for team in wst:
        if not sr_v.with_surgery_team[team]:
            need_stable = False
            break
    if sr_v.value is not None:
        '''if sr_v.value != sr_v.value_in_update and wst:
            if need_stable:
                # cost = cost_factor * len(wst)  # todo this - is the parameter that will define the stability of the schedule
                cost = sr_v.value.surgery_type.utility * 2
            elif sr_v.value in no_good_sr:
                cost = sr_v.value.surgery_type.utility * (no_good_sr[sr_v.value]/current_NCLO)
        else:
            if not need_stable:
                # cost = cost_factor * len(wst)  # todo this - is the parameter that will define the stability of the schedule
                cost = sr_v.value.surgery_type.utility * 2'''
        if sr_v.value != sr_v.value_in_update and wst:
            if need_stable:
                # cost = cost_factor * len(wst)  # todo this - is the parameter that will define the stability of the schedule
                cost = sr_v.value.surgery_type.utility * 2
                if sr_v.value in no_good_sr:
                    cost += sr_v.value.surgery_type.utility * (no_good_sr[sr_v.value]/current_NCLO) * 1
            elif sr_v.value in no_good_sr:
                cost = sr_v.value.surgery_type.utility * (no_good_sr[sr_v.value]/current_NCLO) * 1
        else:
            if not need_stable:
                # cost = cost_factor * len(wst)  # todo this - is the parameter that will define the stability of the schedule
                cost = sr_v.value.surgery_type.utility * 2
                if sr_v.value in no_good_sr:
                    cost += sr_v.value.surgery_type.utility * (no_good_sr[sr_v.value]/current_NCLO) * 1
    sr_v.constraints['dro']["stable_schedule"].prices[sr_v.get_constraint_dro_key()] = cost
    return cost


def set_dro_prices(sr_v, s_v, ward, num_surgeries_day, mutual, ty, with_surgery_Team, no_good_sr, stable_schedule_flag,
                   current_NCLO):
    """
    calculates and updates all the  'dro' date_room_order constraints with cost
    :param no_good_sr: set of surgery requests we don't want to choose (not able to find nurse/anesthetist)
    :param ty: the type of variable updated : surgery request variable type/ surgeon variable type/ none if init sol
    :param mutual: boolean true if it is init solution or any other case of mutual sr_v and s_v update
     false any other case - i.e.single variable was updated
    :param sr_v: surgery request variable of a specific date room order
    :param s_v: surgery variable that matches the sr_v
    :param ward: a ward object
    :param num_surgeries_day: current num of surgeries in the day
    :param current_NCLO: current NCLO of agent - to normalize no_good cost
    :return cost of dro constraints
    """
    dro_cost = 0
    if mutual or isinstance(sr_v, ty):  # if the init_sol cost is calculated or sr_v was updated
        dro_cost += _set_surgery_order_price(sr_v, ward, num_surgeries_day) #BARAK V
        dro_cost += _set_surgery_date_price(sr_v, ward)#BARAK V
        dro_cost += _set_surgeon_patient_price(sr_v, s_v, ward)#BARAK V
        dro_cost += _set_all_diff_price(sr_v)
        # dro_cost += _set_schedule_gap_price(sr_v, ward)
        # dro_cost = dro_cost /6
        if stable_schedule_flag:
            dro_cost += _set_stable_schedule_price(sr_v, with_surgery_Team, no_good_sr, current_NCLO)
    else:  # only s_v was updated
        dro_cost += _set_surgeon_patient_price(sr_v, s_v, ward)

    return dro_cost


def set_dr_prices(var_list, ward, mutual, ty, by_units):
    """
    calculates and updated all the dr date room constraints wit cost
    :param by_units: boolean if true counts num unique units if false counts num unique surgery types
    :param ty: the type of variable updated : surgery request variable type/ surgeon variable type/ none if init sol
    :param mutual: boolean true if it is init solution false any other case - i.e. variable was updated
    :param var_list: list of variables tuple : (sr_v,s_v) with same indexes. sr_v surgery request variable,
    s-v surgeon variable - the list will be of the tuples concerning a certain day and room
    :param ward: ward object
    """
    dr_cost = 0
    if mutual or isinstance(var_list[0][0], ty):
        sr_v_list = [t[0] for t in var_list]
        dr_cost += _set_homo_hetero_price(sr_v_list, ward, by_units)
        dr_cost += _set_total_duration_price(sr_v_list, ward)
        dr_cost += _set_efficiency_price(sr_v_list, ward)
    return dr_cost


def set_d_prices(d_dict, s_v, ty,  sr_v, next=False):
    """
    calculates and updated all the d date constraints wit cost
    :param next: boolean defines if an update was done in the variable value or prior price is calculated
    :param ty: the type of variable updated : surgery request variable type/ surgeon variable type/ none if init sol
    or tuple update
    :param d_dict: dictionary of format key - room value - list of tuples [(sr_v,sr),(sr_v,sr)...]
    rooms concerning a certain ward in a certain day
    :param s_v: surgoen variable or None if sent from init_day and not value update
    """
    d_cost = 0
    s_v_list = []  # list of all the surgeon variables in a certain day
    for lt in list(d_dict.values()):  # lt list of tuples
        for t in lt:
            s_v_list.append(t[1])

    if s_v is None:  # for calculating init day cost
        surgeon_set = set(s_v.value for s_v in s_v_list)  # set of the different surgeons assigned to surgery this day
        for surgeon in surgeon_set:
            if surgeon is not None:
                surgeon_v_list = _s_v_list_by_surgeon(s_v_list, surgeon)
                # list of surgeon variables of a certain surgeon in a day
                if len(surgeon_v_list) > 0:
                    d_cost += set_overlapping_prices(surgeon_v_list, None)
    else:  # when costs need to be updated by a single change
        if s_v.value is not None or s_v.prior_value is not None:
            # - we want to check overlapping also when sr_v is updated because times change
            if isinstance(s_v, ty) or ty == type(None):
                # surgeon variable was updated or tuple was updated cks current value
                d_cost += surgeon_update_overlapping(s_v, s_v_list, next)
            else:  # surgery request variable updated -
                # if with specific senior value or prior value the certain senior needs to be checked for overlapping
                # if len(sr_v.value.specific_senior) == 9:
                if sr_v.value.specific_senior is not None:
                    d_cost += surgeon_update_overlapping(s_v, s_v_list, next)
                elif sr_v.prior_value is not None:
                    # if len(sr_v.prior_value.specific_senior) == 9:
                    if sr_v.prior_value.specific_senior is not None:
                        d_cost += surgeon_update_overlapping(s_v, s_v_list, next)
                # times changed need to check overlapping for the rest of the
                # surgeries in the day because their time changed - wand to check also before change and after change
                room_num = s_v.room.num
                for j in range(s_v.order, len(d_dict[room_num]) + 1):
                    s_v = d_dict[room_num][j - 1][1]  # s_v
                    if s_v.value is None:
                        break
                    surgeon_v_list = _s_v_list_by_surgeon(s_v_list, s_v.value)
                    d_cost += set_overlapping_prices(surgeon_v_list, s_v)

    return d_cost


def surgeon_update_overlapping(s_v, s_v_list, next):
    d_cost = 0
    if s_v.value is not None:
        surgeon_v_list = _s_v_list_by_surgeon(s_v_list, s_v.value)
        # list of surgeon variables of a certain surgeon in a day
        d_cost += set_overlapping_prices(surgeon_v_list, s_v)
    # checks prior value
    if (s_v.prior_value is not None) and next:
        surgeon_v_list = _s_v_list_by_surgeon(s_v_list, s_v.prior_value)
        if len(surgeon_v_list) > 0:
            d_cost += set_overlapping_prices(surgeon_v_list, None)
    return d_cost


def _s_v_list_by_surgeon(s_v_list, surgeon):
    """
    formats a list of surgeon variables of a certain surgeon in a day
    :param s_v_list: list of surgeon variables
    :param surgeon: surgeon object
    :return: list of surgeon variables of a specific surgeon
    """
    surgeon_v_list = []
    for s_v in s_v_list:
        if s_v.value == surgeon:
            surgeon_v_list.append(s_v)
    return surgeon_v_list


def _set_efficiency_price(sr_v_list, ward):
    """
    calculates the cost of the rooms efficiency
    :param sr_v_list: list of all the surgery request variables in a certain day and room
    :param ward: ward object - that recieved this room in the allocation
    :return: cost of the room's efficiency
    """
    room_total_duration = 0
    cost = 0
    cons_w = ward.constraints_w["efficiency"]  # overall constraint weight
    for sr_v in sr_v_list:
        if sr_v.value is not None:
            room_total_duration += sr_v.value.duration
    if room_total_duration < ward.d_duration:
        cost = cons_w * (ward.d_duration - room_total_duration)/ward.d_duration
    sr_v_list[0].constraints['dr']['efficiency'].prices[sr_v_list[0].get_constraint_dr_key()] = cost * cost_factor
    return cost * cost_factor


def _set_schedule_gap_price(sr_v, ward):
    cost = 0
    cons_w = ward.constraints_w['schedule_gap']  # overall constraint weight
    schedule_gaps = ward.schedule_gaps
    max_delta = max(schedule_gaps)
    if sr_v.value is not None:
        if sr_v.value.schedule_date is not None:
            time_delta = abs((sr_v.value.schedule_date - sr_v.day).days)
        else:
            time_delta = 1
        if time_delta > max_delta:
            cost = inf_price / cost_factor
        else:
            for gap in schedule_gaps:
                if time_delta <= gap:
                    cost = schedule_gaps[gap] * cons_w * (time_delta / max_delta)
                    break
    # todo maybe for the demo need that the cost will be cost * cost_factor * 10 - so we keep noa & yair schedules
    sr_v.constraints['dro']['schedule_gap'].prices[sr_v.get_constraint_dro_key()] = cost * cost_factor * 400
    return cost * cost_factor * 100


def _set_surgery_order_price(sr_v, ward, current_num_surgeries):
    """
           calculates the cost of the variable order concerning the current surgery request value,
            inserts to constraint price table - the cost is normalized between 0 to 1 - hence duplicated by 100
           :param sr_v: surgery request variable object
           :param ward: the ward of the current surgery request
           :param current_num_surgeries: the current num of surgeries in the variable room

    """
    cost = 0
    order_cost = 0
    if sr_v.value is not None:
        max_c = ward.max_attributes["cancellations"]  # max cancellation in ward current RTG
        min_c = ward.min_attributes["cancellations"]  # min cancellation in ward current RTG
        param_c_w = ward.parameter_w["surgery_order"]["num_cancellations"]  # cancellation parameter weight in cost
        max_i = ward.max_attributes["complexity"]  # max importance/complexity in ward current RTG
        min_i = ward.min_attributes["complexity"]  # min importance/complexity in ward current RTG
        param_i_w = ward.parameter_w["surgery_order"][
            "complexity"]  # importance/complexity parameter weight in cost
        max_u = ward.max_attributes["urgency"]  # max urgency in ward current RTG
        min_u = ward.min_attributes["urgency"]  # min urgency in ward current RTG
        param_u_w = ward.parameter_w["surgery_order"]["urgency"]  # cancellation parameter weight in cost
        max_age_cut = max(ward.max_attributes["min_birth_d"].age_cut(sr_v.day),
                          ward.max_attributes["max_birth_d"].age_cut(sr_v.day))  # max age_cut in ward current RTG
        min_age_cut = min(ward.max_attributes["min_birth_d"].age_cut(sr_v.day),
                          ward.max_attributes["max_birth_d"].age_cut(sr_v.day))  # min age_cut in ward current RTG
        param_a_w = ward.parameter_w["surgery_order"]["age"]  # age cut parameter weight in cost
        cons_w = ward.constraints_w["surgery_order"]  # overall constraint weight
        age_cut = sr_v.value.age_cut(sr_v.day)
        param_w_sum = param_a_w + param_u_w + param_i_w + param_c_w

        # param_l_w * ((max_l - sr_v.value.duration) / max_l - min_l) + @took down BARAK
        # cost= cons_w * (((((max_c - sr_v.value.num_cancellations) / (max_c - min_c + sys.float_info.epsilon))**param_c_w) *
        #     (((int(max_i) - int(sr_v.value.complexity)) / (int(max_i) - int(min_i) + sys.float_info.epsilon))**param_i_w) *
        #     (((max_u - sr_v.value.urgency) / (max_u - min_u + sys.float_info.epsilon))**param_u_w) *
        #     (((max_age_cut - age_cut) / (max_age_cut - min_age_cut + sys.float_info.epsilon))**param_a_w))**(1/param_w_sum))
        #### withot age cat
        cost = cons_w * (((((max_c - sr_v.value.num_cancellations) / (
                    max_c - min_c + sys.float_info.epsilon)) ** param_c_w) *
                          (((int(max_i) - int(sr_v.value.complexity)) / (
                                      int(max_i) - int(min_i) + sys.float_info.epsilon)) ** param_i_w) *
                          (((max_u - sr_v.value.urgency) / (max_u - min_u + sys.float_info.epsilon)) ** param_u_w)) ** (
                                     1 / param_w_sum))
        order_cost = cost*((current_num_surgeries + sys.float_info.epsilon - sr_v.order) / current_num_surgeries)
    sr_v.constraints['dro']["surgery_order"].prices[sr_v.get_constraint_dro_key()] = order_cost*cost_factor
    return order_cost*cost_factor


def _set_surgery_date_price(sr_v, ward):
    """
           calculates the cost of the variable date concerning the current surgery request value,
            inserts to constraint price table - the cost is normalized between 0 to 1 - hence duplicated by 100
           :param sr_v: surgery request variable object
           :param ward:  the ward of the current surgery request
           """
    cost = 0
    if sr_v.value is not None:
        cons_w = ward.constraints_w["surgery_date"]  # overall constraint weight
        max_c = ward.max_attributes["cancellations"]  # max cancellation in ward current RTG
        min_c = ward.min_attributes["cancellations"]  # min cancellation in ward current RTG
        param_c_w = ward.parameter_w["surgery_date"]["num_cancellations"]  # cancellation parameter weight in cost
        max_u = ward.max_attributes["urgency"]  # max urgency in ward current RTG
        min_u = ward.min_attributes["urgency"]  # min urgency in ward current RTG
        param_u_w = ward.parameter_w["surgery_date"]["urgency"]  # cancellation parameter weight in cost
        max_rdc = ward.max_attributes["entrance_d_cut"]  # earliest entrance date in current RTG, rdc - referral date cut
        min_rdc = ward.min_attributes["entrance_d_cut"]  # latest entrance date in current RTG, rdc - referral date cut
        param_rdc_w = ward.parameter_w["surgery_date"][
            "entrance_date"]  # referral date cut parameter weight in cost
        param_w_sum = param_rdc_w + param_u_w + param_c_w
        max_wait = ward.max_attributes["entrance_d"].calc_waiting_days(sr_v.day)  # max waiting time in days in
        min_wait = ward.min_attributes["entrance_d"].calc_waiting_days(sr_v.day)  # min waiting time in days in
        # current RTG
        my_wait = sr_v.value.calc_waiting_days(sr_v.day)  # current surgery request in variable waiting time
        # referring to the date of the variable

        cost = cons_w * (((((max_c - sr_v.value.num_cancellations) / (max_c - min_c +sys.float_info.epsilon))**param_c_w) *
               (((max_u - sr_v.value.urgency) /(max_u - min_u +sys.float_info.epsilon))**param_u_w)*
               (((max_rdc - sr_v.value.entrance_date_cut) / (max_rdc- min_rdc +sys.float_info.epsilon))**param_rdc_w))**(1/param_w_sum)) * \
               ((max_wait + sys.float_info.epsilon - my_wait) / max_wait)

    sr_v.constraints['dro']['surgery_date'].prices[sr_v.get_constraint_dro_key()] = cost*cost_factor
    return cost*cost_factor


def _set_surgeon_patient_price(sr_v, s_v, ward):
    """
    calculates the binary cost of the surgeon_patient_skill  concerning the current surgery request value,
    inserts to constraint price table - the cost is normalized between 0 to 1 - hence duplicated by 100
    if the surgeon does not have the appropiate skill then price is inf (10,000)
    :param sr_v: surgery request variable object
    :param s_v: surgeon variable object
    :param ward: ward object
    :return:
    """
    cost = 0
    if sr_v.value is not None and s_v.value is not None:
        if sr_v.value.surgery_type not in s_v.value.surgical_grades:
            cost += inf_price / cost_factor
        else:
            cons_w = ward.constraints_w["surgeon_patient"]
            max_i = ward.max_attributes["complexity"]
            min_i = ward.min_attributes["complexity"]
            param_i_w = ward.parameter_w["surgeon_patient"]["complexity"]
            max_grade_skill = ward.max_attributes["skill"][sr_v.value.surgery_type.st_id]
            min_grade_skill = ward.min_attributes["skill"][sr_v.value.surgery_type.st_id]
            param_s_w = ward.parameter_w['surgeon_patient']['skill']
            param_w_sum = param_i_w + param_s_w
            cost = cons_w * (((((int(max_i) - int(sr_v.value.complexity)) / (int(max_i) - int(min_i) + sys.float_info.epsilon))**param_i_w)*
                             (((max_grade_skill - s_v.value.surgical_grades[sr_v.value.surgery_type]) /
                                          (max_grade_skill - min_grade_skill + sys.float_info.epsilon))**param_s_w))**(1/param_w_sum))
    sr_v.constraints['dro']['surgeon_patient'].prices[sr_v.get_constraint_dro_key()] = cost*cost_factor
    return cost*cost_factor


def _set_all_diff_price(sr_v):
    """
    Hard constraint if the surgery request of this variable is already assigned to a different variable cost is inf i.e
    inf price in the constraint dictionary for every surgery slot that contains a surgery request that is assigned more
    than once
    :param sr_v: surgery request variable object
    """
    cost = 0
    if sr_v.value is not None:
        if len(sr_v.value.assigned) > 1:
            for key in sr_v.value.assigned:
                sr_v.constraints['dro']['all_diff'].prices[key] = inf_price
                cost += inf_price
            return cost
        else:
            sr_v.constraints['dro']['all_diff'].prices[sr_v.get_constraint_dro_key()] = 0
            return 0
    else:
        sr_v.constraints['dro']['all_diff'].prices[sr_v.get_constraint_dro_key()] = 0
        return 0


def _set_homo_hetero_price(sr_v_list, ward, by_units):
    """
    calculates the global cost of the homo_herero constraint  concerning the current surgery requests in a certain room,
    inserts once to constraint price table - the cost is normalized between 0 to 1 - hence duplicated by cost factor
    :param sr_v_list: list of surgery request variable objects - all the variables in a certain day and room
    :param ward: ward object
    :param by_units: boolean if true counts num unique units if false counts num unique surgery types
    """
    cons_w = ward.constraints_w["homo_hetero"]  # constraint weight
    num_unique_surgeries, num_surgeries = num_unique_surgeries_room(sr_v_list, ward, by_units)
    cost = 0
    if num_unique_surgeries == 1:
        cost = 0

    elif num_surgeries > 0:
        cost = cons_w * (num_unique_surgeries / num_surgeries)

    else:
        print('lets see')
    sr_v_list[0].constraints['dr']['homo_hetero'].prices[sr_v_list[0].get_constraint_dr_key()] = cost * cost_factor
    return cost * cost_factor


def num_unique_surgeries_room(sr_v_list, ward, by_units):
    """
    counts the number of unique surgery types from a list
    :param ward: ward object
    :param by_units: boolean - if true counts num unique units if false counts num unique surgery types
    :param sr_v_list: list of surgery request variable objects
    :return number of unique surgeries in the room, num of surgeries in the room
    """
    num_surgeries = 0
    surgery_type_list = []
    for sr_v in sr_v_list:
        if sr_v.value is None:
            break
        else:
            num_surgeries += 1
            if by_units:
                surgery_type_list.append(ward.get_unit_st(sr_v.value.surgery_type))
            else:
                surgery_type_list.append(sr_v.value.surgery_type)
    num_unique_surgery = len(set(surgery_type_list))
    return num_unique_surgery, num_surgeries


def _set_total_duration_price(sr_v_list, ward):
    """
    Hard Constraint - gives a price of inf_price to a room which the total duration of it is larger than the wards day
    surgical duration
    :param sr_v_list: list of surgery request variable objects - all the variables in a certain day and room
    :param ward: ward object
    """

    td = 0  # total duration
    max_surgery_time = ward.d_duration
    for sr_v in sr_v_list:
        if sr_v.value is None:
            break
        else:
            td += sr_v.value.duration
        if td > max_surgery_time:
            sr_v.constraints['dr']['total_duration'].prices[sr_v.get_constraint_dr_key()] = inf_price * (td - max_surgery_time)
            return inf_price * (td - max_surgery_time)
        else:
            sr_v.constraints['dr']['total_duration'].prices[sr_v.get_constraint_dr_key()] = 0
    return 0


def set_overlapping_prices(surgeon_v_list, s_v):
    """
    :param surgeon_v_list: list of surgeon variables of a certain surgeon in a day
    :param s_v: surgeon variable
    :return: price
    """
    if s_v is None:  # for initial day and prior value
        s_id = surgeon_v_list[0].value.id
        for i in range(len(surgeon_v_list) - 1):
            for j in range(i + 1, len(surgeon_v_list)):
                if surgeon_v_list[i].room != surgeon_v_list[j].room:
                    overlapping = check_overlapping(surgeon_v_list[i], surgeon_v_list[j])
                    if overlapping:
                        surgeon_v_list[i].constraints['d']['overlapping'].prices[
                            surgeon_v_list[i].get_constraint_d_key(s_id)] = inf_price
                        return inf_price
        surgeon_v_list[0].constraints['d']['overlapping'].prices[surgeon_v_list[0].get_constraint_d_key(s_id)] = 0
    else:
        for v in surgeon_v_list:  # checks overlapping with chosen s_v
            if v != s_v and v.room != s_v.room:
                overlapping = check_overlapping(v, s_v)  # boolean
                if overlapping:
                    s_v.constraints['d']['overlapping'].prices[s_v.get_constraint_d_key(s_v.value.id)] = inf_price
                    return inf_price
        surgeon_v_list.remove(s_v)
        for i in range(len(surgeon_v_list)):
            for j in range(i + 1, len(surgeon_v_list)):
                if surgeon_v_list[i].room != surgeon_v_list[j].room:
                    overlapping = check_overlapping(surgeon_v_list[i], surgeon_v_list[j])
                    if overlapping:
                        s_v.constraints['d']['overlapping'].prices[s_v.get_constraint_d_key(s_v.value.id)] = inf_price
                        return inf_price
        s_v.constraints['d']['overlapping'].prices[s_v.get_constraint_d_key(s_v.value.id)] = 0
    return 0


def check_overlapping(s_v1, s_v2):
    # import SSP_initialization

    if s_v1.start_time <= s_v2.start_time:
        first_v = s_v1
        second_v = s_v2
    else:
        first_v = s_v2
        second_v = s_v1

    if second_v.start_time <= calc_end_time(start_time=first_v.end_time, duration_min=30):
        return True
    else:
        return False


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
    end_time = time(hour=hour, minute=minute)
    return end_time
