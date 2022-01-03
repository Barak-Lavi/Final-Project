from Allocating_Agent import Allocating_Agent
from N_in_Surgery import SurgeryVariable_Nurse
from Constraint import Constraint
from copy import deepcopy
from datetime import datetime
from decimal import Decimal
import Nurse_Prices
import Chief_Agent
import Static
import random
import math
import Prices

# random.seed(4)

cost_factor = 1_000
inf_price = 1_000_000


def by_highest_grade(nurse_grade_tuple):
    """
    resolves as key parameter to sorted
    :param nurse_grade_tuple: tuple (nurse, grade)
    :return: grade
    """
    return nurse_grade_tuple[1]


def reduce_domain(co_nurse, domain):
    """
    removes from a domain of nurses tuple the co nurse of surgeries
    important that not the actuoal domain will be sent
    :param co_nurse: nurse object
    :param domain: list of tuples (nurse, grade)
    :return: the formatted domain
    """
    for nt in domain:
        if nt[0] == co_nurse:
            domain.remove(nt)
    return domain


def tuple_updated(tu):
    tuple_u = False
    if (tu[0].value is None and tu[1].value is None) or (tu[0].prior_value is None and tu[1].prior_value is None):
        tuple_u = True
    return tuple_u


class Nurse_Agent(Allocating_Agent):

    def __init__(self, day, hospital, general_post_office):
        self.nurses = hospital.nurses  # set
        self.factor_weights = {'nurse_suitability': 0.3, 'schedule_ratio': 0.5, 'discrimination': 0.2}
        # init_factor_grades
        # {'nurse_suitability' : weight, 'schedule_ratio' : weight , 'discrimination' : weight}
        super(Nurse_Agent, self).__init__(day, hospital, general_post_office, 'n')
        self._init_solution_day(by_highest_grade)
        self.score = self.calc_utility(with_cost_update=True)
        self.simulated_annealing_by_day()
        self.init_solution = deepcopy(self.v_dict)
        self.counter = 0  # got incremented in SA
        '''self.schedules_by_counter[self.next_schedule_by_counter] = self.init_solution
        self.utility_by_counter[self.next_schedule_by_counter] = 0
        self.next_schedule_by_counter += self.steps_for_counter'''

    def init_variables(self):
        """
        initiates the data structure the variables will be kept in
        :return: dictioanry {'schedule_date' : {ward : {room : [(SN_V, CN_V),(SN_V, CN_V)..], r2: [(SN_V, CN_V)..]},
        """
        variables = {self.schedule_date: {}}
        n_constraints = self._init_constraints()
        for w in self.room_allocations:
            variables[self.schedule_date][w] = {}
            for r in self.room_allocations[w]:
                variables[self.schedule_date][w][r] = []
                max_slots_rd, initial_surgery_duration = w.max_slots_room_day(self.schedule_date, r)
                # initial_surgery_duration = w.d_duration // max_slots_rd
                start_time = w.start_d_hour
                for i in range(max_slots_rd):
                    end_time = Prices.calc_end_time(start_time, initial_surgery_duration)
                    variables[self.schedule_date][w][r].append(
                        (SurgeryVariable_Nurse(room=r, day=self.schedule_date, order=i,
                                               domain=['SN', w.w_id, self.nurses], constraints=n_constraints,
                                               start_time=start_time, end_time=end_time),
                         SurgeryVariable_Nurse(room=r, day=self.schedule_date, order=i,
                                               domain=['CN', w.w_id, self.nurses], constraints=n_constraints,
                                               start_time=start_time, end_time=end_time)))
                    start_time = end_time

        return variables

    def _init_constraints(self):
        """
        initializes the different constraint objects (each one holding an empty dictionary of prices) and gathers them into
        dictionary
        :return: dictionary of constraints of variable for modularity the main key is
         d_r_o / d_r / d (date, room , order) then each key holds a dictionary of all
        the constraints that suits this key - the key is the name of the constraint and the value is the constraint object
        """
        # Soft Constraints
        min_nurse_per_room_constraint = Constraint({})  # global key - date_room
        stable_schedule_constraint = Constraint({})
        # Hard Constraints
        nurse_overlapping_constraint = Constraint({})
        # binary key - date (pass through all the the surgeon variables of this date of a specific ward)
        nurse_sr_constraint = Constraint({})

        n_constraints = {'dro': {'nurse_sr': nurse_sr_constraint, 'stable_schedule': stable_schedule_constraint},
                         'dr': {'min_nurse': min_nurse_per_room_constraint},
                         'd': {'overlapping': nurse_overlapping_constraint}}

        return n_constraints

    def simulated_annealing_by_day(self, init_sol_param=None, genetic=False, stable_schedule_flag=True,
                                   random_selection=True):
        """
            performs SA on a single day
            could receive d_dict of a specific ward and d - but just to be able to play with it for now...
            :param stable_schedule_flag: boolean - if to take into acount stable schedule costs
            :param genetic: boolean - True if sa is used for init population in genetic algorithm , if so list of all the schedules
            is returned
            :param random_selection: if to select randomly from domain or by max DeltaE
            :param init_sol_param: function that determines the parameter which according to it the initial state will be
            generated
            """
        random.seed(4)
        if genetic:
            g = []
        global t
        plot_list = []
        current_value = self.calc_utility \
            (with_cost_update=True, next=True, stable_schedule_flag=stable_schedule_flag)  # float the value of the day
        best_value = current_value
        best_schedule = deepcopy(self.v_dict)
        plot_list.append([0, current_value])
        if genetic:
            g.append((deepcopy(self.v_dict), current_value))  # tuple of schedule and its value
        num_changes = 0
        for t in Static.infinity():
            T = Chief_Agent.sa_schedule(t, "Linear")
            st = Chief_Agent.sa_stopping_temp("Linear")
            if T <= st:  # none hard constraints are broken - not sure best_value > 0 is needed
                if genetic:
                    return g
                else:
                    self.update_schedule(best_schedule, stable_schedule_flag=stable_schedule_flag)
                    # return best_value, t, num_changes, plot_list, best_schedule
                    # Static.simulated_graphs(plot_list, 'Nurses')
                    break
            else:
                chosen_v, delta_E, tu, ward = self._select_successor(stable_schedule_flag=stable_schedule_flag,
                                                                     random_selection=random_selection)
                if delta_E > 0:
                    self.increment_counter()
                    num_changes += 1
                    current_value = self.calc_value(chosen_v)
                    plot_list.append([t, current_value])
                    if genetic:
                        g.append((deepcopy(self.v_dict), current_value))
                    if current_value > best_value:
                        best_value = current_value
                        best_schedule = deepcopy(self.v_dict)
                else:
                    p = Decimal(math.exp(delta_E / T))
                    rnd = random.uniform(0, 1)
                    if rnd < p:  # change but only to feasible solutions
                        current_value = self.calc_value(chosen_v)
                        if current_value > best_value:
                            best_value = current_value
                            best_schedule = deepcopy(self.v_dict)
                        plot_list.append([t, current_value])
                        if genetic:
                            g.append((deepcopy(self.v_dict), current_value))
                        num_changes += 1
                    else:  # don't change
                        self.return_to_prior_value(chosen_v, tu, ward, stable_schedule_flag=stable_schedule_flag)
                    self.increment_counter()

    def _init_solution_day(self, parameter):
        """
        generates an initial solution - the solution is determined by the parameter. updates d-dict variables
        assigning each variable a certain value from its domain. General heuristic same nurse for whole day in a room -
        and full days for each ward - not taking into consideration option of half days
        :param parameter: the parameter that according to it the solution is generated - it will be the name of a
        function that serves as a key for the sort comparison
        """
        random.seed(4)
        allocated_nurses = []
        for w in self.v_dict[self.schedule_date]:
            for r in self.v_dict[self.schedule_date][w]:
                cn_v = self.v_dict[self.schedule_date][w][r][0][1]
                sn_v = self.v_dict[self.schedule_date][w][r][0][0]
                cn_sorted_domain = \
                    sorted(cn_v.domain, key=parameter, reverse=True)
                sn_sorted_domain = \
                    sorted(sn_v.domain, key=parameter, reverse=True)
                cn_nurse = None
                sn_nurse = None
                if self.with_sr[w]:  # heuristic - domain is reduced to only nurses certified for the surgery type of
                    # surgery request in variable - the same nurse won't be in different rooms but their can
                    # be more than a single nurse for room if the "best" nurse isn't certified for a certain surgery
                    r_allocated_nurses = set()
                    for nt in self.v_dict[self.schedule_date][w][r]:
                        cn_sorted_domain = self.with_sr_reduce_domain(cn_sorted_domain, nt[1], w.w_id)
                        sn_sorted_domain = self.with_sr_reduce_domain(sn_sorted_domain, nt[0], w.w_id)
                        if len(cn_sorted_domain) > 1:  # None is included in the domain
                            for cn in cn_sorted_domain:
                                if (cn[0] not in allocated_nurses) and (cn[0] is not None):
                                    nt[1].value = cn[0]
                                    r_allocated_nurses.add(cn[0])
                                    break
                            if nt[1].value is not None and len(sn_sorted_domain) > 1:
                                for sn in sn_sorted_domain:
                                    if (sn[0] not in allocated_nurses) and (sn[0] != nt[1].value) and (
                                            sn[0] is not None):
                                        nt[0].value = sn[0]
                                        r_allocated_nurses.add(sn[0])
                                        break
                            if nt[0].value is None:
                                nt[1].value = None
                    allocated_nurses.extend(r_allocated_nurses)
                else:
                    for cn in cn_sorted_domain:
                        if (cn[0] not in allocated_nurses) and (cn[0] is not None):
                            cn_nurse = cn[0]
                            allocated_nurses.append(cn[0])
                            break
                    if cn_nurse is not None:
                        for sn in sn_sorted_domain:
                            if (sn[0] not in allocated_nurses) and (sn[0] is not None):
                                sn_nurse = sn[0]
                                allocated_nurses.append(sn[0])
                                break
                        if sn_nurse is None:
                            cn_nurse = None
                    for nt in self.v_dict[self.schedule_date][w][r]:  # nt - nurse tuple
                        nt[0].value = sn_nurse
                        nt[1].value = cn_nurse

    def calc_utility(self, with_cost_update, next=False, stable_schedule_flag=True):
        """
        calculates the value of a day overall utility - over all costs of all the variables in the day. Calculates the value
        from zero with no prior value calculated, that means not the value after an update of a single variable
        :param stable_schedule_flag: boolean - if to take into account stable schedule costs
        :return: double the price of the day - total utility - total costs
        """
        init_day_cost = 0
        s_n_u = 0  # suitability nurse utility
        w_r_u = 0  # ward_ratio_utility
        if with_cost_update:
            init_day_cost += Nurse_Prices.set_d_prices(self.v_dict[self.schedule_date], n_v=None, next=next)
        scheduled_wards = 0
        num_rooms = 0
        for w in self.v_dict[self.schedule_date]:
            w_s_n_u = 0  # ward suitability nurse utility
            total_w_num_surgeries = 0
            scheduled_w_num_surgeries = 0
            for r in self.v_dict[self.schedule_date][w]:
                num_rooms += 1
                if with_cost_update:
                    init_day_cost += Nurse_Prices.set_dr_prices(self.v_dict[self.schedule_date][w][r])
                for t in self.v_dict[self.schedule_date][w][r]:  # t- tuple (sr_v, s_v)
                    sn_v = t[0]
                    cn_v = t[1]
                    if self.with_sr[w]:
                        if sn_v.surgery_request is not None:
                            total_w_num_surgeries += 1
                    else:
                        total_w_num_surgeries += 1
                    if with_cost_update:
                        init_day_cost += Nurse_Prices.set_dro_prices \
                            (sn_v, w.w_id, self.with_sr[w], stable_schedule_flag=stable_schedule_flag)
                        init_day_cost += Nurse_Prices.set_dro_prices \
                            (cn_v, w.w_id, self.with_sr[w], stable_schedule_flag=stable_schedule_flag)
                    if sn_v.value is not None:
                        scheduled_w_num_surgeries += 1
                        # {day: {'CN': {ward_id: {r1: grade, r2: grade...}, 'SN': {ward_id: {r1: grade, r2: grade...}}
                        w_s_n_u += \
                            sn_v.value.ward_grades[datetime.strptime(self.schedule_date, '%Y-%m-%d').date()][
                                sn_v.n_type][
                                w.w_id][r.num]
                    if cn_v.value is not None:
                        w_s_n_u += \
                            cn_v.value.ward_grades[datetime.strptime(self.schedule_date, '%Y-%m-%d').date()][
                                cn_v.n_type][
                                w.w_id][r.num]
            if total_w_num_surgeries > 0:
                w_r_u += self.ward_strategy_grades[w.w_id] * (scheduled_w_num_surgeries / total_w_num_surgeries)
            if w_s_n_u > 0:
                s_n_u += w_s_n_u / (scheduled_w_num_surgeries * 2)
            if scheduled_w_num_surgeries > 0:
                scheduled_wards += 1
        if scheduled_wards == 0:
            init_day_utility = 0
        else:
            init_day_utility = (cost_factor * num_rooms * 2) * \
                               (self.factor_weights['nurse_suitability'] * (s_n_u / scheduled_wards) +
                                self.factor_weights['schedule_ratio'] * w_r_u +
                                self.factor_weights['discrimination'] * (scheduled_wards / len(self.room_allocations)))
        return init_day_utility - init_day_cost

    def _select_successor(self, stable_schedule_flag=True, random_selection=True):
        """
        selects a random variable and changes its value randomly from the domain , calculates the difference of the
        total solution price due to the change in the solution. \
        The difference is calculated by the subtraction of prior price from next price
        prior price - utility - cost of the specific variable that changed
        next price - utility - cost after the change
        :param stable_schedule_flag: boolean - if to take into acount stable schedule costs
        :return: chosen variable and the difference in the total price of the solution, and the tuple chosen
        """

        d_dict = self.v_dict[self.schedule_date]
        ward = random.choice(list(d_dict))
        room_num = random.choice(list(d_dict[ward]))
        t = random.choice(d_dict[ward][room_num])  # tuple (sn_v, cn_v) - surgery
        chosen_v = random.choice(list(t))  # chosen variable
        delta_E = 0
        if chosen_v.value is None:  # two options - 1: variable with no assignation 2: the domain is empty
            delta, change = self.update_tuple_value(t, ward, stable_schedule_flag=stable_schedule_flag,
                                                    random_selection=random_selection)
            delta_E += delta
            return chosen_v, delta_E, t, ward
        else:
            delta_E += self.update_variable_value(chosen_v, t, ward, stable_schedule_flag=stable_schedule_flag,
                                                  random_selection=random_selection)
            return chosen_v, delta_E, t, ward

    def single_variable_change(self, random_selection=True, stable_schedule_flag=True):
        """
        changes the nurses of a single surgery - the first one which current nurses does not match the
        wards needs and deltaE > 0.
        """
        for w in self.v_dict[self.schedule_date]:
            for r in self.v_dict[self.schedule_date][w]:
                for t in self.v_dict[self.schedule_date][w][r]:
                    for v in t:
                        if v.need_stable and (v.value != v.value_in_update):
                            if v.value is None:
                                deltaE, change = self.update_tuple_value(t, w, random_selection=random_selection,
                                                                         stable_schedule_flag=stable_schedule_flag)
                            elif v.value_in_update is None:
                                deltaE = self.update_variable_value(v, t, w, random_selection=random_selection,
                                                                    stable_schedule_flag=stable_schedule_flag)
                            else:
                                continue
                        elif (not v.need_stable) and (v.value == v.value_in_update):
                            if v.value is None:
                                deltaE, change = self.update_tuple_value(t, w, random_selection=random_selection,
                                                                         stable_schedule_flag=stable_schedule_flag)
                            else:
                                deltaE = self.update_variable_value(v, t, w, random_selection=random_selection,
                                                                    stable_schedule_flag=stable_schedule_flag)
                        else:
                            continue
                        if deltaE > 0:
                            self.increment_counter()
                            self.score = self.calc_utility(with_cost_update=True, next=True,
                                                           stable_schedule_flag=stable_schedule_flag)
                            return v, t, w
                        elif deltaE != 0:
                            self.return_to_prior_value(v, t, w, stable_schedule_flag=stable_schedule_flag)
                        self.increment_counter()
        return False, False, False

    def single_variable_change_explore(self, random_selection=True, stable_schedule_flag=True):
        """
        matches the DSA without stable schedule chooses a random variable to change more exploration
        """
        for i in range(200):
            chosen_v, delta_E, tu, ward = self._select_successor(random_selection=random_selection,
                                                                 stable_schedule_flag=stable_schedule_flag)
            if delta_E > 0:
                self.score = self.calc_utility(with_cost_update=True, next=True,
                                               stable_schedule_flag=stable_schedule_flag)
                self.increment_counter()
                return chosen_v, tu, ward
            elif delta_E != 0:
                self.return_to_prior_value(chosen_v, tu, ward, stable_schedule_flag=stable_schedule_flag)
            self.increment_counter()
        return False, False, False

    def count_schedule_changes(self, schedule):
        num_changes = 0
        for w, b_w in zip(self.v_dict[self.schedule_date], schedule[self.schedule_date]):
            for r, b_r in zip(self.v_dict[self.schedule_date][w], schedule[self.schedule_date][b_w]):
                for t, b_t in zip(self.v_dict[self.schedule_date][w][r], schedule[self.schedule_date][b_w][b_r]):
                    for v, b_v in zip(t, b_t):
                        if (v.value is None) and (b_v.value is None):
                            continue
                        elif (v.value is None) or (b_v.value is None):
                            num_changes += 1
                        elif v.value.id != b_v.value.id:
                            num_changes += 1
        return num_changes

    def dsa_sc_iteration(self, mail, change_func=None, random_selection=True, stable_schedule_flag=True, no_good_flag=True):
        for m in mail:
            self.update_schedule_by_ward(m.content['schedule'], m.content['ward'], m.content['ward_copy'],
                                         stable_schedule_flag=stable_schedule_flag)
        self.calc_score_updated_schedule(stable_schedule_flag=stable_schedule_flag)
        for m in mail:
            self.update_counter(m.content['counter'])
        # curr_schedule = deepcopy(self.v_dict)
        stable_schedule_price = self.get_stable_schedule_costs()  # we do not want to count these costs because they
        # were updated in the update schedule by ward and do not reflect the last iteration
        curr_score = self.score
        chosen_v, tu, w = getattr(self, change_func)(random_selection=random_selection,
                                                     stable_schedule_flag=stable_schedule_flag)
        if chosen_v:
            if self.score < curr_score:  # alternative schedule is worst then curr schedule
                self.return_to_prior_value(chosen_v, tu, w, stable_schedule_flag=stable_schedule_flag)
                self.score = curr_score
            else:  # alternative schedule is equal valued or better than curr schedule
                change_probability = random.random()
                if change_probability > 0.7:  # will keep alternative schedule only for a chance of 70%
                    self.return_to_prior_value(chosen_v, tu, w, stable_schedule_flag=stable_schedule_flag)
                    self.score = curr_score
        self.send_mail()
        # we return the cost that resulted from the scheduled of the last iteration after receiving the allocating
        # agents schedule i.e. the real price
        # return {'schedule': curr_schedule, 'score': curr_score + stable_schedule_price}
        return {'score': curr_score + stable_schedule_price}

    def update_tuple_value(self, t, ward, stable_schedule_flag=True, random_selection=True):
        """
        chooses new values for a tuple of variables which had none values and calculates the difference in the sol_value
        utility - cost
        :param stable_schedule_flag: boolean - if to take into account stable schedule costs
        :param ward: ward object of the ward which received room in room allocation
        :param t: tuple (sn_v,cn_v)
        :param random_selection: boolean if to choose randomly from domain or by max deltaE
        :return: delta E the difference of the total price of solution
        """
        sn_v = t[0]
        cn_v = t[1]
        change = False
        if len(sn_v.domain) > 1 and len(cn_v.domain) > 1:
            change = True
        else:
            # change isn't possible
            return 0, change
        original_cn_prior_value = sn_v.value
        original_sn_prior_value = cn_v.value
        if self.with_sr[ward]:
            if sn_v.surgery_request is None:
                # No surgery so no need of nurse
                return 0, False
            cn_v_domain = self.with_sr_reduce_domain(cn_v.domain, cn_v, ward.w_id)
            cn_v_domain.remove(tuple([None, 0]))
            if not len(cn_v_domain) > 0:  # only none in domain
                return 0, False
            if random_selection:
                chosen_cn_v_value = random.choice(cn_v_domain)
                sn_v_domain = self.with_sr_reduce_domain(sn_v.domain, sn_v, ward.w_id, chosen_cn_v_value[0])
                sn_v_domain.remove(tuple([None, 0]))
                if not len(sn_v_domain) > 0:  # only none in domain - only a single nurse for both sn and cn
                    return 0, False
                self.increment_counter()
                chosen_sn_v_value = random.choice(sn_v_domain)
                sn_v.prior_value = sn_v.value
                cn_v.prior_value = cn_v.value
                prior_price = self.calc_price_by_variable([sn_v, cn_v], ward, stable_schedule_flag=stable_schedule_flag)
            else:
                next_price = None
                chosen_cn_v_value = None
                chosen_sn_v_value = None
                sn_v_domain = self.with_sr_reduce_domain(sn_v.domain, sn_v, ward.w_id)
                sn_v_domain.remove(tuple([None, 0]))
                if not len(sn_v_domain) > 1:
                    return 0, False
                sn_v.prior_value = sn_v.value
                cn_v.prior_value = cn_v.value
                prior_price = self.calc_price_by_variable([sn_v, cn_v], ward, stable_schedule_flag=stable_schedule_flag)
                for cn in cn_v_domain:
                    # self.increment_counter()
                    for sn in sn_v_domain:
                        if sn[0] != cn[0]:
                            self.increment_counter()
                            temp_next_price = self.calc_price_by_variable([sn_v, cn_v], ward, next=True,
                                                                          stable_schedule_flag=stable_schedule_flag)
                        else:
                            continue
                        if next_price is None:
                            next_price = temp_next_price
                            chosen_sn_v_value = sn
                            chosen_cn_v_value = cn
                        elif next_price < temp_next_price:
                            next_price = temp_next_price
                            chosen_sn_v_value = sn
                            chosen_cn_v_value = cn
                        cn_v.prior_value = original_cn_prior_value
                        sn_v.piror_value = original_sn_prior_value
                        self.return_to_prior_value(cn_v, t, ward, stable_schedule_flag)
                    '''while True:
                        temp_sn_v_value = random.choice(sn_v_domain)
                        if temp_sn_v_value[0] != cn[0]:
                            break
                    temp_next_price = self.calc_price_by_variable([sn_v, cn_v], ward, next=True,
                                                                  stable_schedule_flag=stable_schedule_flag)
                    if next_price is None:
                        next_price = temp_next_price
                        chosen_sn_v_value = temp_sn_v_value
                        chosen_cn_v_value = cn
                    elif next_price < temp_next_price:
                        next_price = temp_next_price
                        chosen_sn_v_value = temp_sn_v_value
                        chosen_cn_v_value = cn
                    self.return_to_prior_value(cn_v, t, ward, stable_schedule_flag)'''

        else:
            cn_v_domain = cn_v.domain.copy()
            cn_v_domain.remove(tuple([None, 0]))
            chosen_cn_v_value = random.choice(cn_v_domain)
            sn_v_domain = sn_v.domain.copy()
            sn_v_domain.remove(tuple([None, 0]))
            sn_v_domain = reduce_domain(chosen_cn_v_value[0], sn_v_domain)
            if not len(sn_v_domain) > 0:
                return 0, False
            chosen_sn_v_value = random.choice(sn_v_domain)
            sn_v.prior_value = sn_v.value
            cn_v.prior_value = cn_v.value
            prior_price = self.calc_price_by_variable([sn_v, cn_v], ward, stable_schedule_flag=stable_schedule_flag)

        sn_v.value = chosen_sn_v_value[0]
        cn_v.value = chosen_cn_v_value[0]
        sn_v.prior_value = original_sn_prior_value
        cn_v.prior_value = original_cn_prior_value
        next_price = self.calc_price_by_variable([sn_v, cn_v], ward, next=True,
                                                 stable_schedule_flag=stable_schedule_flag)
        return next_price - prior_price, change

    def with_sr_reduce_domain(self, domain, n_v, w_id, co_nurse=None):
        """
        creates a list of reduced domain - the domain will only contain the nurses which are certified for the
        surgery request of the variable also the domain will not contain the co nurse of surgery because any way
        it wouldn't be a legal selection, and we exclude from the domain overlapping nurses
        :param co_nurse: nurse object - co nurse in surgery
        :param domain: set of nurse tuples (nurse object, grade) - original domain of variable
        :param n_v: nurse variable object
        :param w_id: int ward id of ward which recieved room of variable in room allocation
        :return: list of tuples - reduced domain
        """
        surgery_type = n_v.surgery_request.surgery_type
        domain.remove(tuple([None, 0]))
        overlapping_nurses = self.get_overlapping_variables(n_v)
        reduced_sorted_domain = [t for t in domain if (surgery_type.st_id in t[0].skills[n_v.n_type][w_id]) and
                                 (t[0] not in overlapping_nurses)]
        domain.append(tuple([None, 0]))
        if co_nurse is not None:
            for nt in reduced_sorted_domain:
                if nt[0] == co_nurse:
                    reduced_sorted_domain.remove(nt)
                    break
        reduced_sorted_domain.append(tuple([None, 0]))
        return reduced_sorted_domain

    def update_variable_value(self, chosen_v, t, ward, stable_schedule_flag=True, random_selection=True):
        """
            chooses randomly new value for variable from domain and calculates the difference in the sol_value utility - cost
            :param ward: ward object of the ward which recieved room in room allocation
            :param chosen_v: chosen variable to change value
            :param t: tuple (sn_v,cn_v)
            :param stable_schedule_flag: boolean - if to take into acount stable schedule costs
            :param random_selection: boolean if to select randomly from domain or by max of next price
            :return: delta E the difference of the total price of solution
            """

        i = t.index(chosen_v)
        if len(chosen_v.domain) > 1:
            chosen_v_original_prior_value = chosen_v.value
            while True:
                co_nurse = t[abs(i - 1)].value  # can not be None .. (would have gone to update tuple value)
                if self.with_sr[ward]:
                    if chosen_v.surgery_request is None:
                        return 0  # there is no surgery so no nurse is needed no need for sa changing
                    domain = self.with_sr_reduce_domain(chosen_v.domain, chosen_v, ward.w_id, co_nurse)
                    chosen_v.prior_value = chosen_v.value
                    prior_price = self.calc_price_by_variable([chosen_v], ward,
                                                              stable_schedule_flag=stable_schedule_flag)
                    if random_selection:
                        chosen_value = random.choice(domain)
                        self.increment_counter()
                        '''if chosen_value[0] == chosen_v.value:
                            continue'''
                        if chosen_value[0] is None:
                            delta_e = self.cancel_nurses_in_surgery(t, ward, stable_schedule_flag=stable_schedule_flag)
                            return delta_e
                    else:
                        next_price = None
                        chosen_value = None
                        for n in domain:
                            self.increment_counter()
                            if n[0] == chosen_v.value:
                                continue
                            if n[0] is None:
                                temp_next_price = self.cancel_nurses_in_surgery(t, ward, stable_schedule_flag) \
                                                  + prior_price
                            else:
                                chosen_v.value = n[0]
                                temp_next_price = self.calc_price_by_variable([chosen_v], ward, next=True,
                                                                              stable_schedule_flag=stable_schedule_flag)
                            if next_price is None:
                                next_price = temp_next_price
                                chosen_value = n
                            elif next_price < temp_next_price:
                                chosen_value = n
                                next_price = temp_next_price
                            chosen_v.prior_value = chosen_v_original_prior_value
                            self.return_to_prior_value(chosen_v, t, ward, stable_schedule_flag)

                else:
                    chosen_v.prior_value = chosen_v.value
                    prior_price = self.calc_price_by_variable([chosen_v], ward,
                                                              stable_schedule_flag=stable_schedule_flag)
                    while True:
                        chosen_value = random.choice(list(chosen_v.domain))
                        if chosen_value[0] is None:
                            delta_e = self.cancel_nurses_in_surgery(t, ward, stable_schedule_flag=stable_schedule_flag)
                            return delta_e
                        if chosen_value[0] != co_nurse:
                            break
                if chosen_value[0] != chosen_v.value:
                    break
        else:  # only None in domain
            # no legal change so (no change was done) there is no need of sa changing
            return 0
        chosen_v.prior_value = chosen_v_original_prior_value
        chosen_v.value = chosen_value[0]
        next_price = self.calc_price_by_variable([chosen_v], ward, next=True,
                                                 stable_schedule_flag=stable_schedule_flag)
        return next_price - prior_price


    def calc_price_by_variable(self, l_n_v, ward, next=False, with_utility=True, stable_schedule_flag=True):
        """
        calculates the difference in the schedule value depending only on the value of this variable. takes in
        the calculation in to account only what is affected by this value
        :param with_utility: boolean - if utility calculation is needed
        :param ward: ward object - the ward which received the variable room in room allocation
        :param l_n_v: list of nurse variable object = the relevant variable updated : if tuple updated the list will
        consist of CN_v and SN_v if single value updated the list will have a single element
        :param next: defines if an update was done in the variable value or prior price is calculated
        :param stable_schedule_flag: boolean - if to take into acount stable schedule costs
        :return: new utility - new cost
        """
        cost = 0
        for n_v in l_n_v:
            cost += Nurse_Prices.set_dro_prices \
                (n_v, ward.w_id, self.with_sr[ward], stable_schedule_flag=stable_schedule_flag)
            cost += Nurse_Prices.set_d_prices(self.v_dict[self.schedule_date], n_v, next)

        cost += Nurse_Prices.set_dr_prices(self.v_dict[self.schedule_date][ward][l_n_v[0].room])
        if with_utility:
            utility = self.calc_utility_by_variable(l_n_v, ward, next)
            return utility - cost
        else:
            return cost

    def calc_utility_by_variable(self, l_n_v, ward, next=False):
        """
        calculates a heuristic of the utility difference caused by the change of the variables value
        :param l_n_v: list of nurse variable object = the relevant variable updated : if tuple updated the list will
        consist of CN_v and SN_v if single value updated the list will have a single element
        :param ward: ward object - the ward which recieved the variable room in room allocation
        :param next: boolean True if next price is calculated i.e. the prior value is taken into account comparing
        to the current variable value
        :return: the utility difference
        """
        utility = 0
        if l_n_v[0].value is not None:
            for n_v in l_n_v:
                if n_v.value is None:
                    print('problem')
                # the grade of the nurse
                utility += self.factor_weights['nurse_suitability'] * \
                           n_v.value.ward_grades[datetime.strptime(self.schedule_date, '%Y-%m-%d').date()][n_v.n_type][
                               ward.w_id][n_v.room.num]
            if len(l_n_v) > 1:
                utility = utility / 2
            if next:
                if len(l_n_v) > 1:
                    if l_n_v[0].prior_value is None:
                        # ratio of ward surgeries enlarged
                        utility += self.factor_weights['schedule_ratio'] * self.ward_strategy_grades[ward.w_id] * 2
                        if not self.ward_is_scheduled(ward, l_n_v[0]):
                            # this variable is the single one scheduled for the ward
                            utility += self.factor_weights['discrimination']
        elif next:
            if len(l_n_v) > 1:
                if l_n_v[0].prior_value is not None:
                    # ratio of schedule got smaller
                    utility -= self.factor_weights['schedule_ratio'] * self.ward_strategy_grades[ward.w_id]
                    if not self.ward_is_scheduled(ward):  # the ward was scheduled to a single surgery and now it is not
                        utility -= self.factor_weights['discrimination']
        return utility * cost_factor * 2

    def ward_is_scheduled(self, ward, n_v=None):
        """
        checks if a certain ward has nurses scheduled for its operations -
        :param ward: ward object - the ward we want to check for
        :param n_v: nurse variable object - if not None the check will be done if any other surgeries are scheduled with
        nurses except this variable if it is None it will be a general check to see if the ward surgeries are scheduled
        with nurses
        :return: boolean True - if ward is scheduled false- if not scheduled
        """

        for r in self.v_dict[self.schedule_date][ward]:
            for t in self.v_dict[self.schedule_date][ward][r]:
                nv = t[0]
                if nv.value is not None:
                    if n_v is not None:
                        if nv != n_v:
                            return True
                    else:
                        return True
        return False

    def cancel_nurses_in_surgery(self, t, ward, stable_schedule_flag=True):
        """
        updates the values of a surgery to be None and caculates the difference in the price
        :param t: tuple (sn_v, cn_v)
        :param ward: ward object - the ward which recieived the room of the variable in room allocation
        :param stable_schedule_flag: boolean - if to take into account stable schedule costs
        :return: difference in price - delta e
        """
        for v in t:
            v.prior_value = v.value
        prior_price = self.calc_price_by_variable(list(t), ward, stable_schedule_flag=stable_schedule_flag)
        for v in t:
            v.value = None
        next_price = self.calc_price_by_variable(list(t), ward, next=True, stable_schedule_flag=stable_schedule_flag)
        return next_price - prior_price

    def calc_value(self, v=None):
        """
        calculates the total cost of the solution node
        :param v: nurse variable object holding the constraints dictionary
        :return: float total cost
        """
        if not v:
            w = list(self.room_allocations.keys())[0]
            r = next(iter(self.room_allocations[w]))
            v = self.v_dict[self.schedule_date][w][r][0][0]
        cost = 0
        constraints = v.constraints
        for con_key in constraints:
            for cons in constraints[con_key]:
                cost += sum(constraints[con_key][cons].prices.values())
        utility = self.calc_utility(with_cost_update=False)
        return utility - cost

    def return_to_prior_value(self, chosen_v, tu, ward, stable_schedule_flag=True):
        """
        returns the solution to the prior solution changes the values of the concerned variables back
        :param ward:
        :param tu: tuple of (sr_v, s_v) of the chosen_v
        :param stable_schedule_flag: boolean - if to take into account stable schedule costs
        :param chosen_v: tuple/variable depends if there was a mutual change -in case of adding new surgery in the day
        """
        if tuple_updated(tu):
            l_n_v = [tu[0], tu[1]]
        else:
            l_n_v = [chosen_v]
        for v in l_n_v:
            prior_update = v.value
            v.value = v.prior_value
            v.prior_value = prior_update
        self.calc_price_by_variable \
            (l_n_v, ward, next=True, with_utility=False, stable_schedule_flag=stable_schedule_flag)

    def update_schedule_by_ward(self, schedule, ward, ward_copy, stable_schedule_flag=True):
        """
        updates v_dict by schedule given from ward - the update includes adding the surgery request for each variable,
        updating the surgery times, cancelling the nurses for variables with no surgery request i.e. no surgery,
        updating the field with_sr to True so calculations will be done matchingly. The function also updates the
        value of the schedule i.e. updates all constraint dictionaries with new prices
        :param ward: ward object of ward whom's schedule is being updated
        :param schedule: dictionary of wards schedule - deep copy of original schedule
        {ward_object: {date : room: [tuples(sr_v, s_v)]}}
        :param ward_copy: copy of ward object as it is in schedule
        :param stable_schedule_flag: boolean - if to take into account stable schedule costs
        :return: the updated value of the schedule
        """
        full_solution = True
        for r in self.v_dict[self.schedule_date][ward]:
            t_list = self.v_dict[self.schedule_date][ward][r]
            for i in range(len(t_list)):
                start_time = schedule[ward_copy][self.schedule_date][r.num][i][0].start_time
                end_time = schedule[ward_copy][self.schedule_date][r.num][i][0].end_time
                sr = schedule[ward_copy][self.schedule_date][r.num][i][0].value
                t = self.v_dict[self.schedule_date][ward][r][i]
                # old version:
                # for v in t:
                for j in range(len(t)):
                    v = t[j]
                    v.start_time = start_time
                    v.end_time = end_time
                    v.surgery_request = sr
                    v.prior_value = None
                    if sr is None:
                        # v.prior_value = None
                        v.value = None
                        v.need_stable = True
                    elif v.value is not None:
                        if v.surgery_request.surgery_type.st_id in v.value.skills[v.n_type][ward.w_id]:
                            # allocation that works
                            v.need_stable = True
                        else:
                            # new
                            v.value = None
                            if j:  # j = 1
                                co_nurse_var = t[j - 1]
                            else:  # j = 0
                                co_nurse_var = t[j + 1]
                            co_nurse_var.value = None
                            co_nurse_var.need_stable = False
                            co_nurse_var.value_in_update = co_nurse_var.value
                            # end new
                            v.need_stable = False
                            full_solution = False
                    else:  # v.value is None but v.surgery request is not None
                        v.need_stable = False
                        full_solution = False
                    v.value_in_update = v.value

        self.with_sr[ward] = True
        # bv, t, nc, pl, bs = self.simulated_annealing_by_day(by_highest_grade)
        # Static.simulated_graphs(pl, 'Nurses_withSR_' + str(ward.w_id))
        return full_solution

    def update_schedule(self, best_schedule, stable_schedule_flag=True):
        """
        updates v_dict to have best_schedule values (best schedule is a deep copy so we want to continue working
        with the same objects and not new ones)
        :param stable_schedule_flag: boolean - if to take into account stable schedule costs
        :param best_schedule: dictionary same format of v_dict but deep copied hence new objects
        """
        for w, b_w in zip(self.v_dict[self.schedule_date], best_schedule[self.schedule_date]):
            for r, b_r in zip(self.v_dict[self.schedule_date][w], best_schedule[self.schedule_date][b_w]):
                for t, b_t in zip(self.v_dict[self.schedule_date][w][r], best_schedule[self.schedule_date][b_w][b_r]):
                    for i in range(len(t)):
                        if b_t[i].value is not None:
                            t[i].value = self.find_nurse(b_t[i].value.id)
                        else:
                            t[i].value = None
                        if b_t[i].prior_value is not None:
                            t[i].prior_value = self.find_nurse(b_t[i].prior_value.id)
                        else:
                            t[i].prior_value = None
                        if self.with_sr[w]:
                            t[i].start_time = b_t[i].start_time
                            t[i].end_time = b_t[i].end_time
                            if b_t[i].surgery_request is not None:
                                t[i].surgery_request = w.find_surgery_request(b_t[i].surgery_request.request_num)
                            else:
                                t[i].surgery_request = None
        self.score = self.calc_utility(with_cost_update=True, next=True, stable_schedule_flag=stable_schedule_flag)

    def find_nurse(self, nurse_id):
        """
        searches for nurse with this nurse_id
        :param nurse_id:  int nurse id
        :return: nurse object with nurse_id
        """
        for n in self.nurses:
            if n.id == nurse_id:
                return n

    def get_overlapping_variables(self, n_v):

        overlapping_var_nurses = set()
        for w in self.v_dict[self.schedule_date]:
            for r in self.v_dict[self.schedule_date][w]:
                if r != n_v.room:
                    for t in self.v_dict[self.schedule_date][w][r]:
                        if t[0].start_time > n_v.end_time:
                            break
                        elif t[0].end_time < n_v.start_time:
                            continue
                        elif Prices.check_overlapping(n_v, t[0]):
                            if t[0].value is not None:
                                overlapping_var_nurses.add(t[0].value)
                                overlapping_var_nurses.add(t[1].value)
        return overlapping_var_nurses

    def get_stable_schedule_costs(self):
        for w in self.v_dict[self.schedule_date]:
            for r in self.v_dict[self.schedule_date][w]:
                for t in self.v_dict[self.schedule_date][w][r]:
                    stable_schedule_price = sum(t[0].constraints['dro']['stable_schedule'].prices.values())
                    return stable_schedule_price

    def cancel_overlapping_nurses(self, overlap_d_keys):
        """
        cancels nurses allocation if overlapping after ward update - the overlapping occured because of change in ward
        schedule- we don't want to take these allocations into account
        cancels only if the variable actually has overlapping and only one of the overlapped not both.
        param overlap_d_keys: list of d_keys 'YYYY-MM-DD-NID' of nurses which have an overlap
        """
        for key in overlap_d_keys:
            n_id = int(key[11:])
            n_t_list = self.get_nurse_tup(n_id)  # all tuples nurse has been allocated in - necessarily sr is not None
            for i in range(len(n_t_list) - 1):
                for j in range(i + 1, len(n_t_list)):
                    if n_t_list[i][0].room != n_t_list[j][0].room:
                        n_i_var = [v for v in n_t_list[i] if v.value.id == n_id][0]
                        n_j_var = [v for v in n_t_list[j] if v.value.id == n_id][0]
                        overlapping = Prices.check_overlapping(n_i_var, n_j_var)
                        if overlapping:
                            co_nurse_var = [v for v in n_t_list[i] if v.value.id != n_id][0]
                            n_i_var.value = None
                            n_i_var.value_in_update = n_i_var.value
                            n_i_var.need_stable = False
                            co_nurse_var.value = None
                            co_nurse_var.value_in_update = co_nurse_var.value
                            co_nurse_var.need_stable = False
                            break

    def get_nurse_tup(self, n_id):
        """
        :param n_id: int nurse id
        :return: list of tuples in whom the nurse is allocated
        """
        n_tup_list = []
        for w in self.v_dict[self.schedule_date]:
            for r in self.v_dict[self.schedule_date][w]:
                for t in self.v_dict[self.schedule_date][w][r]:
                    for v in t:
                        if v.value is not None:
                            if v.value.id == n_id:
                                n_tup_list.append(t)
                                break
                        else:
                            break
        return n_tup_list

    def get_overlapping_d_keys(self):
        """
        :return: list of d_keys of overlapping nurses
        """
        for w in self.v_dict[self.schedule_date]:
            for r in self.v_dict[self.schedule_date][w]:
                for t in self.v_dict[self.schedule_date][w][r]:
                    overlapping_price = sum(t[0].constraints['d']['overlapping'].prices.values())
                    if overlapping_price > 0:
                        return [n for n in t[0].constraints['d']['overlapping'].prices if
                                t[0].constraints['d']['overlapping'].prices[n] > 0]
                    else:
                        return []

    def calc_score_updated_schedule(self, full_solution=None, stable_schedule_flag=True):
        """
        calculates the score of schedule after being updated by all wards new surgery requests - cancells
        nurses overlapping surgeries - caused by change of times of surgeries by wards - because change of surgery
        requests allocated.
        :param full_solution: boolean - checks if needed if a full solution is kept- if nurses allocation
        were cancelled because of overlapping than full solution is false
        :param stable_schedule_flag: boolean - if to take into account stable schedule costs
        :return: full solution
        """
        updated_value = self.calc_utility(with_cost_update=True, next=True, stable_schedule_flag=stable_schedule_flag)
        nurses_with_overlap = self.get_overlapping_d_keys()
        if nurses_with_overlap:  # list of d_keys of nurses which overlapping price > 0
            self.cancel_overlapping_nurses(nurses_with_overlap)
            full_solution = False
            updated_value = self.calc_utility(with_cost_update=True, next=True,
                                              stable_schedule_flag=stable_schedule_flag)
        self.score = updated_value
        return full_solution
