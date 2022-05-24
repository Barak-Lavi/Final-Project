from decimal import Decimal
import math
import random
from copy import deepcopy
import Prices
import Static
from Message import Message
from Agent import Participating_Agent
from Constraint import Constraint
from R_in_Surgery import SurgeryVariable_SurgeryRequest
from S_in_Surgery import SurgeryVariable_Surgeon
from datetime import timedelta

# random.seed(4)
random.SystemRandom()
# cost_factor = 1_000


def reduct_sr_domain(sr_domain):
    """
    reduction of domain that all diff constraint is satisfied
    :param sr_domain: set of surgery requests
    :return: set of surgery requests
    """
    legit_domain = set()
    for sr in sr_domain:
        if len(sr.assigned) < 1:
            legit_domain.add(sr)
    if len(legit_domain) > 0:
        return legit_domain
    else:
        return False


def reduct_s_domain(s_domain, st):
    """
    reduces the domain to only values of surgeons which have the skill to operate this surgery type
    :param domain: set of surgeons
    :param st: surgery type
    :return: set of surgeons
    """
    legit_domain = set()
    for s in s_domain:
        if st in s.surgical_grades:
            legit_domain.add(s)
    if len(legit_domain) > 0:
        return legit_domain
    else:
        return False


def by_waiting_time(sr):
    """
    support function for a key in sorted
    :param sr:A surgery request
    :return: the entrance date of the sr - so the domain can be sorted by it
    """
    return sr.entrance_date


def by_schedule_date(sr):
    """
    support function for a key in sorted
    :param sr:A surgery request
    :return: the schedule date of the sr - so the domain can be sorted by it
    """
    return sr.schedule_date


def sa_schedule(t, param):
    """
    generates a schedule for simulated annealing algo
    determining T0
    :param t: int current iteration
    :param param: string - type of schedule Linear or Exponential
    :return: current temp
    """
    if param == "Exponential":
        T0 = 100
        a = 0.9  # cooling ratio
        return T0 * math.pow(a, t)
    if param == "Linear":
        # ToDo find a way to calculate delta_Cmax which is the max difference between two schedule neighbours and let it
        # be the value of T0
        # work with about 2000 iterations check if u prefer do divide by 10 both- the way it is now lets big changes happend
        T0 = 0.1 * 10000
        a = 0.0005 * 1000  # decrement factor
        return T0 - a * t


def sa_stopping_temp(param):
    """
    generates a stopping condition
    :param param: string - type of schedule Linear or Exponential
    :return: the stopping temperature
    """
    if param == "Exponential":
        return 1.00E-05
    if param == "Linear":
        return 1.66


class Ward_Agent(Participating_Agent):
    # v_dict: format: {w1:{d1:{r1:[(sr_v,sr),(sr_v,sr)...], r2:[(sr_v,sr),(sr_v,sr)]...},
    # d2:{r1:[(sr_v,sr),(sr_v,sr)...], r2:[(sr_v,sr),(sr_v,sr)...]}}, w2:{d1:{r1:[(sr_v,sr),(sr_v,sr)...],.... }
    def __init__(self, ward, day, case, general_post_office):
        self.ward = ward  # ward object
        self.case = case
        self.with_surgery_Team = {'Nurse': False, 'Anesthetist': False, 'Equipment': False}
        self.no_good_sr = {}
        super(Ward_Agent, self).__init__(day, general_post_office, ward.w_id)
        self.num_surgeries = self._init_solution_day(by_waiting_time)
        self.score = self._set_init_value_day(self.num_surgeries)
        # todo addition for not full allocation
        if self.num_surgeries:
            self.simulated_annealing_by_day()
        self.neighbours = ['a', 'e', 'n']
        self.init_solution = deepcopy(self.v_dict)
        self.init_num_surgeries = deepcopy(self.num_surgeries)
        self.counter = 1  # got incremented in SA
        # self.schedules_by_counter[self.next_schedule_by_counter] = \
        #    {'schedule': self.init_solution, 'num_surgeries': self.init_num_surgeries}
        # self.utility_by_counter[self.next_schedule_by_counter] = 0
        # self.next_schedule_by_counter += self.steps_for_counter

        # self.send_mail()

    def init_value(self):
        self.with_surgery_Team = {'Nurse': False, 'Anesthetist': False, 'Equipment': False}
        self.update_schedule(self.init_solution, self.init_num_surgeries)
        self.no_good_sr = {}
        self.clear_counter(chief_agent=True)

    def init_variables(self):
        variables = {}
        d_dict = {}
        w_dict = {}
        v = []
        sr_constraints, s_constraints = self._init_constraints()
        room_allocation = self.ward.room_allocation
        # set of surgery types objects - by the surgeon shifts in a day
        surgeon_st = self.ward.st_surgeons_day(self.schedule_date)
        if self.schedule_date in room_allocation:
            for room in room_allocation[self.schedule_date]:
                v.clear()
                max_slots, initial_surgery_duration = self.ward.max_slots_room_day(self.schedule_date, room)
                for i in range(1, max_slots + 1):
                    v.append((SurgeryVariable_SurgeryRequest(room, day=self.schedule_date, order=i,
                                                             sr_domain=[self.ward.RTG, surgeon_st, self.ward],
                                                             constraints=sr_constraints, ),
                              SurgeryVariable_Surgeon(room, day=self.schedule_date, order=i,
                                                      surgeon_domain=self.ward.ward_surgeons,
                                                      constraints=s_constraints)))
                d_dict[room.num] = v.copy()
        w_dict[self.schedule_date] = d_dict
        variables[self.ward] = w_dict
        return variables

    def _init_constraints(self):
        """
        initializes the different constraint objects (each one holding an empty dictionary of prices) and gathers them into
        dictionary
        :return: two dictionaries of constraints one for every type of variable surgery_request_variable and surgeon
        variable for modularity the main key is d_r_o / d_r / d (date, room , order) then each key holds a dictionary of all
        the constraints that suits this key - the key is the name of the constraint and the value is the constraint object
        """
        # soft constraints
        surgery_order_constraint = Constraint({})  # unary key - date_room_order
        surgery_date_constraint = Constraint({})  # unary key - date_room_order
        surgeon_patient_constraint = Constraint({})  # binary key - date_room_order
        homogeneity_heterogeneity_constraint = Constraint({})  # global key - date_room
        efficiency_constraint = Constraint({})  # global key - date_room
        schedule_gap_constraint = Constraint({})  # unary key - date_room_order
        stable_schedule_constraint = Constraint({})

        # hard constraints
        # surgeon_skill_constraint = Constraint({}) # binary key - date_room_order
        all_diff_constraint = Constraint({})  # binary key - date_room_order (if len(assigned) of sr>1 than inf)
        surgeon_overlapping_constraint = Constraint({})
        # binary key - date (pass through all the the surgeon variables of this date of a specific ward)
        total_duration_constraint = Constraint({})  # global key date_room

        sr_constraints = {'dro': {'surgery_order': surgery_order_constraint, 'surgery_date': surgery_date_constraint,
                                  'surgeon_patient': surgeon_patient_constraint, 'all_diff': all_diff_constraint,
                                  'schedule_gap': schedule_gap_constraint,
                                  'stable_schedule': stable_schedule_constraint},
                          'dr': {'homo_hetero': homogeneity_heterogeneity_constraint,
                                 'total_duration': total_duration_constraint,
                                 'efficiency': efficiency_constraint}}
        s_constraints = {'dro': {'surgeon_patient': surgeon_patient_constraint},
                         'd': {'overlapping': surgeon_overlapping_constraint}}

        return sr_constraints, s_constraints

    def simulated_annealing_by_day(self, init_sol_param=None, genetic=False, stable_schedule_flag=True,
                                   random_selection=True):
        """
        performs SA on a single day
        :param genetic: boolean - True if sa is used for init population in genetic algorithm , if so list of all the schedules
        is returned
        :param init_sol_param: function that determines the parameter which according to it the initial state will be
        generated
        """
        if genetic:
            g = []
        global t
        plot_list = []
        num_surgeries = self.num_surgeries
        # dictionary {key- room num : value - current number of surgeries}
        current_value = self._set_init_value_day(num_surgeries,
                                                 stable_schedule_flag=stable_schedule_flag)  # float the value of the day -
        best_value = current_value
        best_schedule = deepcopy(self.v_dict)
        b_num_surgeries = num_surgeries.copy()
        '''if self.with_surgery_Team['Nurse'] and self.with_surgery_Team['Anesthetist']:
            self.update_no_good_surgery_requests()'''
        if genetic:
            g.append((deepcopy(self.v_dict), current_value, deepcopy(num_surgeries)))  # tuple of schedule and its value
        num_changes = 0
        for t in Static.infinity():
            T = sa_schedule(t, "Linear")
            st = sa_stopping_temp("Linear")
            if T <= st:  # none hard constraints are broken - not sure best_value > 0 is needed
                if genetic:
                    return g
                else:
                    self.update_schedule(best_schedule, b_num_surgeries, stable_schedule_flag=stable_schedule_flag)
                    # return best_value, t, num_changes, plot_list, best_schedule
                    # Static.simulated_graphs(plot_list, 'ward_' + str(self.ward.w_id))
                    break
            else:
                chosen_v, delta_E, tu = self._select_successor(num_surgeries, stable_schedule_flag=stable_schedule_flag,
                                                               random_selection=random_selection)
                if delta_E > 0:
                    # self.increment_counter(chief_agent=True)
                    num_changes += 1
                    current_value = self.calc_value(tu)
                    plot_list.append([t, current_value])
                    if genetic:
                        g.append((deepcopy(self.v_dict), current_value, deepcopy(num_surgeries)))
                    if current_value > best_value:
                        best_value = current_value
                        best_schedule = deepcopy(self.v_dict)
                        b_num_surgeries = num_surgeries.copy()
                else:
                    p = Decimal(math.exp(delta_E / T))
                    rnd = random.uniform(0, 1)
                    if rnd < p:  # change but only to feasible solutions
                        current_value = self.calc_value(tu)
                        if current_value > best_value:
                            best_value = current_value
                            best_schedule = deepcopy(self.v_dict)
                            b_num_surgeries = num_surgeries.copy()
                        plot_list.append([t, current_value])
                        if genetic:
                            g.append((deepcopy(self.v_dict), current_value, deepcopy(num_surgeries)))
                        num_changes += 1
                    else:  # don't change
                        self.return_to_prior_value(chosen_v, num_surgeries, tu,
                                                   stable_schedule_flag=stable_schedule_flag)
                    # self.increment_counter(chief_agent=True)

    def _init_solution_day(self, parameter):
        """
            generates an initial solution - the solution is determined by the parameter. updates d-dict variables
            assigning each variable a certain value from its domain
            :param parameter: the parameter that according to it the solution is generated - it will be the name of a
            function that serves as a key for the sort comparison
            :return num_surgeries - dictionary {key- room num : value - current number of surgeries}
            could receive d_dict of a specific ward and d - but just to be able to play with it for now..
            """
        # general heuristic for now all_diff constraint and total duration constraints are kept. i.e the same sr values will
        # not be assigned to different variables (to avoid all the variables to receive the same value) and the total
        # duration of a certain room will not exceed the max duration.

        num_surgeries = {}
        d_dict = self.v_dict[self.ward][
            self.schedule_date]  # a dictionary -key - room value - list of [(sr_v,s_v),(sr_v,s_v)..]
        # dictionary of variables - representing room allocation of a single day for a certain ward.
        for room_num in d_dict:
            room_duration = 0
            prior_end_time = self.ward.start_d_hour
            for t, i in zip(d_dict[room_num], range(len(d_dict[room_num]))):  # t - tuple
                sr_v = t[0]
                s_v = t[1]
                if self.case == 1:
                    legit_sr_domain = reduct_sr_domain(sr_v.domain)
                    if legit_sr_domain != False:
                        sorted_domain = sorted(legit_sr_domain, key=parameter)  # returns a list
                        sorted_domain = self.scheduled_surgery_request_first(sorted_domain)
                    else:
                        '''sorted_domain = sorted(sr_v.domain, key=parameter)  # returns a list
                        sorted_domain = self.scheduled_surgery_request_first(sorted_domain)'''
                        sorted_domain = []
                else:
                    sorted_domain = sorted(sr_v.domain, key=parameter)
                    sorted_domain = self.scheduled_surgery_request_first(sorted_domain)
                if len(sorted_domain) > 0:
                    for sr in sorted_domain:  # sr - surgery request
                        if room_duration + sr.duration <= self.ward.d_duration:  # total duration global hard constraint
                            sr_v.value = sr
                            sr.assigned.add(sr_v.get_constraint_dro_key())
                            sr_v.set_surgery_time(start_time=prior_end_time, duration=sr_v.value.duration)
                            s_v.set_surgery_time(start_time=prior_end_time, duration=sr_v.value.duration)
                            prior_end_time = sr_v.end_time
                            if len(s_v.domain) > 0:
                                if sr_v.value.specific_senior is not None:
                                    # if len(sr_v.value.specific_senior) == 9:
                                    s_v.value = self.ward.find_surgeon_by_id(sr_v.value.specific_senior)
                                else:
                                    if self.case == 1 or self.case == 2:
                                        legit_s_domain = reduct_s_domain(s_v.domain, sr_v.value.surgery_type)
                                        # only values of surgeons which have the skill to operate this surgery type
                                        if legit_s_domain != False:
                                            s_v.value = random.choice(tuple(legit_s_domain))
                                        else:
                                            s_v.value = random.choice(tuple(s_v.domain))
                                    else:
                                        s_v.value = random.choice(tuple(s_v.domain))
                                room_duration = room_duration + sr.duration
                                break  # assigned from domain no need to continue looking for value
                if sr_v.value is None:
                    num_surgeries[room_num] = sr_v.order - 1
                    break
                if i == (len(d_dict[room_num]) - 1):
                    num_surgeries[room_num] = sr_v.order
        return num_surgeries

    def scheduled_surgery_request_first(self, domain):
        with_schedule_date = []  # sr with schedule date different than this one
        without_schedule_date = []  # not scheduled yet
        this_schedule_date = []  # scheduled to this schedule date
        for sr in domain:
            if sr.schedule_date is None:
                without_schedule_date.append(sr)
            elif str(sr.schedule_date) == self.schedule_date:
                this_schedule_date.append(sr)
            else:
                with_schedule_date.append(sr)
        with_schedule_date = sorted(with_schedule_date, key=by_schedule_date)
        this_schedule_date.extend(with_schedule_date)
        this_schedule_date.extend(without_schedule_date)
        return this_schedule_date

    def _set_init_value_day(self, num_surgeries, stable_schedule_flag=True):
        """
        calculates the value of a day overall utility - over all costs of all the variables in the day. Calculates the value
        from zero with no prior value calculated, that means not the value after an update of a single variable
        Can't be preformed while determining init_sol because looks their are constraints that need the whole solution
        :param num_surgeries:  dictionary {key- room num : value - current number of surgeries}
        :return: float the price of the day and the utility of the day
        could receive d_dict of a specific ward and d - but just to be able to play with it for now..
        """
        init_day_cost = 0
        init_day_utility = 0
        d_dict = self.v_dict[self.ward][self.schedule_date]
        init_day_cost += Prices.set_d_prices(d_dict, None, None, None)
        for room_num in d_dict:
            init_day_cost += Prices.set_dr_prices(d_dict[room_num], self.ward, True, None, by_units=True)
            for t in d_dict[room_num]:  # t- tuple (sr_v, s_v)
                sr_v = t[0]
                s_v = t[1]
                init_day_cost += Prices.set_dro_prices(sr_v, s_v, self.ward, num_surgeries[room_num], True, None,
                                                       self.with_surgery_Team, self.no_good_sr,
                                                       stable_schedule_flag=stable_schedule_flag,
                                                       current_NCLO=self.counter)
                if sr_v.value is not None:
                    init_day_utility += sr_v.value.surgery_type.utility
                    sr_v.best_value = sr_v.value
                    s_v.best_value = s_v.value
        return init_day_utility - init_day_cost

    def _select_successor(self, num_surgeries, stable_schedule_flag=True, random_selection=True):
        """
        selects a random variable and changes its value randomly from the domain , calculates the difference of the total
        solution price due to the change in the solution. \
        The difference is calculated by the subtraction of prior price from next price
        prior price - utility - cost of the specific variable that changed
        next price - utility - cost after the change
        :param num_surgeries:  dictionary {key- room num : value - current number of surgeries}
        :return: chosen variable/tuple and the difference in the total price of the solution, and the tuple chosen
        """

        d_dict = self.v_dict[self.ward][self.schedule_date]
        room_num = random.choice(list(d_dict))
        t = random.choice(d_dict[room_num])  # tuple (sr_v, s_v)
        chosen_v = random.choice(list(t))  # chosen variable
        num_surgeries_day = num_surgeries[room_num]
        delta_E = 0
        # if mutual update sr_v and s_v together
        if chosen_v.value is None:  # two options - 1: variable with no assignation 2: the domain is empty
            # if the value is none than we want to choose the next empty variable so the surgeries are in good order
            t = self.find_t_by_index(chosen_v.room.num, num_surgeries_day + 1)
            delta, change = self.update_tuple_value(t, num_surgeries_day + 1, room_num, num_surgeries,
                                                    stable_schedule_flag=stable_schedule_flag,
                                                    random_selection=random_selection)
            delta_E += delta
            if change:  # if the domain was of the chosen v was not empty
                num_surgeries[room_num] += 1
            return t, delta_E, t
        else:
            delta_E += self.update_variable_value(chosen_v, t, num_surgeries, room_num,
                                                  stable_schedule_flag=stable_schedule_flag,
                                                   random_selection=random_selection)
            return chosen_v, delta_E, t

    def find_t_by_index(self, room_num, order):
        """
        finds a tuple of variables by the order argument
        :param room_num: index of room
        :param order: int the order of the variable/surgery in the day
        :return: tuple (sr_v,s_v) of the index in the input
        """
        d_dict = self.v_dict[self.ward][self.schedule_date]
        for t in d_dict[room_num]:
            if t[0].order == order:
                return t

    def update_tuple_value_genetic(self, t, num_surgeries_day, room_num, genetic_value=False,
                                   stable_schedule_flag=True):
        """
        chooses new values for a tuple of variables which had none values and calculates the difference in the sol_value
        utility - cost
        :param genetic_value: value chosen for tuple variables from genteic algorithm
        :param t: tuple (sr_v,s_v)
        :return: delta E the difference of the total price of solution
        """
        if t is None:
            print('problem')
        sr_v = t[0]
        s_v = t[1]
        change = False
        if genetic_value == False:
            if len(sr_v.domain) > 0 and len(s_v.domain) > 0:
                change = True
            else:
                return 0, change
            if change:
                if self.case == 1:
                    r_sr_domain = reduct_sr_domain(sr_v.domain)
                    if r_sr_domain != False:
                        chosen_sr_v_value = random.choice(list(r_sr_domain))
                    else:
                        change = False
                        return 0, change
                else:
                    chosen_sr_v_value = random.choice(list(sr_v.domain))
                # if len(chosen_sr_v_value.specific_senior) == 9:
                if chosen_sr_v_value.specific_senior is not None:
                    chosen_s_v_value = self.ward.find_surgeon_by_id(chosen_sr_v_value.specific_senior)
                else:
                    if self.case == 2 or self.case == 1:
                        r_s_domain = reduct_s_domain(s_v.domain, chosen_sr_v_value.surgery_type)
                        if r_s_domain != False:
                            chosen_s_v_value = random.choice(list(r_s_domain))
                        else:
                            change = False
                            return 0, change
                    else:
                        chosen_s_v_value = random.choice(list(s_v.domain))
        if genetic_value != False or change:
            sr_v.prior_value = sr_v.value
            s_v.prior_value = s_v.value
        if genetic_value == False:
            if change:
                prior_price = self.calc_price_by_variable \
                    (t, room_num, type(None), num_surgeries_day - 1, mutual=True,
                     stable_schedule_flag=stable_schedule_flag)
                sr_v.value = chosen_sr_v_value
        else:
            sr_v.value = genetic_value[0]
        if sr_v.prior_value is not None and (change or genetic_value != False):
            sr_v.prior_value.assigned.remove(sr_v.get_constraint_dro_key())
            if len(sr_v.prior_value.assigned) == 1:
                sr_v.constraints['dro']['all_diff'].prices[sr_v.prior_value.assigned[0]] = 0
        sr_v.value.assigned.append(sr_v.get_constraint_dro_key())
        if genetic_value == False:
            if change:
                s_v.value = chosen_s_v_value
        else:
            s_v.value = genetic_value[1]
        if sr_v.order > 1 and (change or genetic_value != False):
            prev_t = self.find_t_by_index(room_num=sr_v.room.num, order=sr_v.order - 1)
            start_time = prev_t[0].end_time
        else:
            start_time = self.ward.start_d_hour
        if change or genetic_value != False:
            sr_v.set_surgery_time(start_time, sr_v.value.duration)
            s_v.set_surgery_time(start_time, sr_v.value.duration)
            next_price = self.calc_price_by_variable \
                (t, room_num, type(None), num_surgeries_day, mutual=True, next=True,
                 stable_schedule_flag=stable_schedule_flag)
        if genetic_value == False and change:
            return next_price - prior_price, change
        if genetic_value == False and not change:
            return 0, change

    '''def update_tuple_value(self, t, num_surgeries_day, room_num, stable_schedule_flag=True, random_selection=True):
        """
        chooses new values for a tuple of variables which had none values and calculates the difference in the sol_value
        utility - cost
        :param t: tuple (sr_v,s_v)
        :param num_surgeries_day: int current number of surgeries in room of room num
        :param room_num: int room id of tuple
        :param stable_schedule_flag: boolean if to consider stable schedule prices
        :param random_selection: boolean if to select randomly from domain or by max of next price
        :return: delta E the difference of the total price of solution
        """
        if t is None:
            print('problem')
        sr_v = t[0]
        s_v = t[1]
        change = False
        if len(sr_v.domain) > 0 and len(s_v.domain) > 0:
            change = True
            r_sr_domain = reduct_sr_domain(sr_v.domain)
            if r_sr_domain:
                    chosen_sr_v_value = random.choice(list(r_sr_domain))
            else:
                change = False
                return 0, change
            if chosen_sr_v_value.specific_senior is not None:
                chosen_s_v_value = self.ward.find_surgeon_by_id(chosen_sr_v_value.specific_senior)
            else:
                r_s_domain = reduct_s_domain(s_v.domain, chosen_sr_v_value.surgery_type)
                if r_s_domain:
                    chosen_s_v_value = random.choice(list(r_s_domain))
                else:
                    change = False
                    return 0, change
            sr_v.prior_value = sr_v.value
            s_v.prior_value = s_v.value
            prior_price = self.calc_price_by_variable(t, room_num, type(None), num_surgeries_day - 1, mutual=True,
                                                      stable_schedule_flag=stable_schedule_flag)
            sr_v.value = chosen_sr_v_value
            if sr_v.prior_value is not None:
                sr_v.prior_value.assigned.remove(sr_v.get_constraint_dro_key())
                if len(sr_v.prior_value.assigned) == 1:
                    sr_v.constraints['dro']['all_diff'].prices[sr_v.prior_value.assigned[0]] = 0
            sr_v.value.assigned.append(sr_v.get_constraint_dro_key())
            s_v.value = chosen_s_v_value
            if sr_v.order > 1:
                prev_t = self.find_t_by_index(room_num=sr_v.room.num, order=sr_v.order - 1)
                start_time = prev_t[0].end_time
            else:
                start_time = self.ward.start_d_hour
            sr_v.set_surgery_time(start_time, sr_v.value.duration)
            s_v.set_surgery_time(start_time, sr_v.value.duration)
            next_price = self.calc_price_by_variable(t, room_num, type(None), num_surgeries_day, mutual=True, next=True,
                                                     stable_schedule_flag=stable_schedule_flag)
            return next_price - prior_price, change
        else:
            return 0, change'''

    def update_tuple_value(self, t, num_surgeries_day, room_num, num_surgeries, stable_schedule_flag=True, random_selection=True):
        """
        chooses new values for a tuple of variables which had none values and calculates the difference in the sol_value
        utility - cost
        :param t: tuple (sr_v,s_v)
        :param num_surgeries_day: int current number of surgeries in room of room num
        :param room_num: int room id of tuple
        :param stable_schedule_flag: boolean if to consider stable schedule prices
        :param random_selection: boolean if to select randomly from domain or by max of next price
        :return: delta E the difference of the total price of solution
        """
        if t is None:
            print('problem')
        sr_v = t[0]
        s_v = t[1]
        change = False
        if len(sr_v.domain) > 0 and len(s_v.domain) > 0:
            # todo not sure if because i moved it earlier will do balagan until change=True
            sr_v.prior_value = sr_v.value
            s_v.prior_value = s_v.value
            prior_price = self.calc_price_by_variable(t, room_num, type(None), num_surgeries_day - 1, mutual=True,
                                                      stable_schedule_flag=stable_schedule_flag)
            change = True
            r_sr_domain = reduct_sr_domain(sr_v.domain)
            if r_sr_domain:
                if random_selection:
                    chosen_sr_v_value = random.choice(list(r_sr_domain))
                    next_price = self.calc_tuple_DeltaE(chosen_sr_v_value, s_v, sr_v, room_num, num_surgeries_day,
                                                        stable_schedule_flag)
                    self.increment_counter(chief_agent=True)
                    if len(next_price) > 1:
                        return next_price[0], next_price[1]
                    else:
                        next_price = next_price[0]
                else:
                    next_price = None
                    chosen_sr_v_value = None
                    original_prior_value = sr_v.prior_value
                    s_original_prior_value = s_v.prior_value
                    for sr in r_sr_domain:
                        sr_v.prior_value = sr_v.value
                        temp_next_price = self.calc_tuple_DeltaE(sr, s_v, sr_v, room_num, num_surgeries_day,
                                                                 stable_schedule_flag)
                        self.increment_counter(chief_agent=True)
                        if len(temp_next_price) > 1:
                            continue
                        elif next_price is None:
                            next_price = temp_next_price[0]
                            chosen_sr_v_value = sr
                        elif temp_next_price[0] > next_price:
                            next_price = temp_next_price[0]
                            chosen_sr_v_value = sr
                        sr_v.prior_value = original_prior_value
                        s_v.prior_value = s_original_prior_value
                        self.return_to_prior_value(t, num_surgeries.copy(), t, stable_schedule_flag)
                    sr_v.prior_value = original_prior_value
                    s_v.prior_value = s_original_prior_value
                    if next_price is None:
                        return 0, False
                    else:
                        sr_v.prior_value = original_prior_value
                        s_v.prior_value = s_original_prior_value
                        next_price = self.calc_tuple_DeltaE(chosen_sr_v_value, s_v, sr_v, room_num, num_surgeries_day,
                                                            stable_schedule_flag)
                    if len(next_price) > 1:
                        return next_price[0], next_price[1]
                    else:
                        next_price = next_price[0]
                return next_price - prior_price, change
            else:
                change = False
                return 0, change
        else:
            return 0, change

    def calc_tuple_DeltaE(self, chosen_sr_v_value, s_v, sr_v, room_num, num_surgeries_day, stable_schedule_flag):
        if chosen_sr_v_value.specific_senior is not None:
            chosen_s_v_value = self.ward.find_surgeon_by_id(chosen_sr_v_value.specific_senior)
        else:
            r_s_domain = reduct_s_domain(s_v.domain, chosen_sr_v_value.surgery_type)
            if r_s_domain:
                chosen_s_v_value = random.choice(list(r_s_domain))
            else:
                change = False
                return [0, change]
        sr_v.value = chosen_sr_v_value
        if sr_v.prior_value is not None:
            sr_v.prior_value.assigned.remove(sr_v.get_constraint_dro_key())
            if len(sr_v.prior_value.assigned) == 1:
                sr_v.constraints['dro']['all_diff'].prices[sr_v.prior_value.assigned[0]] = 0
        sr_v.value.assigned.add(sr_v.get_constraint_dro_key())
        s_v.value = chosen_s_v_value
        if sr_v.order > 1:
            prev_t = self.find_t_by_index(room_num=sr_v.room.num, order=sr_v.order - 1)
            start_time = prev_t[0].end_time
        else:
            start_time = self.ward.start_d_hour
        sr_v.set_surgery_time(start_time, sr_v.value.duration)
        s_v.set_surgery_time(start_time, sr_v.value.duration)
        next_price = self.calc_price_by_variable((sr_v, s_v), room_num, type(None), num_surgeries_day, mutual=True, next=True,
                                                 stable_schedule_flag=stable_schedule_flag)
        return [next_price]

    def calc_price_by_variable(self, t, room_num, ty, num_surgeries_day, mutual, next=False, stable_schedule_flag=True):
        """
        :param next: defines if an update was done in the variable value or prior price is calculated
        :param t: t: tuple (sr_v,s_v)
        :param room_num: int
        :param ty: type of the updated variable or none if mutual variables of tuple are updated
        :param num_surgeries_day: int - current num of surgeries
        :param mutual: boolean - if both variables in the tuple have been updated
        :return: new utility - new cost
        """
        d_dict = self.v_dict[self.ward][self.schedule_date]
        cost = 0
        utility = 0
        sr_v = t[0]
        s_v = t[1]
        cost += Prices.set_dro_prices(sr_v, s_v, self.ward, num_surgeries_day, mutual, ty, self.with_surgery_Team,
                                      self.no_good_sr, stable_schedule_flag=stable_schedule_flag,
                                      current_NCLO=self.counter)
        cost += Prices.set_dr_prices(d_dict[room_num], self.ward, mutual, ty, by_units=True)
        cost += Prices.set_d_prices(d_dict, s_v, ty, sr_v, next=next)
        if isinstance(sr_v, ty) or mutual:
            if sr_v.value is not None:  # in the case prior value was None and the calculation of prior price
                utility += sr_v.value.surgery_type.utility
        return utility - cost

    def update_variable_value_genetic(self, chosen_v, t, num_surgeries_day, room_num, genetic_value=False,
                                      stable_schedule_flag=True):
        """
        chooses randomly new value for variable from domain and calculates the difference in the sol_value utility - cost
        :param room_num: int resembles the surgery room number
        :param num_surgeries_day: int num of surgeries currently in room_num
        :param genetic_value: value chosen for the variable from genteic algorithm
        :param chosen_v: chosen variable to change value
        :param t: tuple (sr_v,s_v)
        :return: delta E the difference of the total price of solution
        """
        mutual = False
        if genetic_value == False:
            if len(chosen_v.domain) > 0:
                i = t.index(chosen_v)
                if i == 1:  # surgeon variable
                    # if len(t[0].value.specific_senior) == 9:
                    if t[0].value.specific_senior is not None:
                        chosen_value = self.ward.find_surgeon_by_id(t[0].value.specific_senior)
                    else:
                        if self.case == 1 or self.case == 2:
                            r_s_domain = reduct_s_domain(chosen_v.domain, t[0].value.surgery_type)
                            #  only values of surgeons which have the skill to operate this surgery type
                            if r_s_domain != False:
                                chosen_value = random.choice(list(r_s_domain))
                            else:
                                return 0
                        else:
                            chosen_value = random.choice(list(chosen_v.domain))  # value of the variable
                else:  # surgery request variable
                    if self.case == 1:
                        r_sr_domain = reduct_sr_domain(chosen_v.domain)
                        if r_sr_domain != False:
                            chosen_value = random.choice(list(r_sr_domain))
                            # if len(chosen_value.specific_senior) == 9:
                            if chosen_value.specific_senior is not None:
                                t[1].prior_value = t[1].value
                                mutual = True
                        else:
                            return 0
                    else:
                        chosen_value = random.choice(list(chosen_v.domain))  # value of the variable
                chosen_v.prior_value = chosen_v.value
                prior_price = self.calc_price_by_variable \
                    (t, room_num, type(chosen_v), num_surgeries_day, mutual=mutual,
                     stable_schedule_flag=stable_schedule_flag)
                chosen_v.value = chosen_value
                if mutual:
                    t[1].value = self.ward.find_surgeon_by_id(chosen_value.specific_senior)
        else:
            i = t.index(chosen_v)
            if i == 1:
                # if len(t[0].value.specific_senior) != 9:
                if t[0].value.specific_senior is not None:
                    chosen_v.prior_value = chosen_v.value
                    chosen_v.value = genetic_value
                else:
                    chosen_v.prior_value = chosen_v.value
                    chosen_v.value = self.ward.find_surgeon_by_id(t[0].value.specific_senior)
            else:
                chosen_v.prior_value = chosen_v.value
                chosen_v.value = genetic_value
                # if len(genetic_value.specific_senior) == 9:
                if genetic_value.specific_senior is not None:
                    t[1].prior_value = t[1].value
                    t[1].value = self.ward.find_surgeon_by_id(genetic_value.specific_senior)
                    mutual = True
        if genetic_value != False or len(chosen_v.domain) > 0:
            if i == 0:  # surgery request variable
                chosen_v.value.assigned.append(chosen_v.get_constraint_dro_key())
                chosen_v.prior_value.assigned.remove(chosen_v.get_constraint_dro_key())
                if len(chosen_v.prior_value.assigned) == 1:  # all diff constraint satisfied
                    chosen_v.constraints['dro']['all_diff'].prices[chosen_v.prior_value.assigned[0]] = 0
                self.update_surgeries_time(chosen_v.room.num, chosen_v.order - 1)

            next_price = self.calc_price_by_variable(t, room_num, type(chosen_v), num_surgeries_day, mutual=mutual,
                                                     next=True, stable_schedule_flag=stable_schedule_flag)
        if genetic_value == False:
            if len(chosen_v.domain) > 0:
                return next_price - prior_price
            else:
                return 0

    '''def update_variable_value(self, chosen_v, t, num_surgeries_day, room_num, stable_schedule_flag=True):
        """
        chooses randomly new value for variable from domain and calculates the difference in the sol_value utility - cost
        :param room_num: int resembles the surgery room number
        :param num_surgeries_day: int num of surgeries currently in room_num
        :param chosen_v: chosen variable to change value
        :param t: tuple (sr_v,s_v)
        :return: delta E the difference of the total price of solution
        """
        mutual = False
        next_price = 0
        prior_price = 0
        if len(chosen_v.domain) > 0:
            i = t.index(chosen_v)
            if i == 1:  # surgeon variable
                if t[0].value.specific_senior is not None:
                    chosen_value = self.ward.find_surgeon_by_id(t[0].value.specific_senior)
                else:
                    r_s_domain = reduct_s_domain(chosen_v.domain, t[0].value.surgery_type)
                    #  only values of surgeons which have the skill to operate this surgery type
                    if r_s_domain:
                        chosen_value = random.choice(list(r_s_domain))
                    else:
                        return 0
            else:  # surgery request variable
                r_sr_domain = reduct_sr_domain(chosen_v.domain)
                if r_sr_domain:
                    chosen_value = random.choice(list(r_sr_domain))
                    if chosen_value.specific_senior is not None:
                        t[1].prior_value = t[1].value
                        mutual = True
                else:
                    return 0
            chosen_v.prior_value = chosen_v.value
            prior_price = self.calc_price_by_variable(t, room_num, type(chosen_v), num_surgeries_day, mutual=mutual,
                                                      stable_schedule_flag=stable_schedule_flag)
            chosen_v.value = chosen_value
            if mutual:
                t[1].value = self.ward.find_surgeon_by_id(chosen_value.specific_senior)
            if i == 0:  # surgery request variable
                chosen_v.value.assigned.append(chosen_v.get_constraint_dro_key())
                chosen_v.prior_value.assigned.remove(chosen_v.get_constraint_dro_key())
                if len(chosen_v.prior_value.assigned) == 1:  # all diff constraint satisfied
                    chosen_v.constraints['dro']['all_diff'].prices[chosen_v.prior_value.assigned[0]] = 0
                self.update_surgeries_time(chosen_v.room.num, chosen_v.order - 1)

            next_price = self.calc_price_by_variable(t, room_num, type(chosen_v), num_surgeries_day, mutual=mutual,
                                                     next=True, stable_schedule_flag=stable_schedule_flag)
        if len(chosen_v.domain) > 0:
            return next_price - prior_price
        else:
            return 0'''

    def update_variable_value(self, chosen_v, t, num_surgeries, room_num, stable_schedule_flag=True,
                              random_selection=True):
        """
        chooses randomly new value for variable from domain and calculates the difference in the sol_value utility - cost
        :param room_num: int resembles the surgery room number
        :param stable_schedule_flag: boolean if to consider stable schedule prices
        :param random_selection: if to select randomly a value from domain or the one which maximizes it
        ::param num_surgeries: dictionary {key- room num : value - current number of surgeries}
        :param chosen_v: chosen variable to change value
        :param t: tuple (sr_v,s_v)
        :return: delta E the difference of the total price of solution
        """
        mutual = False
        if len(chosen_v.domain) > 0:
            i = t.index(chosen_v)
            if i == 1:  # surgeon variable
                if t[0].value.specific_senior is not None:
                    chosen_value = self.ward.find_surgeon_by_id(t[0].value.specific_senior)
                    self.increment_counter(chief_agent=True)
                    chosen_v.prior_value = chosen_v.value
                    prior_price = self.calc_price_by_variable(t, room_num, type(chosen_v), num_surgeries[room_num],
                                                              mutual=mutual,
                                                              stable_schedule_flag=stable_schedule_flag)
                    next_price = self.calc_DeltaE_by_var(chosen_v, chosen_value, mutual, t, stable_schedule_flag, i,
                                                         num_surgeries[room_num], room_num)

                else:
                    r_s_domain = reduct_s_domain(chosen_v.domain, t[0].value.surgery_type)
                    #  only values of surgeons which have the skill to operate this surgery type
                    if r_s_domain:
                        chosen_v.prior_value = chosen_v.value
                        prior_price = self.calc_price_by_variable(t, room_num, type(chosen_v), num_surgeries[room_num],
                                                                  mutual=mutual,
                                                                  stable_schedule_flag=stable_schedule_flag)
                        if random_selection:
                            chosen_value = random.choice(list(r_s_domain))
                            self.increment_counter(chief_agent=True)
                            next_price = self.calc_DeltaE_by_var(chosen_v, chosen_value, mutual, t,
                                                                 stable_schedule_flag, i, num_surgeries[room_num],
                                                                 room_num)
                        else:
                            original_prior_value = chosen_v.prior_value
                            next_price = None
                            chosen_value = None
                            for s in r_s_domain:
                                temp_next_price = self.calc_DeltaE_by_var(chosen_v, s, mutual, t, stable_schedule_flag,
                                                                          i, num_surgeries[room_num], room_num)
                                self.increment_counter(chief_agent=True)
                                if next_price is None:
                                    next_price = temp_next_price
                                    chosen_value = s
                                elif next_price < temp_next_price:
                                    chosen_value = s
                                    next_price = temp_next_price
                                chosen_v.prior_value = original_prior_value
                                self.return_to_prior_value(chosen_v, num_surgeries, t, stable_schedule_flag)
                            chosen_v.prior_value = original_prior_value
                            next_price = self.calc_DeltaE_by_var(chosen_v, chosen_value, mutual, t,
                                                                 stable_schedule_flag, i, num_surgeries[room_num],
                                                                 room_num)
                    else:
                        return 0
            else:  # surgery request variable
                r_sr_domain = reduct_sr_domain(chosen_v.domain)
                if r_sr_domain:
                    chosen_v.prior_value = chosen_v.value
                    prior_price = self.calc_price_by_variable(t, room_num, type(chosen_v), num_surgeries[room_num],
                                                              mutual=mutual,
                                                              stable_schedule_flag=stable_schedule_flag)
                    if random_selection:
                        self.increment_counter(chief_agent=True)
                        chosen_value = random.choice(list(r_sr_domain))
                        if chosen_value.specific_senior is not None:
                            t[1].prior_value = t[1].value
                            mutual = True
                        next_price = self.calc_DeltaE_by_var(chosen_v, chosen_value, mutual, t, stable_schedule_flag, i,
                                                             num_surgeries[room_num], room_num)
                    else:
                        next_price = None
                        chosen_value = None
                        original_prior_value = chosen_v.prior_value
                        for sr in r_sr_domain:
                            temp_mutual = False
                            if sr.specific_senior is not None:
                                temp_mutual = True
                                t[1].prior_value = t[1].value
                            temp_next_price = self.calc_DeltaE_by_var(chosen_v, sr, temp_mutual, t,
                                                                      stable_schedule_flag, i, num_surgeries[room_num],
                                                                      room_num)
                            self.increment_counter(chief_agent=True)
                            if next_price is None:  # first option in domain
                                next_price = temp_next_price
                                chosen_value = sr
                            elif next_price < temp_next_price:
                                next_price = temp_next_price
                                chosen_value = sr
                            chosen_v.prior_value = original_prior_value
                            self.return_to_prior_value(chosen_v, num_surgeries, t, stable_schedule_flag)
                        if chosen_value.specific_senior is not None:
                            mutual = True
                            t[1].prior_value = t[1].value
                        chosen_v.prior_value = original_prior_value
                        next_price = self.calc_DeltaE_by_var(chosen_v, chosen_value, mutual, t, stable_schedule_flag, i,
                                                             num_surgeries[room_num], room_num)
                else:
                    return 0

            return next_price - prior_price
        else:
            return 0

    def calc_DeltaE_by_var(self, chosen_v, chosen_value, mutual, t, stable_schedule_flag, i, num_surgeries_day, room_num):
        chosen_v.value = chosen_value
        if mutual:
            t[1].value = self.ward.find_surgeon_by_id(chosen_value.specific_senior)
        if i == 0:  # surgery request variable
            dro_key = chosen_v.get_constraint_dro_key()
            chosen_v.value.assigned.add(dro_key)
            if dro_key in chosen_v.prior_value.assigned:
                chosen_v.prior_value.assigned.remove(chosen_v.get_constraint_dro_key())
            if len(chosen_v.prior_value.assigned) == 1:  # all diff constraint satisfied
                # chosen_v.constraints['dro']['all_diff'].prices[chosen_v.prior_value.assigned[0]] = 0
                chosen_v.constraints['dro']['all_diff'].prices[next(iter(chosen_v.prior_value.assigned))] = 0
            self.update_surgeries_time(chosen_v.room.num, chosen_v.order - 1)

        next_price = self.calc_price_by_variable(t, room_num, type(chosen_v), num_surgeries_day, mutual=mutual,
                                                 next=True, stable_schedule_flag=stable_schedule_flag)
        return next_price

    def update_surgeries_time(self, room_num, order):
        """
        updates all the surgery times from a specific surgery in the day and on - the surgery value in this order has been
        change so all the surgeries time after it also changed
        :param room_num: int room number that needs the times updated
        :param order: int index of the order of the surgery in the day - that from it the times need to be updated
        """
        d_dict = self.v_dict[self.ward][self.schedule_date]
        prior_end_time = d_dict[room_num][order][0].start_time
        for j in range(order, len(d_dict[room_num])):  # update the times of all the surgeries
            # after the updated surgery variable because the times changed
            sr_v = d_dict[room_num][j][0]  # sr_v
            s_v = d_dict[room_num][j][1]  # s_v
            if sr_v.value is None:
                break
            else:
                sr_v.set_surgery_time(start_time=prior_end_time, duration=sr_v.value.duration)
                s_v.set_surgery_time(start_time=prior_end_time, duration=sr_v.value.duration)
                prior_end_time = sr_v.end_time

    def calc_value(self, tu):
        """
        calcs the total cost of the solution node
        :param tu: tuple (sr_v, s_v)
        :return: float total cost
        """
        cost = 0
        # utility = current_u
        utility = 0
        sr_v = tu[0]
        s_v = tu[1]
        constraints = sr_v.constraints
        for con_key in constraints:
            for cons in constraints[con_key]:
                cost += sum(constraints[con_key][cons].prices.values())

        cost += sum(s_v.constraints['d']['overlapping'].prices.values())  # s_v constraint which is not sr_v constraint
        for w in self.v_dict:
            for d in self.v_dict[w]:
                for rn in self.v_dict[w][d]:
                    for t in self.v_dict[w][d][rn]:
                        sr_v = t[0]
                        if sr_v.value is not None:
                            utility += sr_v.value.surgery_type.utility
        return utility - cost

    def return_to_prior_value(self, chosen_v, num_surgeries, tu, stable_schedule_flag=True):
        """
        returns the solution to the prior solution changes the values of the concerned variables back and updates
        num_surgeries if needed
        :param tu: tuple of (sr_v, s_v) of the chosen_v
        :param d_dict: dictionary format {room_num : [(sr_v,s_v),(sr_v,s_v)...]
        :param num_surgeries: dictionary {key- room num : value - current number of surgeries}
        :param chosen_v: tuple/variable depends if there was a mutual change -in case of adding new surgery in the day

        """
        mutual = False
        if isinstance(chosen_v, tuple):  # tuple (sr_v,s_v)
            sr_v = chosen_v[0]
            s_v = chosen_v[1]
            room_num = sr_v.room.num
            s_v_prior_update = chosen_v[1].value
            s_v.value = chosen_v[1].prior_value
            s_v.prior_value = s_v_prior_update
            # sr_v.value.assigned -= 1
            sr_v.value.assigned.remove(sr_v.get_constraint_dro_key())
            if len(sr_v.value.assigned) == 1:
                sr_v.constraints['dro']['all_diff'].prices[sr_v.value.assigned[0]] = 0
            if sr_v.prior_value is not None:
                # sr_v.prior_value.assigned += 1
                sr_v.prior_value.assigned.add(sr_v.get_constraint_dro_key())
                sr_v.set_surgery_time(start_time=sr_v.start_time, duration=sr_v.value.duration)
                s_v.set_surgery_time(start_time=sr_v.start_time, duration=sr_v.value.duration)
            else:
                num_surgeries[sr_v.room.num] -= 1
                sr_v.nullify_surgery_time()
                s_v.nullify_surgery_time()
            sr_v_prior_update = sr_v.value
            sr_v.value = sr_v.prior_value
            sr_v.prior_value = sr_v_prior_update
            price = self.calc_price_by_variable(tu, room_num, type(None), num_surgeries[room_num], mutual=True,
                                                next=True, stable_schedule_flag=stable_schedule_flag)

        else:
            if isinstance(chosen_v, SurgeryVariable_SurgeryRequest):
                # chosen_v.prior_value.assigned += 1
                if chosen_v.prior_value is None:
                    print('stop')
                chosen_v.prior_value.assigned.add(chosen_v.get_constraint_dro_key())
                # chosen_v.value.assigned -= 1
                chosen_v.value.assigned.remove(chosen_v.get_constraint_dro_key())
                if len(chosen_v.value.assigned) == 1:
                    chosen_v.constraints['dro']['all_diff'].prices[chosen_v.value.assigned[0]] = 0
                prior_update = chosen_v.value
                chosen_v.value = chosen_v.prior_value
                chosen_v.prior_value = prior_update
                self.update_surgeries_time(chosen_v.room.num, chosen_v.order - 1)
                if prior_update.specific_senior is not None:
                    s_prior_update = tu[1].value
                    tu[1].value = tu[1].prior_value
                    tu[1].prior_value = s_prior_update
                    mutual = True
            else:
                prior_update = chosen_v.value
                chosen_v.value = chosen_v.prior_value
                chosen_v.prior_value = prior_update
            price = self.calc_price_by_variable(tu, chosen_v.room.num, type(chosen_v),
                                                num_surgeries[chosen_v.room.num], mutual=mutual, next=True,
                                                stable_schedule_flag=stable_schedule_flag)

    def update_schedule(self, best_schedule, num_surgeries, stable_schedule_flag=True):
        """
        updates v_dict to have best_schedule values (best schedule is a deep copy so we want to continue working
        with the same objects and not new ones)
        :param num_surgeries: dictionary{room num: number of surgeries}
        :param best_schedule: dictionary same format of v_dict but deep copied hence new objects
        """
        self.clear_all_schedule_assigned()
        for w, b_w in zip(self.v_dict, best_schedule):
            for r, b_r in zip(self.v_dict[w][self.schedule_date], best_schedule[b_w][self.schedule_date]):
                for t, b_t in zip(self.v_dict[w][self.schedule_date][r], best_schedule[b_w][self.schedule_date][b_r]):
                    for i in range(len(t)):
                        if b_t[i].value is not None:
                            if i == 0:
                                sr_v = t[i]
                                sr_v.value = w.find_surgery_request(b_t[i].value.request_num)
                                sr_v.value.assigned.add(sr_v.get_constraint_dro_key())
                            else:
                                t[i].value = w.find_surgeon_by_id(b_t[i].value.id)
                        else:
                            t[i].value = None
                        if b_t[i].prior_value is not None:
                            if i == 0:
                                t[i].prior_value = w.find_surgery_request(b_t[i].prior_value.request_num)
                            else:
                                t[i].prior_value = w.find_surgeon_by_id(b_t[i].prior_value.id)
                        else:
                            t[i].prior_value = None
                        t[i].start_time = b_t[i].start_time
                        t[i].end_time = b_t[i].end_time
        self.score = self._set_init_value_day(num_surgeries, stable_schedule_flag=stable_schedule_flag)
        self.num_surgeries = num_surgeries.copy()

    def update_schedule_with_nurses(self, nurse_schedule):
        """
        receives the nurses schedule and marks the surgeries which received nurse.
        Notice the method does not check that the surgery requests are the same - relies that from the main algorithm
        it is the same one i.e. the same surgery requests that are in the nurses variables are the ones in
        the ward schedule
        :param nurse_schedule: dictionary {room object:[(SN, CN) , (SN, CN)...]..}}
        the schedule given will only be of the current ward and not all the wards
        """
        full_solution = True
        for r in nurse_schedule:
            for i in range(len(nurse_schedule[r])):
                nt = nurse_schedule[r][i]
                if nt[0].value is None:  # no nurse for surgery
                    flag = False
                else:
                    flag = True
                sr_v = self.v_dict[self.ward][self.schedule_date][r.num][i][0]
                if flag:
                    sr_v.with_surgery_team['Nurse'] = nt
                else:
                    sr_v.with_surgery_team['Nurse'] = flag
                sr_v.value_in_update = sr_v.value
                sr_v.surgery_team_in_update['Nurse'] = nt
                full_solution = full_solution and flag

        self.with_surgery_Team['Nurse'] = True
        return full_solution

    def update_schedule_with_nurses_DSA(self, nurse_schedule):
        """
        receives the nurses schedule and marks the surgeries which received nurse.
        Notice the method relies on the fact that it may be that the schedules are not synchronized i.e. the surgery
        requests for the different surgeries are not the same in self.v_dict and the schedule given. Hence
        needs to check that each nurse is good for the specified surgery or not.
        :param nurse_schedule: dictionary {room object:[(SN, CN) , (SN, CN)...]..}}
        the schedule given will only be of the current ward and not all the wards
        """
        for r in nurse_schedule:
            for i in range(len(nurse_schedule[r])):
                sr_v = self.v_dict[self.ward][self.schedule_date][r.num][i][0]
                nt = nurse_schedule[r][i]
                flag = True
                for nv in nt:
                    if nv.value is None:  # no nurse for surgery
                        flag = False
                        break
                    elif (sr_v.value is not None) and \
                            (sr_v.value.surgery_type.st_id not in nv.value.skills[nv.n_type][self.a_id]):
                        flag = False
                        break
                if flag:
                    sr_v.with_surgery_team['Nurse'] = nt
                else:
                    sr_v.with_surgery_team['Nurse'] = flag
                sr_v.value_in_update = sr_v.value
                sr_v.surgery_team_in_update['Nurse'] = nt

        self.with_surgery_Team['Nurse'] = True

    def update_schedule_with_anesthetists(self, anesthetist_schedule):
        """
        receives the anesthetists schedule and marks the surgeries which received an anesthetist.
        Notice the method does not check that the surgery requests are the same - relies that from the main algorithm
        it is the same one i.e. the same surgery requests that are in the anesthetist variables are the ones in
        the ward schedule
        :param anesthetist_schedule: dictionary
        {room object:(Room Manager Variable, [av1, av2...])..}
        the schedule given will only be of the current ward and not all the wards
        """
        full_solution = True
        for r in anesthetist_schedule:
            for i in range(len(anesthetist_schedule[r][1])):
                av = anesthetist_schedule[r][1][i]
                if av.value is None:
                    flag = False
                else:
                    flag = True
                sr_v = self.v_dict[self.ward][self.schedule_date][r.num][i][0]
                if flag:
                    sr_v.with_surgery_team['Anesthetist'] = av.value
                else:
                    sr_v.with_surgery_team['Anesthetist'] = flag
                sr_v.value_in_update = sr_v.value
                sr_v.surgery_team_in_update['Anesthetist'] = av.value
                full_solution = full_solution and flag

        self.with_surgery_Team['Anesthetist'] = True
        return full_solution

    def update_schedule_with_anesthetists_DSA(self, anesthetist_schedule):
        """
        receives the anesthetists schedule and marks the surgeries which received an anesthetist.
        the method relies on the fact that it may be that the schedules are not synchronized i.e. the surgery requests
        for the different surgeries are not the same in self.v_dict and the schedule given.
        Hence needs to check that each anesthetist is good for the specified surgery or not
        :param anesthetist_schedule: dictionary
        {room object:(Room Manager Variable, [av1, av2...])..}
        the schedule given will only be of the current ward and not all the wards
        """
        for r in anesthetist_schedule:
            for i in range(len(anesthetist_schedule[r][1])):
                sr_v = self.v_dict[self.ward][self.schedule_date][r.num][i][0]
                av = anesthetist_schedule[r][1][i]
                if av.value is None:
                    flag = False
                elif av.value.rank == 'Stagiaire':
                    if (sr_v.value is not None) and (sr_v.value.surgery_type.st_id in av.value.skills[self.a_id]):
                        flag = True
                    else:
                        flag = False
                else:
                    flag = True
                if flag:
                    sr_v.with_surgery_team['Anesthetist'] = av.value
                else:
                    sr_v.with_surgery_team['Anesthetist'] = flag
                sr_v.value_in_update = sr_v.value
                sr_v.surgery_team_in_update['Anesthetist'] = av.value

        self.with_surgery_Team['Anesthetist'] = True

    def update_schedule_with_equipment_DSA(self, equipment_schedule):
        """
        receives the equipments schedule and marks the surgeries which received the necessary equipments.
        Notice the method relies on the fact that it may be that the schedules are not synchronizes i.e. the surgery
        requests for the different surgeries are not the same in self.v_dict and the schedule given.
        Hence needs to check what are the equipment required for the surgery and check if it got all of them
        :param equipment_schedule: dictionary {room:[(ev1, ev2..evn),..]...}
        """
        for r in equipment_schedule:
            for i in range(len(equipment_schedule[r])):
                sr_v = self.v_dict[self.ward][self.schedule_date][r.num][i][0]
                flag = True
                equipments_in_update = []
                if sr_v.value is not None:
                    equipments = sr_v.value.equipments
                    for ev in equipment_schedule[r][i]:
                        if equipments:  # not empty list
                            if ev.equipment.id in equipments:
                                flag = flag and ev.value
                        if ev.value:
                            equipments_in_update.append(ev.equipment.id)
                else:
                    flag = False
                sr_v.with_surgery_team['Equipment'] = flag
                sr_v.value_in_update = sr_v.value
                sr_v.surgery_team_in_update['Equipment'] = equipments_in_update

        self.with_surgery_Team['Equipment'] = True

    def update_schedule_with_equipment(self, equipment_schedule):
        """
        receives the equipments schedule and marks the surgeries which received the necessary equipments.
        Notice the method does not check that the surgery requests are the same - relies that from the main algorithm
        it is the same one i.e. the same surgery requests that are in the anesthetist variables are the ones in
        the ward schedule
        can be updated in the future to have several dates
        :param equipment_schedule: dictionary {room:[(ev1, ev2..evn),..]...}
        """
        full_solution = True
        for r in equipment_schedule:
            for i in range(len(equipment_schedule[r])):
                sr_v = self.v_dict[self.ward][self.schedule_date][r.num][i][0]
                flag = False
                if sr_v.value is not None:
                    equipments = sr_v.value.equipments
                    if equipments:  # non empty list
                        for ev in equipment_schedule[r][i]:
                            if ev.equipment.id in equipments:
                                flag = ev.value
                                break  # or all the equipments are scheduled or none
                    else:  # no equipments needed for surgery - so we can mark equipment is given
                        flag = True
                sr_v.with_surgery_team['Equipment'] = flag
                sr_v.value_in_update = sr_v.value
                sr_v.surgery_team_in_update['Equipment'] = [ev.equipment.id for ev in equipment_schedule[r][i] if
                                                            ev.value]
                full_solution = full_solution and flag

        self.with_surgery_Team['Equipment'] = True
        return full_solution

    def clear_all_schedule_assigned(self):
        """
        clears all assigned list for the current schedule - all assigned lists for all surgery request will
        be empty
        """
        for w in self.v_dict:
            for r in self.v_dict[w][self.schedule_date]:
                for t in self.v_dict[w][self.schedule_date][r]:
                    if t[0].value is not None:
                        t[0].value.assigned.clear()

    def update_no_good_surgery_requests(self):
        # self.no_good_sr.clear()
        for w in self.v_dict:
            for r in self.v_dict[w][self.schedule_date]:
                for t in self.v_dict[w][self.schedule_date][r]:
                    if (not t[0].with_surgery_team['Nurse']) or (not t[0].with_surgery_team['Anesthetist']):
                        # self.no_good_sr.add(t[0].value)
                        self.no_good_sr[t[0].value] = self.counter  # todo verify that by value and not by pointer
                    elif t[0].value in self.no_good_sr:
                        del self.no_good_sr[t[0].value]

    def classifier_update_schedule(self, content, DSA):
        """
        sends to the corresponding update schedule method, depending on the agent the message was sent from
        :param DSA: boolean if for DSA algo
        :param content: {'a_id': e/n/a , 'schedule': concerned v_dict}
        :return: ea_fs - extra agent full solution - boolean - True - if a full solution was got from the update
        i.e. all surgeries received nurses ect.
        """
        m_a_id = content['a_id']
        schedule = content['schedule']
        ea_fs = None
        if m_a_id == 'a':
            if DSA:
                self.update_schedule_with_anesthetists_DSA(schedule)
            else:
                ea_fs = self.update_schedule_with_anesthetists(schedule)
        elif m_a_id == 'n':
            if DSA:
                self.update_schedule_with_nurses_DSA(schedule)
            else:
                ea_fs = self.update_schedule_with_nurses(schedule)
        elif m_a_id == 'e':
            if DSA:
                self.update_schedule_with_equipment_DSA(schedule)
            else:
                ea_fs = self.update_schedule_with_equipment(schedule)
        return ea_fs

    def NG_iteration(self, mail, stable_schedule_flag=True, stop_fs_flag=True, random_selection=True, no_good_flag=True):
        fs = True  # full solution
        for m in mail:
            ea_fs = self.classifier_update_schedule(m.content, DSA=False)  # extra schedule full solution
            fs = fs and ea_fs
        if no_good_flag:
            self.update_no_good_surgery_requests()
        for m in mail:
            self.update_counter(m.content['counter'], chief_agent=True)
        # curr_schedule = deepcopy(self.v_dict)
        curr_score = self.calc_dsa_value()
        if stop_fs_flag:
            if not fs:
                self.simulated_annealing_by_day(stable_schedule_flag=stable_schedule_flag,
                                                random_selection=random_selection)
                self.n_changes += 1
        else:
            self.simulated_annealing_by_day(stable_schedule_flag=stable_schedule_flag, random_selection=random_selection)
            self.n_changes += 1
        self.send_mail()
        # return {'fs': fs, 'schedule': curr_schedule, 'score': curr_score}
        return {'fs': fs, 'score': curr_score}

    def dsa_sa_iteration(self, mail, stable_schedule_flag=True, random_selection=True, no_good_flag=True):
        for m in mail:
            self.classifier_update_schedule(m.content, DSA=True)
        if no_good_flag:
            self.update_no_good_surgery_requests()
        for m in mail:
            self.update_counter(m.content['counter'], chief_agent=True)
        curr_schedule = deepcopy(self.v_dict)
        curr_num_surgeries = self.num_surgeries.copy()
        curr_score = self.calc_dsa_value()
        self.simulated_annealing_by_day(stable_schedule_flag=stable_schedule_flag, random_selection=random_selection)
        if self.calc_dsa_value() < curr_score:  # alternative schedule is worst then curr schedule
            self.update_schedule(curr_schedule, curr_num_surgeries, stable_schedule_flag=stable_schedule_flag)
        else:  # alternative schedule is equal valued or better than curr schedule
            change_probability = random.random()
            if change_probability > 0.7:  # will keep alternative schedule only for a chance of 70%
                self.update_schedule(curr_schedule, curr_num_surgeries, stable_schedule_flag=stable_schedule_flag)
        self.send_mail()
        num_schedule_changes = self.count_schedule_changes(curr_schedule)
        # we return the cost that resulted from the scheduled of the last iteration after receiving the allocating
        # agents schedule i.e. the real price
        # return {'schedule': curr_schedule, 'num_surgeries': curr_num_surgeries, 'score': curr_score}
        return {'score': curr_score, 'num_changes': num_schedule_changes}

    def count_schedule_changes(self, schedule):
        num_changes = 0
        for w, b_w in zip(self.v_dict, schedule):
            for r, b_r in zip(self.v_dict[w][self.schedule_date], schedule[b_w][self.schedule_date]):
                for t, b_t in zip(self.v_dict[w][self.schedule_date][r], schedule[b_w][self.schedule_date][b_r]):
                    if (t[0].value is None) and (b_t[0].value is None):
                        continue
                    elif (t[0].value is None) or (b_t[0].value is None):
                        num_changes += 1
                    elif t[0].value.request_num != b_t[0].value.request_num:
                        num_changes += 1
                    if (t[1].value is None) and (b_t[1].value is None):
                        continue
                    elif (t[1].value is None) or (b_t[1].value is None):
                        num_changes += 1
                    elif t[1].value.id != b_t[1].value.id:
                        num_changes += 1
        return num_changes

    def dsa_sc_iteration(self, mail, change_func, random_selection=True, stable_schedule_flag=True, no_good_flag=True):
        for m in mail:
            self.classifier_update_schedule(m.content, DSA=True)
        if no_good_flag:
            self.update_no_good_surgery_requests()
        for m in mail:
            self.update_counter(m.content['counter'], chief_agent=True)
        # curr_schedule = deepcopy(self.v_dict)
        # curr_num_surgeries = self.num_surgeries.copy()
        curr_score = self.calc_dsa_value()
        chosen_v, tu = getattr(self, change_func)(random_selection=random_selection,
                                                  stable_schedule_flag=stable_schedule_flag)
        if chosen_v:
            if self.calc_dsa_value() < curr_score:  # alternative schedule is worst then curr schedule
                self.return_to_prior_value(chosen_v, self.num_surgeries, tu, stable_schedule_flag=stable_schedule_flag)
                self.score = self._set_init_value_day(self.num_surgeries, stable_schedule_flag=stable_schedule_flag)
            else:  # alternative schedule is equal valued or better than curr schedule
                change_probability = random.random()
                if change_probability > 0.7:  # will keep alternative schedule only for a chance of 70%
                    self.return_to_prior_value(chosen_v, self.num_surgeries, tu,
                                               stable_schedule_flag=stable_schedule_flag)
                    self.score = self._set_init_value_day(self.num_surgeries, stable_schedule_flag=stable_schedule_flag)
        self.send_mail()
        # we return the cost that resulted from the scheduled of the last iteration after receiving the allocating
        # agents schedule i.e. the real price
        # return {'schedule': curr_schedule, 'num_surgeries': curr_num_surgeries, 'score': curr_score}
        return {'score': curr_score}

    def NG_sc_iteration(self, mail, change_func, stop_fs_flag=True, random_selection=True, stable_schedule_flag=True,
                        no_good_flag=True):
        fs = True  # full solution
        for m in mail:
            ea_fs = self.classifier_update_schedule(m.content, DSA=False)  # extra schedule full solution
            fs = fs and ea_fs
        if no_good_flag:
            self.update_no_good_surgery_requests()
        for m in mail:
            self.update_counter(m.content['counter'], chief_agent=True)
        # curr_schedule = deepcopy(self.v_dict)
        curr_score = self.calc_dsa_value()
        if stop_fs_flag:
            if not fs:
                getattr(self, change_func)(random_selection, stable_schedule_flag)
                self.n_changes += 1
        else:
            getattr(self, change_func)(random_selection, stable_schedule_flag)
            self.n_changes += 1
        self.send_mail()
        # return {'fs': fs, 'schedule': curr_schedule, 'score': curr_score}
        return {'fs': fs, 'score': curr_score}

    def single_variable_change(self, random_selection, stable_schedule_flag=True):
        """
        pass through the schedule find the first surgery request variable without a full surgery team
        and changes its value - returns the first variable which the change resulted in an improvement.
        :return:
        """
        d_dict = self.v_dict[self.ward][self.schedule_date]
        for room_num in d_dict:
            update_tuple_try = False  # try to add a new surgery - only once
            for tu in d_dict[room_num]:
                change = False  # if tuple has been updated
                sr_v = tu[0]
                need_stable = True
                for team in self.with_surgery_Team:
                    if not sr_v.with_surgery_team[team]:
                        need_stable = False
                        break
                if not need_stable:
                    if sr_v.value is not None:
                        delta_E = self.update_variable_value(sr_v, tu, self.num_surgeries, room_num,
                                                             random_selection=random_selection,
                                                             stable_schedule_flag=stable_schedule_flag)
                    elif self.find_t_by_index(room_num, self.num_surgeries[room_num])[0].end_time < \
                            Prices.calc_end_time(self.ward.start_d_hour, self.ward.d_duration):
                        delta_E, change = self.update_tuple_value(tu, self.num_surgeries[room_num] + 1, room_num,
                                                                  self.num_surgeries, random_selection=random_selection,
                                                                  stable_schedule_flag=stable_schedule_flag)
                        if change:
                            self.num_surgeries[room_num] += 1
                            sr_v = tu
                        update_tuple_try = True
                    else:
                        break
                    if delta_E > 0:
                        self.score = self._set_init_value_day(self.num_surgeries,
                                                              stable_schedule_flag=stable_schedule_flag)
                        # self.increment_counter(chief_agent=True)
                        return sr_v, tu
                    elif delta_E != 0:
                        self.return_to_prior_value(sr_v, self.num_surgeries, tu,
                                                   stable_schedule_flag=stable_schedule_flag)
                    # self.increment_counter(chief_agent=True)
                if update_tuple_try:
                    break
        return False, False  # all the surgeries have a full team

    def single_variable_change_explore(self, random_selection, stable_schedule_flag=True):
        """
        matches the DSA without stable schedule chooses a random variable to change more exploration
        :return:
        """
        for i in range(200):
            chosen_v, delta_E, tu = self._select_successor(self.num_surgeries, random_selection=random_selection,
                                                           stable_schedule_flag=stable_schedule_flag)
            if delta_E > 0:
                self.score = self._set_init_value_day(self.num_surgeries, stable_schedule_flag=stable_schedule_flag)
                # self.increment_counter(chief_agent=True)
                return chosen_v, tu
            elif delta_E != 0:
                self.return_to_prior_value(chosen_v, self.num_surgeries, tu, stable_schedule_flag=stable_schedule_flag)
            # self.increment_counter(chief_agent=True)
        return False, False

    def send_mail(self):
        for n in self.neighbours:
            self.gpo.append(Message(to_agent=n, content=self.create_message_content()))

    def calc_dsa_value(self, counter=False):
        """
        the score of the schedule will only take into account the cost and utility of surgeries whom received nurse/
        anesthetist/equipment. or ones who did not receive but their value was changed
        :return:
        """
        cost = 0
        utility = 0
        num_of_full_surgeries = 0  # num of surgeries with full staff and equipment
        for w in self.v_dict:
            for d in self.v_dict[w]:
                for rn in self.v_dict[w][d]:
                    for t in self.v_dict[w][d][rn]:
                        sr_v = t[0]
                        s_v = t[1]
                        if sr_v.value is not None:
                            if False not in sr_v.with_surgery_team.values():  # need Stable
                                if sr_v.value != sr_v.value_in_update:  # check if new surgery request is good for us
                                    count_in_value = self.check_srv_new_value(sr_v)
                                else:
                                    count_in_value = True
                            else:  # not need stable
                                if sr_v.value == sr_v.value_in_update:
                                    count_in_value = False
                                elif sr_v.with_surgery_team['Equipment']:
                                    count_in_value = \
                                        set(sr_v.value.equipments).issubset(set(sr_v.value_in_update.equipments))
                                else:
                                    if counter:  # when calculating value for non concurrent logical steps
                                        count_in_value = False
                                    else:
                                        count_in_value = True
                            if count_in_value:
                                num_of_full_surgeries += 1
                                utility += sr_v.value.surgery_type.utility
                                cost += self.get_srv_cost(sr_v)
                                cost += s_v.constraints['d']['overlapping'].prices[
                                    s_v.get_constraint_d_key(s_v.value.id)]
                            '''else:
                                cost += cost_factor * list(sr_v.with_surgery_team.values()).count(False)'''
        return utility - cost

    def check_srv_new_value(self, sr_v):
        """
        checks if the Nurses, Anesthetist and Equipment given to the original surgery request are good
        for this one
        :param sr_v: surgery request variable that needed to be stable but valued changed
        :return: boolean
        """
        for nv in sr_v.with_surgery_team['Nurse']:
            if sr_v.value.surgery_type.st_id not in nv.value.skills[nv.n_type][self.a_id]:
                return False
        if sr_v.with_surgery_team['Anesthetist'].rank == 'Stagiaire':
            if sr_v.value.surgery_type.st_id not in sr_v.with_surgery_team['Anesthetist'].skills[self.a_id]:
                return False
        if not set(sr_v.value.equipments).issubset(set(sr_v.value_in_update.equipments)):
            return False
        return True

    def get_srv_cost(self, sr_v):
        cost = 0
        for con_key in sr_v.constraints:
            for con in sr_v.constraints[con_key]:
                if con_key == 'dro':
                    if con != 'stable_schedule':
                        cost += sr_v.constraints[con_key][con].prices[sr_v.get_constraint_dro_key()]
                else:
                    cost += sr_v.constraints[con_key][con].prices[sr_v.get_constraint_dr_key()]
        return cost

    def create_message_content(self):

        schedule = deepcopy(self.v_dict)
        ward = self.ward
        ward_copy = list(schedule.keys())[0]
        return {'schedule': schedule, 'ward': ward, 'ward_copy': ward_copy, 'counter': self.counter}
