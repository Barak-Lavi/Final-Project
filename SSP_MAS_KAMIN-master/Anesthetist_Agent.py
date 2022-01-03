from Allocating_Agent import Allocating_Agent
from General_Variable import General_Variable
from A_in_Surgery import SurgeryVariable_Anesthetist
from byRoom_Variable import Room_Variable
from Constraint import Constraint
from copy import deepcopy
from decimal import Decimal
from Stagiaire_anesthetist import Stag_Anesth
from Message import Message
import math
import Prices
import random
import Anesthetist_Prices
import Static
import Chief_Agent
import Nurse_Prices

cost_factor = 1_000
inf_price = 1_000_000


# random.seed(4)


def by_ward_grade(anesthetist, w_id):
    if anesthetist is not None:
        return anesthetist.ward_grades[w_id]
    else:
        return 0


def reduce_oa_domain(chosen_v, w_id, new_room):
    """
    creates a set of the available current domain for operating anesthetist taking into account current assignation.
    Only anesthetists that aren't floor manager or room manager. if oa is the first surgery in the room i.e. new
    room - than if available will send only anesthetist not operating yet- so they can operate in all the
    surgeries of room. if not available then sends all the available domain.
    :param chosen_v: operating aneshtetist variable
    :param new_room: boolean true if first surgery in room
    :return: set of anesthetist
    """
    reduce_domain = chosen_v.domain.copy()
    reduce_domain.discard(chosen_v.value)
    for a in chosen_v.domain:
        if a is not None:
            if a.assigned['FM']:
                reduce_domain.discard(a)
                continue
            if a.assigned['RM']:
                reduce_domain.discard(a)
            if (chosen_v.surgery_request is not None) and (isinstance(a, Stag_Anesth)):
                if chosen_v.surgery_request.surgery_type.st_id not in a.skills[w_id]:
                    reduce_domain.discard(a)
    return reduce_domain


def reduce_fm_domain(chosen_v):
    reduce_domain = chosen_v.domain.copy()
    reduce_domain.discard(chosen_v.value)
    for a in chosen_v.domain:
        if a is not None:
            if a.assigned['RM']:
                reduce_domain.discard(a)
                continue
            if a.assigned['OA']:
                reduce_domain.discard(a)
    return reduce_domain


class Anes_Agent(Allocating_Agent):

    def __init__(self, day, hospital, general_post_office):
        self.anesthetists = hospital.anesthetists  # set
        self.num_rooms_RM = 2
        self.factor_weights = {'anesthetist_suitability': 0.3, 'schedule_ratio': 0.5, 'discrimination': 0.2}
        # init_factor_grades
        # {'nurse_suitability' : weight, 'schedule_ratio' : weight , 'discrimination' : weight}
        super(Anes_Agent, self).__init__(day, hospital, general_post_office, 'a')
        self._init_solution_day(by_ward_grade)
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
        :return: dictioanry
        {'schedule_date' : (Floor_Manager_Varialbe, {ward : {room :(Room_Manager_Variable, [anesthetist_variable1, anesthetist_variable2...])...
        """
        a_constraints, fm_constraints, rm_constraints = self._init_constraints()
        variables = {self.schedule_date:
                         (General_Variable(self.schedule_date, self.anesthetists, fm_constraints), {})}
        for w in self.room_allocations:
            variables[self.schedule_date][1][w] = {}
            for r in self.room_allocations[w]:
                variables[self.schedule_date][1][w][r] = \
                    (Room_Variable(self.schedule_date, r, self.anesthetists, rm_constraints), [])
                max_slots_rd, initial_surgery_duration = w.max_slots_room_day(self.schedule_date, r)
                # initial_surgery_duration = w.d_duration // max_slots_rd
                start_time = w.start_d_hour
                for i in range(max_slots_rd):
                    end_time = Prices.calc_end_time(start_time, initial_surgery_duration)
                    variables[self.schedule_date][1][w][r][1].append(
                        SurgeryVariable_Anesthetist(room=r, day=self.schedule_date, order=i,
                                                    domain=[w.w_id, self.anesthetists],
                                                    constraints=a_constraints,
                                                    start_time=start_time, end_time=end_time))
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
        min_anesthetist_per_room_constraint = Constraint({})
        min_rm_constraint = Constraint({})
        stable_schedule_constraint = Constraint({})

        # Hard Constraints
        anesthetist_overlapping_constraint = Constraint({})
        all_diff_constraint = Constraint({})
        stag_sr_constraint = Constraint({})

        a_constraints = {'dro': {'all_diff': all_diff_constraint, 'stable_schedule': stable_schedule_constraint,
                                 'stag_sr': stag_sr_constraint},
                         'dr': {'min_anesthetist': min_anesthetist_per_room_constraint},
                         'd': {'overlapping': anesthetist_overlapping_constraint}}
        fm_constraints = {'d': {'all_diff': all_diff_constraint}}

        rm_constraints = {'dr': {'all_diff': all_diff_constraint},
                          'd': {'min_rm': min_rm_constraint}}

        return a_constraints, fm_constraints, rm_constraints

    def _init_solution_day(self, parameter):
        allocated_anesthetists = []
        num_rooms_current_RM = self.num_rooms_RM
        current_RM = None

        # initial value for floor manager
        fm_v = self.v_dict[self.schedule_date][0]
        fm_v.value = random.choice(list(fm_v.domain))
        fm_v.prior_value = fm_v.value
        allocated_anesthetists.append(fm_v.value)
        fm_v.value.assigned['FM'].append(fm_v.get_constraint_d_key())

        for w in self.v_dict[self.schedule_date][1]:
            for r in self.v_dict[self.schedule_date][1][w]:
                rm_v = self.v_dict[self.schedule_date][1][w][r][0]
                # value for room manager
                if (num_rooms_current_RM < 2) and (current_RM is not None):
                    rm_v.value = current_RM
                    current_RM.assigned['RM'].append(rm_v.get_constraint_dr_key())
                    num_rooms_current_RM += 1
                else:
                    current_RM = None
                    rm_sorted_domain = sorted(rm_v.domain, key=lambda anes: parameter(anes, w.w_id), reverse=True)
                    if len(rm_sorted_domain) > 1:
                        for a in rm_sorted_domain:
                            if (a not in allocated_anesthetists) and (a is not None):
                                rm_v.value = a
                                allocated_anesthetists.append(a)
                                a.assigned['RM'].append(rm_v.get_constraint_dr_key())
                                num_rooms_current_RM = 1
                                current_RM = a
                                break

                # value of operating anesthetist
                if current_RM is not None:
                    oa_anes = None
                    oa_v = self.v_dict[self.schedule_date][1][w][r][1][0]
                    oa_sorted_domain = sorted(oa_v.domain, key=lambda anes: parameter(anes, w.w_id), reverse=True)
                    for oa in oa_sorted_domain:
                        if (oa not in allocated_anesthetists) and (oa is not None):
                            oa_anes = oa
                            allocated_anesthetists.append(oa)
                            break
                    if oa_anes is not None:
                        for av in self.v_dict[self.schedule_date][1][w][r][1]:
                            av.value = oa_anes
                            oa_anes.assigned['OA'].append(av.get_constraint_dro_key())
                    else:  # couldn't find an operating anesthetist for this room --> no need of room manager
                        rm_v.value = None
                        allocated_anesthetists.remove(current_RM)
                        current_RM.assigned['RM'].remove(rm_v.get_constraint_dr_key())
                        num_rooms_current_RM -= 1

    def simulated_annealing_by_day(self, init_sol_param=None, genetic=False, stable_schedule_flag=True,
                                   random_selection=True):
        """
        performs SA on a single day
        could receive d_dict of a specific ward and d - but just to be able to play with it for now...
        :param genetic: boolean - True if sa is used for init population in genetic algorithm , if so list of all the schedules
        is returned
        :param stable_schedule_flag: boolean - if to take into account stable schedule costs
        :param init_sol_param: function that determines the parameter which according to it the initial state will be
        generated
        :param random_selection: boolean if to choose randomly from domain or by max deltaE
        """
        if genetic:
            g = []
        global t
        plot_list = []
        current_value = self.calc_utility(with_cost_update=True, next=True, stable_schedule_flag=stable_schedule_flag)
        # float the value of the day -
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
                    # Static.simulated_graphs(plot_list, 'Anesthetists')
                    break
            else:
                chosen_v, delta_E = self._select_successor(stable_schedule_flag=stable_schedule_flag)
                if delta_E > 0:
                    num_changes += 1
                    current_value = self.calc_value()
                    self.increment_counter()
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
                        current_value = self.calc_value()
                        if current_value > best_value:
                            best_value = current_value
                            best_schedule = deepcopy(self.v_dict)
                        plot_list.append([t, current_value])
                        if genetic:
                            g.append((deepcopy(self.v_dict), current_value))
                        num_changes += 1
                    else:  # don't change
                        self.return_to_prior_value(chosen_v, stable_schedule_flag=stable_schedule_flag)
                    self.increment_counter()

    def calc_utility(self, with_cost_update, next=False, stable_schedule_flag=True):
        """
        calculates the value of a day overall utility - over all costs of all the variables in the day. Calculates the value
        from zero with no prior value calculated, that means not the value after an update of a single variable
        :param with_cost_update: boolean - determines if only to calculate utility or also take into consideration the
        costs
        :param stable_schedule_flag: boolean - if to take into account stable schedule costs
        :param next:  defines if an update was done before the calculation
        :return: double the price of the day - total utility - total costs
        """
        day_cost = 0
        s_n_u = 0
        w_r_u = 0
        if with_cost_update:
            day_cost += Anesthetist_Prices.set_d_prices(self.v_dict[self.schedule_date], None, self.num_rooms_RM, next)
        scheduled_wards = 0
        num_rooms = 0

        for w in self.v_dict[self.schedule_date][1]:
            w_s_rm_u = 0
            w_s_a_u = 0  # ward suitability anesthetist utility
            total_w_num_surgeries = 0
            scheduled_w_num_surgeries = 0
            scheduled_w_rooms = 0
            for r in self.v_dict[self.schedule_date][1][w]:
                rm_v = self.v_dict[self.schedule_date][1][w][r][0]
                if rm_v.value is not None:
                    w_s_rm_u += rm_v.value.ward_grades[w.w_id]
                    scheduled_w_rooms += 1
                num_rooms += 1
                if with_cost_update:
                    day_cost += Anesthetist_Prices.set_dr_prices(self.v_dict[self.schedule_date][1][w][r][1])
                for v in self.v_dict[self.schedule_date][1][w][r][1]:
                    if self.with_sr[w]:
                        if v.surgery_request is not None:
                            total_w_num_surgeries += 1
                    else:
                        total_w_num_surgeries += 1
                    if v.value is not None:
                        scheduled_w_num_surgeries += 1
                        w_s_a_u += v.value.ward_grades[w.w_id]
                        if with_cost_update:
                            day_cost += Anesthetist_Prices.set_dro_prices \
                                (v, self.with_sr[w], w.w_id, stable_schedule_flag=stable_schedule_flag)
            if total_w_num_surgeries > 0:
                w_r_u += self.ward_strategy_grades[w.w_id] * (scheduled_w_num_surgeries / total_w_num_surgeries)
            if scheduled_w_num_surgeries > 0:
                if scheduled_w_rooms == 0:
                    print('problem')
                s_n_u += 0.5 * (w_s_a_u / scheduled_w_num_surgeries) + 0.5 * (w_s_rm_u / scheduled_w_rooms)
                scheduled_wards += 1
        if scheduled_wards == 0:
            day_utility = 0
        else:
            day_utility = (cost_factor * num_rooms * 2) * \
                          (self.factor_weights['anesthetist_suitability'] * (s_n_u / scheduled_wards) +
                           self.factor_weights['schedule_ratio'] * w_r_u +
                           self.factor_weights['discrimination'] * (scheduled_wards / len(self.room_allocations)))
        return day_utility - day_cost

    def _select_successor(self, stable_schedule_flag=True):
        """
        selects a random variable and changes its value randomly from the domain , calculates the difference of the
        total solution price due to the change in the solution. \
        The difference is calculated by the subtraction of prior price from next price
        prior price - utility - cost of the specific variable that changed
        next price - utility - cost after the change
        :param stable_schedule_flag: boolean - if to take into account stable schedule costs
        :return: chosen variable and the difference in the total price of the solution, and the tuple chosen
        """
        var_list = self.create_var_list()
        chosen_v = random.choice(var_list)  # chosen variable
        w = None
        if isinstance(chosen_v, Room_Variable):
            r = chosen_v.room
            w = self.get_ward_from_room_allocation(r)
            if self.with_sr[w]:
                if isinstance(chosen_v, SurgeryVariable_Anesthetist):
                    if chosen_v.surgery_request is None:
                        # return [chosen_v], inf_price * -1  # No surgery so no need of anesthetist
                        return [chosen_v], 0
                elif not self.room_is_scheduled(r, w):
                    # return [chosen_v], inf_price * -1  # No surgeries in room
                    return [chosen_v], 0
        if w is not None:
            var, delta_E = self.update_variable_value(chosen_v, w.w_id, stable_schedule_flag=stable_schedule_flag)
        else:
            var, delta_E = self.update_variable_value(chosen_v, stable_schedule_flag=stable_schedule_flag)
        return var, delta_E

    def single_variable_change(self, random_selection=True, stable_schedule_flag=True):
        """
        changes the anesthetist of a single surgery - the first one which current equipment does not match the
        wards needs and deltaE > 0.
        """
        for w in self.v_dict[self.schedule_date][1]:
            for r in self.v_dict[self.schedule_date][1][w]:
                for v in self.v_dict[self.schedule_date][1][w][r][1]:
                    if v.need_stable and \
                            ((v.value != v.value_in_update) or
                             ((v.value is not None) and
                              v.constraints['d']['overlapping'].prices[v.value.get_d_key(self.schedule_date)] > 0)):
                        if (v.value is None) or (v.value_in_update is None):
                            chosen_v, deltaE = self.update_variable_value(v, w.w_id,
                                                                          stable_schedule_flag=stable_schedule_flag)
                        elif v.constraints['d']['overlapping'].prices[v.value.get_d_key(self.schedule_date)] > 0:
                            chosen_v, deltaE = self.update_variable_value(v, w.w_id,
                                                                          stable_schedule_flag=stable_schedule_flag)
                        else:
                            continue
                    elif (not v.need_stable) and (v.value == v.value_in_update):
                        chosen_v, deltaE = self.update_variable_value(v, w.w_id,
                                                                      stable_schedule_flag=stable_schedule_flag)
                    else:
                        continue
                    if deltaE > 0:
                        self.score = self.calc_utility(with_cost_update=True, next=True,
                                                       stable_schedule_flag=stable_schedule_flag)
                        self.increment_counter()
                        return chosen_v
                    elif deltaE != 0:
                        self.return_to_prior_value(chosen_v, stable_schedule_flag=stable_schedule_flag)
                    self.increment_counter()
        return False

    def single_variable_change_explore(self, random_selection=True, stable_schedule_flag=True):
        """
        matches the DSA without stable schedule chooses a random variable to change more exploration
        """
        for i in range(200):
            chosen_v, deltaE = self._select_successor(stable_schedule_flag=stable_schedule_flag)
            if deltaE > 0:
                self.score = self.calc_utility(with_cost_update=True, next=True,
                                               stable_schedule_flag=stable_schedule_flag)
                self.increment_counter()
                return chosen_v
            elif deltaE != 0:
                self.return_to_prior_value(chosen_v, stable_schedule_flag=stable_schedule_flag)
            self.increment_counter()
        return False

    def dsa_sc_iteration(self, mail, change_func=None, random_selection=True, stable_schedule_flag=True,
                         no_good_flag=True):
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
        chosen_v = getattr(self, change_func)(stable_schedule_flag=stable_schedule_flag)
        if chosen_v:
            if self.score < curr_score:  # alternative schedule is worst then curr schedule
                self.return_to_prior_value(chosen_v, stable_schedule_flag=stable_schedule_flag)
                self.score = curr_score
            else:  # alternative schedule is equal valued or better than curr schedule
                change_probability = random.random()
                if change_probability > 0.7:  # will keep alternative schedule only for a chance of 70%
                    self.return_to_prior_value(chosen_v, stable_schedule_flag=stable_schedule_flag)
                    self.score = curr_score
        self.send_mail()
        # we return the cost that resulted from the scheduled of the last iteration after receiving the allocating
        # agents schedule i.e. the real price
        # return {'schedule': curr_schedule, 'score': curr_score+stable_schedule_price}
        return {'score': curr_score + stable_schedule_price}

    def create_var_list(self):
        """
        creates a list with all the variables i.e - floor manager variable, all the room manager variables, and all the
        operating anesthetist variables
        :return: list of different variables
        """

        var_list = [self.v_dict[self.schedule_date][0]]
        for w in self.v_dict[self.schedule_date][1]:
            for r in self.v_dict[self.schedule_date][1][w]:
                var_list.append(self.v_dict[self.schedule_date][1][w][r][0])
                var_list.extend(self.v_dict[self.schedule_date][1][w][r][1])
        return var_list

    def update_variable_value(self, chosen_v, w_id=None, stable_schedule_flag=True):
        feasible_domain = self.check_feasibility(chosen_v, w_id)
        if feasible_domain:
            weights = []
            if isinstance(chosen_v, SurgeryVariable_Anesthetist):  # operating Anesthetist
                if len(feasible_domain) > 1:
                    return self.new_room(chosen_v, feasible_domain, w_id, stable_schedule_flag=stable_schedule_flag)
                else:
                    domain = feasible_domain[0]
                    weights = self.calc_domain_weights(domain, chosen_v)
                    key_assigned = 'OA'
                    value_assigned = chosen_v.get_constraint_dro_key()
            elif isinstance(chosen_v, Room_Variable):  # room manager
                if chosen_v.value is None:
                    return self.new_room(chosen_v, feasible_domain, w_id, stable_schedule_flag=stable_schedule_flag)
                else:
                    domain = feasible_domain[0]
                    key_assigned = 'RM'
                    value_assigned = chosen_v.get_constraint_dr_key()
            else:  # floor manager
                domain = feasible_domain[0]
                key_assigned = 'FM'
                value_assigned = chosen_v.get_constraint_d_key()

            if weights:
                chosen_value = random.choices(list(domain), weights=weights, k=1)[0]
            else:
                chosen_value = random.choice(list(domain))
            if chosen_value is None:
                if isinstance(chosen_v, SurgeryVariable_Anesthetist):
                    if self.num_manned_surgeries(chosen_v.room) == 1:
                        return self.cancel_room(chosen_v, stable_schedule_flag=stable_schedule_flag)
                elif isinstance(chosen_v, Room_Variable):
                    return self.cancel_room(chosen_v, stable_schedule_flag=stable_schedule_flag)

            chosen_v.prior_value = chosen_v.value
            prior_price = self.calc_price_by_variable(chosen_v, stable_schedule_flag=stable_schedule_flag)
            if chosen_v.prior_value is not None:
                chosen_v.prior_value.assigned[key_assigned].remove(value_assigned)
                Anesthetist_Prices.set_all_diff_price(chosen_v, self.num_rooms_RM)
            chosen_v.value = chosen_value
            if chosen_value is not None:
                chosen_v.value.assigned[key_assigned].append(value_assigned)
            next_price = self.calc_price_by_variable(chosen_v, next=True, stable_schedule_flag=stable_schedule_flag)
            return [chosen_v], next_price - prior_price
        else:
            # return [chosen_v], inf_price * -1
            return [chosen_v], 0
            # no legal change is optional - (no change was done) - no need of sa changing

    def calc_domain_weights(self, domain, chosen_v):
        """
        assignees for each anesthetist in operating anesthetist domain a weight for random selection - anesthetist
        assigned already to room have highest weight --> anesthetist free with no operation --> the rest.
        """
        weights = []
        for oa in domain:
            if oa is None:
                weights.append(5)
            elif not oa.assigned['OA']:
                weights.append(15)
            elif [key for key in oa.assigned['OA'] if key[11] == chosen_v.room.num]:  # oa is scheduled in room
                weights.append(30)
            else:
                weights.append(5)
        return weights

    def check_feasibility(self, chosen_v, w_id):
        """
        considers all the possible options for variable's new value . in case of room manager
        checks if their is an option of adding another room - takes into consideration also option of room manager
        and also option of operating aneshtetist, in case of floor manager checks if their is a senior available, in
        case of operating aneshtetist checks if it is the first surgery in the room and if it is - then checks if their
        is an option for room manager for the room
        :param chosen_v: aneshtetist variable (one of the three fm/rm/oa) which value is currently None
        :return:list of available reduced domain i.e. list of sets - if rm or oa then both their domains
        """

        if isinstance(chosen_v, SurgeryVariable_Anesthetist):  # operating anesthetist
            r = chosen_v.room
            if self.num_manned_surgeries(r) == 0:  # first surgery in room --> need room manager for room
                oa_available_domain = reduce_oa_domain(chosen_v, w_id, new_room=True)
                if len(oa_available_domain) > 0:
                    w = self.get_ward_from_room_allocation(r)
                    rm_v = self.v_dict[self.schedule_date][1][w][r][0]
                    if rm_v.value is not None:
                        print('problem')
                    rm_available_domain = self.reduce_rm_domain(rm_v)
                    if len(rm_available_domain) > 0:
                        return [rm_available_domain, oa_available_domain]
                    else:
                        return []
                else:
                    return []
            else:
                oa_available_domain = reduce_oa_domain(chosen_v, w_id, new_room=False)
                if len(oa_available_domain) > 0:
                    return [oa_available_domain]
                else:
                    return []
        elif isinstance(chosen_v, Room_Variable):  # room manager
            rm_available_domain = self.reduce_rm_domain(chosen_v)
            if len(rm_available_domain) > 0:
                if chosen_v.value is None:
                    r = chosen_v.room
                    w = self.get_ward_from_room_allocation(r)
                    oa_available_domain = reduce_oa_domain(self.v_dict[self.schedule_date][1][w][r][1][0],
                                                           w_id, new_room=True)
                    if len(oa_available_domain) > 0:
                        return [rm_available_domain, oa_available_domain]
                    else:
                        return []
                else:
                    return [rm_available_domain]
            else:
                return []
        else:  # floor manager
            fm_available_domain = reduce_fm_domain(chosen_v)
            if len(fm_available_domain) > 0:
                return [fm_available_domain]
            else:
                return []

    def reduce_rm_domain(self, chosen_v):
        """
        creates a set of the available current domain for room manager taking into account current assignation.
        Only anesthetists not operating, not floor manager, and room managing less then num_rooms_RM rooms can be
        available
        :param chosen_v: room manager variable
        :return: set of available room managers anesthetists
        """
        reduce_domain = chosen_v.domain.copy()
        reduce_domain.discard(chosen_v.value)
        for a in chosen_v.domain:
            if a is not None:
                if a.assigned['FM']:
                    reduce_domain.discard(a)
                    continue
                if a.assigned['OA']:
                    reduce_domain.discard(a)
                    continue
                if len(a.assigned['RM']) >= self.num_rooms_RM:
                    reduce_domain.discard(a)
                    continue
        return reduce_domain

    def num_manned_surgeries(self, r):
        """
        :param r room object
        """
        num_manned_surgeries_r = 0
        w = self.get_ward_from_room_allocation(r)
        for v in self.v_dict[self.schedule_date][1][w][r][1]:
            if v.value is not None:
                num_manned_surgeries_r += 1
        return num_manned_surgeries_r

    def calc_price_by_variable(self, chosen_v, next=False, with_utility=True, stable_schedule_flag=True):
        """
        calculates the difference in the schedule value depending only on the value of this variable. takes in
        the calculation in to account only what is affected by this value
        :param chosen_v: anesthetist variable object - the variable which value was updated
        :param with_utility: boolean - if utility calculation is needed
        :param next: defines if an update was done in the variable value or prior price is calculated
        :return: new utility - new cost
        """
        cost = 0
        utility = 0
        cost += Anesthetist_Prices.set_d_prices(self.v_dict[self.schedule_date], chosen_v, self.num_rooms_RM, next)
        if isinstance(chosen_v, Room_Variable):
            r = chosen_v.room
            w = self.get_ward_from_room_allocation(r)
            if isinstance(chosen_v, SurgeryVariable_Anesthetist):
                cost += Anesthetist_Prices.set_dr_prices(self.v_dict[self.schedule_date][1][w][r][1])
                cost += Anesthetist_Prices.set_dro_prices \
                    (chosen_v, self.with_sr[w], w.w_id, stable_schedule_flag=stable_schedule_flag)
            if with_utility:  # floor manager does not contribute to utility
                utility = self.calc_utility_by_variable(chosen_v, w, next)
        if with_utility:
            return utility - cost
        else:
            return cost

    def calc_utility_by_variable(self, chosen_v, ward, next=False):
        """
        calculates a heuristic of the utility difference caused by the change of the variables value
        :param chosen_v: the relevant variable updated
        :param ward: ward object - the ward which recieved the variable room in room allocation
        :param next: boolean True if next price is calculated i.e. the prior value is taken into account comparing
        to the current variable value
        :return: the utility difference
        """
        utility = 0
        if chosen_v.value is not None:
            utility += self.factor_weights['anesthetist_suitability'] * chosen_v.value.ward_grades[ward.w_id]
            if next:
                if isinstance(chosen_v, SurgeryVariable_Anesthetist):
                    if chosen_v.prior_value is None:
                        # ratio of ward surgeries enlarged
                        utility += self.factor_weights['schedule_ratio'] * self.ward_strategy_grades[ward.w_id]
                        if not self.ward_is_scheduled(ward, chosen_v):
                            # this variable is the single one scheduled for the ward
                            utility += self.factor_weights['discrimination']
        elif next:
            if isinstance(chosen_v, SurgeryVariable_Anesthetist):
                if chosen_v.prior_value is not None:
                    # ratio of schedule got smaller
                    utility -= self.factor_weights['schedule_ratio'] * self.ward_strategy_grades[ward.w_id]
                    if not self.ward_is_scheduled(ward):  # the ward was scheduled to a single surgery and now it is not
                        utility -= self.factor_weights['discrimination']
        return utility * cost_factor * 5

    def ward_is_scheduled(self, ward, chosen_v=None):
        """
        checks if a certain ward has anesthetist scheduled for its operations -
        :param ward: ward object - the ward we want to check for
        :param chosen_v: nurse variable object - if not None the check will be done if any other surgeries are scheduled with
        nurses except this variable if it is None it will be a general check to see if the ward surgeries are scheduled
        with nurses
        :return: boolean True - if ward is scheduled false- if not scheduled
        """

        for r in self.v_dict[self.schedule_date][1][ward]:
            for v in self.v_dict[self.schedule_date][1][ward][r][1]:
                if v.value is not None:
                    if chosen_v is not None:
                        if v != chosen_v:
                            return True
                    else:
                        return True
        return False

    def new_room(self, chosen_v, feasible_domain, w_id, stable_schedule_flag=True):
        """
        schedules with anesthetists a new room that is not yet scheduled. i.e. chooses a value for the room manager
        and for the possible surgeries in the room the same operating anesthetist is chosen. The only surgeries
        chosen for the room are the ones that don't overlap with the chosen operating anesthetis current surgeries.
        The method knows to check different values for the opperating aneshtetist if the chosen one has overlapping
        :param chosen_v: room manager aneshtetist object or operatin aneshtetist object
        :param feasible_domain:list[room manager domain - set, operating aneshtetist domain -set]
        :param stable_schedule_flag: boolean - if to take into account stable schedule costs
        :return: next price - prior price
        """
        tested_oa = set()  # all the operating anesthetists that have been already checked
        rm_available_domain = feasible_domain[0]
        oa_available_domain = feasible_domain[1]
        while True:
            # choose values
            if len(oa_available_domain) < len(rm_available_domain):
                oa_value = random.choice(list(oa_available_domain))
                tested_oa.add(oa_value)
                if oa_value in rm_available_domain:
                    rm_available_domain.discard(oa_value)
                rm_value = random.choice(list(rm_available_domain))
            elif len(rm_available_domain) < len(oa_available_domain):
                rm_value = random.choice(list(rm_available_domain))
                if rm_value in oa_available_domain:
                    oa_available_domain.discard(rm_value)
                oa_value = random.choice(list(oa_available_domain))
                tested_oa.add(oa_value)
            else:  # same length
                rm_value = random.choice(list(rm_available_domain))
                if rm_value in oa_available_domain:
                    oa_available_domain.discard(rm_value)
                if len(oa_available_domain) > 0:
                    oa_value = random.choice(list(oa_available_domain))
                    tested_oa.add(oa_value)
                else:
                    return [chosen_v], 0
                    # return [chosen_v], inf_price * -1
                    # only a single value in rm_domain and oa_domain and it is the same one

            # choose variables to update - only oa_v that don't overlap the oa_value current surgeries
            r = chosen_v.room
            w = self.get_ward_from_room_allocation(r)
            var_list = self.v_dict[self.schedule_date][1][w][r][1].copy()
            oa_is_assigned = oa_value.assigned['OA']
            for av in self.v_dict[self.schedule_date][1][w][r][1]:
                if self.with_sr[w]:
                    if av.surgery_request is None:  # remove surgeries with no surgery request
                        var_list.remove(av)
                        continue
                    if isinstance(oa_value, Stag_Anesth):
                        if av.surgery_request.surgery_type.st_id not in oa_value.skills[w_id]:
                            var_list.remove(av)
                            continue
                if oa_is_assigned:
                    for dro_key in oa_value.assigned['OA']:
                        assigned_var = self.get_var_by_dro_key(dro_key)
                        overlapping = Prices.check_overlapping(av, assigned_var)
                        if overlapping:
                            var_list.remove(av)
                            break
            if len(var_list) > 0:  # their are surgeries that don't overlap room can be opened
                break
            else:
                oa_available_domain.discard(oa_value)
                if (not oa_value.assigned['OA']) and (not oa_value.rank == 'Stagiaire'):
                    rm_available_domain.add(oa_value)
                if (rm_value not in tested_oa) and (not rm_value.assigned['RM']):
                    oa_available_domain.add(rm_value)  # will only add it if this value hasn't yet been tested
            if not len(oa_available_domain) > 0:
                return [chosen_v], 0
                # return [chosen_v], inf_price * -1
                # all oa domain was tested and all of them overlap all the surgeries in room

        rm_v = self.v_dict[self.schedule_date][1][w][r][0]
        var_list.append(rm_v)
        # rm_v.prior_value = rm_v.value

        # calc delta_e
        prior_price = 0
        for v in var_list:
            v.prior_value = v.value  # don't delete this row if for loop is deleted
            prior_price += self.calc_price_by_variable(v, stable_schedule_flag=stable_schedule_flag)
        next_price = 0
        for v in var_list:
            if isinstance(v, SurgeryVariable_Anesthetist):
                v.value = oa_value
                v.value.assigned['OA'].append(v.get_constraint_dro_key())
            else:
                v.value = rm_value
                v.value.assigned['RM'].append(v.get_constraint_dr_key())
            next_price += self.calc_price_by_variable(v, next=True, stable_schedule_flag=stable_schedule_flag)
        return var_list, next_price - prior_price

    def get_var_by_dro_key(self, dro_key):
        """
        finds the variable represented by the day_room_order key. i.e. the variable representing the surgery in a
        certain room and order
        :param dro_key: String 'YYYY-MM-DD_roomnum_order'
        :return: operating anesthetist variable
        """
        order_index = dro_key.index("_", 11)
        room_num = int(dro_key[11:order_index])
        order = int(dro_key[order_index + 1:])
        room = self.get_room_from_room_allocation(room_num)
        ward = self.get_ward_from_room_allocation(room)

        return self.v_dict[self.schedule_date][1][ward][room][1][order]

    def cancel_room(self, chosen_v, stable_schedule_flag=True):

        room = chosen_v.room
        ward = self.get_ward_from_room_allocation(room)
        var_list = []
        rm_v = self.v_dict[self.schedule_date][1][ward][room][0]
        if rm_v.value is None:
            print('problem')
        rm_v.prior_value = rm_v.value
        var_list.append(rm_v)
        for av in self.v_dict[self.schedule_date][1][ward][room][1]:
            if av.value is not None:
                var_list.append(av)
                av.prior_value = av.value

        prior_price = 0
        for v in var_list:
            prior_price += self.calc_price_by_variable(v, stable_schedule_flag=stable_schedule_flag)
        next_price = 0
        for v in var_list:
            if isinstance(v, SurgeryVariable_Anesthetist):
                v.prior_value.assigned['OA'].remove(v.get_constraint_dro_key())
                Anesthetist_Prices.set_all_diff_price(v, self.num_rooms_RM)
            else:
                v.prior_value.assigned['RM'].remove(v.get_constraint_dr_key())
                Anesthetist_Prices.set_all_diff_price(v, self.num_rooms_RM)
            v.value = None
        for v in var_list:
            next_price += self.calc_price_by_variable(v, next=True, stable_schedule_flag=stable_schedule_flag)

        return var_list, next_price - prior_price

    def calc_value(self):
        """
        calculates the total cost of the solution node - sums the costs on all constraint objects
        :return: float total cost
        """
        w = list(self.room_allocations.keys())[0]
        r = next(iter(self.room_allocations[w]))
        fm_v = self.v_dict[self.schedule_date][0]
        rm_v = self.v_dict[self.schedule_date][1][w][r][0]
        oa_v = self.v_dict[self.schedule_date][1][w][r][1][0]
        cost = 0

        cost += sum(fm_v.constraints['d']['all_diff'].prices.values())
        cost += sum(rm_v.constraints['d']['min_rm'].prices.values())
        cost += sum(oa_v.constraints['dr']['min_anesthetist'].prices.values())
        cost += sum(oa_v.constraints['d']['overlapping'].prices.values())
        cost += sum(oa_v.constraints['dro']['stable_schedule'].prices.values())
        cost += sum(oa_v.constraints['dro']['stag_sr'].prices.values())
        utility = self.calc_utility(with_cost_update=False)
        return utility - cost

    def return_to_prior_value(self, chosen_v, stable_schedule_flag=True):
        """
        returns the solution to the prior solution changes the values of the concerned variables back and updates
        num_surgeries if needed
        :param stable_schedule_flag: boolean - if to take into account stable schedule costs
        :param chosen_v: list of variable depends if there was a single variable value change or cancel_room/new_room
        """

        for v in chosen_v:
            if isinstance(v, SurgeryVariable_Anesthetist):  # operating Anesthetist
                key_assigned = 'OA'
                value_assigned = v.get_constraint_dro_key()
            elif isinstance(v, Room_Variable):  # room manager
                key_assigned = 'RM'
                value_assigned = v.get_constraint_dr_key()
            else:  # floor manager
                key_assigned = 'FM'
                value_assigned = v.get_constraint_d_key()
            if v.prior_value is not None:
                v.prior_value.assigned[key_assigned].append(value_assigned)
            if v.value is not None:
                v.value.assigned[key_assigned].remove(value_assigned)
                Anesthetist_Prices.set_all_diff_price(v, self.num_rooms_RM)
            prior_update = v.value
            v.value = v.prior_value
            v.prior_value = prior_update
            self.calc_price_by_variable(v, next=True, with_utility=False, stable_schedule_flag=stable_schedule_flag)

    def update_schedule(self, best_schedule, stable_schedule_flag=True):
        """
        updates v_dict to have best_schedule values (best schedule is a deep copy so we want to continue working
        with the same objects and not new ones)
        :param stable_schedule_flag: boolean - if to take into account stable schedule costs
        :param best_schedule: dictionary same format of v_dict but deep copied hence new objects
        """

        self.clear_all_assigned()
        fm_v = self.v_dict[self.schedule_date][0]
        b_fm_v = best_schedule[self.schedule_date][0]
        self.update_var_from_best_schedule(fm_v, b_fm_v)

        for w, b_w in zip(self.v_dict[self.schedule_date][1], best_schedule[self.schedule_date][1]):
            for r, b_r in zip(self.v_dict[self.schedule_date][1][w], best_schedule[self.schedule_date][1][b_w]):
                rm_v = self.v_dict[self.schedule_date][1][w][r][0]
                b_rm_v = best_schedule[self.schedule_date][1][b_w][b_r][0]
                self.update_var_from_best_schedule(rm_v, b_rm_v)
                for v, b_v in zip(self.v_dict[self.schedule_date][1][w][r][1],
                                  best_schedule[self.schedule_date][1][b_w][b_r][1]):
                    self.update_var_from_best_schedule(v, b_v)
                    if self.with_sr[w]:
                        v.start_time = b_v.start_time
                        v.end_time = b_v.end_time
                        if b_v.surgery_request is not None:
                            v.surgery_request = w.find_surgery_request(b_v.surgery_request.request_num)
                        else:
                            v.surgery_request = None
        self.score = self.calc_utility(with_cost_update=True, next=True, stable_schedule_flag=stable_schedule_flag)

    def update_var_from_best_schedule(self, var, b_var):
        """
        updates variable value and prior value like the values in b_var - variable object representing the same
        variable as var from best_schedule which is a deep copy of self.v_dict
        :param var: anesthetist variable
        :param b_var: best var - same variable from best_schedule
        :return:
        """
        if b_var.value is not None:
            var.value = self.find_anesthetist(b_var.value.id)
            var.value.assigned = deepcopy(b_var.value.assigned)
        else:
            var.value = None
        if b_var.prior_value is not None:
            var.prior_value = self.find_anesthetist(b_var.prior_value.id)
            var.prior_value.assigned = deepcopy(b_var.prior_value.assigned)
        else:
            var.prior_value = None

    def clear_all_assigned(self):
        """"
        clears all the assigned keys of all the anesthetists
        """
        for a in self.anesthetists:
            for key in a.assigned:
                a.assigned[key].clear()

    def find_anesthetist(self, anesthetist_id):
        """
        searches for an anesthetist with this id
        :param anesthetist_id: int id
        :return: anesthetist object with anesthetist_id
        """
        for a in self.anesthetists:
            if a.id == anesthetist_id:
                return a

    def update_schedule_by_ward(self, schedule, ward, ward_copy, stable_schedule_flag=True):
        """
        updates v_dict by schedule given from ward - the update includes adding the surgery request for each variable,
        updating the surgery times, cancelling the anesthetists for variables with no surgery request i.e. no surgery
        updating the field with_sr to True so calculations will be done matchingly. The function also updates the
        value of the schedule i.e. updates all constraint dictionaries with new prices
        :param schedule: dictionary of ward's schedule deep copy of original schedule
        {ward_object: {date : room: [tuples(sr_v, s_v)]}}
        :param stable_schedule_flag: boolean - if to take into account stable schedule costs
        :param ward_copy: copy of ward object as it is in schedule
        :return: the updated value of the schedule
        """
        full_solution = True
        for r in self.v_dict[self.schedule_date][1][ward]:
            room_is_scheduled = False
            for i in range(len(self.v_dict[self.schedule_date][1][ward][r][1])):
                start_time = schedule[ward_copy][self.schedule_date][r.num][i][0].start_time
                end_time = schedule[ward_copy][self.schedule_date][r.num][i][0].end_time
                sr = schedule[ward_copy][self.schedule_date][r.num][i][0].value
                v = self.v_dict[self.schedule_date][1][ward][r][1][i]
                v.start_time = start_time
                v.end_time = end_time
                v.surgery_request = sr
                v.prior_value = None
                if sr is None:
                    # v.prior_value = None
                    if v.value is not None:
                        v.value.assigned['OA'].remove(v.get_constraint_dro_key())
                    v.value = None
                    v.need_stable = True
                else:
                    # from old version:
                    # room_is_scheduled = True
                    if v.value is None:
                        v.need_stable = False
                        full_solution = False
                    else:
                        if isinstance(v.value, Stag_Anesth):
                            if v.surgery_request.surgery_type.st_id in v.value.skills[ward.w_id]:
                                v.need_stable = True
                                # new
                                room_is_scheduled = True
                                # end new
                            else:
                                v.value.assigned['OA'].remove(v.get_constraint_dro_key())
                                v.value = None
                                # end new
                                v.need_stable = False
                                full_solution = False
                        else:
                            # new
                            room_is_scheduled = True
                            # end new
                            v.need_stable = True
                v.value_in_update = v.value
            if not room_is_scheduled:  # if there weren't any surgeries from ward in room no need of room manager for
                # this room
                rm_v = self.v_dict[self.schedule_date][1][ward][r][0]
                rm_v.prior_value = None
                if rm_v.value is not None:
                    rm_v.value.assigned['RM'].remove(rm_v.get_constraint_dr_key())
                rm_v.value = None

        self.with_sr[ward] = True
        '''updated_value = self.calc_utility(with_cost_update=True, next=True, stable_schedule_flag=stable_schedule_flag)
        anes_with_overlap = self.get_overlapping_d_keys()
        if anes_with_overlap:
            self.cancel_overlapping_anesthetists(anes_with_overlap)
            full_solution = False
            updated_value = self.calc_utility(with_cost_update=True, next=True, stable_schedule_flag=stable_schedule_flag)
        self.score = updated_value'''
        # bv, t, nc, pl, bs = self.simulated_annealing_by_day(by_ward_grade)
        # Static.simulated_graphs(pl, 'Anesthetists_withSR' + str(ward.w_id))
        return full_solution

    def room_is_scheduled(self, r, w):
        """
        checks if a room has surgery requests scheduled - when with_sr[w] = True
        :param r: room object
        :param w: ward object
        :return: boolean - True when their are surgery requests in room and false otherwise
        """
        for av in self.v_dict[self.schedule_date][1][w][r][1]:
            if av.surgery_request is not None:
                return True
        return False

    def count_schedule_changes(self, schedule):
        num_changes = 0
        if not ((self.v_dict[self.schedule_date][0].value is None) and (schedule[self.schedule_date][0].value is None)):
            if (self.v_dict[self.schedule_date][0].value is None) or (schedule[self.schedule_date][0].value is None):
                num_changes += 1
            elif self.v_dict[self.schedule_date][0].value.id != schedule[self.schedule_date][0].value.id:
                num_changes += 1
        for w, b_w in zip(self.v_dict[self.schedule_date][1], schedule[self.schedule_date][1]):
            for r, b_r in zip(self.v_dict[self.schedule_date][1][w], schedule[self.schedule_date][1][b_w]):
                t = self.v_dict[self.schedule_date][1][w][r]
                b_t = schedule[self.schedule_date][1][b_w][b_r]
                if not ((t[0].value is None) and (b_t[0].value is None)):
                    if (t[0].value is None) or (b_t[0].value is None):
                        num_changes += 1
                    elif t[0].value.id != b_t[0].value.id:
                        num_changes += 1
                for v, b_v in zip(t[1], b_t[1]):
                    if (v.value is None) and (b_v.value is None):
                        continue
                    elif (v.value is None) or (b_v.value is None):
                        num_changes += 1
                    elif v.value.id != b_v.value.id:
                        num_changes += 1
        return num_changes

    def send_mail(self):
        for w in self.wards:
            # todo addition for not full allocation
            if w in self.v_dict[self.schedule_date][1]:
                self.gpo.append(Message(to_agent=w.w_id,
                                        content={'a_id': self.a_id,
                                                 'schedule': deepcopy(self.v_dict[self.schedule_date][1][w]),
                                                 'counter': self.counter}))

    def get_ward_from_schedule(self, schedule, w_id):
        """
        :param w: int ward id of the ward we want the schedule of
        :param schedule: a copy instance of the agent's schedule (in earlier iteration)
        :return: the schedule regarding the given wardthe schedule regarding the given ward of the given agent i.e. if anesthetist so aneshtetist schedule
        for given ward
        """
        for w in schedule[self.schedule_date][1]:
            if w.w_id == w_id:
                return schedule[self.schedule_date][1][w]

    def get_stable_schedule_costs(self):

        for w in self.v_dict[self.schedule_date][1]:
            for r in self.v_dict[self.schedule_date][1][w]:
                for v in self.v_dict[self.schedule_date][1][w][r][1]:
                    stable_schedule_price = sum(v.constraints['dro']['stable_schedule'].prices.values())
                    return stable_schedule_price

    def get_overlapping_d_keys(self):
        """
        :return: list of d_keys of overlapping anesthetists
        """
        for w in self.v_dict[self.schedule_date][1]:
            for r in self.v_dict[self.schedule_date][1][w]:
                for v in self.v_dict[self.schedule_date][1][w][r][1]:
                    overlapping_price = sum(v.constraints['d']['overlapping'].prices.values())
                    if overlapping_price > 0:
                        return [a for a in v.constraints['d']['overlapping'].prices if
                                v.constraints['d']['overlapping'].prices[a] > 0]
                    else:
                        return []

    def cancel_overlapping_anesthetists(self, overlap_d_keys):
        """
        cancels anesthetist allocation if overlapping after ward update - the overlapping occured because of change in ward
        schedule- we don't want to take these allocations into account
        cancels only if the variable actually has overlapping and only one of the overlapped not both.
        param overlap_d_keys: list of d_keys 'YYYY-MM-DD-NID' of anesthetist which have an overlap
        """
        for key in overlap_d_keys:
            a_id = int(key[11:])
            av_list = self.get_anes_var(a_id)
            # all oa variables a_id has been allocated in -necessarily sr is not None
            for i in range(len(av_list) - 1):
                for j in range(i + 1, len(av_list)):
                    if av_list[i].room != av_list[j].room:
                        overlapping = Prices.check_overlapping(av_list[i], av_list[j])
                        if overlapping:
                            av = av_list[i]
                            av.value.assigned['OA'].remove(av.get_constraint_dro_key())
                            av.value = None
                            av.value_in_update = av_list[i].value
                            av.need_stable = False
                            if self.num_manned_surgeries(av.room) == 0:
                                rm_v = self.v_dict[self.schedule_date][1][self.get_ward_from_room_allocation(av.room)][
                                    av.room][0]
                                rm_v.prior_value = None
                                if rm_v.value is not None:
                                    rm_v.value.assigned['RM'].remove(rm_v.get_constraint_dr_key())
                                rm_v.value = None
                            break

    def get_anes_var(self, a_id):
        """
        :param a_id: int anesthetist id
        :return: list of var in whom the anes is allocated
        """
        av_list = []
        for w in self.v_dict[self.schedule_date][1]:
            for r in self.v_dict[self.schedule_date][1][w]:
                for v in self.v_dict[self.schedule_date][1][w][r][1]:
                    if v.value is not None:
                        if v.value.id == a_id:
                            av_list.append(v)
        return av_list

    def calc_score_updated_schedule(self, full_solution=None, stable_schedule_flag=True):
        """
        calculates the score of schedule after being updated by all wards new surgery requests - cancels
        anesthetists overlapping surgeries - caused by change of times of surgeries by wards - because change of surgery
        requests allocated.
        :param full_solution: boolean - checks if needed if a full solution is kept- if nurses allocation
        were cancelled because of overlapping than full solution is false
        :param stable_schedule_flag: boolean - if to take into account stable schedule costs
        :return: full solution
        """
        updated_value = self.calc_utility(with_cost_update=True, next=True, stable_schedule_flag=stable_schedule_flag)
        anes_with_overlap = self.get_overlapping_d_keys()
        if anes_with_overlap:
            self.cancel_overlapping_anesthetists(anes_with_overlap)
            full_solution = False
            updated_value = self.calc_utility(with_cost_update=True, next=True,
                                              stable_schedule_flag=stable_schedule_flag)
        self.score = updated_value
        return full_solution
