from Allocating_Agent import Allocating_Agent
from Constraint import Constraint
from E_in_Surgery import SurgeryVariable_Equipment
from datetime import datetime, timedelta
from copy import deepcopy
from decimal import Decimal
import pandas as pd
import Prices
import random
import Static
import Chief_Agent
import math
import Nurse_Prices


cost_factor = 1_000
inf_price = 1_000_000


class Equipment_Agent(Allocating_Agent):

    def __init__(self, day, hospital, general_post_office):
        self.equipments = hospital.equipment  # set
        self.factor_weights = {'schedule_ratio': 0.7, 'discrimination': 0.3}
        self.min_period = 30
        super(Equipment_Agent, self).__init__(day, hospital, general_post_office, 'e')
        self.day = datetime.strptime(self.schedule_date, '%Y-%m-%d').date()
        self.tracking_table = self.init_tracking_table(hospital)
        self.update_tracking_table_by_schedule()
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
        :return: {schedule_date : {ward: {room: [(v_type0, v_type1, ...v_typen), (v_type0, v_type1..v_typen)], room...}}}
        """
        variables = {self.schedule_date: {}}
        e_constraints = self._init_constraints()
        for w in self.room_allocations:
            variables[self.schedule_date][w] = {}
            for r in self.room_allocations[w]:
                variables[self.schedule_date][w][r] = []
                max_slots_rd, initial_surgery_duration = w.max_slots_room_day(self.schedule_date, r)
                # initial_surgery_duration = w.d_duration // max_slots_rd
                start_time = w.start_d_hour
                for i in range(max_slots_rd):
                    end_time = Prices.calc_end_time(start_time, initial_surgery_duration)
                    ot = []
                    for e in self.equipments:
                        ot.append(SurgeryVariable_Equipment(room=r, day=self.schedule_date, order=i,
                                                            constraints=e_constraints, equipment=e,
                                                            start_time=start_time, end_time=end_time))
                    # The variables are all initiated with an initial value of False i.e. no equipment in no surgery
                    start_time = end_time
                    variables[self.schedule_date][w][r].append(tuple(ot))
        return variables

    def _init_constraints(self):
        """
        initializes the different constraint objects (each one holding an empty dictionary of prices) and gathers them
        into a dictionary
        :return: ictionary of constraints of variable for modularity the main key is d_r_o / d_r / d
        (date, room , order) then each key holds a dictionary of all the constraints that suits this key - the key is
        the name of the constraint and the value is the constraint object
        """
        # has a single hard constraint
        max_units_constraints = Constraint({})  # global key - d
        stable_schedule_constraint = Constraint({})

        e_constraints = {'dro': {'stable_schedule': stable_schedule_constraint},
                         'd': {'max_units': max_units_constraints}}
        return e_constraints

    def calc_utility(self, with_cost_update, next=False, stable_schedule_flag=True):
        init_day_cost = 0
        w_r_u = 0
        if with_cost_update:
            init_day_cost += self.set_d_prices(e_t=None)
        scheduled_wards = 0
        num_rooms = 0
        for w in self.v_dict[self.schedule_date]:
            total_w_num_surgeries = 0
            scheduled_w_num_surgeries = 0
            for r in self.v_dict[self.schedule_date][w]:
                num_rooms += 1
                for t in self.v_dict[self.schedule_date][w][r]:
                    # tuple which holds all the equipment variables for a certain surgery order
                    if with_cost_update and stable_schedule_flag:
                        for v in t:
                            init_day_cost += Nurse_Prices.set_stable_schedule_price(v, self.with_sr[w])
                    if self.with_sr[w]:
                        if t[0].surgery_request is not None:
                            total_w_num_surgeries += 1
                            if t[0].surgery_request.equipments:
                                e_id = t[0].surgery_request.equipments[0]  # enough to check a single equipment variable
                                # or all the equipments for surgery are allocated for surgery or none - but not partial
                                e_v = [ev for ev in t if ev.equipment.id == e_id][0]
                                '''if with_cost_update and stable_schedule_flag:
                                    for v in t:
                                        init_day_cost += Nurse_Prices.set_stable_schedule_price(v, self.with_sr[w])'''
                                if e_v.value is not False:
                                    scheduled_w_num_surgeries += 1
                            else:
                                scheduled_w_num_surgeries += 1
                    else:
                        total_w_num_surgeries += len(t)
                        for e_v in t:
                            if e_v.value:  # True - (domain is true or false)
                                scheduled_w_num_surgeries += 1
                                # break
            if total_w_num_surgeries > 0:
                w_r_u += self.ward_strategy_grades[w.w_id] * (scheduled_w_num_surgeries / total_w_num_surgeries)
            if scheduled_w_num_surgeries > 0:
                scheduled_wards += 1
        if scheduled_wards == 0:
            init_day_utility = 0
        else:
            init_day_utility = (cost_factor * num_rooms * 2) * (
                    self.factor_weights['schedule_ratio'] * w_r_u +
                    self.factor_weights['discrimination'] * (scheduled_wards / len(self.room_allocations)))
        return init_day_utility - init_day_cost

    def init_tracking_table(self, hospital):
        """
        creates a table (pd data frame) that holds how many units of each equipment is used in every time period of
        30 min of the surgery day. The table will be initiated with 0 in all cells.
        :param hospital: hospital object
        :return: pd data frame (columns - 08:00-08:30, 08:30-09:00 .... 14:30-15:00, rows - equipments id)
        """
        max_day_duration = hospital.max_d_duration()
        first_start_d_hour = hospital.get_earliest_start_hour()
        rows = [e.id for e in self.equipments]
        columns = [datetime.combine(self.day, first_start_d_hour) + timedelta(minutes=p) for p in
                   range(0, max_day_duration, self.min_period)]
        columns = [c.time() for c in columns]
        df = pd.DataFrame(columns=columns, index=rows)
        for col in df.columns:
            df[col].values[:] = 0
        return df

    def set_d_prices(self, e_t):
        """
        calculates and updates all the d date constraints with cost
        :param e_t: equipment variable tuple - of equipment variables of a certain surgery (all or partial). partial in
        the case where not with_sr so their is no need for all the equipments in sr
        :return: d_cost
        """
        d_cost = 0
        if e_t is None:  # calculating whole problem cost
            equipment_set = set([e.id for e in self.equipments])
            columns = list(self.tracking_table.columns)
            for w in self.v_dict[self.schedule_date]:
                for r in self.v_dict[self.schedule_date][w]:
                    for t in self.v_dict[self.schedule_date][w][r]:
                        e_t = t
                        break
                    break
                break
        else:
            columns = self.tracking_table_columns(e_t[0])
            r = e_t[0].room
            w = self.get_ward_from_room_allocation(r)
            if self.with_sr[w]:
                if e_t[0].surgery_request is not None:
                    if e_t[0].surgery_request.equipments:
                        equipment_set = set(e_t[0].surgery_request.equipments)  # set e_id
                    else:
                        return d_cost
                else:
                    return d_cost
            else:
                equipment_set = set(e_v.equipment.id for e_v in e_t)
        for e_id in equipment_set:
            max_units = self.get_num_units(e_id)
            # self.tracking_table.iloc[e_id] > max_units
            reduced_tracking_table = self.tracking_table.loc[[e_id], columns]
            for col in columns:
                if reduced_tracking_table.at[e_id, col] > max_units:
                    e_t[0].constraints['d']['max_units'].prices[
                        e_t[0].get_constraint_d_key(str(e_id) + '_' + str(col))] = inf_price
                    d_cost += inf_price
                else:
                    e_t[0].constraints['d']['max_units'].prices[
                        e_t[0].get_constraint_d_key(str(e_id) + '_' + str(col))] = 0
        return d_cost

    def get_num_units(self, e_id):
        """
        returns the num of units of the equipment with the id received
        :param e_id: int equipment id
        :return: int num of units
        """
        for e in self.equipments:
            if e_id == e.id:
                return e.num_units

    def simulated_annealing_by_day(self, init_sol_param=None, genetic=False, stable_schedule_flag=True, random_selection=True):
        """
            performs SA on a single day
            could receive d_dict of a specific ward and d - but just to be able to play with it for now...
            :param genetic: boolean - True if sa is used for init population in genetic algorithm , if so list of all the schedules
            is returned
            :param init_sol_param: function that determines the parameter which according to it the initial state will be
            generated
            :param stable_schedule_flag: boolean - if to take into account stable schedule costs
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
                    # Static.simulated_graphs(plot_list, 'Equipment')
                    break
            else:
                var_t, delta_E, ward = self._select_successor(stable_schedule_flag=stable_schedule_flag)
                if delta_E > 0:
                    self.increment_counter()
                    num_changes += 1
                    current_value = self.calc_value(var_t)
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
                        current_value = self.calc_value(var_t)
                        if current_value > best_value:
                            best_value = current_value
                            best_schedule = deepcopy(self.v_dict)
                        plot_list.append([t, current_value])
                        if genetic:
                            g.append((deepcopy(self.v_dict), current_value))
                        num_changes += 1
                    else:  # don't change
                        self.return_to_prior_value(var_t, ward, stable_schedule_flag=stable_schedule_flag)
                    self.increment_counter()

    def _select_successor(self, stable_schedule_flag=True):
        """
        selects a random surgery and changes its value(i.e. if with sr than if equipment wasn't allocated will 
        allocate it and if it was will cancel it, if without sr selects a random equipment to be added or removed
        from surgery). Calculates the difference of the total solution price due to change in solution. 
        The difference is calculated by the subtraction of prior price from next price
        prior price - utility - cost of the specific variable that changed
        next price - utility - cost after the change
        :param stable_schedule_flag: boolean - if to take into account stable schedule costs
        :return: chosen tuple of variables which changed and the difference in the total price of the solution, 
        
        """
        d_dict = self.v_dict[self.schedule_date]
        ward = random.choice(list(d_dict))
        room = random.choice(list(d_dict[ward]))
        t = random.choice(list(d_dict[ward][room]))
        if self.with_sr[ward]:  # no partial allocation of equipment or all the equipment or None
            if (t[0].surgery_request is not None) and t[0].surgery_request.equipments:
                relevant_t = tuple([e_v for e_v in t if e_v.equipment.id in e_v.surgery_request.equipments])
                # relevant_t - tuple of only the equipment variables needed for surgery
                delta = self.update_tuple_value(relevant_t, ward, stable_schedule_flag=stable_schedule_flag)
            else:  # or no surgery in this slot or no need of equipment
                return t, 0, ward
        else:  # selects a random equipment to be added or removed doesn't really matter
            chosen_v = random.choice(list(t))
            relevant_t = tuple([chosen_v])
            delta = self.update_tuple_value(relevant_t, ward, stable_schedule_flag=stable_schedule_flag)
        return relevant_t, delta, ward

    def single_variable_change(self, random_selection=True, stable_schedule_flag=True):
        """
        changes the equipment of a single surgery - the first one which current equipment does not match the
        wards needs and deltaE > 0.
        """
        for w in self.v_dict[self.schedule_date]:
            for room_num in self.v_dict[self.schedule_date][w]:
                for tu in self.v_dict[self.schedule_date][w][room_num]:
                    if (tu[0].surgery_request is not None) and tu[0].surgery_request.equipments:
                        relevant_t = tuple([e_v for e_v in tu if e_v.equipment.id in e_v.surgery_request.equipments])
                        for v in relevant_t:
                            if v.need_stable and (v.value != v.value_in_update):
                                deltaE = self.update_tuple_value(relevant_t, w,
                                                                 stable_schedule_flag=stable_schedule_flag)
                            elif (not v.need_stable) and (v.value == v.value_in_update):
                                deltaE = self.update_tuple_value(relevant_t, w,
                                                                 stable_schedule_flag=stable_schedule_flag)
                            else:
                                continue
                            if deltaE > 0:
                                self.score = self.calc_utility(with_cost_update=True, next=True,
                                                               stable_schedule_flag=stable_schedule_flag)
                                self.increment_counter()
                                return relevant_t, w
                            elif deltaE != 0:
                                self.return_to_prior_value(relevant_t, w, stable_schedule_flag=stable_schedule_flag)
                            self.increment_counter()
        return False, False

    def single_variable_change_explore(self, random_selection=True, stable_schedule_flag=True):
        """
        matches the DSA without stable schedule chooses a random variable to change more exploration
        """
        for i in range(200):
            relevant_t, deltaE, ward = self._select_successor(stable_schedule_flag=stable_schedule_flag)
            if deltaE > 0:
                self.score = self.calc_utility(with_cost_update=True, next=True,
                                               stable_schedule_flag=stable_schedule_flag)
                self.increment_counter()
                return relevant_t, ward
            elif deltaE != 0:
                self.return_to_prior_value(relevant_t, ward, stable_schedule_flag=stable_schedule_flag)
            self.increment_counter()
        return False, False

    def dsa_sc_iteration(self, mail, change_func, random_selection=True, stable_schedule_flag=True, no_good_flag=True):
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
        relevant_t, ward = getattr(self, change_func)(stable_schedule_flag=stable_schedule_flag)
        if relevant_t:
            if self.score < curr_score:  # alternative schedule is worst then curr schedule
                self.return_to_prior_value(relevant_t, ward, stable_schedule_flag=stable_schedule_flag)
                self.score = curr_score
            else:  # alternative schedule is equal valued or better than curr schedule
                change_probability = random.random()
                if change_probability > 0.7:  # will keep alternative schedule only for a chance of 70%
                    self.return_to_prior_value(relevant_t, ward, stable_schedule_flag=stable_schedule_flag)
                    self.score = curr_score
        self.send_mail()
        # we return the cost that resulted from the scheduled of the last iteration after receiving the allocating
        # agents schedule i.e. the real price
        # return {'schedule': curr_schedule, 'score': curr_score + stable_schedule_price}
        return {'score': curr_score + stable_schedule_price}

    def update_tuple_value(self, var_t, ward, stable_schedule_flag=True):
        """
        changes the value of all the variables in var_t - i.e if true so change to false and vi-versa. Calculates the
        difference in the sol_value - utility - cost
        :param ward: ward object - ward of variable surgeries in tuple
        :param var_t: tuple of variables which value we want to change (if with_sr then all the variables of
        the equipments needed in surgery if without sr than a random equipment)
        :param stable_schedule_flag: boolean - if to take into account stable schedule costs
        :return: delta e - the difference of the total price of solution
        """
        for e_v in var_t:
            e_v.prior_value = e_v.value
        prior_price = self.calc_price_by_variable(var_t, ward, stable_schedule_flag=stable_schedule_flag)
        for e_v in var_t:
            e_v.value = not e_v.value
        self.update_tracking_table_by_surgery(var_t)
        next_price = self.calc_price_by_variable(var_t, ward, next=True, stable_schedule_flag=stable_schedule_flag)
        return next_price - prior_price

    def count_schedule_changes(self, schedule):
        num_changes = 0
        for w, b_w in zip(self.v_dict[self.schedule_date], schedule[self.schedule_date]):
            for r, b_r in zip(self.v_dict[self.schedule_date][w], schedule[self.schedule_date][b_w]):
                for t, b_t in zip(self.v_dict[self.schedule_date][w][r], schedule[self.schedule_date][b_w][b_r]):
                    for v, b_v in zip(t, b_t):
                        if v.value != b_v.value:
                            num_changes += 1
        return num_changes

    def calc_price_by_variable(self, var_t, ward, with_utility=True, next=False, stable_schedule_flag=True):
        """
         calculates the difference in the schedule value depending only on the group of values of this variables in the 
         tuple. Takes in the calculation in to account only what is affected by these variables
        :param ward: ward object - ward of variable surgeries in tuple
        :param var_t: tuple of equipment variables (if with_sr then all the variables of the equipments needed in
        surgery if without sr than a random equipment)
        :param with_utility: boolean - if utility calculation is needed
        :param stable_schedule_flag: boolean - if to take into account stable schedule costs
        :return: new utility - new cost
        """
        cost = self.set_d_prices(var_t)
        if stable_schedule_flag:
            for v in var_t:
                cost += Nurse_Prices.set_stable_schedule_price(v, self.with_sr[ward])
        if with_utility and next:
            utility = self.calc_utility_by_tuple(var_t, ward)
            return utility - cost
        else:
            return -cost

    def calc_utility_by_tuple(self, var_t, ward):
        """
        calculates by a heurstic the utility difference caused by the change of the variables value
        to the current variable value
        :param var_t: tuple of equipment variables  (if with_sr then all the variables of the equipments needed in
        surgery if without sr than a random equipment)
        :return:the utility difference
        """
        utility = 0
        if var_t[0].value:  # if value == True - equipment in surgery
            # ratio of ward surgeries enlarged - when not with sr - we will address any additional equipment
            # as enlargement of the ratio
            utility += self.factor_weights['schedule_ratio'] * self.ward_strategy_grades[ward.w_id]
            if not self.ward_is_scheduled(ward, var_t[0]):
                # this variable is the single one scheduled for the ward
                utility += self.factor_weights['discrimination']
        else:
            # equipment was cancelled from surgery
            utility -= self.factor_weights['schedule_ratio'] * self.ward_strategy_grades[ward.w_id]
            if not self.ward_is_scheduled(ward):
                # the ward was scheduled to a single surgery and now it is not
                utility -= self.factor_weights['discrimination']
        return utility * cost_factor * 2

    def ward_is_scheduled(self, ward, e_v=None):
        """
        check if a certain ward has surgeries allocated with their needed equipments to the different surgeries
        :param e_v: equipment variable object - if not None will check if any other surgeries of ward received
        equipment
        :param ward: ward object - the ward we want to check
        :return:
        """
        for r in self.v_dict[self.schedule_date][ward]:
            for t in self.v_dict[self.schedule_date][ward][r]:
                for ev in t:
                    if ev.value:
                        if e_v is not None:
                            if ev != e_v:
                                return True
                        else:
                            return True
        return False

    def calc_value(self, chosen_v=None):
        """
        calculates the total cost of the current solution node
        :param chosen_v: tuple of equipment variables (if with_sr then all the variables of
        the equipments needed in surgery if without sr than a random equipment)
        :return: float total value - utility - cost
        """
        cost = 0
        if not chosen_v:
            w = list(self.room_allocations.keys())[0]
            r = next(iter(self.room_allocations[w]))
            chosen_v = self.v_dict[self.schedule_date][w][r][0]
        constraints = chosen_v[0].constraints
        for con_key in constraints:
            for cons in constraints[con_key]:
                cost += sum(constraints[con_key][cons].prices.values())
        utility = self.calc_utility(with_cost_update=False)
        return utility - cost

    def return_to_prior_value(self, var_t, ward, stable_schedule_flag=True):
        """
        returns the solution to the prior solution changes the values of the concerned variables back
        :param var_t: tuple of equipment variables  (if with_sr then all the variables of the equipments needed in
        surgery if without sr than a random equipment)
        :param stable_schedule_flag: boolean - if to take into account stable schedule costs
        :param ward: ward object
        :return:
        """
        for v in var_t:
            prior_update = v.value
            v.value = v.prior_value
            v.prior_value = prior_update
        self.update_tracking_table_by_surgery(var_t)
        self.calc_price_by_variable\
            (var_t, ward, with_utility=False, next=True, stable_schedule_flag=stable_schedule_flag)

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
                        t[i].value = b_t[i].value
                        t[i].prior_value = b_t[i].prior_value
                        if self.with_sr[w]:
                            t[i].start_time = b_t[i].start_time
                            t[i].end_time = b_t[i].end_time
                            if b_t[i].surgery_request is not None:
                                t[i].surgery_request = w.find_surgery_request(b_t[i].surgery_request.request_num)
                            else:
                                t[i].surgery_request = None
        self.update_tracking_table_by_schedule()
        self.score = self.calc_utility(with_cost_update=True, next=True, stable_schedule_flag=stable_schedule_flag)

    def update_tracking_table_by_surgery(self, var_t):
        """
        updates the tracking table after a change was done - if the equipment was added to surgery will augment the num
        of units in one - in the relevant columns(depending on the timeschedule of the surgery and the relevant rows
        depending on the equipments updated.
        :param var_t: tupple of equipment variables concerning a certain surgery
        """
        columns = self.tracking_table_columns(var_t[0])
        for v in var_t:
            e_id = v.equipment.id
            for c in columns:
                if v.value:
                    self.tracking_table.at[e_id, c] += 1
                else:
                    self.tracking_table.at[e_id, c] -= 1

    def update_tracking_table_by_schedule(self):
        self.nullify_tracking_table()
        debug_time = datetime(year=2021,month=12,day=3,hour=15,minute=0).time()
        for w in self.v_dict[self.schedule_date]:
            for r in self.v_dict[self.schedule_date][w]:
                for t in self.v_dict[self.schedule_date][w][r]:
                    columns = self.tracking_table_columns(t[0])
                    for v in t:
                        if v.value:
                            e_id = v.equipment.id
                            for c in columns:
                                if c == debug_time:
                                    print('stop')
                                self.tracking_table.at[e_id, c] += 1

    def tracking_table_columns(self, v):
        """
        derives the necessary columns of tracking table depending on the duration of the surgery - these will be
        the columns we would like to update.
        :param v: equipment variable
        :return: list of time objects each one representing a column in tracking table
        """
        start_time = v.start_time
        end_time = v.end_time
        duration = (datetime.combine(self.day, end_time) - datetime.combine(self.day, start_time)).total_seconds() / 60
        columns = [datetime.combine(self.day, start_time) + timedelta(minutes=p) for p in
                   range(0, int(duration), self.min_period)]
        columns = [c.time() for c in columns]
        return columns

    def nullify_tracking_table(self):
        for col in self.tracking_table.columns:
            self.tracking_table[col].values[:] = 0

    def update_schedule_by_ward(self, schedule, ward, ward_copy, stable_schedule_flag=True):
        """
        updates v_dict by schedule given from ward - the update includes adding the surgery request for each variable,
        updating the surgery times, adapting the equipments to match the equipments needed
        in surgery_request (cancel equipments not needed if given to surgery, or all equipments if no surgery request
        in variable), updating the field with_sr to True so calculations will be done matchingly. The function also
        updates the value of the schedule i.e. updates all constraint dictionaries with new prices
        :param ward: ward object of ward whom's schedule is being updated
        :param schedule: dictionary of wards schedule deep copy of original schedule
        {ward_object: {date : room: [tuples(sr_v, s_v)]}}
        :param stable_schedule_flag: boolean - if to take into account stable schedule costs
        :param ward_copy: copy of ward object as it is in schedule
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
                values = set()
                for v in t:
                    v.start_time = start_time
                    v.end_time = end_time
                    v.surgery_request = sr
                    '''# todo verify that does not do balagan
                    v.prior_value = None'''
                    if sr is None:
                        # v.prior_value = False
                        v.value = False
                    else:
                        if v.equipment.id not in sr.equipments:
                            v.value = False
                        else:
                            values.add(v.value)
                    v.need_stable = True
                    v.value_in_update = v.value
                if len(values) > 1:  # the initial solution produced a solution where the surgery received only partial
                    # allocation of the equipment needed for surgery
                    for v in t:
                        if v.equipment.id in sr.equipments:
                            # v.value = True
                            v.value = False
                            if not v.value_in_update:
                                v.need_stable = False
                            full_solution = False
                elif len(values) == 1:  # or all equipments in surgery or none are in surgery
                    if False in values:  # equipments were not allocated to surgery
                        for v in t:
                            if v.equipment.id in sr.equipments:
                                v.need_stable = False
                                full_solution = False

        self.with_sr[ward] = True
        '''self.update_tracking_table_by_schedule()
        updated_value = self.calc_utility(with_cost_update=True, next=True, stable_schedule_flag=stable_schedule_flag)
        max_units_exceeds = self.get_max_units_ex_d_keys()
        if max_units_exceeds:
            self.cancel_exceeding_equipment(max_units_exceeds)
            full_solution = False
            self.update_tracking_table_by_schedule()
            updated_value = self.calc_utility(with_cost_update=True, next=True, stable_schedule_flag=stable_schedule_flag)
        self.score = updated_value'''
        # bv, t, nc, pl, bs = self.simulated_annealing_by_day()
        # Static.simulated_graphs(pl, 'Equipment_withSR_' + str(ward.w_id))
        return full_solution

    def get_stable_schedule_costs(self):
        for w in self.v_dict[self.schedule_date]:
            for r in self.v_dict[self.schedule_date][w]:
                t_list = self.v_dict[self.schedule_date][w][r]
                for v in t_list:
                    stable_schedule_prices = sum(v[0].constraints['dro']['stable_schedule'].prices.values())
                    return stable_schedule_prices

    def get_max_units_ex_d_keys(self):
        """
        :return: list of d_keys of exceeding max units equipments
        """
        for w in self.v_dict[self.schedule_date]:
            for r in self.v_dict[self.schedule_date][w]:
                for t in self.v_dict[self.schedule_date][w][r]:
                    max_units_price = sum(t[0].constraints['d']['max_units'].prices.values())
                    if max_units_price > 0:
                        return [e for e in t[0].constraints['d']['max_units'].prices if
                                t[0].constraints['d']['max_units'].prices[e] > 0]
                    else:
                        return []

    def cancel_exceeding_equipment(self, max_units_d_keys):
        """
        cancels equipment allocation if exceed max units after ward update - the exceeding occured because of change in
        ward schedule- we don't want to take these allocations into account
        cancels the number which exceeds the max units and keeps allocation for max units.
        param overlap_d_keys: list of d_keys 'YYYY-MM-DD-NID' of nurses which have an overlap
        """
        for key in max_units_d_keys:
            e_id = int(key[11])
            e_time = key[13:]
            max_units = [e.num_units for e in self.equipments if e.id == e_id][0]
            et_list = self.get_equipment_var(e_id, e_time)
            # list of tuples of eq_var in which e_id has been allocated in in given time
            random.shuffle(et_list)
            et_list = et_list[:len(et_list)-max_units]  # keeps allocation for max_units of variables and cancels
            # allocation of the rest
            for t in et_list:
                for ev in t:
                    if ev.equipment.id in ev.surgery_request.equipments:
                        ev.value = False
                        if not ev.value_in_update:
                            ev.need_stable = False

    def get_equipment_var(self, e_id, e_time):
        """"
        :param e_id: int equipment id
        :param e_time: string of time
        :return: list of variables of equipment of e_id that occurs in time e_time and have been allocated
        """
        et_list = []
        e_time = datetime.strptime(e_time, '%H:%M:%S').time()
        for w in self.v_dict[self.schedule_date]:
            for r in self.v_dict[self.schedule_date][w]:
                for tu in self.v_dict[self.schedule_date][w][r]:
                    break_flag = False
                    for v in tu:
                        if v.equipment.id == e_id:
                            if v.start_time <= e_time < v.end_time and v.value:
                                et_list.append(tu)
                            break
                        if e_time < v.end_time:
                            break_flag = True
                    if break_flag:
                        break
        return et_list

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
        self.update_tracking_table_by_schedule()
        updated_value = self.calc_utility(with_cost_update=True, next=True, stable_schedule_flag=stable_schedule_flag)
        max_units_exceeds = self.get_max_units_ex_d_keys()
        if max_units_exceeds:
            self.cancel_exceeding_equipment(max_units_exceeds)
            full_solution = False
            self.update_tracking_table_by_schedule()
            updated_value = self.calc_utility(with_cost_update=True, next=True,
                                              stable_schedule_flag=stable_schedule_flag)
        self.score = updated_value
        return full_solution

