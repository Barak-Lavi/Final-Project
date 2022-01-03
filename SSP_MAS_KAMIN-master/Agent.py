from datetime import datetime
from copy import deepcopy

class Participating_Agent(object):
    def __init__(self, day, general_post_office, a_id):
        self.schedule_date = day
        # self.schedule_date = datetime.strptime(day, '%Y-%m-%d').date()
        self.v_dict = self.init_variables()  # schedule
        self.score = 0
        self.gpo = general_post_office
        self.n_changes = 0
        self.a_id = a_id
        self.counter = 0
        self.schedules_by_counter = {}  # dictionary of schedules - key counter ; value- schedule for this counter
        self.utility_by_counter = {}
        self.steps_for_counter = 100
        self.next_schedule_by_counter = 0

    def init_variables(self):
        pass

    def _init_constraints(self):
        pass

    def simulated_annealing_by_day(self, init_sol_param=None, genetic=False, stable_schedule_flag=True,
                                   random_selection=True):
        pass

    def _init_solution_day(self, parameter):
        pass

    def update_counter(self, neighbour_counter, chief_agent=False):
        if self.counter < neighbour_counter:
            self.counter = neighbour_counter
        '''while self.counter >= self.next_schedule_by_counter:
            if chief_agent:
                self.schedules_by_counter[self.next_schedule_by_counter] = \
                    {'schedule': deepcopy(self.v_dict), 'num_surgeries': deepcopy(self.num_surgeries)}
                self.utility_by_counter[self.next_schedule_by_counter] = self.calc_dsa_value(counter=True)
            else:
                self.schedules_by_counter[self.next_schedule_by_counter] = deepcopy(self.v_dict)
                self.utility_by_counter[self.next_schedule_by_counter] = self.score + self.get_stable_schedule_costs()
            self.next_schedule_by_counter += self.steps_for_counter'''

    def clear_counter(self, chief_agent=False):
        self.counter = 1
        # self.schedules_by_counter = {}
        # self.utility_by_counter = {}
        self.next_schedule_by_counter = 100
        '''if chief_agent:
            self.schedules_by_counter[self.counter] = \
                {'schedule': self.init_solution, 'num_surgeries': self.init_num_surgeries}
        else:
            self.schedules_by_counter[self.counter] = self.init_solution
        self.utility_by_counter[self.counter] = 0'''

    def increment_counter(self, chief_agent=False):
        self.counter += 1
        '''if self.counter == self.next_schedule_by_counter:
            if chief_agent:
                self.schedules_by_counter[self.next_schedule_by_counter] = \
                    {'schedule': deepcopy(self.v_dict), 'num_surgeries': self.num_surgeries.copy()}
                self.utility_by_counter[self.next_schedule_by_counter] = self.calc_dsa_value(counter=True)
            else:
                self.schedules_by_counter[self.next_schedule_by_counter] = deepcopy(self.v_dict)
                self.utility_by_counter[self.next_schedule_by_counter] = \
                    self.calc_value() + self.get_stable_schedule_costs()
            self.next_schedule_by_counter += self.steps_for_counter'''









