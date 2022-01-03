from Agent import Participating_Agent
from Message import Message
from copy import deepcopy
import random
import Static


class Allocating_Agent(Participating_Agent):
    def __init__(self, day, hospital, general_post_office, a_id):
        # init_ward_grades(day)  # {ward_id: grade} or maybe hospital.ward_grades
        # self.ward_strategy_grades = {0: 0.6, 1: 0.4}
        self.ward_strategy_grades = hospital.ward_strategy_grades
        self.room_allocations = self.init_room_allocations(hospital, day)  # {ward_id :set of room objects}
        self.with_sr = self.init_with_sr()
        self.wards = None
        super(Allocating_Agent, self).__init__(day, general_post_office, a_id)


    def init_value(self):
        self.with_sr = self.init_with_sr()
        self.update_schedule(self.init_solution)
        self.clear_counter()

    def init_room_allocations(self, hospital, day):
        """
        init the field - the keys of the room allocation dictionary are sorted by the importance of the ward-
        this way the init solution will be dependant on the importance of each ward
        :param day: String 'YYYY-MM-DD' the schedule date of the problem - the day we want to receive the room allocation
        :param hospital: hospital object
        :return: ra - room allocation - dictionary {ward_id : set of room objects}
        """
        # day = datetime.strptime(day, '%Y-%m-%d').date()
        ra = {}
        sorted_wards = {k: v for k, v in
                        sorted(self.ward_strategy_grades.items(), key=lambda item: item[1], reverse=True)}
        for w_id in sorted_wards:
            w = hospital.find_ward(w_id)
            if day in w.room_allocation:
                ra[w] = w.room_allocation[day]  # set of rooms
        return ra

    def init_with_sr(self):
        """
        the field shows which ward's surgery are already in the schedule i.e. surgery request in variable
        :return: dictionary {ward_id : boolean}
        """
        with_sr = {}
        for w in self.room_allocations:
            with_sr[w] = False
        return with_sr

    def calc_utility(self, with_cost_update, next=False, stable_schedule_flag=True):
        pass

    def get_room_from_room_allocation(self, room_num):
        """
        searches for a room object with the room_num in the room allocation
        :param room_num: int
        :return: room object
        """
        for w in self.room_allocations:
            for r in self.room_allocations[w]:
                if r.num == room_num:
                    return r

    def get_ward_from_room_allocation(self, r):
        """
        searches for the ward given room r in room allocation i.e. the ward operating in room r
        :param r: room object
        :return: ward object
        """
        for w in self.room_allocations:
            if r in self.room_allocations[w]:
                return w

    def update_schedule_by_ward(self, schedule, ward, ward_copy, stable_schedule_flag=True):
        pass

    def update_schedule(self, schedule, stable_schedule_flag=True):
        pass

    def get_stable_schedule_costs(self):
        pass

    def send_mail(self):
        for w in self.wards:
            # todo addition for not full allocation
            if w in self.v_dict[self.schedule_date]:
                self.gpo.append(Message(to_agent=w.w_id,
                                        content={'a_id': self.a_id, 'schedule': deepcopy(self.v_dict[self.schedule_date][w])
                                                 , 'counter': self.counter}))

    def get_ward_from_schedule(self, schedule,  w_id):
        """
        :param w_id: int ward id of the ward we want the schedule of
        :param schedule: a copy instance of the agent's schedule (in earlier iteration)
        :return: the schedule regarding the given ward of the given agent i.e. if anesthetist so aneshtetist schedule
        for given ward
        """
        for w in schedule[self.schedule_date]:
            if w.w_id == w_id:
                return schedule[self.schedule_date][w]

    def NG_iteration(self, mail, stable_schedule_flag=True, stop_fs_flag=True, random_selection=True, no_good_flag=True):
        fs = True
        for m in mail:
            w_fs = self.update_schedule_by_ward(m.content['schedule'], m.content['ward'], m.content['ward_copy'],
                                                stable_schedule_flag=stable_schedule_flag)
            fs = fs and w_fs
        u_fs = self.calc_score_updated_schedule(full_solution=fs, stable_schedule_flag=stable_schedule_flag)
        fs = fs and u_fs
        for m in mail:
            self.update_counter(m.content['counter'])
        if stop_fs_flag:
            if not fs:
                self.n_changes += 1
                self.simulated_annealing_by_day(stable_schedule_flag=stable_schedule_flag,
                                                random_selection=random_selection)
        else:
            self.n_changes += 1
            self.simulated_annealing_by_day(stable_schedule_flag=stable_schedule_flag, random_selection=random_selection)
        # curr_schedule = deepcopy(self.v_dict)
        curr_score = self.score
        stable_schedule_price = 0
        if stable_schedule_flag:
            stable_schedule_price = self.get_stable_schedule_costs()
        self.send_mail()
        # do not erase - the first one is if we want to track changes in schedule
        # return {'fs': fs, 'schedule': curr_schedule, 'score': curr_score + stable_schedule_price}
        return {'fs': fs, 'score': curr_score + stable_schedule_price}

    def NG_sc_iteration(self, mail, change_func, stop_fs_flag=True, random_selection=True, stable_schedule_flag=True,
                        no_good_flag=True):
        fs = True
        for m in mail:
            w_fs = self.update_schedule_by_ward(m.content['schedule'], m.content['ward'], m.content['ward_copy'],
                                                stable_schedule_flag=stable_schedule_flag)
            fs = fs and w_fs
        u_fs = self.calc_score_updated_schedule(full_solution=fs, stable_schedule_flag=stable_schedule_flag)
        fs = fs and u_fs
        for m in mail:
            self.update_counter(m.content['counter'])
        if stop_fs_flag:
            if not fs:
                self.n_changes += 1
                getattr(self, change_func)(random_selection, stable_schedule_flag)
        else:
            self.n_changes += 1
            getattr(self, change_func)(random_selection, stable_schedule_flag)
        # curr_schedule = deepcopy(self.v_dict)
        curr_score = self.score
        stable_schedule_price = self.get_stable_schedule_costs()
        self.send_mail()
        # do not erase - the first one is if we want to track changes in schedule
        # return {'fs': fs, 'schedule': curr_schedule, 'score': curr_score + stable_schedule_price}
        return {'fs': fs, 'score': curr_score + stable_schedule_price}

    def dsa_sa_iteration(self, mail, stable_schedule_flag, random_selection=True, no_good_flag=True):
        for m in mail:
            self.update_schedule_by_ward(m.content['schedule'], m.content['ward'], m.content['ward_copy'],
                                         stable_schedule_flag=stable_schedule_flag)
        self.calc_score_updated_schedule(stable_schedule_flag=stable_schedule_flag)
        for m in mail:
            self.update_counter(m.content['counter'])
        curr_schedule = deepcopy(self.v_dict)
        stable_schedule_price = 0
        if stable_schedule_flag:
            stable_schedule_price = self.get_stable_schedule_costs()  # we do not want to count these costs because they
        # were updated in the update schedule by ward and do not reflect the last iteration
        curr_score = self.score + stable_schedule_price
        self.simulated_annealing_by_day(stable_schedule_flag=stable_schedule_flag, random_selection=random_selection)
        if self.score < curr_score:  # alternative schedule is worst then curr schedule
            self.update_schedule(curr_schedule, stable_schedule_flag=stable_schedule_flag)
        else:  # alternative schedule is equal valued or better than curr schedule
            change_probability = random.random()
            if change_probability > 0.7:  # will keep alternative schedule only for a chance of 70%
                self.update_schedule(curr_schedule, stable_schedule_flag=stable_schedule_flag)
        num_schedule_changes = self.count_schedule_changes(curr_schedule)
        self.send_mail()
        # we return the cost that resulted from the scheduled of the last iteration after receiving the ward's schedule
        # i.e. the real price
        # return {'schedule': curr_schedule, 'score': curr_score}  # we return the cost that resulted from
        return {'score': curr_score, 'num_changes': num_schedule_changes}  # we return the cost that resulted from

    def count_schedule_changes(self, schedule):
        pass

    def calc_score_updated_schedule(self, full_solution=None, stable_schedule_flag=True):
        pass

