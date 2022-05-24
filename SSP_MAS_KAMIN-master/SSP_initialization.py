import random

from Hospital import Hospital
from Chief_Agent import Ward_Agent
from Ward import Ward
from Nurse_Agent import Nurse_Agent
from Anesthetist_Agent import Anes_Agent
from Equipment_Tracking_Agent import Equipment_Agent
from copy import deepcopy
from scipy.interpolate import interp1d
from scipy.stats import f_oneway
from scipy.stats import ttest_ind
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from datetime import datetime
import itertools
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.font_manager import FontProperties
import os
import pickle
import requests
import sys


class Problem(object):
    def __init__(self, num_wards, schedule_date):
        self.general_post_office = []
        self.schedule_date = schedule_date
        self.schedule_date_date = datetime.strptime(schedule_date, '%Y-%m-%d').date()
        self.agents = self.init_agents(num_wards)
        self.agents_mail_box = self.init_agents_mail_boxes()
        self.max_counter = 1000  # 50_000
        self.not_scheduled_surgery_requests = []

    def init_agents(self, num_wards):
        Soroka = Hospital(1, 'Soroka', DB=False)
        ward_agents = []
        for i in Soroka.ward_strategy_grades.keys():
            ward_agents.append(
                Ward_Agent(Soroka.find_ward(w_id=i), self.schedule_date, case=1,
                           general_post_office=self.general_post_office))

        ta = Equipment_Agent(self.schedule_date, Soroka, self.general_post_office)
        anesthetist_Agent = Anes_Agent(self.schedule_date, Soroka, self.general_post_office)
        nurse_Agent = Nurse_Agent(self.schedule_date, Soroka, self.general_post_office)
        return {'ward_agents': ward_agents, 'extra_agents': [anesthetist_Agent, ta, nurse_Agent]}

    def init_agents_mail_boxes(self):
        amb = {}  # agent mail box
        wards = []
        for wa in self.agents['ward_agents']:
            amb[wa.ward.w_id] = []
            wards.append(wa.ward)
        amb['a'] = []
        amb['n'] = []
        amb['e'] = []
        for ea in self.agents['extra_agents']:
            ea.wards = wards
        return amb

    def post_surgeries_DB(self, max_global_schedule):
        add_surgery_url = "https://www.api.orm-bgu-soroka.com/surgeries_preop_partials/add_surgery"
        update_url = "https://www.api.orm-bgu-soroka.com/surgery_requests/update"
        not_scheduled_rtg_url = "https://www.api.orm-bgu-soroka.com/not_scheduled_surgery_requests/add"
        scheduled_sr_id = []
        not_scheduled_rtg = []
        for w_id in max_global_schedule['ward_agents']:
            best_schedule = max_global_schedule['ward_agents'][w_id]
            for w in best_schedule:
                for r in best_schedule[w][self.schedule_date]:
                    for t in best_schedule[w][self.schedule_date][r]:
                        sr_v = t[0]
                        s_v = t[1]
                        if (sr_v.value is not None) and (False not in sr_v.with_surgery_team.values()):
                            params = {'request_id': sr_v.value.request_num, 'room_id': sr_v.room.num,
                                      'st_time': self.schedule_date + " " + str(sr_v.start_time),
                                      'end_time': self.schedule_date + " " + str(sr_v.end_time),
                                      'duration': sr_v.value.duration, 'senior_id': s_v.value.id,
                                      'anesthetist_id': sr_v.with_surgery_team['Anesthetist'].id,
                                      'circulating_id': sr_v.with_surgery_team['Nurse'][1].value.id,
                                      'scrubbing_id': sr_v.with_surgery_team['Nurse'][0].value.id,
                                      'prior_status': str(sr_v.value.status)}
                            r = requests.post(url=add_surgery_url, json=params)
                            r2 = requests.post(url=update_url,
                                               json={'request_id': sr_v.value.request_num, 'request_status': str(4)})
                            # sr_v.value.status = 4
                            scheduled_sr_id.append(sr_v.value.request_num)
                        elif (sr_v.value is not None) and (str(sr_v.value.schedule_date) == self.schedule_date):
                            r2 = requests.post(url=update_url,
                                               json={'request_id': sr_v.value.request_num, 'request_status': str(1.1),
                                                     'cancellations': sr_v.value.num_cancellations + 1,
                                                     'schedule_date': None})
                            # sr_v.value.status = 1.1
                not_scheduled_rtg = [sr for sr in w.RTG if (str(sr.schedule_date) == self.schedule_date) and
                                     (sr.request_num not in scheduled_sr_id)]
        ns_rtg_json = []
        for sr in not_scheduled_rtg:
            r2 = requests.post(url=update_url, json={'request_id': sr.request_num, 'request_status': str(1.1),
                                                     'cancellations': sr.num_cancellations + 1, 'schedule_date': None})
            ns_rtg_json.append({'request_id': sr.request_num, 'request_status': str(sr.status),
                                'cancellations': sr.num_cancellations, 'schedule_date': str(sr.schedule_date)})
        r3 = requests.post(url=not_scheduled_rtg_url, json=ns_rtg_json)

        rm_url = 'https://www.api.orm-bgu-soroka.com/room_managers/add_manager'
        fm_url = 'https://www.api.orm-bgu-soroka.com/floor_managers/add_manager'
        floor_manager = max_global_schedule['anes'][self.schedule_date][0].value.id
        requests.post(url=fm_url, json={'date': self.schedule_date, 'anesthetist_id': floor_manager})
        for w in max_global_schedule['anes'][self.schedule_date][1]:
            for r in max_global_schedule['anes'][self.schedule_date][1][w]:
                rt = max_global_schedule['anes'][self.schedule_date][1][w][r]
                if rt[0].value is not None:
                    room_manager = rt[0].value.id
                    room_id = rt[0].room.num
                    requests.post(url=rm_url, json={'date': self.schedule_date, 'anesthetist_id': room_manager,
                                                    'room_id': room_id})
        self.not_scheduled_surgery_requests.extend(not_scheduled_rtg)

    def NG(self, single_change, num_iter, stop_fs_flag, change_func=None, init_sol_value=None, random_selection=True,
           stable_schedule_flag=True, no_good_flag=True):
        max_global_cost = 0
        max_global_schedule = {'ward_agents': {}, 'anes': {}}
        global_schedules = {}
        global_costs = {}  # keys - max agent counter - value - global cost
        max_counter = 0
        if init_sol_value:
            global_costs[0] = init_sol_value
        for a in self.agents['ward_agents']:
            # todo addition for not full allocation
            if not a.v_dict[a.ward][self.schedule_date]:
                continue
            a.send_mail()
        i = 0
        for i in range(num_iter):
            if max_counter >= self.max_counter:
                break
            global_schedules[i] = {}
            fs = True  # full solution
            self.distribute_mail()
            for a in self.agents['extra_agents']:
                if single_change:
                    global_schedules[i][a.a_id] = a.NG_sc_iteration \
                        (self.agents_mail_box[a.a_id], change_func=change_func, stop_fs_flag=stop_fs_flag,
                         random_selection=random_selection, stable_schedule_flag=stable_schedule_flag)
                else:
                    global_schedules[i][a.a_id] = a.NG_iteration(self.agents_mail_box[a.a_id],
                                                                 stable_schedule_flag=stable_schedule_flag,
                                                                 stop_fs_flag=stop_fs_flag,
                                                                 random_selection=random_selection)
                a_fs = global_schedules[i][a.a_id]['fs']
                fs = fs and a_fs
            self.distribute_mail()
            max_counter = 0
            for a in list(itertools.chain.from_iterable(self.agents.values())):
                if a.counter > max_counter:
                    max_counter = a.counter
            for a in self.agents['ward_agents']:
                # todo addition for not full allocation
                if not a.v_dict[a.ward][self.schedule_date]:
                    continue
                if single_change:
                    global_schedules[i][a.a_id] = a.NG_sc_iteration \
                        (self.agents_mail_box[a.a_id], change_func=change_func, stop_fs_flag=stop_fs_flag,
                         random_selection=random_selection, stable_schedule_flag=stable_schedule_flag,
                         no_good_flag=no_good_flag)
                else:
                    global_schedules[i][a.a_id] = a.NG_iteration \
                        (self.agents_mail_box[a.a_id], stable_schedule_flag=change_func,
                         stop_fs_flag=stop_fs_flag, random_selection=random_selection, no_good_flag=no_good_flag)
                w_fs = global_schedules[i][a.a_id]['fs']
                fs = fs and w_fs
            '''if fs:
                break'''
            global_cost = self.sum_global_costs(global_schedules[i])
            if global_cost > max_global_cost:
                max_global_cost = global_cost
                for wa in self.agents['ward_agents']:
                    max_global_schedule['ward_agents'][wa.a_id] = deepcopy(wa.v_dict)
                max_global_schedule['anes'] = deepcopy(self.agents['extra_agents'][0].v_dict)
            global_costs[max_counter - 1] = global_cost
            # print('global cost in step ' + str(i) + ':' + str(global_cost) + 'max_counter:' + str(max_counter))
        ############################ RL application #################################

        # divid to two groups of sr. one of the schedualed and the other of those that could and didnt, good sr

        mean_stats_per_ward, max_min_stats_by_ward_schedual = self.schedule_analayse_RL()
        ############### reward function calculation for every ward ! #############
        reward_by_ward, global_norm_reward = self.reward_function_calc(mean_stats_per_ward,
                                                                       max_min_stats_by_ward_schedual)

        return global_costs, global_schedules, i, max_global_schedule, max_global_cost, reward_by_ward, global_norm_reward

    def DSA(self, single_change, num_iter, change_func=None, random_selection=True, stable_schedule_flag=True,
            no_good_flag=True):
        """
        :param change_func - index of function of schedule update to alternative value -
        or via simulated annealing or  via single variable change
        :param random_selection: boolean - if to select randomly from domain or by max DeltaE
        """
        max_global_cost = 0
        max_global_schedule = {'ward_agents': {}, 'anes': {}}
        global_schedules = {}
        global_costs = {}
        max_counter = 0
        num_changes = []  # list of dictionaries index of dict is the iteration

        for a in list(itertools.chain.from_iterable(self.agents.values())):
            # todo addition for not full allocation
            if isinstance(a, Ward_Agent) and (
                    not a.v_dict[a.ward][self.schedule_date]):  # did not receive room in allocation
                continue
            a.send_mail()
        for i in range(num_iter):
            num_iter_change = {}
            if max_counter >= self.max_counter:
                break
            global_schedules[i] = {}
            self.distribute_mail()
            max_counter = 0
            for a in list(itertools.chain.from_iterable(self.agents.values())):
                if a.counter > max_counter:
                    max_counter = a.counter
            for a in list(itertools.chain.from_iterable(self.agents.values())):
                # todo addition for not full allocation
                if isinstance(a, Ward_Agent) and (not a.v_dict[a.ward][self.schedule_date]):
                    continue
                if single_change:
                    global_schedules[i][a.a_id] = a.dsa_sc_iteration \
                        (mail=self.agents_mail_box[a.a_id], change_func=change_func, random_selection=random_selection,
                         stable_schedule_flag=stable_schedule_flag, no_good_flag=no_good_flag)
                else:
                    iteration_dict = a.dsa_sa_iteration \
                        (mail=self.agents_mail_box[a.a_id], stable_schedule_flag=stable_schedule_flag,
                         no_good_flag=no_good_flag, random_selection=random_selection)
                    global_schedules[i][a.a_id] = iteration_dict['score']
                    num_iter_change[a.a_id] = iteration_dict['num_changes']
            num_changes.append(num_iter_change)  # number of total changes done in all hospital schedule in iteration
            global_cost = self.sum_global_costs(global_schedules[i])
            if global_cost > max_global_cost:
                max_global_cost = global_cost
                for wa in self.agents['ward_agents']:
                    max_global_schedule['ward_agents'][wa.a_id] = deepcopy(wa.v_dict)
                max_global_schedule['anes'] = deepcopy(self.agents['extra_agents'][0].v_dict)
            # print('global cost in step ' + str(i) + ':' + str(global_cost) + 'max_counter:' + str(max_counter))
            global_costs[max_counter - 1] = global_cost
        # mean_num_changes = sum(num_changes) / len(num_changes)  # mean of changes in iteration
        ############################ RL application #################################

        # divid to two groups of sr. one of the schedualed and the other of those that could and didnt, good sr

        mean_stats_per_ward, max_min_stats_by_ward_schedual = self.schedule_analayse_RL()
        ############### reward function calculation for every ward ! #############
        reward_by_ward, global_norm_reward = self.reward_function_calc(mean_stats_per_ward,
                                                                       max_min_stats_by_ward_schedual)

        return global_costs, global_schedules, max_global_schedule, max_global_cost, reward_by_ward, global_norm_reward  # , num_changes

    def sum_global_costs(self, global_schedules):
        gc = 0
        for a in global_schedules:
            gc += global_schedules[a].get('score')
        return gc

    def distribute_mail(self):
        for a_id in self.agents_mail_box:
            self.agents_mail_box[a_id].clear()
        for m in self.general_post_office:
            self.agents_mail_box[m.to_agent].append(m)
        self.general_post_office.clear()

    def clear_problem(self):
        self.general_post_office.clear()
        for mb in self.agents_mail_box:
            self.agents_mail_box[mb].clear()
        for a in list(itertools.chain.from_iterable(self.agents.values())):
            a.init_value()

    def NCLO_costs(self, stable_schedule_flag=True):
        """
        non concurrent logical operation graph. calculates the global costs of the different counter values for the
        graph.
        :param stable_schedule_flag: boolean - if to take into account stable schedule costs
        :return:
        """
        # turns out that if i look at the schedules in counter X of all agents - it is not guaranteed that they will
        # all be with the same schedule  - and it is ok - because we want to see where each agent arrived in counter
        # X and it is possible that w1 int X steps arrived to the schedule received in iteration 2 and in the other hand
        # w0 --> that also a/n/e in X steps only arrived to the schedule received in iteration 1.
        # when i want to look at the global schedules then use the output by iter and not by counter - we still want
        # to see the global cost by counter and not by iter
        # to conclude if we want the value of the schedule in counter X - we do not want the global value of all the
        # schedules saved in counter X but the sum of each agent's value of its schedule in counter X.
        # this funciton results with the sum of the global value of the different schedules when unified to a global
        # schedule
        global_costs = {}  # dictionary {counter value: global cost}
        global_schedules = {}  # {counter value: {a_id : {'schedule' : , 'score' : , 'num surgeries' for ward agents}}}
        shortest_schedule_by_counter = None
        for a in list(itertools.chain.from_iterable(self.agents.values())):
            if shortest_schedule_by_counter is None:
                shortest_schedule_by_counter = a.schedules_by_counter
            else:
                if len(a.schedules_by_counter) < len(shortest_schedule_by_counter):
                    shortest_schedule_by_counter = a.schedules_by_counter
        for counter in shortest_schedule_by_counter:
            ea_flag = True
            global_costs[counter] = 0
            global_schedules[counter] = {}
            for a in self.agents['ward_agents']:
                a.update_schedule(
                    a.schedules_by_counter[counter]['schedule'], a.schedules_by_counter[counter]['num_surgeries'],
                    stable_schedule_flag=stable_schedule_flag)
                # Updates v_dict to be the saved schedule in schedules by counter field
                a_schedule = a.schedules_by_counter[counter]['schedule']
                a_ward = a.ward
                a_ward_copy = list(a_schedule.keys())[0]
                for ea in self.agents['extra_agents']:
                    a.classifier_update_schedule({'a_id': ea.a_id,
                                                  'schedule': ea.get_ward_from_schedule
                                                  (ea.schedules_by_counter[counter], a.a_id)}, True)
                    if ea_flag:  # for the first ward agent update extra agent to counter schedule
                        ea.update_schedule(ea.schedules_by_counter[counter], stable_schedule_flag=stable_schedule_flag)
                    ea.update_schedule_by_ward(a_schedule, a_ward, a_ward_copy,
                                               stable_schedule_flag=stable_schedule_flag)
                ea_flag = False
                a_score = a.calc_dsa_value()
                global_schedules[counter][a.a_id] = \
                    {'schedule': deepcopy(a.v_dict), 'score': a_score, 'num_surgeries': deepcopy(a.num_surgeries)}
                global_costs[counter] += a.calc_dsa_value()
            for ea in self.agents['extra_agents']:
                ssc = ea.get_stable_schedule_costs()
                global_schedules[counter][ea.a_id] = {'schedule': deepcopy(ea.v_dict), 'score': ea.score + ssc}
                global_costs[counter] += ea.score + ssc
        return global_costs, global_schedules

    def sum_NCLO_gc(self):
        global_costs_by_counter = {}
        shortest_cost_by_counter = None
        for a in list(itertools.chain.from_iterable(self.agents.values())):
            if shortest_cost_by_counter is None:
                shortest_cost_by_counter = a.utility_by_counter
            else:
                if len(a.utility_by_counter) < len(shortest_cost_by_counter):
                    shortest_cost_by_counter = a.utility_by_counter
        for counter in shortest_cost_by_counter:
            global_costs_by_counter[counter] = 0
            for a in list(itertools.chain.from_iterable(self.agents.values())):
                global_costs_by_counter[counter] += a.utility_by_counter[counter]
        return global_costs_by_counter

    def schedule_analayse_RL(self):
        """
        after DSA return its best scheduale, the function divide it to two groups from the surgeries requests.
        for each ward the function takes surgeries that was ready to go (rtg) and schedule to one group.
        and rtg surgeries that didnt schedule but was able to (without hard constraints)
        :return: max_min_stats_by_ward_schedual {} - min max features value from all available rtg surgeries without hard constraints
        mean_stats_per_ward {} - mean features value only from the schedule surgeries
        """

        mean_stats_per_ward = {}
        max_min_stats_by_ward_schedual = {}
        for wr in self.agents['ward_agents']:
            max_c, max_u, max_waiting_days = 0, 0, 0
            min_c, min_u, min_waiting_days = 10, 10, 366
            good_sr_rtg = []
            for rtg in wr.ward.RTG:
                if rtg not in wr.no_good_sr:
                    good_sr_rtg.append(rtg)
                    try:
                        if max_c < rtg.num_cancellations:
                            max_c = rtg.num_cancellations
                        if min_c > rtg.num_cancellations:
                            min_c = rtg.num_cancellations
                    except:
                        continue
                    try:
                        if max_u < rtg.urgency:
                            max_u = rtg.urgency
                        if min_u > rtg.urgency:
                            min_u = rtg.urgency
                    except:
                        continue
                    try:
                        waiting_days = rtg.calc_waiting_days(self.schedule_date)
                        if max_waiting_days < waiting_days:
                            max_waiting_days = waiting_days
                        if min_waiting_days > waiting_days:
                            min_waiting_days = waiting_days
                    except:
                        continue
            max_min_stats_by_ward_schedual[wr.ward] = {'max_c': max_c, 'min_c': min_c, 'max_u': max_u, 'min_u': min_u,
                                                       'max_waiting_days': max_waiting_days,
                                                       'min_waiting_days': min_waiting_days}

            ###
            schedule_sr_today = []
            mean_c, mean_u, mean_waiting_days = 0, 0, 0
            for sr in wr.v_dict[wr.ward].values():
                for room in sr.values():
                    for single_sched in room:
                        schedule_sr_today.append(single_sched[0].value)
                        try:
                            mean_c += single_sched[0].value.num_cancellations
                        except:
                            continue
                        try:
                            mean_u += single_sched[0].value.urgency
                        except:
                            continue
                        try:
                            mean_waiting_days += single_sched[0].value.calc_waiting_days(self.schedule_date)
                        except:
                            continue

            mean_c = mean_c / len(schedule_sr_today)
            mean_u = mean_u / len(schedule_sr_today)
            mean_waiting_days = mean_waiting_days / len(schedule_sr_today)
            mean_stats_per_ward[wr.ward] = {'mean_c': mean_c, 'mean_u': mean_u, 'mean_waiting_days': mean_waiting_days}

        return mean_stats_per_ward, max_min_stats_by_ward_schedual

    def reward_function_calc(self, mean_stats_per_ward, max_min_stats_by_ward_schedual):
        """
        Reward function calculation for schadule of a single ward.
        calc with Composite Desirability function- for each feature, the mean value of schedule surgeries
        minus the min value of all rtg surgeries divide by max minus min of all rtg surgeries.
        :return: reward_by_ward {} - reward value for each ward in the problem
        """
        reward_by_ward = {}
        reward = 0
        for ward in mean_stats_per_ward:
            wr_reward = (((mean_stats_per_ward[ward]['mean_c'] - max_min_stats_by_ward_schedual[ward]['min_c']) /
                          (max_min_stats_by_ward_schedual[ward]['max_c'] - max_min_stats_by_ward_schedual[ward][
                              'min_c'] + sys.float_info.epsilon)) *
                         ((mean_stats_per_ward[ward]['mean_u'] - max_min_stats_by_ward_schedual[ward]['min_c']) /
                          (max_min_stats_by_ward_schedual[ward]['max_u'] - max_min_stats_by_ward_schedual[ward][
                              'min_u'] + sys.float_info.epsilon)) *
                         ((mean_stats_per_ward[ward]['mean_waiting_days'] - max_min_stats_by_ward_schedual[ward][
                             'min_waiting_days']) /
                          (max_min_stats_by_ward_schedual[ward]['max_waiting_days'] -
                           max_min_stats_by_ward_schedual[ward][
                               'min_waiting_days'] + sys.float_info.epsilon))) ** (1 / 3)
            reward_by_ward[ward] = [wr_reward]
            reward += wr_reward
            ward.reward = wr_reward
        ############### norm rewad for global value #############
        global_norm_reward = reward / len(reward_by_ward.keys())

        return reward_by_ward, global_norm_reward


def NCLO_graph_DSA_SC_gc(DSA_SC_gc):
    DSA_SC_gc1 = standarize_gc(DSA_SC_gc.values())
    counter = len(DSA_SC_gc)
    plt.plot(range(0, counter * 100, 100), DSA_SC_gc1[:counter], zorder=2, c='g', label='dsa_sc')
    plt.legend(loc='lower right')
    plt.xlabel('non concurrent logical operation')
    plt.ylabel('utility')
    plt.show()


def NCLO_graph(DSA_SC_gc, DSA_SC_E_gc, DSA_gc, DSA_SS_gc, NG_gc, NG_ss_gc):
    DSA_SC_gc1 = standarize_gc(DSA_SC_gc.values())
    DSA_SC_E_gc1 = standarize_gc(DSA_SC_E_gc.values())
    DSA_gc1 = standarize_gc(DSA_gc.values())
    DSA_ss_gc1 = standarize_gc(DSA_SS_gc.values())
    NG_gc1 = standarize_gc(NG_gc.values())
    NG_ss_gc1 = standarize_gc(NG_ss_gc.values())
    counter = min(len(DSA_SC_gc), len(DSA_SC_E_gc), len(DSA_gc), len(DSA_SS_gc), len(NG_gc), len(NG_ss_gc))
    plt.plot(range(0, counter * 100, 100), DSA_gc1[:counter], zorder=2, c='b', label='dsa_sa')
    plt.plot(range(0, counter * 100, 100), NG_gc1[:counter], zorder=2, c='r', label='ssp_sa')
    plt.plot(range(0, counter * 100, 100), DSA_ss_gc1[:counter], zorder=2, c='m', label='dsa_sa_ss')
    plt.plot(range(0, counter * 100, 100), NG_ss_gc1[:counter], zorder=2, c='k', label='ssp_sa_ss')
    plt.plot(range(0, counter * 100, 100), DSA_SC_gc1[:counter], zorder=2, c='g', label='dsa_sc')
    plt.plot(range(0, counter * 100, 100), DSA_SC_E_gc1[:counter], zorder=2, c='c', label='dsa_sc_e')
    plt.legend(loc='lower right')
    plt.xlabel('non concurrent logical operation')
    plt.ylabel('utility')
    plt.show()


'''def NCLO_iter_graph(DSA_SC_gc, DSA_SC_E_gc, DSA_gc, DSA_SS_gc, NG_gc, NG_ss_gc):

    DSA_SC_gc1 = standarize_gc(DSA_SC_gc.values())
    DSA_SC_E_gc1 = standarize_gc(DSA_SC_E_gc.values())
    DSA_gc1 = standarize_gc(DSA_gc.values())
    DSA_ss_gc1 = standarize_gc(DSA_SS_gc.values())
    NG_gc1 = standarize_gc(NG_gc.values())
    NG_ss_gc1 = standarize_gc(NG_ss_gc.values())
    counter = min(list(DSA_SC_gc.keys())[-1], list(DSA_SC_E_gc.keys())[-1], list(DSA_gc.keys())[-1],
                  list(DSA_SS_gc.keys())[-1], list(NG_gc.keys())[-1], list(NG_ss_gc.keys())[-1])
    DSA_gc_counter = [c for c in DSA_gc.keys() if c <= counter]
    NG_gc_counter = [c for c in NG_gc.keys() if c <= counter]
    DSA_SS_gc_counter = [c for c in DSA_SS_gc.keys() if c <= counter]
    NG_ss_gc_counter = [c for c in NG_ss_gc.keys() if c <= counter]
    DSA_SC_gc_counter = [c for c in DSA_SC_gc.keys() if c <= counter]
    DSA_SC_E_gc_counter = [c for c in DSA_SC_E_gc.keys() if c <= counter]
    # counter = min(len(DSA_SC_gc), len(DSA_SC_E_gc), len(DSA_gc), len(DSA_SS_gc), len(NG_gc), len(NG_ss_gc))
    plt.plot(DSA_gc_counter, DSA_gc1[:len(DSA_gc_counter)], zorder=2, c='b', label='dsa_sa')
    plt.plot(NG_gc_counter, NG_gc1[:len(NG_gc_counter)], zorder=2, c='r', label='ssp_sa')
    plt.plot(DSA_SS_gc_counter, DSA_ss_gc1[:len(DSA_SS_gc_counter)], zorder=2, c='m', label='dsa_sa_ss')
    plt.plot(NG_ss_gc_counter, NG_ss_gc1[:len(NG_ss_gc_counter)], zorder=2, c='k', label='ssp_sa_ss')
    plt.plot(DSA_SC_gc_counter, DSA_SC_gc1[:len(DSA_SC_gc_counter)], zorder=2, c='g', label='dsa_sc')
    plt.plot(DSA_SC_E_gc_counter, DSA_SC_E_gc1[:len(DSA_SC_E_gc_counter)], zorder=2, c='c', label='dsa_sc_e')
    plt.plot(list(DSA_gc.keys()), DSA_gc1, zorder=2, c='b', label='dsa_sa')
    plt.plot(list(NG_gc.keys()), NG_gc1, zorder=2, c='r', label='ssp_sa')
    plt.plot(list(DSA_SS_gc.keys()), DSA_ss_gc1, zorder=2, c='m', label='dsa_sa_ss')
    plt.plot(list(NG_ss_gc.keys()), NG_ss_gc1, zorder=2, c='k', label='ssp_sa_ss')
    plt.plot(list(DSA_SC_gc.keys()), DSA_SC_gc1, zorder=2, c='g', label='dsa_sc')
    plt.plot(list(DSA_SC_E_gc.keys()), DSA_SC_E_gc1, zorder=2, c='c', label='dsa_sc_e')
    plt.legend(loc='lower right')
    plt.xlabel('non concurrent logical operation')
    plt.ylabel('utility')
    plt.show()
    print('last check')'''


def NCLO_iter_graph_sp(DSA_SC_gc, DSA_SC_E_gc, DSA_gc, DSA_SS_gc, NG_gc, NG_ss_gc, NG_SC_gc, NG_SC_E_gc):
    DSA_SC_gc1 = standarize_gc(DSA_SC_gc.values())
    DSA_SC_E_gc1 = standarize_gc(DSA_SC_E_gc.values())
    DSA_gc1 = standarize_gc(DSA_gc.values())
    DSA_ss_gc1 = standarize_gc(DSA_SS_gc.values())
    NG_gc1 = standarize_gc(NG_gc.values())
    NG_ss_gc1 = standarize_gc(NG_ss_gc.values())
    NG_SC_gc1 = standarize_gc(NG_SC_gc.values())
    NG_SC_E_gc1 = standarize_gc(NG_SC_E_gc.values())
    counter = min(list(DSA_SC_gc.keys())[-1], list(DSA_SC_E_gc.keys())[-1], list(DSA_gc.keys())[-1],
                  list(DSA_SS_gc.keys())[-1])

    DSA_SC_gc_counter = [c for c in DSA_SC_gc.keys() if c <= counter]
    DSA_SC_E_gc_counter = [c for c in DSA_SC_E_gc.keys() if c <= counter]
    DSA_gc_counter = [c for c in DSA_gc.keys() if c <= counter]
    DSA_SS_gc_counter = [c for c in DSA_SS_gc.keys() if c <= counter]
    f, (ax1, ax2) = plt.subplots(1, 2, sharey='all', sharex='all')

    if list(NG_gc.keys())[-1] < counter:
        dashed_NG_x = [list(NG_gc.keys())[-1], counter]
        dashed_NG_y = [list(NG_gc.values())[-1], list(NG_gc.values())[-1]]
        ax1.plot(dashed_NG_x, dashed_NG_y, '--', zorder=2, c='r')
        NG_gc_counter = [c for c in NG_gc.keys()]
    else:
        NG_gc_counter = [c for c in NG_gc.keys() if c <= counter]
    if list(NG_ss_gc.keys())[-1] < counter:
        dashed_NG_ss_x = [list(NG_ss_gc.keys())[-1], counter]
        dashed_NG_ss_y = [list(NG_ss_gc.values())[-1], list(NG_ss_gc.values())[-1]]
        ax1.plot(dashed_NG_ss_x, dashed_NG_ss_y, '--', zorder=2, c='k')
        NG_ss_gc_counter = [c for c in NG_ss_gc.keys()]
    else:
        NG_ss_gc_counter = [c for c in NG_ss_gc.keys() if c <= counter]
    if list(NG_SC_gc.keys())[-1] < counter:
        dashed_NG_SC_x = [list(NG_SC_gc.keys())[-1], counter]
        dashed_NG_SC_y = [list(NG_SC_gc.values())[-1], list(NG_SC_gc.values())[-1]]
        ax2.plot(dashed_NG_SC_x, dashed_NG_SC_y, '--', zorder=2, c='y')
        NG_SC_gc_counter = [c for c in NG_SC_gc.keys()]
    else:
        NG_SC_gc_counter = [c for c in NG_SC_gc.keys() if c <= counter]
    if list(NG_SC_E_gc.keys())[-1] < counter:
        dashed_NG_SC_E_x = [list(NG_SC_E_gc.keys())[-1], counter]
        dashed_NG_SC_E_y = [list(NG_SC_E_gc.values())[-1], list(NG_SC_E_gc.values())[-1]]
        ax2.plot(dashed_NG_SC_E_x, dashed_NG_SC_E_y, '--', zorder=2, c='tab:orange')
        NG_SC_E_gc_counter = [c for c in NG_SC_E_gc.keys()]
    else:
        NG_SC_E_gc_counter = [c for c in NG_SC_E_gc.keys() if c <= counter]

    ax1.plot(DSA_gc_counter, DSA_gc1[:len(DSA_gc_counter)], zorder=2, c='b', label='dsa_sa')
    ax1.plot(NG_gc_counter, NG_gc1[:len(NG_gc_counter)], zorder=2, c='r', label='ssp_sa')
    ax1.plot(DSA_SS_gc_counter, DSA_ss_gc1[:len(DSA_SS_gc_counter)], zorder=2, c='m', label='dsa_sa_ss')
    ax1.plot(NG_ss_gc_counter, NG_ss_gc1[:len(NG_ss_gc_counter)], zorder=2, c='k', label='ssp_sa_ss')
    ax2.plot(DSA_SC_gc_counter, DSA_SC_gc1[:len(DSA_SC_gc_counter)], zorder=2, c='g', label='dsa_sc')
    ax2.plot(DSA_SC_E_gc_counter, DSA_SC_E_gc1[:len(DSA_SC_E_gc_counter)], zorder=2, c='c', label='dsa_sc_e')
    ax2.plot(NG_SC_gc_counter, NG_SC_gc1[:len(NG_SC_gc_counter)], zorder=2, c='y', label='ssp_sc')
    ax2.plot(NG_SC_E_gc_counter, NG_SC_E_gc1[:len(NG_SC_E_gc_counter)], zorder=2, c='tab:orange', label='ssp_sc_e')
    ax1.legend(loc='lower right')
    ax2.legend(loc='lower right')

    f.text(0.5, 0.02, 'non concurrent logical operation', ha='center', va='center')
    f.text(0.02, 0.5, 'utility', ha='center', va='center', rotation='vertical')
    plt.show()
    print('last check')


def NCLO_iter_graph_by_dict(nclo_gc_dict, color_dict, line_dict):
    standarized_dict = {}
    max_counter_list = []
    algo_counter = {}
    for algo in nclo_gc_dict:
        standarized_dict[algo] = standarize_gc(nclo_gc_dict[algo].values())
        max_counter_list.append(list(nclo_gc_dict[algo].keys())[-1])
    counter = min(max_counter_list)
    for algo in nclo_gc_dict:
        algo_counter[algo] = [c for c in nclo_gc_dict[algo].keys() if c <= counter]
    f, ax = plt.subplots()
    for algo in nclo_gc_dict:
        ax.plot(algo_counter[algo], standarized_dict[algo][:len(algo_counter[algo])], zorder=2, c=color_dict[algo],
                label=algo, linestyle=line_dict[algo])
    fontP = FontProperties()
    fontP.set_size('xx-small')
    ax.legend(loc='center right', bbox_to_anchor=(1.31, 0.5), prop=fontP)
    f.text(0.43, 0.01, 'iterations', ha='center', va='center')
    f.text(0.02, 0.5, 'Utility', ha='center', va='center', rotation='vertical')
    plt.legend(loc='upper right')
    plt.show()


def NCLO_iter_graph(DSA_SC_gc, DSA_SC_E_gc, DSA_gc, DSA_SS_gc, NG_gc, NG_ss_gc, NG_SC_gc, NG_SC_E_gc):
    DSA_SC_gc1 = standarize_gc(DSA_SC_gc.values())
    DSA_SC_E_gc1 = standarize_gc(DSA_SC_E_gc.values())
    DSA_gc1 = standarize_gc(DSA_gc.values())
    DSA_ss_gc1 = standarize_gc(DSA_SS_gc.values())
    NG_gc1 = standarize_gc(NG_gc.values())
    NG_ss_gc1 = standarize_gc(NG_ss_gc.values())
    NG_SC_gc1 = standarize_gc(NG_SC_gc.values())
    NG_SC_E_gc1 = standarize_gc(NG_SC_E_gc.values())

    counter = min(list(DSA_SC_gc.keys())[-1], list(DSA_SC_E_gc.keys())[-1], list(DSA_gc.keys())[-1],
                  list(DSA_SS_gc.keys())[-1])
    DSA_SC_gc_counter = [c for c in DSA_SC_gc.keys() if c <= counter]
    DSA_SC_E_gc_counter = [c for c in DSA_SC_E_gc.keys() if c <= counter]
    DSA_gc_counter = [c for c in DSA_gc.keys() if c <= counter]
    DSA_SS_gc_counter = [c for c in DSA_SS_gc.keys() if c <= counter]
    f, ax = plt.subplots()

    if list(NG_gc.keys())[-1] < counter:
        dashed_NG_x = [list(NG_gc.keys())[-1], counter]
        dashed_NG_y = [list(NG_gc.values())[-1], list(NG_gc.values())[-1]]
        ax.plot(dashed_NG_x, dashed_NG_y, '--', zorder=2, c='r')
        NG_gc_counter = [c for c in NG_gc.keys()]
    else:
        NG_gc_counter = [c for c in NG_gc.keys() if c <= counter]
    if list(NG_ss_gc.keys())[-1] < counter:
        dashed_NG_ss_x = [list(NG_ss_gc.keys())[-1], counter]
        dashed_NG_ss_y = [list(NG_ss_gc.values())[-1], list(NG_ss_gc.values())[-1]]
        ax.plot(dashed_NG_ss_x, dashed_NG_ss_y, '--', zorder=2, c='k')
        NG_ss_gc_counter = [c for c in NG_ss_gc.keys()]
    else:
        NG_ss_gc_counter = [c for c in NG_ss_gc.keys() if c <= counter]
    if list(NG_SC_gc.keys())[-1] < counter:
        dashed_NG_SC_x = [list(NG_SC_gc.keys())[-1], counter]
        dashed_NG_SC_y = [list(NG_SC_gc.values())[-1], list(NG_SC_gc.values())[-1]]
        ax.plot(dashed_NG_SC_x, dashed_NG_SC_y, '--', zorder=2, c='y')
        NG_SC_gc_counter = [c for c in NG_SC_gc.keys()]
    else:
        NG_SC_gc_counter = [c for c in NG_SC_gc.keys() if c <= counter]
    if list(NG_SC_E_gc.keys())[-1] < counter:
        dashed_NG_SC_E_x = [list(NG_SC_E_gc.keys())[-1], counter]
        dashed_NG_SC_E_y = [list(NG_SC_E_gc.values())[-1], list(NG_SC_E_gc.values())[-1]]
        ax.plot(dashed_NG_SC_E_x, dashed_NG_SC_E_y, '--', zorder=2, c='tab:orange')
        NG_SC_E_gc_counter = [c for c in NG_SC_E_gc.keys()]
    else:
        NG_SC_E_gc_counter = [c for c in NG_SC_E_gc.keys() if c <= counter]

    ax.plot(DSA_gc_counter, DSA_gc1[:len(DSA_gc_counter)], zorder=2, c='b', label='dsa_sa')
    ax.plot(NG_gc_counter, NG_gc1[:len(NG_gc_counter)], zorder=2, c='r', label='ssp_sa')
    ax.plot(DSA_SS_gc_counter, DSA_ss_gc1[:len(DSA_SS_gc_counter)], zorder=2, c='m', label='dsa_sa_ss')
    ax.plot(NG_ss_gc_counter, NG_ss_gc1[:len(NG_ss_gc_counter)], zorder=2, c='k', label='ssp_sa_ss')
    ax.plot(DSA_SC_gc_counter, DSA_SC_gc1[:len(DSA_SC_gc_counter)], zorder=2, c='g', label='dsa_sc')
    ax.plot(DSA_SC_E_gc_counter, DSA_SC_E_gc1[:len(DSA_SC_E_gc_counter)], zorder=2, c='c', label='dsa_sc_e')
    ax.plot(NG_SC_gc_counter, NG_SC_gc1[:len(NG_SC_gc_counter)], zorder=2, c='y', label='ssp_sc')
    ax.plot(NG_SC_E_gc_counter, NG_SC_E_gc1[:len(NG_SC_E_gc_counter)], zorder=2, c='tab:orange', label='ssp_sc_e')
    ax.legend(loc='lower right')

    f.text(0.5, 0.02, 'non concurrent logical operation', ha='center', va='center')
    f.text(0.02, 0.5, 'utility', ha='center', va='center', rotation='vertical')
    plt.show()
    print('last check')


def NCLO_average_graph(average_outputs, x_range, color_dict, line_dict, title):
    # f, (ax1, ax2) = plt.subplots(1,2,sharey='all')
    f, ax = plt.subplots()
    right_side = ax.spines["right"]
    right_side.set_visible(False)
    top_side = ax.spines["top"]
    top_side.set_visible(False)
    for algo in color_dict.keys():
        '''if not 'sf' in algo:
            continue'''
        ax.plot(x_range, average_outputs[algo], zorder=2, c=color_dict[algo], linestyle=line_dict[algo], label=algo)
    # fontP = FontProperties()
    # fontP.set_size('small')
    # ax.legend(loc='center right', bbox_to_anchor=(1.31, 0.5), prop=fontP, frameon=False)
    # space = 1.5  # for without ng & opt / only opt
    space = 0.6  # with ng, without opt
    ax.legend(frameon=False, bbox_to_anchor=(1.05, 1), labelspacing=space)
    # ax2.legend(frameon=False, bbox_to_anchor=(1.05, 1), labelspacing=ax2_space)
    f.text(0.43, 0.02, 'NCLO', ha='center', va='center')
    f.text(0.015, 0.5, 'Utility', ha='center', va='center', rotation='vertical')
    # ax.set_title(title)
    # plt.savefig('ss100.pdf')
    plt.yticks(range(65_000, 120_000, 5_000))
    plt.show()


def iter_graph(DSA_SC_gc, DSA_SC_E_gc, DSA_gc, DSA_SS_gc, NG_gc, NG_ss_gc, NG_SC_gc, NG_SC_E_gc):
    DSA_SC_gc1 = standarize_gc(DSA_SC_gc.values())
    DSA_SC_E_gc1 = standarize_gc(DSA_SC_E_gc.values())
    DSA_gc1 = standarize_gc(DSA_gc.values())
    DSA_ss_gc1 = standarize_gc(DSA_SS_gc.values())
    NG_gc1 = standarize_gc(NG_gc.values())
    NG_ss_gc1 = standarize_gc(NG_ss_gc.values())
    NG_SC_gc1 = standarize_gc(NG_SC_gc.values())
    NG_SC_E_gc1 = standarize_gc(NG_SC_E_gc.values())
    fig, ((ax1, ax2, ax3, ax7), (ax4, ax5, ax6, ax8)) = plt.subplots(2, 4, sharey='all')

    ax1.plot(range(len(DSA_gc1)), DSA_gc1)
    ax2.plot(range(len(NG_gc1)), NG_gc1)
    ax3.plot(range(len(DSA_SC_gc1)), DSA_SC_gc1)
    ax4.plot(range(len(DSA_ss_gc1)), DSA_ss_gc1)
    ax5.plot(range(len(NG_ss_gc1)), NG_ss_gc1)
    ax6.plot(range(len(DSA_SC_E_gc1)), DSA_SC_E_gc1)
    ax7.plot(range(len(NG_SC_gc1)), NG_SC_gc1)
    ax8.plot(range(len(NG_SC_E_gc1)), NG_SC_E_gc1)

    fig.text(0.5, 0.02, 'iteration', ha='center', va='center')
    fig.text(0.02, 0.5, 'utility', ha='center', va='center', rotation='vertical')

    ax1.set_title('dsa_sa')
    ax2.set_title('ssp_sa')
    ax3.set_title('dsa_sc')
    ax4.set_title('dsa_sa_ss')
    ax5.set_title('ssp_sa_ss')
    ax6.set_title('dsa_sc_e')
    ax7.set_title('ssp_sc')
    ax8.set_title('ssp_sc_e')

    plt.show()


def iter_graph_Barak_test1(DSA_SC):
    DSA_SC = standarize_gc(DSA_SC.values())
    fig, (ax1), = plt.subplots(2, 4, sharey='all')

    ax1.plot(range(len(DSA_SC)), DSA_SC)

    fig.text(0.5, 0.02, 'iteration', ha='center', va='center')
    fig.text(0.02, 0.5, 'utility', ha='center', va='center', rotation='vertical')

    ax1.set_title('DSA_SC')

    plt.show()


def standarize_gc(gc):
    negative_costs = [c for c in gc if c < 0]
    if len(negative_costs) > 0:
        max_neg_cost = abs(max(negative_costs))
        stan_gc = []
        for c in gc:
            if c < 0:
                stan_gc.append((c + max_neg_cost) / 10_000)
            else:
                stan_gc.append(c)
        return stan_gc
    else:
        return list(gc)


def convert_from_nclo_to_iter_keys(nclo_dict, num_iter):
    """
    converts dictioanry of form {nclo_counter: utility..} to {num_iteration: utility}
    because in NG their is an option that the global steps will stop growing if a fs
    is reached
    nclo - non concurrent logical operation
    """
    iter_dict = {}
    i = 0
    for nclo_counter in nclo_dict:
        iter_dict[i] = nclo_dict[nclo_counter]
        i += 1
    if (i - 1) < num_iter:
        last_value = list(nclo_dict.values())[-1]
        if i == num_iter:
            iter_dict[num_iter] = last_value
        else:
            for j in range(i, num_iter + 1):
                iter_dict[j] = last_value
    return iter_dict


def statistical_evaluation(read_path, no_good, write_path, by_max=False):
    data_dict = create_data_for_eval(read_path, no_good,
                                     by_max)  # dictionary key: algo_name, value: list of scores of all 50 instances
    F, p = f_oneway(*data_dict.values())
    if True:  # p < 0.05:
        scores = list(itertools.chain(*data_dict.values()))
        num_samples = 0
        for k in data_dict:
            num_samples = len(data_dict[k])
            break
        groups = np.repeat(list(data_dict.keys()), repeats=num_samples)
        df = pd.DataFrame({'score': scores, 'group': groups})
        tukey = pairwise_tukeyhsd(endog=df['score'], groups=df['group'], alpha=0.05)
        print(tukey)
        results_df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
        results_df.to_csv(write_path)
        print('lets see')


def create_data_for_eval(path, no_good, by_max=False):
    files = os.listdir(path)
    outputs = []
    AAAI_algo = ['Dsa_sc_e', 'Dsa_sc_e_sf', 'Dsa_sc', 'Dsa_sc_sf', 'Dsa_sa', 'Dsa_sa_sf', 'QRDSA_sc_e', 'QRDSA_sc_e_sf',
                 'QRDSA_sc', 'QRDSA_sc_sf', 'QRDSA_sa', 'QRDSA_sa_sf']
    for f in files:
        if f in no_good:
            continue
        pa = path + '\\' + f
        infile = open(pa, 'rb')
        p = pickle.load(infile)
        outputs.append(p)
        infile.close()
    data_dict = {}
    for out in outputs:
        for algo in out:
            if any(n < 0 for n in out[algo].values()):
                print('yassu')
            if algo not in AAAI_algo:
                continue
            if algo not in data_dict:
                data_dict[algo] = []
            if by_max:  # Anytime evaluation
                data_dict[algo].append(max(out[algo].values()))
            else:
                max_iter = max(out[algo].keys())
                data_dict[algo].append(out[algo][max_iter])
    return data_dict


def t_test(read_path, no_good, write_path, algo1, algo2, by_max=False):
    data_dict = create_data_for_eval(read_path, no_good,
                                     by_max)  # dictionary key: algo_name, value: list of scores of all 50 instances
    sample1 = data_dict[algo1]
    sample2 = data_dict[algo2]
    t, p = ttest_ind(sample1, sample2)
    print(p)


def read_pickle_files(read_path):
    files = os.listdir(read_path)
    outputs = []
    # read files
    for f in files:
        pa = read_path + '\\' + f
        infile = open(pa, 'rb')
        p = pickle.load(infile)
        outputs.append(p)
        infile.close()
    return outputs


def plot_num_changes(read_path):
    outputs = read_pickle_files(read_path)
    agents_mean_outputs = []
    # mean of all problems
    for o in outputs:
        ol = []  # list of output o - list of dictionaries for every iteration mean and max
        for iter in o['nc_ss']:
            ol.append({'mean': sum(iter.values()) / len(iter.values()), 'max': max(iter.values())})
        agents_mean_outputs.append(ol)
    iteration_mean_outputs = {'mean': [], 'max': []}
    for i in range(16):
        iter_out = {'mean': [], 'max': []}
        for o in agents_mean_outputs:
            try:
                iter_out['mean'].append(o[i]['mean'])
                iter_out['max'].append(o[i]['max'])
            except:
                IndexError

        iteration_mean_outputs['mean'].append(sum(iter_out['mean']) / len(iter_out['mean']))
        iteration_mean_outputs['max'].append(sum(iter_out['max']) / len(iter_out['max']))
    problem_name = read_path[-5:]
    create_graph_num_changes(iteration_mean_outputs, problem_name)
    print(problem_name)
    print('mean:' + str(sum(iteration_mean_outputs['mean']) / len(iteration_mean_outputs['mean'])))
    print('max:' + str(sum(iteration_mean_outputs['max']) / len(iteration_mean_outputs['max'])))


def create_graph_num_changes(iteration_mean_outputs, problem):
    f, ax = plt.subplots()
    right_side = ax.spines["right"]
    right_side.set_visible(False)
    top_side = ax.spines["top"]
    top_side.set_visible(False)
    ax.plot(range(len(iteration_mean_outputs['mean'])), iteration_mean_outputs['mean'], zorder=2, c='tab:green',
            label='mean')
    ax.plot(range(len(iteration_mean_outputs['max'])), iteration_mean_outputs['max'], zorder=2, c='tab:red',
            label='max')
    space = 1.5
    ax.legend(frameon=False, bbox_to_anchor=(1.05, 1), labelspacing=space)
    f.text(0.43, 0.02, 'Iteration', ha='center', va='center')
    f.text(0.015, 0.5, 'Number of Changes', ha='center', va='center', rotation='vertical')
    ax.set_title(problem + ' - SA Number Of Changes')
    plt.show()

def reinforcement_iteration_action(reward_by_ward):
    for ward_RL in reward_by_ward:
        if (ward_RL.reward >= ward_RL.max_reward):
            ward_RL.best_state = ward_RL.curr_state
            ward_RL.max_reward = ward_RL.reward
            # good action will continue to next available actions
            ward_RL.available_actions(ward_RL.curr_state)
            if len(ward_RL.available_actions_list) == 0:
                ward_RL.choice_random_action()
                ward_RL.update_action(ward_RL.curr_state)
            else:
                action = random.choice(ward_RL.available_actions_list)
                ward_RL.update_action(action)
        else:
            # bad action, take another action from available actions
            if len(ward_RL.available_actions_list) > 0:
                action = random.choice(ward_RL.available_actions_list)
                ward_RL.update_action(action)
            else:
                ward_RL.choice_random_action()
                ward_RL.update_action(ward_RL.curr_state)

def DSA_RL_train(rl_iter):
    global_cost_iter = {}
    global_reward_iter = {}
    reward_by_ward_iter = {}
    DSA_best_states_per_ward ={}
    for iter in range(0, rl_iter):
        print('DSA_RL iter ' + str(iter))

        dsa_sc_sat_iter_gc, dsa_sc_sat_iter_gs, dsa_sc_sat_mgs, max_global_cost, reward_by_ward, global_norm_reward = p.DSA(
            single_change=True, num_iter=5000,
            change_func='single_variable_change',
            random_selection=True, stable_schedule_flag=True,
            no_good_flag=True)

        global_cost_iter[iter] = max_global_cost
        global_reward_iter[iter] = global_norm_reward
        reward_by_ward_iter[iter] = reward_by_ward
        # reinforcement ward agent action
        reinforcement_iteration_action(reward_by_ward)
        if iter == 0:
            DSA_before_RL = dsa_sc_sat_iter_gc

        p.clear_problem()

    for ward in reward_by_ward:
        DSA_best_states_per_ward[ward] = [ward.best_state, ward.max_reward]


    return DSA_best_states_per_ward ,DSA_before_RL, reward_by_ward_iter, global_reward_iter

def QRDSA_RL_train(rl_iter,init_sol_value):
    global_cost_iter = {}
    global_reward_iter = {}
    reward_by_ward_iter = {}
    QRDSA_best_states_per_ward ={}
    for iter in range(0, rl_iter):
        print('QRDSA_RL iter ' + str(iter))

        ssp_sc_sat_iter_gc, ssp_sc_sat_iter_gs, ssp_sc_sat_iter, ssp_sc_sat_mgs, max_global_cost, reward_by_ward, global_norm_reward = p.NG(
            single_change=True, num_iter=5000,
            change_func='single_variable_change',
            stop_fs_flag=False,
            init_sol_value= init_sol_value,
            random_selection=True,
            stable_schedule_flag=True,
            no_good_flag=True)

        global_cost_iter[iter] = max_global_cost
        global_reward_iter[iter] = global_norm_reward
        reward_by_ward_iter[iter] = reward_by_ward
        # reinforcement ward agent action
        reinforcement_iteration_action(reward_by_ward)

        if iter == 0:
            QRDSA_before_RL = ssp_sc_sat_iter_gc

        p.clear_problem()

    for ward in reward_by_ward:
        QRDSA_best_states_per_ward[ward] = [ward.best_state, ward.max_reward]

    return QRDSA_best_states_per_ward ,QRDSA_before_RL, reward_by_ward_iter, global_reward_iter

### Reinforcement Learning plot ###
def Reinforcement_Learning_iter_plot(reward_by_ward_iter, global_reward_iter,algo):
    iterations = list(reward_by_ward_iter.keys())
    wards_dict = reward_by_ward_iter[0]
    for i in range(1, len(iterations)):
        for j in wards_dict.keys():
            wards_dict[j] += reward_by_ward_iter[i][j]
    print('wards_dict')
    print(wards_dict)

    for i in wards_dict.keys():
        y1 = wards_dict[i]
        plt.plot(iterations, y1, label='ward ' + str(i.w_id))

    plt.plot(iterations, list(global_reward_iter.values()), label='total schedule')
    # naming the x axis
    plt.xlabel('RL iterations')
    # naming the y axis
    plt.ylabel('Reward')
    # giving a title to my graph
    plt.title('Train RL Ward Agent ' + str(algo) + ' algorithm')

    # show a legend on the plot
    plt.legend(loc='upper right')

    # function to show the plot
    plt.show()


# Full Experiment Script:
p = Problem(num_wards=2, schedule_date='2021-11-07')

DSA_best_states_per_ward ,DSA_before_RL ,DSA_reward_by_ward_iter, DSA_global_reward_iter= DSA_RL_train(10)
for wa in p.agents['ward_agents']:
    wa.ward.curr_state = [1,1,1]
    wa.ward.max_reward = 0
    wa.ward.reward = None
    wa.ward.visited_states = [[1, 1, 1]]
    wa.ward.available_actions_list = []
    wa.ward.best_state = None

QRDSA_best_states_per_ward ,QRDSA_before_RL ,QRDSA_reward_by_ward_iter, QRDSA_global_reward_iter = QRDSA_RL_train(10,DSA_before_RL[0])

### RL Train plot ###
Reinforcement_Learning_iter_plot(DSA_reward_by_ward_iter, DSA_global_reward_iter,' DSA')
Reinforcement_Learning_iter_plot(QRDSA_reward_by_ward_iter, QRDSA_global_reward_iter,' QRDSA')

### schedule after RL training with best solution found

for ward in DSA_best_states_per_ward.keys():
    print('best state for ward ' + str(ward.w_id) + ' is: ' + str(DSA_best_states_per_ward[ward][0]) + ' with max reward: ' +
          str(DSA_best_states_per_ward[ward][1]) )
    ward.update_best_state(DSA_best_states_per_ward[ward][0])

DSA_after_RL, dsa_sc_sat_iter_gs, dsa_sc_sat_mgs, max_global_cost, reward_by_ward, global_norm_reward = p.DSA(
    single_change=True, num_iter=5000,
    change_func='single_variable_change',
    random_selection=True, stable_schedule_flag=True,
    no_good_flag=True)

for ward in QRDSA_best_states_per_ward.keys():
    print('best state for ward ' + str(ward.w_id) + ' is: ' + str(QRDSA_best_states_per_ward[ward][0]) + ' with max reward: ' +
          str(QRDSA_best_states_per_ward[ward][1]) )
    ward.update_best_state(QRDSA_best_states_per_ward[ward][0])

QRDSA_after_RL, ssp_sc_sat_iter_gs, ssp_sc_sat_iter, ssp_sc_sat_mgs, max_global_cost, reward_by_ward, global_norm_reward =\
    p.NG(single_change=True, num_iter=5000, change_func='single_variable_change', stop_fs_flag=False,
        init_sol_value=DSA_before_RL[0], random_selection=True, stable_schedule_flag=True, no_good_flag=True)

pickle_dict = {'DSA_before_RL': DSA_before_RL, 'DSA_after_RL': DSA_after_RL,'QRDSA_befor_RL': QRDSA_before_RL, 'QRDSA_after_RL': QRDSA_after_RL}
outfile = open(r'C:\Users\User\Desktop\Final-Project\Final-Project\output\ex3_output\demo1', 'wb')
pickle.dump(pickle_dict, outfile)
outfile.close()
color_dict = {'DSA_before_eRL': 'navy', 'DSA_after_RL': 'navy','QRDSA_before_RL': 'magenta', 'QRDSA_after_RL': 'magenta'}
line_dict = {'DSA_before_RL': 'dashed', 'DSA_after_RL': 'solid','QRDSA_before_RL': 'dashed', 'QRDSA_after_RL': 'solid'}

NCLO_iter_graph_by_dict(pickle_dict, color_dict, line_dict)


# p.clear_problem()

### befor tyr qrdsa
# pickle_dict = {'DSA_befor_RL': DSA_befor_RL, 'DSA_after_RL': DSA_after_RL}
# outfile = open(r'C:\Users\User\Desktop\Final-Project\Final-Project\output\ex3_output\demo1', 'wb')
# pickle.dump(pickle_dict, outfile)
# outfile.close()
# color_dict = {'DSA_befor_RL': 'navy', 'DSA_after_RL': 'red'}
# line_dict = {'DSA_befor_RL': 'dashed', 'DSA_after_RL': 'dashed'}
#
# NCLO_iter_graph_by_dict(pickle_dict, color_dict, line_dict)
#


# pickle_dict = {'DSA_befor_RL': DSA_befor_RL}
# outfile = open(r'C:\Users\User\Desktop\Final-Project\Final-Project\output\ex3_output\demo1', 'wb')
# pickle.dump(pickle_dict, outfile)
# outfile.close()
# color_dict = {'DSA_befor_RL': 'navy'}
# line_dict = {'DSA_befor_RL': 'dashed'}
#
path = r'C:\Users\User\Desktop\Final-Project\Final-Project\output\ex3_output'

files = os.listdir(path)
outputs = []

''''

# DSA_sc_sat_ss
print('DSA_SC_SAT_SS')
dsa_sc_sat_ss_iter_gc, dsa_sc_sat_ss_iter_gs, dsa_sc_sat_ss_mgs = p.DSA(single_change=True, num_iter=5000,
                                                                        change_func='single_variable_change',
                                                                        random_selection=False,
                                                                        stable_schedule_flag=True,
                                                                        no_good_flag=False)
p.clear_problem()
# DSA_sc_sat_ss_ng
print('DSA_sc_sat_ss_ng')
dsa_sc_sat_ss_ng_iter_gc, dsa_sc_sat_ss_ng_iter_gs, dsa_sc_sat_ss_ng_mgs = p.DSA(single_change=True, num_iter=5000,
                                                                                 change_func='single_variable_change',
                                                                                 random_selection=False,
                                                                                 stable_schedule_flag=True,
                                                                                 no_good_flag=True)
p.clear_problem()
# DSA_sc
print('DSA_sc')
dsa_sc_iter_gc, dsa_sc_iter_gs, dsa_sc_mgs = p.DSA(single_change=True, num_iter=5000,
                                                   change_func='single_variable_change',
                                                   random_selection=True, stable_schedule_flag=False,
                                                   no_good_flag=False)
p.clear_problem()
# DSA_sc_ss
print('DSA_sc_ss')
dsa_sc_ss_iter_gc, dsa_sc_ss_iter_gs, dsa_sc_ss_mgs = p.DSA(single_change=True, num_iter=5000,
                                                            change_func='single_variable_change',
                                                            random_selection=True, stable_schedule_flag=True,
                                                            no_good_flag=False)
p.clear_problem()
# DSA_sc_ss_ng
print('DSA_sc_ss_ng')
dsa_sc_ss_ng_iter_gc, dsa_sc_ss_ng_iter_gs, dsa_sc_ss_ng_mgs = p.DSA(single_change=True, num_iter=5000,
                                                                     change_func='single_variable_change',
                                                                     random_selection=True, stable_schedule_flag=True,
                                                                     no_good_flag=True)
p.clear_problem()
# DSA_sc_e_sat
print('DSA_sc_e_sat')
dsa_sc_e_sat_iter_gc, dsa_sc_e_sat_iter_gs, dsa_sc_e_sat_mgs = p.DSA(single_change=True, num_iter=5000,
                                                                     change_func='single_variable_change_explore',
                                                                     random_selection=False, stable_schedule_flag=False,
                                                                     no_good_flag=False)
p.clear_problem()
# DSA_sc_e_sat_ss
print('DSA_sc_e_sat_ss')
dsa_sc_e_sat_ss_iter_gc, dsa_sc_e_sat_ss_iter_gs, dsa_sc_e_sat_ss_mgs = p.DSA(single_change=True, num_iter=5000,
                                                                              change_func=
                                                                              'single_variable_change_explore',
                                                                              random_selection=False,
                                                                              stable_schedule_flag=True,
                                                                              no_good_flag=False)
p.clear_problem()
# DSA_sc_e_sat_ss_ng
print('DSA_sc_e_sat_ss_ng')
dsa_sc_e_sat_ss_ng_iter_gc, dsa_sc_e_sat_ss_ng_iter_gs, dsa_sc_e_sat_ss_ng_mgs = p.DSA(single_change=True,
                                                                                       num_iter=5000,
                                                                                       change_func=
                                                                                       'single_variable_change_explore',
                                                                                       random_selection=False,
                                                                                       stable_schedule_flag=True,
                                                                                       no_good_flag=True)
p.clear_problem()
# DSA_sc_e
print('DSA_sc_e')
dsa_sc_e_iter_gc, dsa_sc_e_iter_gs, dsa_sc_e_mgs = p.DSA(single_change=True,
                                                         num_iter=5000,
                                                         change_func=
                                                         'single_variable_change_explore',
                                                         random_selection=True,
                                                         stable_schedule_flag=False,
                                                         no_good_flag=False)
p.clear_problem()
# DSA_sc_e_ss
print('DSA_sc_e_ss')
dsa_sc_e_ss_iter_gc, dsa_sc_e_ss_iter_gs, dsa_sc_e_ss_mgs = p.DSA(single_change=True,
                                                                  num_iter=5000,
                                                                  change_func=
                                                                  'single_variable_change_explore',
                                                                  random_selection=True,
                                                                  stable_schedule_flag=True,
                                                                  no_good_flag=False)
p.clear_problem()
# DSA_sc_e_ss_ng
print('DSA_sc_e_ss_ng')
dsa_sc_e_ss_ng_iter_gc, dsa_sc_e_ss_ng_iter_gs, dsa_sc_e_ss_ng_mgs = p.DSA(single_change=True,
                                                                           num_iter=5000,
                                                                           change_func=
                                                                           'single_variable_change_explore',
                                                                           random_selection=True,
                                                                           stable_schedule_flag=True,
                                                                           no_good_flag=True)
p.clear_problem()'''
'''# DSA_sa
print('DSA_SA')
dsa_sa_iter_gc, dsa_sa_iter_gs, dsa_sa_mgs = p.DSA(single_change=False, num_iter=5000, random_selection=True,
                                                   stable_schedule_flag=False, no_good_flag=False)
p.clear_problem()'''
# DSA_sa_ss
'''print('DSA_sa_ss')
dsa_sa_ss_iter_gc, dsa_sa_ss_iter_gs, dsa_sa_ss_mgs, nc_ss = p.DSA(single_change=False, num_iter=5000, random_selection=True,
                                                            stable_schedule_flag=True, no_good_flag=False)
p.clear_problem()
# DSA_sa_ss_ng
print('DSA_sa_ss_ng')
dsa_sa_ss_ng_iter_gc, dsa_sa_ss_ng_iter_gs, dsa_sa_ss_ng_mgs, nc_ng = p.DSA(single_change=False, num_iter=5000,
                                                                     random_selection=True, stable_schedule_flag=True,
                                                                       no_good_flag=True)
p.clear_problem()
nc_dict = {'nc_ss': nc_ss, 'nc_ng': nc_ng}'''
# outfile = open(r'C:\Users\noam\Desktop\thesis\experiments\output\demo_V2', 'wb')
'''pickle.dump(pickle_dict, outfile)
outfile.close()'''
# NG_sc_sat
'''print('NG_sc_sat')
ssp_sc_sat_iter_gc, ssp_sc_sat_iter_gs, ssp_sc_sat_iter, ssp_sc_sat_mgs = p.NG(single_change=True, num_iter=5000,
                                                                               change_func='single_variable_change',
                                                                               stop_fs_flag=False,
                                                                               init_sol_value=dsa_sa_iter_gc[0],
                                                                               random_selection=False,
                                                                               stable_schedule_flag=False,
                                                                               no_good_flag=False)
p.clear_problem()
# NG_sc_sat_ss
print('NG_sc_sat_ss')
ssp_sc_sat_ss_iter_gc, ssp_sc_sat_ss_iter_gs, ssp_sc_sat_ss_iter, ssp_sc_sat_ss_mgs = p.NG(single_change=True,
                                                                                           num_iter=5000,
                                                                                           change_func=
                                                                                           'single_variable_change',
                                                                                           stop_fs_flag=False,
                                                                                           init_sol_value=
                                                                                           dsa_sa_iter_gc[0],
                                                                                           random_selection=False,
                                                                                           stable_schedule_flag=True,
                                                                                           no_good_flag=False)
p.clear_problem()
# NG_sc_sat_ss_ng
print('NG_sc_sat_ss_ng')
ssp_sc_sat_ss_ng_iter_gc, ssp_sc_sat_ss_ng_iter_gs, ssp_sc_sat_ss_ng_iter, ssp_sc_sat_ss_ng_mgs = p.NG(
    single_change=True, num_iter=5000, change_func='single_variable_change', stop_fs_flag=False,
    init_sol_value=dsa_sa_iter_gc[0], random_selection=False, stable_schedule_flag=True, no_good_flag=True)
p.clear_problem()
# NG_sc
print('NG_SC')
ssp_sc_iter_gc, ssp_sc_iter_gs, ssp_sc_iter, ssp_sc_mgs = p.NG(
    single_change=True, num_iter=5000, change_func='single_variable_change', stop_fs_flag=False,
    init_sol_value=dsa_sa_iter_gc[0], random_selection=True, stable_schedule_flag=False, no_good_flag=False)
p.clear_problem()
# NG_sc_ss
print('NG_sc_ss')
ssp_sc_ss_iter_gc, ssp_sc_ss_iter_gs, ssp_sc_ss_iter, ssp_sc_ss_mgs = p.NG(
    single_change=True, num_iter=5000, change_func='single_variable_change', stop_fs_flag=False,
    init_sol_value=dsa_sa_iter_gc[0], random_selection=True, stable_schedule_flag=True, no_good_flag=False)
p.clear_problem()
# NG_sc_ss_ng
# todo init_sol_value=dsa_sa_iter_gc[0]
print('NG_sc_ss_ng')
ssp_sc_ss_ng_iter_gc, ssp_sc_ss_ng_iter_gs, ssp_sc_ss_ng_iter, ssp_sc_ss_ng_mgs = p.NG(
    single_change=True, num_iter=5000, change_func='single_variable_change', stop_fs_flag=False,
    init_sol_value=dsa_sa_iter_gc[0], random_selection=True, stable_schedule_flag=True, no_good_flag=True)
p.clear_problem()
# NG_sc_e_sat
print('NG_sc_e_sat')
ssp_sc_e_sat_iter_gc, ssp_sc_e_sat_iter_gs, ssp_sc_e_sat_iter, ssp_sc_e_sat_mgs = p.NG(single_change=True,
                                                                                       num_iter=5000,
                                                                                       change_func=
                                                                                       'single_variable_change_explore',
                                                                                       stop_fs_flag=False,
                                                                                       init_sol_value=dsa_sa_iter_gc[0],
                                                                                       random_selection=False,
                                                                                       stable_schedule_flag=False,
                                                                                       no_good_flag=False)
p.clear_problem()
# NG_sc_e_sat_ss
print('NG_sc_e_sat_ss')
ssp_sc_e_sat_ss_iter_gc, ssp_sc_e_sat_ss_iter_gs, ssp_sc_e_sat_ss_iter, ssp_sc_e_sat_ss_mgs = p.NG(single_change=True,
                                                                                                   num_iter=5000,
                                                                                                   change_func=
                                                                                                   'single_variable_change_explore',
                                                                                                   stop_fs_flag=False,
                                                                                                   init_sol_value=dsa_sa_iter_gc[0],
                                                                                                   random_selection=False,
                                                                                                   stable_schedule_flag=True,
                                                                                                   no_good_flag=False)
p.clear_problem()
# NG_sc_e_sat_ss_ng
print('NG_sc_e_sat_ss_ng')
ssp_sc_e_sat_ss_ng_iter_gc, ssp_sc_e_sat_ss_ng_iter_gs, ssp_sc_e_sat_ss_ng_iter, ssp_sc_e_sat_ss_ng_mgs = p.NG(
    single_change=True, num_iter=5000, change_func='single_variable_change_explore', stop_fs_flag=False,
    init_sol_value=dsa_sa_iter_gc[0], random_selection=False, stable_schedule_flag=True, no_good_flag=True)
p.clear_problem()
# NG_sc_e
print('NG_sc_e')
ssp_sc_e_iter_gc, ssp_sc_e_iter_gs, ssp_sc_e_iter, ssp_sc_e_mgs = p.NG(single_change=True,
                                                                       num_iter=5000,
                                                                       change_func=
                                                                       'single_variable_change_explore',
                                                                       stop_fs_flag=False,
                                                                       init_sol_value=dsa_sa_iter_gc[0],
                                                                       random_selection=True,
                                                                       stable_schedule_flag=False,
                                                                       no_good_flag=False)
p.clear_problem()
# NG_sc_e_ss
print('NG_sc_e_ss')
ssp_sc_e_ss_iter_gc, ssp_sc_e_ss_iter_gs, ssp_sc_e_ss_iter, ssp_sc_e_ss_mgs = p.NG(single_change=True,
                                                                                   num_iter=5000,
                                                                                   change_func=
                                                                                   'single_variable_change_explore',
                                                                                   stop_fs_flag=False,
                                                                                   init_sol_value=dsa_sa_iter_gc[0],
                                                                                   random_selection=True,
                                                                                   stable_schedule_flag=True,
                                                                                   no_good_flag=False)
p.clear_problem()
# NG_sc_e_ss_ng
print('NG_sc_e_ss_ng')
ssp_sc_e_ss_ng_iter_gc, ssp_sc_e_ss_ng_iter_gs, ssp_sc_e_ss_ng_iter, ssp_sc_e_ss_ng_mgs = p.NG(single_change=True,
                                                                                               num_iter=5000,
                                                                                               change_func=
                                                                                               'single_variable_change_explore',
                                                                                               stop_fs_flag=False,
                                                                                               init_sol_value=
                                                                                               dsa_sa_iter_gc[0],
                                                                                               random_selection=True,
                                                                                               stable_schedule_flag=True,
                                                                                               no_good_flag=True)
p.clear_problem()
# NG_sa
print('NG_sa')
ssp_sa_iter_gc, ssp_sa_iter_gs, ssp_sa_iter, ssp_sa_mgs = p.NG(single_change=False, num_iter=5000, stop_fs_flag=False,
                                                               init_sol_value=dsa_sa_iter_gc[0], random_selection=True,
                                                               stable_schedule_flag=False, no_good_flag=False)
p.clear_problem()
# NG_sa_ss
print('NG_sa_ss')
ssp_sa_ss_iter_gc, ssp_sa_ss_iter_gs, ssp_sa_ss_iter, ssp_sa_ss_mgs = p.NG(single_change=False, num_iter=5000,
                                                                           stop_fs_flag=False,
                                                                           init_sol_value=dsa_sa_iter_gc[0],
                                                                           random_selection=True,
                                                                           stable_schedule_flag=True,
                                                                           no_good_flag=False)
p.clear_problem()
# NG_sa_ss_ng
print('NG_sa_ss_ng')
ssp_sa_ss_ng_iter_gc, ssp_sa_ss_ng_iter_gs, ssp_sa_ss_ng_iter, ssp_sa_ss_ng_mgs = p.NG(single_change=False,
                                                                                       num_iter=5000,
                                                                                       stop_fs_flag=False,
                                                                                       init_sol_value=dsa_sa_iter_gc[0],
                                                                                       random_selection=True,
                                                                                       stable_schedule_flag=True,
                                                                                       no_good_flag=True)
p.clear_problem()
pickle_dict = {'DSA_sc_sat': dsa_sc_sat_iter_gc, 'DSA_sc_sat_sf': dsa_sc_sat_ss_iter_gc,
               'DSA_sc_sat_sf_ng': dsa_sc_sat_ss_ng_iter_gc, 'Dsa_sc': dsa_sc_iter_gc, 'Dsa_sc_sf': dsa_sc_ss_iter_gc,
               'Dsa_sc_sf_ng': dsa_sc_ss_ng_iter_gc, 'Dsa_sc_e_sat': dsa_sc_e_sat_iter_gc,
               'Dsa_sc_e_sat_sf': dsa_sc_e_sat_ss_iter_gc, 'Dsa_sc_e_sat_sf_ng': dsa_sc_e_sat_ss_ng_iter_gc,
               'Dsa_sc_e': dsa_sc_e_iter_gc, 'Dsa_sc_e_sf': dsa_sc_e_ss_iter_gc,
               'Dsa_sc_e_sf_ng': dsa_sc_e_ss_ng_iter_gc,
               'QRDSA_sc_sat': ssp_sc_sat_iter_gc, 'QRDSA_sc_sat_sf': ssp_sc_sat_ss_iter_gc,
               'QRDSA_sc_sat_sf_ng': ssp_sc_sat_ss_ng_iter_gc, 'QRDSA_sc': ssp_sc_iter_gc,
               'QRDSA_sc_sf': ssp_sc_ss_iter_gc,
               'QRDSA_sc_sf_ng': ssp_sc_ss_ng_iter_gc, 'QRDSA_sc_e_sat': ssp_sc_e_sat_iter_gc,
               'QRDSA_sc_e_sat_sf': ssp_sc_e_sat_ss_iter_gc, 'QRDSA_sc_e_sat_sf_ng': ssp_sc_e_sat_ss_ng_iter_gc,
               'QRDSA_sc_e': ssp_sc_e_iter_gc, 'QRDSA_sc_e_sf': ssp_sc_e_ss_iter_gc,
               'QRDSA_sc_e_sf_ng': ssp_sc_e_ss_ng_iter_gc,
               'Dsa_sa': dsa_sa_iter_gc, 'Dsa_sa_sf': dsa_sa_ss_iter_gc, 'Dsa_sa_sf_ng': dsa_sa_ss_iter_gc,
               'QRDSA_sa': ssp_sa_iter_gc, 'QRDSA_sa_sf': ssp_sa_ss_iter_gc, 'QRDSA_sa_ss_ng': ssp_sa_ss_ng_iter_gc}'''

# outfile = open(r'C:\Users\noam\Desktop\thesis\experiments\output\demo_V2', 'wb')
# pickle.dump(pickle_dict, outfile)
# outfile.close()
# line dict & color_dict - full experiments
'''color_dict = {'DSA_sc_sat': 'navy', 'DSA_sc_sat_sf': 'navy',
              'DSA_sc_sat_sf_ng': 'navy', 'DSA_sc': 'slateblue', 'DSA_sc_sf': 'slateblue',
              'DSA_sc_sf_ng': 'slateblue', 'DSA_sc_e_sat': 'purple',
              'DSA_sc_e_sat_sf': 'purple', 'DSA_sc_e_sat_sf_ng': 'purple',
              'DSA_sc_e': 'mediumorchid', 'DSA_sc_e_sf': 'mediumorchid', 'DSA_sc_e_sf_ng': 'mediumorchid',
              'QRDSA_sc_sat': 'lightcoral', 'QRDSA_sc_sat_sf': 'lightcoral',
              'QRDSA_sc_sat_sf_ng': 'lightcoral', 'QRDSA_sc': 'firebrick', 'QRDSA_sc_sf': 'firebrick',
              'QRDSA_sc_sf_ng': 'firebrick', 'QRDSA_sc_e_sat': 'darkorange',
              'QRDSA_sc_e_sat_sf': 'darkorange', 'QRDSA_sc_e_sat_sf_ng': 'darkorange',
              'QRDSA_sc_e': 'orange', 'QRDSA_sc_e_sf': 'orange', 'QRDSA_sc_e_sf_ng': 'orange',
              'DSA_sa': 'forestgreen', 'DSA_sa_sf': 'forestgreen', 'DSA_sa_sf_ng': 'forestgreen',
              'QRDSA_sa': 'gold', 'QRDSA_sa_sf': 'gold', 'QRDSA_sa_sf_ng': 'gold'}'''

'''line_dict = {'DSA_sc_sat': 'dotted', 'DSA_sc_sat_sf': 'dashed',
             'DSA_sc_sat_sf_ng': 'solid', 'DSA_sc': 'dotted', 'DSA_sc_sf': 'dashed',
             'DSA_sc_sf_ng': 'solid', 'DSA_sc_e_sat': 'dotted',
             'DSA_sc_e_sat_sf': 'dashed', 'DSA_sc_e_sat_sf_ng': 'solid',
             'DSA_sc_e': 'dotted', 'DSA_sc_e_sf': 'dashed', 'DSA_sc_e_sf_ng': 'solid',
             'QRDSA_sc_sat': 'dotted', 'QRDSA_sc_sat_sf': 'dashed',
             'QRDSA_sc_sat_sf_ng': 'solid', 'QRDSA_sc': 'dotted', 'QRDSA_sc_sf': 'dashed',
             'QRDSA_sc_sf_ng': 'solid', 'QRDSA_sc_e_sat': 'dotted',
             'QRDSA_sc_e_sat_sf': 'dashed', 'QRDSA_sc_e_sat_sf_ng': 'solid',
             'QRDSA_sc_e': 'dotted', 'QRDSA_sc_e_sf': 'dashed', 'QRDSA_sc_e_sf_ng': 'solid',
             'DSA_sa': 'dotted', 'DSA_sa_sf': 'dashed', 'DSA_sa_sf_ng': 'solid',
             'QRDSA_sa': 'dotted', 'QRDSA_sa_sf': 'dashed', 'QRDSA_sa_sf_ng': 'solid'}'''
# line dict - paper experiments
'''line_dict = {'DSA_sc_e': 'dashed', 'DSA_sc_e_sf': 'solid',
             'DSA_sc': 'dashed', 'DSA_sc_sf': 'solid',
             'DSA_sa': 'dashed', 'DSA_sa_sf': 'solid',
             'QRDSA_sc_e': 'dashed', 'QRDSA_sc_e_sf': 'solid',
             'QRDSA_sc': 'dashed', 'QRDSA_sc_sf': 'solid',
             'QRDSA_sa': 'dashed', 'QRDSA_sa_sf': 'solid'}

color_dict = {'DSA_sc_e': 'mediumorchid', 'DSA_sc_e_sf': 'mediumorchid',
              'DSA_sc': 'royalblue', 'DSA_sc_sf': 'royalblue',
              'DSA_sa': 'navy', 'DSA_sa_sf': 'navy',
              'QRDSA_sc_e': 'tab:orange', 'QRDSA_sc_e_sf': 'tab:orange',
              'QRDSA_sc': 'tab:green', 'QRDSA_sc_sf': 'tab:green',
              'QRDSA_sa': 'tab:red', 'QRDSA_sa_sf': 'tab:red'}

thesis_color_dict = {'DSA_sc_e': 'mediumorchid', 'DSA_sc_e_sf': 'mediumorchid', 'DSA_sc_e_sf_ng' : 'mediumorchid',
              'DSA_sc': 'royalblue', 'DSA_sc_sf': 'royalblue', 'DSA_sc_sf_ng' : 'royalblue',
              'DSA_sa': 'navy', 'DSA_sa_sf': 'navy', 'DSA_sa_sf_ng' : 'navy',
              'QRDSA_sc_e': 'tab:orange', 'QRDSA_sc_e_sf': 'tab:orange', 'QRDSA_sc_e_sf_ng': 'tab:orange',
              'QRDSA_sc': 'tab:green', 'QRDSA_sc_sf': 'tab:green', 'QRDSA_sc_sf_ng': 'tab:green',
              'QRDSA_sa': 'tab:red', 'QRDSA_sa_sf': 'tab:red', 'QRDSA_sa_sf_ng': 'tab:red'}

thesis_line_dict = {'DSA_sc_e': 'dashed', 'DSA_sc_e_sf': 'solid', 'DSA_sc_e_sf_ng': 'dotted',
             'DSA_sc': 'dashed', 'DSA_sc_sf': 'solid', 'DSA_sc_sf_ng': 'dotted',
             'DSA_sa': 'dashed', 'DSA_sa_sf': 'solid', 'DSA_sa_sf_ng': 'dotted',
             'QRDSA_sc_e': 'dashed', 'QRDSA_sc_e_sf': 'solid', 'QRDSA_sc_e_sf_ng': 'dotted',
             'QRDSA_sc': 'dashed', 'QRDSA_sc_sf': 'solid', 'QRDSA_sc_sf_ng': 'dotted',
             'QRDSA_sa': 'dashed', 'QRDSA_sa_sf': 'solid', 'QRDSA_sa_sf_ng': 'dotted'}

thesis_opt_color_dict = {'DSA_sc_e_opt': 'mediumorchid', 'DSA_sc_e_opt_sf': 'mediumorchid', 'DSA_sc_e_opt_sf_ng': 'mediumorchid',
              'DSA_sc_opt': 'royalblue', 'DSA_sc_opt_sf': 'royalblue', 'DSA_sc_opt_sf_ng': 'royalblue',
              'QRDSA_sc_e_opt': 'tab:orange', 'QRDSA_sc_e_opt_sf': 'tab:orange', 'QRDSA_sc_e_opt_sf_ng': 'tab:orange',
              'QRDSA_sc_opt': 'tab:green', 'QRDSA_sc_opt_sf': 'tab:green', 'QRDSA_sc_opt_sf_ng': 'tab:green'}

thesis_opt_line_dict = {'DSA_sc_e_opt': 'dashed', 'DSA_sc_e_opt_sf': 'solid', 'DSA_sc_e_opt_sf_ng': 'dotted',
             'DSA_sc_opt': 'dashed', 'DSA_sc_opt_sf': 'solid', 'DSA_sc_opt_sf_ng': 'dotted',
             'QRDSA_sc_e_opt': 'dashed', 'QRDSA_sc_e_opt_sf': 'solid', 'QRDSA_sc_e_opt_sf_ng': 'dotted',
             'QRDSA_sc_opt': 'dashed', 'QRDSA_sc_opt_sf': 'solid', 'QRDSA_sc_opt_sf_ng': 'dotted'}
'''
'''color_dict1 = {'DSA_sc_sat': 'navy', 'DSA_sc_sat_sf': 'navy',
               'DSA_sc_sat_sf_ng': 'navy', 'Dsa_sc': 'slateblue', 'Dsa_sc_sf': 'slateblue',
               'Dsa_sc_sf_ng': 'slateblue', 'Dsa_sc_e_sat': 'purple',
               'Dsa_sc_e_sat_sf': 'purple', 'Dsa_sc_e_sat_sf_ng': 'purple',
               'Dsa_sc_e': 'mediumorchid', 'Dsa_sc_e_sf': 'mediumorchid', 'Dsa_sc_e_sf_ng': 'mediumorchid',
               'QRDSA_sc_sat': 'lightcoral', 'QRDSA_sc_sat_sf': 'lightcoral',
               'QRDSA_sc_sat_sf_ng': 'lightcoral', 'QRDSA_sc': 'firebrick', 'QRDSA_sc_sf': 'firebrick',
               'QRDSA_sc_sf_ng': 'firebrick', 'QRDSA_sc_e_sat': 'darkorange',
               'QRDSA_sc_e_sat_sf': 'darkorange', 'QRDSA_sc_e_sat_sf_ng': 'darkorange',
               'QRDSA_sc_e': 'orange', 'QRDSA_sc_e_sf': 'orange', 'QRDSA_sc_e_sf_ng': 'orange',
               'Dsa_sa': 'forestgreen', 'Dsa_sa_sf': 'forestgreen', 'Dsa_sa_sf_ng': 'forestgreen',
               'QRDSA_sa': 'gold', 'QRDSA_sa_sf': 'gold', 'QRDSA_sa_ss_ng': 'gold'}'''

'''line_dict1 = {'DSA_sc_sat': 'dotted', 'DSA_sc_sat_sf': 'dashed',
             'DSA_sc_sat_sf_ng': 'solid', 'Dsa_sc': 'dotted', 'Dsa_sc_sf': 'dashed',
             'Dsa_sc_sf_ng': 'solid', 'Dsa_sc_e_sat': 'dotted',
             'Dsa_sc_e_sat_sf': 'dashed', 'Dsa_sc_e_sat_sf_ng': 'solid',
             'Dsa_sc_e': 'dotted', 'Dsa_sc_e_sf': 'dashed', 'Dsa_sc_e_sf_ng': 'solid',
             'QRDSA_sc_sat': 'dotted', 'QRDSA_sc_sat_sf': 'dashed',
             'QRDSA_sc_sat_sf_ng': 'solid', 'QRDSA_sc': 'dotted', 'QRDSA_sc_sf': 'dashed',
             'QRDSA_sc_sf_ng': 'solid', 'QRDSA_sc_e_sat': 'dotted',
             'QRDSA_sc_e_sat_sf': 'dashed', 'QRDSA_sc_e_sat_sf_ng': 'solid',
             'QRDSA_sc_e': 'dotted', 'QRDSA_sc_e_sf': 'dashed', 'QRDSA_sc_e_sf_ng': 'solid',
             'Dsa_sa': 'dotted', 'Dsa_sa_sf': 'dashed', 'Dsa_sa_sf_ng': 'solid',
             'QRDSA_sa': 'dotted', 'QRDSA_sa_sf': 'dashed', 'QRDSA_sa_ss_ng': 'solid'}'''

'''line_dict1 = {'DSA_sc_sat': 'dashed', 'DSA_sc_sat_sf': 'solid',
              'DSA_sc_sat_sf_ng': 'dotted', 'Dsa_sc': 'dashed', 'Dsa_sc_sf': 'solid',
              'Dsa_sc_sf_ng': 'solid', 'Dsa_sc_e_sat': 'dashed',
              'Dsa_sc_e_sat_sf': 'solid', 'Dsa_sc_e_sat_sf_ng': 'solid',
              'Dsa_sc_e': 'dashed', 'Dsa_sc_e_sf': 'solid', 'Dsa_sc_e_sf_ng': 'solid',
              'QRDSA_sc_sat': 'dashed', 'QRDSA_sc_sat_sf': 'solid',
              'QRDSA_sc_sat_sf_ng': 'solid', 'QRDSA_sc': 'dashed', 'QRDSA_sc_sf': 'solid',
              'QRDSA_sc_sf_ng': 'solid', 'QRDSA_sc_e_sat': 'dashed',
              'QRDSA_sc_e_sat_sf': 'solid', 'QRDSA_sc_e_sat_sf_ng': 'solid',
              'QRDSA_sc_e': 'dashed', 'QRDSA_sc_e_sf': 'solid', 'QRDSA_sc_e_sf_ng': 'solid',
              'Dsa_sa': 'dashed', 'Dsa_sa_sf': 'solid', 'Dsa_sa_sf_ng': 'solid',
              'QRDSA_sa': 'dashed', 'QRDSA_sa_sf': 'solid', 'QRDSA_sa_ss_ng': 'solid'}'''
# NCLO_iter_graph_by_dict(pickle_dict, color_dict, line_dict)

'''color_dict = {'DSA_SC_iter_gc': 'orange', 'DSA_SC_E_iter_gc': 'forestgreen',
              'DSA_SA_iter_gc': 'gold', 'DSA_SA_iter_ss_gc': 'darkorange',
              'NG_iter_gc': 'navy', 'NG_iter_gc_ss': 'firebrick', 'NG_SC_iter_gc': 'mediumorchid',
              'NG_SC_E_iter_gc': 'lightcoral'}
line_dict = {'DSA_SC_iter_gc': 'solid', 'DSA_SC_E_iter_gc': 'dashed',
              'DSA_SA_iter_gc': 'dotted', 'DSA_SA_iter_ss_gc': 'solid',
              'NG_iter_gc': 'dashed', 'NG_iter_gc_ss': 'dotted', 'NG_SC_iter_gc': 'solid',
              'NG_SC_E_iter_gc': 'dashed'}'''
# read Pickles
# path = r'C:\Users\User\Desktop\final project\output\ex2_output'
# # path = r'C:\Users\User\Desktop\thesis\system modeling\experiments\output\July Experiments\num changes\by agent'
# files = os.listdir(path)
# outputs = []
# dsa_sc_sat_iter_gc, dsa_sc_sat_iter_gs, dsa_sc_sat_mgs
# without sat & ng
'''no_good = ['NG_iter_gc1', 'NG_iter_gc_ss1', 'NG_SC_iter_gc1', 'NG_SC_E_iter_gc1', 'DSA_sc_sat', 'DSA_sc_sat_sf',
           'DSA_sc_sat_sf_ng', 'Dsa_sc_e_sat', 'Dsa_sc_e_sat_sf', 'Dsa_sc_e_sat_sf_ng', 'QRDSA_sc_sat',
           'QRDSA_sc_sat_sf',
           'QRDSA_sc_sat_sf_ng', 'QRDSA_sc_e_sat', 'QRDSA_sc_e_sat_sf', 'QRDSA_sc_e_sat_sf_ng', 'Dsa_sc_sf_ng',
           'Dsa_sc_e_sf_ng', 'QRDSA_sc_sf_ng', 'QRDSA_sc_e_sf_ng', 'Dsa_sa_sf_ng', 'QRDSA_sa_ss_ng']'''
no_good = ['NG_iter_gc1', 'NG_iter_gc_ss1', 'NG_SC_iter_gc1', 'NG_SC_E_iter_gc1']
##  with schedule date
# July Experiments 420_30min
'''all_no_good = ['e5', 'e6', 'e19']
partially_no_good = ['e29', 'e12']'''
# July Experiments ss_5
'''all_no_good = []
partially_no_good = ['e29', 'e44', 'e8']'''
# July Experiments ss_10
'''all_no_good = []
partially_no_good = []'''
# July Experiments ng_5
'''all_no_good = []
partially_no_good = ['e29', 'e38', 'e42', 'e8']'''
# July Experiments ng_10
'''all_no_good = ['e47']
partially_no_good = ['e38', 'e42', 'e44']'''
# July Experiments ngSs_100
'''all_no_good = ['e47']
partially_no_good = ['e38']'''
# July Experiments ng_100
'''all_no_good = ['e42', 'e45', 'e47']
partially_no_good = ['e9']'''
# July Experiments ss_100
'''all_no_good = []
partially_no_good = ['e29']'''

## without schedule date
# 420min
'''all_no_good = []
partially_no_good = ['e37', 'e45']'''
# 600min
'''all_no_good = ['e22', 'e45']
partially_no_good = ['e30', 'e31', 'e40', 'e46']'''
# 240min
all_no_good = ['e13', 'e24']
partially_no_good = ['e14']
# 25rooms
'''all_no_good = []
partially_no_good = ['e11', 'e14', 'e15', 'e2', 'e28', 'e4', 'e38', 'e48']'''
# 5rooms
'''all_no_good = []
partially_no_good = []'''
# ss100
'''all_no_good = []
partially_no_good = ['e42', 'e44', 'e45']'''
# ng100
'''all_no_good = ['e37', 'e44']
partially_no_good = []'''
# ssng100
'''all_no_good = []
partially_no_good = ['e21', 'e38', 'e42', 'e44', 'e45', 'e47', 'e8']'''

# mean of num of changes
for f in files:
    pa = path + '\\' + f
    infile = open(pa, 'rb')
    p = pickle.load(infile)
    outputs.append(p)
    infile.close()
'''num_changes = []
for d in outputs:
    num_changes.append(d['nc_ss'])
print(sum(num_changes) / len(num_changes))
print('lets see')
'''
# interpolation of experiments output
outputs_by_algo = {}
for out, f in zip(outputs, files):
    ov = list(out.values())
    bad_algo = []
    # NCLO_iter_graph_by_dict(out, color_dict1, line_dict1)
    if f in all_no_good:
        continue
    elif f in partially_no_good:
        # bad_algo.extend([algo for algo in out if any(n < 0 for n in out[algo].values())])
        continue
    for algo in out:
        if (algo in no_good) or (algo in bad_algo):
            continue
        if algo in outputs_by_algo:
            outputs_by_algo[algo].append(out[algo])
        else:
            outputs_by_algo[algo] = [out[algo]]

outputs_by_algo_by_steps = {}
for algo in outputs_by_algo:
    outputs_by_algo_by_steps[algo] = {'X': [], 'Y': []}
    for exp in outputs_by_algo[algo]:
        exp_dict = {}
        interp_func = interp1d(np.fromiter(exp.keys(), dtype=float), np.fromiter(exp.values(), dtype=float))
        # outputs_by_algo_by_steps[algo]['X'].append(np.arange(0, 50_000, 100))
        outputs_by_algo_by_steps[algo]['X'].append(np.arange(0, 1000, 100))
        # outputs_by_algo_by_steps[algo]['Y'].append(interp_func(np.arange(0, 50_000, 100)))
        outputs_by_algo_by_steps[algo]['Y'].append(interp_func(np.arange(0, 1000, 100)))
        try:
            outputs_by_algo_by_steps[algo]['Y'].append(interp_func(np.arange(0, 50_000, 100)))
        except ValueError:
            print('lets see')
# NCLO_graph_DSA_SC_gc(outputs_by_algo_by_steps)
# print('lets see')

average_outputs = {}
for algo in outputs_by_algo_by_steps:
    if algo[0] == 'D':
        algo1 = algo[0:3].upper() + algo[3:]
    else:
        algo1 = algo
    if algo == 'QRDSA_sa_ss_ng':
        algo1 = 'QRDSA_sa_sf_ng'
    if 'sat' in algo1:
        algo1 = algo1.replace("sat", "opt")
    average_outputs[algo1] = np.mean(outputs_by_algo_by_steps[algo]['Y'], axis=0)
# NCLO_average_graph(average_outputs, np.arange(0, 50_000, 100), thesis_opt_color_dict, thesis_opt_line_dict, '')
# NCLO_average_graph(average_outputs, np.arange(0, 50_000, 100), color_dict, line_dict, '')


# iter_graph_Barak_test1(outputs_by_algo_by_steps)
