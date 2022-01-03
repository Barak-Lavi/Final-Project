import requests
from SSP_initialization import Problem
from datetime import datetime




# Main
p = Problem(num_wards=2, schedule_date='2021-04-04')
print('algo start')
DSA_SC_E_iter_gc, DSA_SC_E_iter_gs, DSA_SC_E_mgs = p.DSA(single_change=True, num_iter=2000,
                                                         change_func='single_variable_change_explore')
p.post_surgeries_DB(DSA_SC_E_mgs)


# script of single experiment - before random selection feature
p = Problem(num_wards=10)
DSA_SA_iter_gc, DSA_SA_iter_gs = p.DSA(single_change=False, num_iter=100, change_func=False)  # 15
# DSA_SA_NCLO_gc, DSA_SA_NCLO_gs = p.NCLO_costs(stable_schedule_flag=False)
# DSA_SA_NCLO_gc_new = p.sum_NCLO_gc()
p.clear_problem()
DSA_SA_iter_ss_gc, DSA_SA_iter_ss_gs = p.DSA(single_change=False, num_iter=100, change_func=True)  # 15
# DSA_SA_NCLO_ss_gc, DSA_SA_NCLO_ss_gs = p.NCLO_costs()
# DSA_SA_NCLO_gc_ss_new = p.sum_NCLO_gc()
p.clear_problem()
DSA_SC_E_iter_gc, DSA_SC_E_iter_gs = p.DSA(single_change=True, num_iter=2000,
                                           change_func='single_variable_change_explore')
# num iter = 150
# DSA_SC_E_NCLO_gc, DSA_SC_E_NCLO_gs = p.NCLO_costs()
# DSA_SC_E_NCLO_gc_new = p.sum_NCLO_gc()
p.clear_problem()
DSA_SC_iter_gc, DSA_SC_iter_gs = p.DSA(single_change=True, num_iter=5000, change_func='single_variable_change')  # 3750
# DSA_SC_NCLO_gc, DSA_SC_NCLO_gs = p.NCLO_costs()
# DSA_SC_NCLO_gc_new = p.sum_NCLO_gc()
p.clear_problem()
# num iter = 1000
NG_SC_iter_gc, NG_SC_iter_gs, NG_SC_iter = p.NG(num_iter=5000, change_func='single_variable_change', stop_fs_flag=True,
                                    init_sol_value=DSA_SA_iter_gc[0], single_change=True)
p.clear_problem()
NG_SC_E_iter_gc, NG_SC_E_iter_gs, NG_SC_E_iter = p.NG(num_iter=2000, change_func='single_variable_change_explore',
                                                      stop_fs_flag=True, init_sol_value=DSA_SA_iter_gc[0],
                                                      single_change=True)
p.clear_problem()
NG_iter_gc, NG_iter_gs, NG_iter = p.NG(num_iter=150, change_func=False, stop_fs_flag=True,
                                       init_sol_value=DSA_SA_iter_gc[0], single_change=False)  # 10
# init_sol_value=DSA_SA_iter_gc[0]
# NG_NCLO_gc, NG_NCLO_gs = p.NCLO_costs(stable_schedule_flag=False)
# NG_NCLO_gc_new = p.sum_NCLO_gc()
p.clear_problem()
NG_iter_gc_ss, NG_iter_gs_ss, NG_ss_iter = p.NG(num_iter=150, change_func=True, stop_fs_flag=True,
                                                init_sol_value=DSA_SA_iter_gc[0], single_change=False)  # 10
# NG_NCLO_gc_ss, NG_NCLO_gs_ss = p.NCLO_costs()
# NG_NCLO_gc_ss_new = p.sum_NCLO_gc()
NG_iter_gc_ss1 = convert_from_nclo_to_iter_keys(NG_iter_gc_ss, NG_ss_iter)
NG_iter_gc1 = convert_from_nclo_to_iter_keys(NG_iter_gc, NG_iter)
NG_SC_iter_gc1 = convert_from_nclo_to_iter_keys(NG_SC_iter_gc, NG_SC_iter)
NG_SC_E_iter_gc1 = convert_from_nclo_to_iter_keys(NG_SC_E_iter_gc, NG_SC_E_iter)
# NCLO_graph(DSA_SC_NCLO_gc, DSA_SC_E_NCLO_gc, DSA_SA_NCLO_gc, DSA_SA_NCLO_ss_gc, NG_NCLO_gc, NG_NCLO_gc_ss)
# NCLO_graph(DSA_SC_NCLO_gc_new, DSA_SC_E_NCLO_gc_new, DSA_SA_NCLO_gc_new, DSA_SA_NCLO_gc_ss_new, NG_NCLO_gc_new, NG_NCLO_gc_ss_new)
pickle_dict = {'DSA_SC_iter_gc': DSA_SC_iter_gc, 'DSA_SC_E_iter_gc': DSA_SC_E_iter_gc,
               'DSA_SA_iter_gc': DSA_SA_iter_gc, 'DSA_SA_iter_ss_gc': DSA_SA_iter_ss_gc,
               'NG_iter_gc1': NG_iter_gc1, 'NG_iter_gc_ss1': NG_iter_gc_ss1, 'NG_iter_gc': NG_iter_gc,
               'NG_iter_gc_ss': NG_iter_gc_ss, 'NG_SC_iter_gc': NG_SC_iter_gc, 'NG_SC_E_iter_gc': NG_SC_E_iter_gc,
               'NG_SC_iter_gc1': NG_SC_iter_gc1, 'NG_SC_E_iter_gc1': NG_SC_E_iter_gc1}
outfile = open(r'C:\Users\noam\Desktop\thesis\experiments\output\600minday\e43', 'wb')
pickle.dump(pickle_dict, outfile)
outfile.close()
print('what')
iter_graph(DSA_SC_iter_gc, DSA_SC_E_iter_gc, DSA_SA_iter_gc, DSA_SA_iter_ss_gc, NG_iter_gc1, NG_iter_gc_ss1,
           NG_SC_iter_gc1, NG_SC_E_iter_gc1)
NCLO_iter_graph(DSA_SC_iter_gc, DSA_SC_E_iter_gc, DSA_SA_iter_gc, DSA_SA_iter_ss_gc, NG_iter_gc, NG_iter_gc_ss,
                NG_SC_iter_gc, NG_SC_E_iter_gc)
NCLO_iter_graph_sp(DSA_SC_iter_gc, DSA_SC_E_iter_gc, DSA_SA_iter_gc, DSA_SA_iter_ss_gc, NG_iter_gc, NG_iter_gc_ss,
                   NG_SC_iter_gc, NG_SC_E_iter_gc)

