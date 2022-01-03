from enum import IntEnum, Enum


class Urgency(IntEnum):
    Urgent = 6
    Day = 5
    Half_Day = 4
    Semi_Elective = 3
    Elective = 2
    Routine = 1


class Complexity(IntEnum):
    Super_Complex = 6
    Complex = 5
    Semi_Complex = 4
    Regular = 3
    Semi_Simple = 2
    Simple = 1


class Gender(Enum):
    Male = 1
    Female = 2
    Other = 3


class Status(IntEnum):
    Open = 1  # The patient was not summoned to surgery
    Closed = 2  # The surgery got cancelled
    Performed = 3  # The surgery was performed and the request was closed
    Assigned = 4  # The date of surgery passed but the surgery was not performed -
    # the patient was notified and ward approves


class SurgicalGrade(IntEnum):
    Excellent = 6
    Very_Good = 5
    Good = 4
    Average = 3
    Sufficient = 2
    Bad = 1


class Wards(Enum):
    Orthopedic = 1
    Surgical = 2


class RoomNum(Enum):
    One = 1
    Two = 2
    Three = 3
    Four = 4
    Five = 5
    Six = 6
    Seven = 7
    Eight = 8
    Nine = 9
    Ten = 10
    Eleven = 11


class SurgeryOrder(IntEnum):
    First = 1
    Second = 2
    Third = 3
    Fourth = 4
    Fifth = 5


# # # genetic functions for chief Agent # # #
def calc_utility_of_day_schedule(schedule, ward, day):
    """
    could receive d_dict of a specific ward and d - but just to be able to play with it for now...
    calculates the total utility of a ward's daily schedule
    :param day: the day the schedule is calculated to
    :param ward: ward object - the ward that the schedule is calculated to
    :param schedule: dictionary : {ward : day : {room_num :{ [(sr_v, s_v)...]}}}
    :return: int total utility of the schedule
    """
    u = 0
    d_dict = schedule[ward][day]
    for room in d_dict:
        for t in d_dict[room]:
            if t[0].value is not None:
                u += t[0].value.surgery_type.utility
    return u


def genetic_algorithm(pm, size_population, case, max_generation, ward, day):
    """
    preforms genetic algorithm
    :param day: the day that the algorithm is preformed
    :param ward:  the ward that the algorithm is preformed in
    :param pm: float -  probability for mutation
    :param size_population: int - parameter of the size of the population
    :param case:  int resembles the solution space we want to work with
    {1 - all diff + surgeon patient, 2 - surgeon feasible , 3- None} default - 0  not taken into consideration when
    determining T0 of simulated Annealing
    :param max_generation: stop condition of number of generations
    :return: list of all the generation values - when each value consists of a list of the best value of each gneration
    and the average value of the generation. and the best chrom i.e. best schedule
    """
    # chrom - tuplelist of tuples (schedule, value, num_surgeries, p_begin, p_end)
    generation_values = []
    population = genetic_initialize_population(size_population, case, ward, day)  # list of tuples (schedule, value)
    num_generation = 0
    best_chrom = max(population, key=lambda chrom: chrom[1])  # best schedule
    while True:
        new_population = []
        generation_values.append([max(population, key=lambda chrom: chrom[1])[1],
                                  sum(chrom[1] for chrom in population) / len(population)])
        for i in range(len(population)):
            parents = genetic_random_selection(population)  # returns list of schedules
            child, child_num_surgeries = genetic_reproduce(parents, case)
            if random.uniform(0, 1) < pm:
                child, child_num_surgeries = genetic_mutate(child, child_num_surgeries, case, day)
            child_value = genetic_calc_value(child)
            new_population.append((child, child_value, child_num_surgeries))
        genetic_selection_probability(new_population)
        population = deepcopy(new_population)
        best_of_generation = max(population, key=lambda chrom: chrom[1])
        if best_of_generation[1] > best_chrom[1]:
            best_chrom = deepcopy(best_of_generation)
        if num_generation > max_generation:
            return generation_values, best_chrom
        else:
            num_generation += 1


def genetic_initialize_population(size, case, ward, day):
    """
    initializes a populcation of chromosomes when each chromosome is a schedule i.e v_dict
    the population is initialized from schedules outputted from simulated annealing algorithm
    :param size: the size of the population
    :param day: the day that the algorithm is preformed
    :param ward:  the ward that the algorithm is preformed in
    :param case: int resembles the solution space we want to work with
    {1 - all diff + surgeon patient, 2 - surgeon feasible , 3- None} default - 0  not taken into consideration when
    determining T0 of simulated Annealing
    :return: population - list of tuples (schedule, value, num_surgeries, p_begin, p_end)
    """
    population = []
    Soroka = Hospital(1, "Soroka")
    var_dict = init_variables_day('2020-07-07', ward)
    sa_schedules = simulated_annealing_by_day(var_dict, day, ward, by_waiting_time, case=case, genetic=True)
    sample_index = len(sa_schedules) // size
    for i in range(len(sa_schedules)):
        if i % sample_index == 0:
            population.append(sa_schedules[i])
    genetic_selection_probability(population)
    return population


def genetic_selection_probability(population):
    """
    for every schedule in population Cumulative distribution values are assigned.
    first the density probability is assigned and from there the cumulative one.
    the range of p is added to each tuple of chrom in the population
    :param population: list of tuples (schedule, value)
    :return: void - updates population to be list of tuples (schedule, value, p_begin, p_end)
    """
    min_value = min(population, key=lambda item: item[1])[1]
    sum_value = sum(chrom[1] for chrom in population)
    sum_v = 0
    list_v = []
    for chrom in population:  # chrom = tuple
        v = (abs(min_value - 1 - chrom[1])) / sum_value
        sum_v += v
        list_v.append(v)

    p_prior = 0
    for i in range(len(population)):
        p = list_v[i] / sum_v
        population[i] = population[i] + (p_prior, p_prior + p,)
        p_prior += p


def genetic_random_selection(population):
    """
    select parents from population
    :param population: list of tuples (schedule, value, num_surgeries, p_begin, p_end)
    :return: list of  tuples (parent schedules, parents num of surgeries)
    """
    parents = []
    for i in range(2):
        p = random.uniform(0, 1)
        for chrom in population:
            if chrom[3] < p < chrom[4]:
                parents.append((chrom[0], chrom[2]))
                break
    return parents


def genetic_reproduce(parents, case):
    """
    generated child schedule from two parents - compares the schedules for every variable in schedule i.e. surgery
    request or surgeon that are not identical in both schedules child chooses randomly one of the parents variables
    values
    :param parents: list of schedules
    :param case: selection method, int resembles the solution space we want to work with
    {1 - all diff + surgeon patient, 2 - surgeon feasible , 3- None} default - 0  not taken into consideration when
    determining T0 of simulated Annealing
    :return: child variable
    """
    parent_a = parents[0][0]
    parent_b = parents[1][0]
    child = deepcopy(parent_a)
    child_num_surgeries = deepcopy(parents[0][1])

    for ward_a, ward_b, ward_c in zip(parent_a, parent_b, child):
        for day in parent_a[ward_a]:
            for room_num in parent_a[ward_a][day]:
                for order in range(len(parent_a[ward_a][day][room_num])):
                    variable_a = parent_a[ward_a][day][room_num][order]  # variable tuple (sr_v, s_v)
                    variable_b = parent_b[ward_b][day][room_num][order]
                    variable_c = child[ward_c][day][room_num][order]
                    if variable_a[0].value != variable_b[0].value:  # surgery request in surgery request variable
                        if variable_b[0].value is not None:
                            p = random.uniform(0, 1)
                            if p < 0.5:
                                if variable_a[0].value is None:
                                    srv_value = genetic_find_child_value(variable_b[0].value, variable_c[0])
                                    sr_value = genetic_find_child_value(variable_b[1].value, variable_c[1])
                                    update_tuple_value(child[ward_c][day][room_num][order], ward_c, child[ward_c][day],
                                                       room_num, case, (srv_value, sr_value))
                                    child_num_surgeries[room_num] += 1
                                    continue
                                else:
                                    srv_value = genetic_find_child_value(variable_b[0].value, variable_c[0])
                                    update_variable_value(child[ward_c][day][room_num][order][0], variable_c, ward_c,
                                                          child[ward_c][day], room_num, case, srv_value)
                        else:
                            continue
                    if variable_a[0].value is not None:
                        if variable_a[1].value != variable_b[1].value:
                            p = random.uniform(0, 1)
                            if p < 0.5:
                                sr_value = genetic_find_child_value(variable_b[1].value, variable_c[1])
                                update_variable_value(child[ward_c][day][room_num][order][1], variable_c, ward_c,
                                                      child[ward_c][day], room_num, case, sr_value)

    return child, child_num_surgeries


def genetic_find_child_value(value, variable):
    """
    finds the surgery request or surgeon i.e. value in anothers varialbe domain
    :param value: surgeon or surgery request object
    :param variable: surgery request variable of surgeon variable object
    :return: the value object from the domain
    """
    for v in variable.domain:
        if v == value:
            return v


def genetic_mutate(child, child_num_surgeries, case, day):
    """
    changes a single value of a single variable in the schedule - or adds a new surgery to the schedule ie updating
    both sr_v and s_v value
    :param child: schedule that is being changed
    :param child_num_surgeries: dictionary {room_num : num of surgeries}
    :param case: seleciton method -int resembles the solution space we want to work with
    {1 - all diff + surgeon patient, 2 - surgeon feasible , 3- None} default - 0  not taken into consideration when
    determining T0 of simulated Annealing
    :return:
    """
    ward = next(iter(child))
    chosen_v, delta_E, t = select_successor(child, day, ward, child_num_surgeries, case)
    return child, child_num_surgeries


def genetic_calc_value(child):
    """
    calcs the value of the child schedule
    :param child: v_dict - dictionary {ward:{day: {room_num: [(sr_v, s_v),...]}}}
    :param ward: ward object
    :param day: day of the schedule example - '2020-08-03'
    :return: value of the schedule utility - cost
    """
    cost = 0
    srv_constraints, sv_constraints = get_constraint_dictionary(child)
    for con_key in srv_constraints:
        for cons in srv_constraints[con_key]:
            cost += sum(srv_constraints[con_key][cons].prices.values())
    cost += sum(sv_constraints['d']['overlapping'].prices.values())  # s_v constraint which is not sr_v constraint

    u = 0  # utility
    for w in child:
        for d in child[w]:
            for room_num in child[w][d]:
                for order in range(len(child[w][d][room_num])):
                    sr_v = child[w][d][room_num][order][0]
                    if sr_v.value is not None:
                        u += sr_v.value.surgery_type.utility
    return u - cost


def get_constraint_dictionary(v_dict):
    """
    returns the dictionaries that hold all the constraints
    :param v_dict: dictionary {ward:{day: {room_num: [(sr_v, s_v),...]}}}
    :param ward: ward object
    :param day: day of the schedule example - '2020-08-03'
    :return: both sr_v and s_v constraints dictionary
    """
    for w in v_dict:
        for d in v_dict[w]:
            for room_num in v_dict[w][d]:
                for order in range(len(v_dict[w][d][room_num])):
                    srv_constraints = v_dict[w][d][room_num][order][0].constraints
                    sv_constraints = v_dict[w][d][room_num][order][1].constraints
                    return srv_constraints, sv_constraints


def geneitc_graphs(g_values):
    for v in g_values:
        if v[0] < 0:
            v[0] = 2_500
    generations = range(len(g_values))
    xtick = np.arange(min(generations), max(generations), 25)
    min_values, avg_values = zip(*g_values)

    # best values graph
    fig = plt.figure()
    ax = fig.add_subplot()
    plt.plot(generations, min_values, )
    plt.title('SSP Genetic Best Values')
    plt.xticks(xtick)
    plt.xlabel('generation')
    plt.ylabel('value')
    for i, v in enumerate(min_values):
        if i < 2:
            ax.text(i, v - 50, "%d" % v, ha="center")
        if i > 50 and i % 25 == 0:
            ax.text(i, v - 50, "%d" % v, ha="center")
    plt.show()

    # average values graph
    fig = plt.figure()
    ax = fig.add_subplot()
    plt.plot(generations, avg_values)
    plt.title('SSP Genetic Average Values')
    plt.xticks(xtick)
    plt.xlabel('generation')
    plt.ylabel('waiting time')
    for i, v in enumerate(avg_values):
        if i > 100 and i % 25 == 0:
            ax.text(i, v - 50, "%d" % v, ha="center")
    plt.show()

