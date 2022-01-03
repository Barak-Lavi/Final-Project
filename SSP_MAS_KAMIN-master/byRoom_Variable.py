from General_Variable import General_Variable


class Room_Variable(General_Variable):
    def __init__(self, day, room, domain, constraints, value=None):
        self.room = room
        super(Room_Variable, self).__init__(day, domain, constraints, value)

    def init_constraints_in_variables(self, c_dict):
        """
        adds the required keys to each dictionary constraint and initializes the price value to 0. via the keys the price of
        the concerning variable will be updated the keys refer to the index of the variable.
        :param dro: key referring to date room order
        :param dr: key referring to date room
        :param d: key referring to date
        :param c_dict: dictionary of constraints of a variable type
        """
        if 'dr' in c_dict:
            dr_cons = c_dict['dr']
            for cons in dr_cons:
                dr_cons[cons].prices[self.get_constraint_dr_key()] = 0

        if 'd' in c_dict:
            d_cons = c_dict['d']
            for cons in d_cons:
                d_key_list = self.get_init_d_key()
                for d_key in d_key_list:
                    d_cons[cons].prices[d_key] = 0

        return c_dict

    def get_constraint_dr_key(self):
        return str(self.day) + '_' + str(self.room.num)  # date_room

    def initialize_domain(self, domain):
        """
        initializes the domain of the variable according to unary hard constraints
        :param domain: set of all anesthetists of the hospital
        :return: set of anesthetists available for surgery on surgery date and ranked Expert or Senior
        """
        new_domain = domain.copy()
        if len(new_domain) > 0:
            for a in domain:
                # anesthetist must be available to surgery on day
                if self.day not in a.surgical_days:
                    new_domain.discard(a)
                    continue
                # Room manager rank must be Expert or Higher
                if a.rank == 'Stagiaire':
                    new_domain.discard(a)
                    continue
        new_domain.add(None)
        return new_domain




