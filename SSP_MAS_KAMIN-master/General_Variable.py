import datetime


class General_Variable(object):

    def __init__(self, day, domain, constraints, value=None):
        self.day = datetime.datetime.strptime(day, '%Y-%m-%d').date()  # converts string to date object day
        self.domain = self.initialize_domain(domain)
        self.constraints = self.init_constraints_in_variables(constraints)
        self.value = value
        self.prior_value = None

    def initialize_domain(self, domain):
        """
        initializes the domain of the variable according to unary hard constraints
        :param domain: set of all anesthetists of the hospital
        :return: set of anesthetists available for surgery on surgery date and ranked Senior
        """

        new_domain = domain.copy()
        if len(new_domain) > 0:
            for a in domain:
                # anesthetist must be available to surgery on day
                if self.day not in a.surgical_days:
                    new_domain.discard(a)
                    continue
                # Floor manager rank must be Senior
                if a.rank != 'Senior':
                    new_domain.discard(a)
                    continue
        return new_domain

    def get_init_d_key(self):
        d_key = [str(self.day)]
        return d_key

    def get_constraint_d_key(self, id=None):
        if id is None:
            return str(self.day)
        else:
            return str(self.day) + '_' + str(id)


    def init_constraints_in_variables(self, c_dict):
        """
        adds the required keys to each dictionary constraint and initializes the price value to 0. via the keys the price of
        the concerning variable will be updated the keys refer to the index of the variable.
        :param dro: key referring to date room order
        :param dr: key referring to date room
        :param d: key referring to date
        :param c_dict: dictionary of constraints of a variable type
        """
        if 'd' in c_dict:
            d_cons = c_dict['d']
            for cons in d_cons:
                d_key_list = self.get_init_d_key()
                for d_key in d_key_list:
                    d_cons[cons].prices[d_key] = 0

        return c_dict
