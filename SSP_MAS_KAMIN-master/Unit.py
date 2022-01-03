
class Unit(object):
    def __init__(self, u_id, name, st_set):
        self.u_id = u_id
        self.u_name = name
        self.surgery_types = st_set  # set of surgery types
