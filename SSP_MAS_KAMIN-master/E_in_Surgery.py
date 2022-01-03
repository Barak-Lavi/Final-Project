from Surgery_Variable import SurgeryVariable
from datetime import time, datetime, timedelta

class SurgeryVariable_Equipment(SurgeryVariable):

    def __init__(self, room, day, order, constraints, equipment, start_time=time(hour=0, minute=0), end_time=time(hour=0, minute=0)):
        self.surgery_request = None
        self.equipment = equipment
        self.value_in_update = None
        self.need_stable = False
        super(SurgeryVariable_Equipment, self).__init__(room, day, order, start_time, end_time, constraints,
                                                        {True, False}, value=False)

    def get_init_d_key(self):
        duration = (datetime.combine(self.day, self.end_time) - datetime.combine(self.day,self.start_time)).total_seconds() / 60
        time_keys = [datetime.combine(self.day, self.start_time) + timedelta(minutes=p) for p in range(0, int(duration), 30)]
        time_keys = [tk.time() for tk in time_keys]
        init_d_key = []
        for tk in time_keys:
            init_d_key.append(str(self.day) + '_' + str(self.equipment.id) + '_' + str(tk))
        return init_d_key
        # return [str(self.day) + '_' + str(self.equipment.id)]

    def get_constraint_dro_key(self):
        dro_key = str(self.day) + '_' + str(self.room.num) + '_' + str(self.order) + '_' + str(self.equipment.id)  # date_room_order
        return dro_key


