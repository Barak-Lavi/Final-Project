import itertools
import matplotlib.pyplot as plt
import random
import pickle
import os
from datetime import datetime, timedelta
x = range(1, 101)
y1 = [random.randint(1, 100) for _ in range(len(x))]
y2 = [random.randint(1, 100) for _ in range(len(x))]
y3 = [random.randint(1, 100) for _ in range(len(x))]
y4 = [random.randint(1, 100) for _ in range(len(x))]
y5 = [random.randint(1, 100) for _ in range(len(x))]
y6 = [random.randint(1, 100) for _ in range(len(x))]


fig = plt.figure()
ax1 = fig.add_subplot(231)
ax2 = fig.add_subplot(232)
ax3 = fig.add_subplot(233)
ax4 = fig.add_subplot(234)
ax5 = fig.add_subplot(235)
ax6 = fig.add_subplot(236)

ax1.plot(x, y1)
ax2.plot(x, y2)
ax3.plot(x, y3)
ax4.plot(x, y4)
ax5.plot(x, y5)
ax6.plot(x, y6)

fig.text(0.5, 0.02, 'common xlabel', ha='center', va='center')
fig.text(0.02, 0.5, 'common ylabel', ha='center', va='center', rotation='vertical')

ax1.set_title('ax1 title')
ax2.set_title('ax2 title')
ax3.set_title('ax2 title')
ax4.set_title('ax2 title')
ax5.set_title('ax2 title')
ax6.set_title('ax2 title')

plt.show()


y6 = [random.randint(1, 100) for _ in range(100)]
y5 = [random.randint(1, 100) for _ in range(50)]
counter = min(len(y6), len(y5))
print(counter)
print(len(y5))
print(y6)
print(y5)
print(type(y5))

plt.plot(range(counter), list(y6[:counter]), zorder=2, c='b', label='dsa_sa')
plt.show()
# How to work with Pickle - Don't erase
pickle_dict = {'DSA_SC_iter_gc': {0: 21.13903267323, 500: 200, 1990: 1200}, 'DSA_SC_E_iter_gc': {0: 21, 500: 200, 1990: 1200},
               'DSA_SA_iter_gc': {0: 21, 500: 200, 1990: 1200}, 'DSA_SA_iter_ss_gc': {0: 21, 500: 200, 1990: 1200},
               'NG_iter_gc1': {0: 21, 500: 200, 1990: 1200}, 'NG_iter_gc_ss1': {0: 21, 500: 200, 1990: 1200}}
# outfile = open(r'C:\Users\noam\Desktop\thesis\experiments\output\e1', 'wb')
pickle.dump(pickle_dict, outfile)
outfile.close()

#pickle_dict2 = {'DSA_SC_iter_gc': {0: 22, 500: 200, 1990: 1200}, 'DSA_SC_E_iter_gc': {0: 22, 500: 200, 1990: 1200},
#               'DSA_SA_iter_gc': {0: 22, 500: 200, 1990: 1200}, 'DSA_SA_iter_ss_gc': {0: 22, 500: 200, 1990: 1200},
#               'NG_iter_gc1': {0: 22, 500: 200, 1990: 1200}, 'NG_iter_gc_ss1': {0: 22, 500: 200, 1990: 1200}}
#outfile2 = open(r'C:\Users\User\Desktop\thesis\system modeling\experiments\output\v2', 'wb')
#pickle.dump(pickle_dict2, outfile2)
#outfile2.close()

path = r'C:\Users\User\Desktop\final project\output'
files = os.listdir(path)
outputs = []
for f in files:
# pa = path + '\\' + f
infile = open(pa, 'rb')
    p = pickle.load(infile)
    outputs.append(p)
    infile.close()


# infile = open(r'C:\Users\User\Desktop\thesis\system modeling\experiments\output\v1', 'rb')
# new_dict = pickle.load(infile)
# infile.close()
# print(new_dict)
l = [5, -5, 10]
if any([n<0 for n in l]):
    print('ok')




