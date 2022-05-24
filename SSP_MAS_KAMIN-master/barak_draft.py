from math import sqrt
import matplotlib.pyplot as plt

# global_cost_iter = {}
# global_reward_iter = {}
# reward_by_ward_iter = {}
# for i in range(0,30):
#     global_cost_iter[i] = 20000 +i*15
#     global_reward_iter[i]= i/30
#     reward_by_ward_iter[i] = {'1':(i/30)*0.4,'2':(i/30)*0.6}
#
# print(global_cost_iter)
# print(global_reward_iter)
# print(reward_by_ward_iter)
#
# def plot_global_cost_quality(global_cost_iter,global_reward_iter):
#
#     # line 1 points
#     y1 = global_cost_iter.values()
#     x1 = global_cost_iter.keys()
#     # plotting the line 1 points
#     plt.plot(x1, y1, label="global_cost_iter")
#
#     # line 2 points
#     y2 = global_reward_iter.values()
#     x2 = global_reward_iter.keys()
#     # plotting the line 2 points
#     plt.plot(x2, y2, label="global_reward_iter")
#
#     # naming the x axis
#     plt.xlabel('RL iteration')
#     # naming the y axis
#     plt.ylabel('utility and quality')
#     # giving a title to my graph
#     plt.title('utility vs quality schedule')
#
#     # show a legend on the plot
#     plt.legend()
#
#     # function to show the plot
#     plt.show()
#
#
# reward_by_ward_iter= {}
# reward_by_ward_iter = {0: {'ward0': 0.463677732198715,'ward1': 0.7709235887874113}, 1: {'ward0': 0.463677732198715,'ward1': 0.7616358205330577}, 2: {'ward0': 0.463677732198715,'ward1': 0.7701042592797926}, 3: {'ward0': 0.463677732198715,'ward1': 0.7705354382241195}, 4: {'ward0': 0.463677732198715,'ward1': 0.7607239349329904}, 5: {'ward0': 0.463677732198715,'ward1': 0.7747122995602448}, 6: {'ward0': 0.463677732198715,'ward1': 0.7658795392486057}, 7: {'ward0': 0.463677732198715,'ward1': 0.764400946781647}, 8: {'ward0': 0.463677732198715,'ward1': 0.7699775303886656}, 9: {'ward0': 0.463677732198715,'ward1': 0.7647432110759145}}
#
# def plot_reward_ward(reward_by_ward_iter):
#     for i in reward_by_ward_iter:
#         for ward in i:
#             x1 = i.key()
#             y1 = ward.value()
#             plt.plot(x1, y1, label="ward "+ str(i))
#
#         # naming the x axis
#     plt.xlabel('RL iteration')
#     # naming the y axis
#     plt.ylabel('schedule quality')
#     # giving a title to my graph
#     plt.title('schedule quality per ward')
#
#     # show a legend on the plot
#     plt.legend()
#
#     # function to show the plot
#     plt.show()

#
# plot_global_cost_quality(global_cost_iter,global_reward_iter)
#
# plot_reward_ward(reward_by_ward_iter)

# global_reward_iter
# {0: 0.6173006604930631, 1: 0.6126567763658863, 2: 0.6168909957392539, 3: 0.6171065852114173, 4: 0.6122008335658526, 5: 0.6191950158794799, 6: 0.6147786357236604, 7: 0.6140393394901811, 8: 0.6168276312936902, 9: 0.6142104716373147}
# reward_by_ward_iter
# {0: {<Ward.Ward object at 0x00000295FFFC9E20>: 0.463677732198715, <Ward.Ward object at 0x00000295FFFC9DF0>: 0.7709235887874113}, 1: {<Ward.Ward object at 0x00000295FFFC9E20>: 0.463677732198715, <Ward.Ward object at 0x00000295FFFC9DF0>: 0.7616358205330577}, 2: {<Ward.Ward object at 0x00000295FFFC9E20>: 0.463677732198715, <Ward.Ward object at 0x00000295FFFC9DF0>: 0.7701042592797926}, 3: {<Ward.Ward object at 0x00000295FFFC9E20>: 0.463677732198715, <Ward.Ward object at 0x00000295FFFC9DF0>: 0.7705354382241195}, 4: {<Ward.Ward object at 0x00000295FFFC9E20>: 0.463677732198715, <Ward.Ward object at 0x00000295FFFC9DF0>: 0.7607239349329904}, 5: {<Ward.Ward object at 0x00000295FFFC9E20>: 0.463677732198715, <Ward.Ward object at 0x00000295FFFC9DF0>: 0.7747122995602448}, 6: {<Ward.Ward object at 0x00000295FFFC9E20>: 0.463677732198715, <Ward.Ward object at 0x00000295FFFC9DF0>: 0.7658795392486057}, 7: {<Ward.Ward object at 0x00000295FFFC9E20>: 0.463677732198715, <Ward.Ward object at 0x00000295FFFC9DF0>: 0.764400946781647}, 8: {<Ward.Ward object at 0x00000295FFFC9E20>: 0.463677732198715, <Ward.Ward object at 0x00000295FFFC9DF0>: 0.7699775303886656}, 9: {<Ward.Ward object at 0x00000295FFFC9E20>: 0.463677732198715, <Ward.Ward object at 0x00000295FFFC9DF0>: 0.7647432110759145}}
# lets see
#
# ward0 = {0: 0.6122079303503416, 1: 0.6134356581568265, 2: 0.6115215183721164, 3: 0.6275763607707762, 4: 0.6219658418318063 }
# ward1 = {0: 0.7122079303503416, 1: 0.7134356581568265, 2: 0.7115215183721164, 3: 0.7275763607707762, 4: 0.7219658418318063 }
#
# wards={'ward0.id':ward0,'ward1.id':ward1}
# print(wards)
# print(wards.values())
# wards_id_list = list(wards.keys())
# wards_dict ={}
# for i in range(0,len(wards.values())): # range(0,len(wards.values()))
#     # wards_id_list[i] = {list(wards.values())[i]}
#     wards_dict[wards_id_list[i]] = list(wards.values())[i]
#     print(wards_dict)
#

dict1 = {0:{'w0':[0.4],'w1':[0.7]},1:{'w0':[0.41],'w1':[0.69]}}
# dict2= {1:{'w0':[0.41],'w1':[0.69]}}

# tatol_d = {k: dict1[k] + dict2[k] for k in dict1}

# print(tatol_d)



# reward_by_ward_iter = {0: {'ward0': 0.463677732198715,'ward1': 0.7709235887874113}, 1: {'ward0': 0.463677732198715,'ward1': 0.7616358205330577}, 2: {'ward0': 0.463677732198715,'ward1': 0.7701042592797926}, 3: {'ward0': 0.463677732198715,'ward1': 0.7705354382241195}, 4: {'ward0': 0.463677732198715,'ward1': 0.7607239349329904}, 5: {'ward0': 0.463677732198715,'ward1': 0.7747122995602448}, 6: {'ward0': 0.463677732198715,'ward1': 0.7658795392486057}, 7: {'ward0': 0.463677732198715,'ward1': 0.764400946781647}, 8: {'ward0': 0.463677732198715,'ward1': 0.7699775303886656}, 9: {'ward0': 0.463677732198715,'ward1': 0.7647432110759145}}
reward_by_ward_iter = {0: {'ward0': [0.463677732198715],'ward1': [0.7709235887874113]}, 1: {'ward0': [0.46367773219871],'ward1': [0.7616358205330577]}, 2: {'ward0': [0.463677732198715],'ward1': [0.7701042592797926]}, 3: {'ward0':[ 0.463677732198715],'ward1':[ 0.7705354382241195]}, 4: {'ward0': [0.463677732198715],'ward1': [0.7607239349329904]}, 5: {'ward0': [0.463677732198715],'ward1': [0.7747122995602448]}, 6: {'ward0': [0.463677732198715],'ward1': [0.7658795392486057]}, 7: {'ward0': [0.463677732198715],'ward1': [0.764400946781647]}, 8: {'ward0': [0.463677732198715],'ward1': [0.7699775303886656]}, 9: {'ward0': [0.463677732198715],'ward1': [0.7647432110759145]}}
iterations= list(reward_by_ward_iter.keys())
wards_dict = reward_by_ward_iter[0]
for i in range(1,len(iterations)):
    for j in wards_dict.keys():
        wards_dict[j]+= reward_by_ward_iter[i][j]
print('wards_dict')
print(wards_dict)

for i in wards_dict.keys():
        y1 = wards_dict[i]
        plt.plot(iterations, y1, label= i)

        # naming the x axis
plt.xlabel('RL iteration')
# naming the y axis
plt.ylabel('schedule quality')
# giving a title to my graph
plt.title('schedule quality per ward')

# show a legend on the plot
plt.legend()

# function to show the plot
plt.show()