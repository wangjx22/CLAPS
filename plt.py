import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text

# class for top
HIV = [0.756,0.769,0.623]
BACE = [0.721,0.723,0.598]
BBBP = [0.939,0.945,0.741]
Tox21 = [0.715,0.723,0.614]
SIDER= [0.744,0.748,0.545]
ClinTox = [0.947,0.956,0.743]
class_name = ['CLAPS (top-10%)',	'CLAPS (top-25%)',	'CLAPS (top-50%)']



"""
绘制折线图方法plot
参数一：y轴
参数二：x轴
"""
# plt.plot(x, y,color='red')
plt.plot(class_name, HIV, "x-", label="HIV", alpha=0.5)
plt.plot(class_name, BACE, "x-", label="BACE", alpha=0.5)
plt.plot(class_name, BBBP, "x-", label="BBBP", alpha=0.5)
plt.plot(class_name, Tox21, "x-", label="Tox21", alpha=0.5)
plt.plot(class_name, SIDER, "x-", label="SIDER", alpha=0.5)
plt.plot(class_name, ClinTox, "x-", label="ClinTox", alpha=0.5)


plt.ylim(0.5, 1)
# plt.xlabel("Datasets")
plt.ylabel("ROC AUC", size=12)
# plt.title("Line Graph Example")
plt.tick_params(labelsize=12)
plt.legend()
plt.show()
plt.clf()


# class for random
HIV = [0.535,0.51,0.501]
BACE = [0.631,0.675,0.545]
BBBP = [0.877,0.904,0.673]
Tox21 = [0.622,0.623,0.51]
SIDER= [0.723,0.734,0.576]
ClinTox = [0.984,0.985,0.733]
class_name = ['CLAPS (random-10%)',	'CLAPS (random-25%)',	'CLAPS (random-50%)']

plt.plot(class_name, HIV, "x-", label="HIV", alpha=0.5)
plt.plot(class_name, BACE, "x-", label="BACE", alpha=0.5)
plt.plot(class_name, BBBP, "x-", label="BBBP", alpha=0.5)
plt.plot(class_name, Tox21, "x-", label="Tox21", alpha=0.5)
plt.plot(class_name, SIDER, "x-", label="SIDER", alpha=0.5)
plt.plot(class_name, ClinTox, "x-", label="ClinTox", alpha=0.5)


plt.ylim(0.5, 1)
# plt.xlabel("Datasets")
plt.ylabel("ROC AUC", size=12)
# plt.title("Line Graph Example")
plt.tick_params(labelsize=12)
plt.legend()
plt.show()
plt.clf()



# regression for top
ESOL = [0.711,0.662,0.855]
FreeSolv = [1.318,1.091,1.783]
Lipo = [0.978,0.954,1.275]

class_name = ['CLAPS (top-10%)',	'CLAPS (top-25%)',	'CLAPS (top-50%)']

plt.plot(class_name, ESOL, "x-", label="ESOL", alpha=0.5)
plt.plot(class_name, FreeSolv, "x-", label="FreeSolv", alpha=0.5)
plt.plot(class_name, Lipo, "x-", label="Lipo", alpha=0.5)
plt.ylim(0.5, 2)
# plt.xlabel("Datasets")
plt.ylabel("RMSE", size=12)
# plt.title("Line Graph Example")
plt.tick_params(labelsize=12)
plt.legend()
plt.show()
plt.clf()


# regression for random
ESOL = [0.774,0.731,0.823]
FreeSolv = [1.344,1.314,1.656]
Lipo = [0.947,0.881,1.079]

class_name = ['CLAPS (random-10%)',	'CLAPS (random-25%)',	'CLAPS (random-50%)']

plt.plot(class_name, ESOL, "x-", label="ESOL", alpha=0.5)
plt.plot(class_name, FreeSolv, "x-", label="FreeSolv", alpha=0.5)
plt.plot(class_name, Lipo, "x-", label="Lipo", alpha=0.5)
plt.ylim(0.5, 1.8)
# plt.xlabel("Datasets")
plt.ylabel("RMSE", size=12)
# plt.title("Line Graph Example")
plt.tick_params(labelsize=12)
plt.legend()
plt.show()
plt.clf()