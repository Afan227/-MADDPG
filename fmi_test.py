import pandas as pd
import numpy as np
import pyfmi
import matplotlib.pyplot as plt

t_interval = 10  # 时间粒度 (min)
hourDivide = int(60 / t_interval)  # 每小时时间步数，按时间粒度分割
start_time = 0
final_time = 60 * 60 * 24 * 365  # (s)
step_size = 60 * t_interval  # (s)
num_step = int((final_time - start_time) / step_size)

fmu = 'model_fmu_0303.fmu'

# model = pyfmi.load_fmu(fmu, enable_logging=True)
# var = model.get_model_variables()

input = [
    'FCUAva',  # 空调开关 01开关
    'Thermostat_Set',  # 恒温器设定温度
    'U_ITE',  # ITE利用率
    # 'Fan_Spd',  # 风机风速
    'StorageAva',  # 蓄电池启用 01开关
    'P_Storage_Charge_Set',  # 蓄电池充电功率 0~200000W
    'P_Storage_Discharge_Set',  # 蓄电池放电功率 0~200000W
]
output = [
    'T_FCU_Outlet',  # 空调供风温度
    'F_FCU_Outlet',  # 空调供风风速
    'T_Outdoor',  # 室外温度
    'I_Solar',  # 太阳辐射
    'T_Zone',  # 室内温度
    'T_ITE_Intlet',  # ITE进风温度
    'T_ITE_Outlet'  # ITE出风温度
    'P_AC_Fan',  # 风机能耗
    'P_AC_Coil',  # 盘管能耗
    'P_AC_Coil_CrankcaseHeater',  # 盘管曲轴箱加热器能耗
    'P_ITE_CPU',  # ITE CPU能耗
    'P_ITE_Fan',  # ITE风扇能耗
    'P_ITE_UPS',  # ITE UPS能耗
    'E_Total_Purchased',  # 总的电网买电能耗
    'E_Total_Net',  # 电网净能耗（+买电，-卖电）
    'P_Total_Demand',  # 用电能耗
    'P_Storage_Supply',  # 蓄电池供电功率
    'P_Storage_Draw',  # 蓄电池用电功率
    'E_Storage_SOC',  # 蓄电池电量
    'E_Storage_Charge',  # 蓄电池充电能耗
    'E_Storage_Discharge',  # 蓄电池放电能耗
    'Num_TimeStep'  # 时间步
]

F_max = 0.22972491  # 最大风速(kg/s)
####风速的设置要用一个要小于该最大风速，推荐风速档位 [0.4, 0.75, 1]*F_max

T = np.zeros((5, num_step))
N = np.zeros((2, num_step))
P = np.zeros((7, num_step))
D = np.zeros(num_step)
TW = np.zeros((12, num_step))
try:
    model = pyfmi.load_fmu(fmu, enable_logging=True)
    # names = model.get_model_variables()
    model.initialize(start_time=start_time, stop_time=final_time)
    model.set('Thermostat_Set',30)
    model.set('FCUAva', 1)
    model.set('U_ITE', 1.2)
    # model.set('Fan_Spd', F_max * 0.8)
    for k in range(num_step):
        t_step = start_time + k * step_size
        model.do_step(current_t=t_step, step_size=step_size, new_step=True)
        # FS[k, :] = np.array(model.get(output)).reshape(-1)
        # model.set(input, list(F_stage[k]))
        # T[0, k] = model.get('E_Total_Purchased')
        # print(T[0,k])
        T[0, k] = model.get('T_Zone')
        T[1, k] = model.get('T_ITE_Intlet')
        T[2, k] = model.get('T_ITE_Outlet')
        # T[0, k] = model.get('P_AC_Fan')
        # T[1, k] = model.get('P_AC_Coil')
        # T[2, k] = model.get('P_AC_Coil_CrankcaseHeater')
        # T[3, k] = model.get('T_Outdoor')
        # T[4, k] = model.get('T_FCU_Outlet')
        N[0, k] = model.get('T_FCU_Outlet')
        N[1, k] = model.get('F_FCU_Outlet')
        P[0, k] = model.get('P_AC_Fan')
        P[1, k] = model.get('P_AC_Coil')
        P[2, k] = model.get('P_AC_Coil_CrankcaseHeater')
        # P[3, k] = model.get('P_ITE_CPU')
        # P[4, k] = model.get('P_ITE_Fan')
        # P[5, k] = model.get('P_ITE_UPS')
        # P[6, k] = model.get('E_Storage_SOC')
        D[k] = model.get('Num_TimeStep')

        i = 0
        # for l in ['Inside', 'Outside']:
        #     for d in ['F', 'W', 'N', 'E', 'S', 'C']:
        #         TW[i, k] = model.get('T_Wall' + l + '_' + d)
        #         i += 1
finally:
    # model.free_instance()
    model.terminate()

P_AC = np.sum(P[0:3, :], axis=0)  # 空调总能耗
P_ITE = np.sum(P[3:6, :], axis=0)  # IT设备总能耗
print(T)
print(P)
plt.figure('T')
plt.plot(T.T)
plt.grid()


plt.figure('P')
plt.plot(P[:3].T)
plt.grid()

plt.show()
#
# plt.figure('T_FCU_Outlet')
# plt.plot(N[0])
# plt.grid()
# plt.figure('F_FCU_Outlet')
# plt.plot(N[1])
# plt.grid()
#
# plt.figure('T_zone')
# plt.plot(T[0])
# plt.grid()
