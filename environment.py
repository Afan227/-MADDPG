import pandas as pd
import numpy as np
import pyfmi
import matplotlib.pyplot as plt
import datetime

t_interval = 10  # 时间粒度 (min)
hourDivide = int(60 / t_interval)  # 每小时时间步数，按时间粒度分割
start_time = 0
final_time = 60 * 60 * 24 * 365  # (s)
step_size = 60 * t_interval  # (s)
num_step = int((final_time - start_time) / step_size)
P_battery = 4800
fmu = 'model_fmu0221.fmu'

# model = pyfmi.load_fmu(fmu, enable_logging=True)
# var = model.get_model_variables()

class HVAC():
    def __init__(self):
        self.t_interval = t_interval  # 时间粒度 (min)
        hourDivide = int(60 / self.t_interval)  # 每小时时间步数，按时间粒度分割
        self.step_size = 60 * t_interval  # (s)
        self.step = 0
        self.U_ITE = 1.2
        # 将字符串类型的时间转换为 datetime 类型
        start = '2006-01-01 00:00:00'
        final = '2006-12-30 00:00:00'
        # 计算时间戳
        start_obj = datetime.datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
        final_obj = datetime.datetime.strptime(final, '%Y-%m-%d %H:%M:%S')
        # 提取起始时间戳的月份与日期
        start_month = int(start_obj.strftime('%m'))
        start_day = int(start_obj.strftime('%d'))
        # 计算这一年的第几个小时
        # self.start_time = (start_obj - datetime.datetime(start_obj.year, 1, 1)).total_seconds()
        # self.final_time = (final_obj - datetime.datetime(final_obj.year, 1, 1)).total_seconds()
        self.start_time = 0
        self.final_time = 60 * 60 * 24 * 365
        # # 计算相差的小时数
        num_step = (self.final_time - self.start_time) / self.step_size
        fmu = 'model_fmu_0303.fmu'
        self.model = pyfmi.load_fmu(fmu)
        self.model.initialize(start_time=self.start_time, stop_time=self.final_time)
        self.P_list = []

    def HVAC_step(self, state,input):
        """ input = actions [action1, action2] action1 = [17- 29] action2 = [-1,1]"""
        T1 = []  #
        T2 = []
        T = []
        reward =[0,0]
        # print(input)
        in_ = input[0]*7 +23
        #print(f'in{in_}')
        self.model.set('Thermostat_Set', in_)
        self.model.set('FCUAva', 1)
        self.model.set('U_ITE', self.U_ITE)
        b_set = [0,0,0]
        if -0.05 <= input[1] <= 0.05:
            b_set = b_set
        elif -1 <= input[1] < -0.05:
            b_set = [1,0,-input[1]]
        elif 0.05 < input[1] <= 1:
            b_set = [1,input[1],0]
        b_set = [0,0,0]


        self.model.set('StorageAva', b_set[0])
        self.model.set('P_Storage_Charge_Set', b_set[1]*P_battery)
        self.model.set('P_Storage_Discharge_Set', b_set[2]*P_battery)

        self.model.do_step(current_t=self.start_time + self.step * self.step_size, step_size=self.step_size,
                           new_step=True)
        day_hour = int((self.step % (24 * 6)) / 6)  # 判断k属于一天中的哪个时段
        price = price_level(day_hour)
        # 空调状态
        T1.append(day_hour/23)  # 时间[0,23]
        #T.append(price)
        T1.append((self.model.get('T_Zone').item()-23)/7)  # 室内温度[10,30]
        T1.append((self.model.get('P_ITE_CPU').item() + self.model.get('P_ITE_UPS').item() + self.model.get('P_ITE_Fan').item())/2000)   # IT设备功率[100,2000]
        T1.append(float(input[1]))   # 蓄电池充放电功率[0,4000]
        T1.append((self.model.get('T_Outdoor').item()+5)/40)   # 室外温度[-5,35]
        T1.append(self.model.get('I_Solar').item()/800)     # 太阳辐射[0,800]
        # 蓄电池状态
        T2.append(day_hour/23)  # 时间
        T2.append(self.model.get('E_Storage_SOC').item()/17280000)  # 蓄电池电量[0,17280000]
        T2.append(self.model.get('T_Zone').item()/20-0.5)  # 室内温度
        T2.append((self.model.get('P_ITE_CPU').item() + self.model.get('P_ITE_UPS').item() + self.model.get(
              'P_ITE_Fan').item())/2000)  # IT设备功率
        # T.append(price)


        # T_Outdoor = self.model.get('T_Outdoor')
        T_Zone = self.model.get('T_Zone').item()
        # T_ITE_Intlet = self.model.get('T_ITE_Intlet')
        T_ITE_Outlet = self.model.get('T_ITE_Outlet').item()
        # T_FCU_Outlet = self.model.get('T_FCU_Outlet')
        P_AC_Fan = self.model.get('P_AC_Fan')
        P_AC_Coil = self.model.get('P_AC_Coil')
        P_AC_Coil_CrankcaseHeater = self.model.get('P_AC_Coil_CrankcaseHeater')
        # P_ITE_CPU = self.model.get('P_ITE_CPU')
        # P_ITE_Fan = self.model.get('P_ITE_Fan')
        # P_ITE_UPS = self.model.get('P_ITE_UPS')
        E_Storage_SOC = self.model.get('E_Storage_SOC')  # 蓄电池电量

        P_AC_All = P_AC_Fan.item() + P_AC_Coil.item() + P_AC_Coil_CrankcaseHeater.item()


        E_Total_Purchased = self.model.get('E_Total_Purchased')
        # print(P_ITE_UPS)
        self.step += 1
        energycost =   - ((E_Total_Purchased.item()) * price)/3600000
        #(E_Total_Purchased.item())
        reward[0] = energycost
        reward[1] = energycost


        # if T_ITE_Outlet > 47 :
        #     reward[0] +=  0.25 * energycost
        #     #print(T_ITE_Outlet)
        #     #print(f'ACACAC{(-45 + T_ITE_Outlet)**2}')
        # if T_Zone > 29.7 :
        #     reward[0] +=  0.1 * energycost

        # if (state[0][0]*23 > 8) and (state[0][1] < 0.286) and (in_ < 23):
        #     reward[0] += 0.05 * energycost
        #     print(state[0][1])
        #     print(in_)
        if (state[1][1] * 17280000 == 17280000) and input[1] > 0:
            reward[1] +=  0.25 * energycost
        if (state[1][1] * 17280000 == 4320000) and input[1] < 0 :
            reward[1] +=  0.25 * energycost
        # if T1[0] * 23 <= 8 and -1 <= input[1] < -0.05 :
        #     reward[1] -= 5
        # elif T1[0] * 23 > 8 and 0.05 < input[1] <= 1:
        #     reward[1] -= 1
        # else :
        #     reward[1] += 0
        # if reward[0] < -100000 or reward[1] < -100000:
        #     print('333')
        #print(P_ITE_UPS)
        # print(self.step)
        # outputs_t = np.array([day_hour, price,T_FCU_Outlet.item(),T_Zone.item(), T_Outdoor.item(), T_ITE_Outlet.item(),T_ITE_Intlet.item(),P_AC_Fan.item(), P_AC_Coil.item(),
        #                       P_AC_Coil_CrankcaseHeater.item(),
        #                       P_ITE_CPU.item(), P_ITE_Fan.item(), P_ITE_UPS.item(),P_ITE_Fan.item()])
        #print(f'energycost:{energycost}')
        T.append(T1)
        T.append(T2)
        # print(f'output{outputs_t}')
        # self.P_list.append(outputs_t)
        print(f'动作{in_}')
        print(f'zhuangtai{T1}')
        print(f'电池{E_Total_Purchased}')
        print(reward[0])
        print(reward[1])


        return T, reward,[False,False],energycost,E_Total_Purchased.item()

    def terminate(self):
        self.model.terminate()

    def reset(self):
        T=[]
        T1 = [0, 0.6938,0.5025,0,0.08,0]
        T2 = [0,1,0.6938,0.5025]
        T.append(T1)
        T.append(T2)
        return T

    def rreset(self):
        self.model.terminate()
        model_temp = self.model
        self.model = None
        del model_temp
        # self.model.free_instance()
        # self.__init__(10)
        fmu = 'model_fmu_0303.fmu'
        self.model = pyfmi.load_fmu(fmu)
        self.model.initialize(start_time=self.start_time, stop_time=self.final_time)
        # self.model.initialize(start_time=self.start_time, stop_time=self.final_time)
        self.step = 0

    def return_P(self):
        outputs_set = pd.DataFrame(self.P_list,
                                   columns=['day_hour', 'price', 'T_FCU_Outlet','T_Zone','T_Outdoor','T_ITE_Outlet','T_ITE_Intlet', 'P_AC_Fan',
                                            'P_AC_Coil',
                                            'P_AC_Coil_CrankcaseHeater',
                                            'P_ITE_CPU', 'P_ITE_Fan', 'P_ITE_UPS','P_ITE_Fan'
                                            ])
        outputs_set.to_csv(
            f'./output_file/Initial stage outputs/every_P_list_1024_episode153_cop3_reward.csv')
        return self.P_list

def price_level(hour):
    if 0 <= hour < 8:
        level = 0.3363
    elif (8 <= hour < 9) or (12 <= hour < 19) or (22 <= hour < 24):
        level = 0.6725
    else:
        level = 1.1096
    return level

