from base import System
from func import init_input,load_capacity_cal_sp
from postprocess import static_performance_cal
import model
import time

filename = ['input', 'orifice_position']
init_args = init_input(filename)
LN_sys = System(init_args)
old_time = time.time()
model.hydrostatic_with_newton_and_reynold(LN_sys)
new_time = time.time()
print('运行时间：', new_time-old_time)
old_time = time.time()
load = static_performance_cal(LN_sys)
print('承载力：', load)
new_time = time.time()
print('运行时间：', new_time-old_time)


