from film.filmsystem import FilmSystem
from dynamic.dysystem import DySystem
from func import init_input
from postprocess import static_performance_cal
import model
import time
old_time = time.time()

init_args = init_input()
LN_sys = FilmSystem(init_args)
new_time = time.time()
print('计算流场运行时间：', new_time-old_time)

model.hydrostatic_with_newton_and_reynold(LN_sys)
new_time = time.time()
print('计算流场运行时间：', new_time-old_time)

old_time = time.time()
load = static_performance_cal(LN_sys)
print('承载力：', load)
new_time = time.time()
print('计算承载力运行时间：', new_time-old_time)

dysystem = DySystem(LN_sys.pass_dy())
model.dynamic_char(dysystem)
