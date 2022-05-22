from base import System
from func import init_input

filename = ['input', 'orifice_position']
init_args = init_input(filename)
LN_sys = System(init_args)
LN_sys.hydrostatic_film_bearing_with_orifice()
LN_sys.save_output_p()

