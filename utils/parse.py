from controller.gtcontroller import GtController
from controller.base_controller import BaseController
from controller.modelcontroller import ModelController

from manipulation.base_manipulation import BaseManipulation
from manipulation.open_bottle import OpenBottleManipulation
from manipulation.open_microwave import OpenMicroWaveManipulation
from manipulation.open_pen import OpenPenManipulation
from manipulation.open_door import OpenDoorManipulation
from manipulation.open_window import OpenWindowManipulation
from manipulation.open_pc import OpenPressureCookerManipulation
from manipulation.open_cm import OpenCoffeeMachineManipulation
from manipulation.open_lamp import OpenLampManipulation
from manipulation.open_safe import OpenSafeManipulation

def parse_controller(args, env, manipulation, cfg, logger):
    try:
        print(args.controller)
        controller = eval(args.controller)(env, manipulation, cfg, logger)
    except NameError as e:
        print(e)
    return controller

def parse_manipulation(args, env, cfg, logger):
    try:
        manipulation = eval(args.manipulation)(env, cfg, logger)
    except NameError as e:
        print(e)
    return manipulation