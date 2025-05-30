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
    # Handle common controller name variations
    controller_mapping = {
        'GTController': 'GtController',  # Handle the common case mismatch
        'ModelController': 'ModelController',
        'BaseController': 'BaseController'
    }
    
    controller_name = controller_mapping.get(args.controller, args.controller)
    
    try:
        print(f"Loading controller: {controller_name}")
        controller = eval(controller_name)(env, manipulation, cfg, logger)
    except NameError as e:
        print(f"Controller error: {e}")
        # Fallback to GtController if the specified controller is not found
        print(f"Controller '{args.controller}' not found, falling back to GtController")
        controller = GtController(env, manipulation, cfg, logger)
    except Exception as e:
        print(f"Error creating controller: {e}")
        # Fallback to GtController for any other errors
        print(f"Error with controller '{args.controller}', falling back to GtController")
        controller = GtController(env, manipulation, cfg, logger)
    return controller

def parse_manipulation(args, env, cfg, logger):
    try:
        manipulation = eval(args.manipulation)(env, cfg, logger)
    except NameError as e:
        print(e)
    return manipulation