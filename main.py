### Imports, Initialize Global Vars
from main_helpers import get_motor_groups
from main_helpers import project_point_onto_line_segment
ser, motor_group, weapon_motor_group = get_motor_groups(False, 1, 3, 4)
running = True
### YOLO Obj Detection init
# Select limbs, set low/high values
# Dummy values for low/high
limb_inits = {"Speed-low": [-200, -100], 
                "Speed-high": [200, 100], 
                "Turn-low":[-200, -100], 
                "Turn-high": [200, 100]
                }

while running:
    
    ### YOLO obj detection main
    # Output selected limb positions
    # Dummy values for current positions
    current_limbs = {"Speed-mover": [200,300], 
                    "Speed-base": [400,400], 
                    "Turn-mover": [400, 500], 
                    "Turn-base": [600, 600]
    }
    
    ### Translate limb positions to Speed, turn values
    relative_limbs = {
        "Speed-rel": [current_limbs["Speed-base"] - current_limbs["Speed-mover"]],
        "Turn-rel": [current_limbs["Turn-base"] - current_limbs["Turn-mover"]]
    }
    
    # Convert 0-1 range to -1 to 1
    speed = project_point_onto_line_segment(relative_limbs["Speed-rel"], limb_inits["Speed-low"], limb_inits["Speed-high"]) * 2 - 1
    turn = project_point_onto_line_segment(relative_limbs["Turn-rel"], limb_inits["Turn-low"], limb_inits["Turn-high"]) * 2 - 1


    ### Transmit Values to Robot
    if turn * -1 > 0:
        motor_group.move(speed * 0.8, turn * -1 * 0.55 + 0.2)
    else:
        motor_group.move(speed * 0.8, turn * -1 * 0.55 - 0.2)
        
    running = False
