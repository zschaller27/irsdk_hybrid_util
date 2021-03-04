# from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout
import irsdk
import time

class iRacing_Client_State:
    ir_connected = False
    last_car_setup_tick = -1

# here we check if we are connected to iracing
# so we can retrieve some data
def check_iracing(ir, state):
    if state.ir_connected and not (ir.is_initialized and ir.is_connected):
        state.ir_connected = False
        # don't forget to reset your State variables
        state.last_car_setup_tick = -1
        # we are shutting down ir library (clearing all internal variables)
        ir.shutdown()
        print('irsdk disconnected')
    elif not state.ir_connected and ir.startup() and ir.is_initialized and ir.is_connected:
        state.ir_connected = True
        print('irsdk connected')

def normalize_straight_fractions(frac_list, straight_epsilon):

    sum = 0
    for elem in frac_list:
        sum += elem[0]
    
    eliminated = [False] * len(frac_list)
    eliminated_sum = sum
    for i, elem in enumerate(frac_list):
        if elem[0] / sum < 0.05:
            eliminated[i] = True
            eliminated_sum -= elem[0]

    for i, elem in enumerate(frac_list):
        if eliminated[i]:
            elem[0] = elem[0] / sum
        else:
            elem[0] = elem[0] / eliminated_sum

    return frac_list

def deploy_hybrid(straights, striaght_epsilon):
    deploy_sum = 0
    for i, straight in enumerate(straights):
        if (lap_deployment * straight[0] >= 100 * straight_epsilon):
            deploy_sum += lap_deployment * straight[0]
            if abs(lap_pct - straight[1]) <= lap_epsilon:
                print("Straight %d Detected. Deploy Up to: %3.1f" %(i, deploy_sum))

if __name__ == '__main__':
    # Initailize ir and state
    ir = irsdk.IRSDK()
    state = iRacing_Client_State()

    ## Initialize Values ##
    # General Values #
    timout = 1/10

    # Epsilon Values #
    # Used to account for small errors in the data
    lap_epsilon = 0.01  # Error value. If abs(x) - abs(y) <= epsilon they will be called equal
    throttle_epsilon = 0.05
    straight_epsilon = 0.05

    # Hybrid Related Values #
    lap_deployment = 100

    # Lap Info #
    lap_count = 0       # Laps done this stint (0 is outlap)
    started_lap = 0
    out_lap = True      # Boolean of if the driver is on outlap

    # Lap Percentages at Full Throttle #
    full_throttle_pcts = []
    straight_fractions = []
    newlap = False
    at_full = (False, 0, 0) # Index 0: active full throttle segment, 1: Start lap percentage, 2: Start speed

    try:
        while True:
            check_iracing(ir, state)

            if state.ir_connected:
                # Freeze the data stream
                ir.freeze_var_buffer_latest()

                if ir['IsOnTrack'] and ir['OnPitRoad']:
                    if lap_count > 1:
                        print("All Values Reset")
                    lap_count = 0
                    full_throttle_pcts = []
                    out_lap = True
                    at_full = (False, 0, 0)
                elif ir['IsOnTrack']:
                    if not out_lap:
                        lap_pct = ir['LapDistPct']
                        throttle_pos = ir['Throttle']
                        speed = ir['Speed']

                        # Check for deployment
                        if not out_lap and len(straight_fractions) > 0:
                            deploy_hybrid(straight_fractions, straight_epsilon) 

                        # Gather Straight Data
                        if 1 - throttle_pos <= throttle_epsilon and not at_full[0]:
                            at_full = (True, lap_pct, speed)
                            # print("Started a Full Throttle Segment at: %1.3f with speed %3.1f" %(lap_pct, speed))
                        
                        if 1 - ir['Throttle'] > throttle_epsilon and at_full[0]:
                            
                            if lap_pct < at_full[1]:
                                full_throttle_pct = 1 - at_full[1] + lap_pct
                            else:
                                full_throttle_pct = lap_pct - at_full[1]

                            full_throttle_pcts.append((full_throttle_pct, at_full[1], at_full[2]))
                            
                            at_full = (False, 0, 0)

                            # print("Started a Full Throttle Segment:", (full_throttle_pct, at_full[1], at_full[2]))
                    
                    # Check for new lap
                    if ir['Lap'] != None and started_lap < ir['Lap']:
                        started_lap += 1
                        lap_count += 1
                        newlap = True
                    
                    # Check if new lap and the start/finish straight is over
                    # The start/finish straight is now added to the array and the lap can be "finished"
                    if newlap and not at_full[0]:
                        print("New Lap\tLap Count: %d" % lap_count)
                        print("\tFull Throttle Pcts Length:", len(full_throttle_pcts))
                        for i, entry in enumerate(full_throttle_pcts):
                            print("\tStraight %d:\n\t\tLength: %1.3f\n\t\tStart: %1.3f\n\t\tEntry Speed: %3.1f" %(i, entry[0], entry[1], entry[2]))
                        
                        # Update straight_fractions
                        if len(full_throttle_pcts) != 0 and (len(straight_fractions) == 0 or len(straight_fractions) != len(full_throttle_pcts)):
                            straight_fractions = []
                            for i in range(len(full_throttle_pcts)):
                                straight_fractions.append([0,0])

                        for i in range(len(straight_fractions)):
                            if straight_fractions[i][0] == 0:
                                factor = 1
                            else:
                                factor = 2

                            straight_fractions[i][0] += full_throttle_pcts[i][0]
                            straight_fractions[i][0] = straight_fractions[i][0] / factor

                            straight_fractions[i][1] += full_throttle_pcts[i][1]
                            straight_fractions[i][1] = straight_fractions[i][1] / factor
                        
                        straight_fractions = normalize_straight_fractions(straight_fractions, straight_epsilon)

                        newlap = False
                        full_throttle_pcts = []
                        # lap_count += 1

                        # Update out_lap status
                        if out_lap:
                            out_lap = False


            # Update 10 times per second (can go up to 60)
            time.sleep(timout)
    except KeyboardInterrupt:
        # press ctrl+c to exit
        pass
