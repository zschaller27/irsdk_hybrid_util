import irsdk
import time

# from hybrid_deploy import Deploy_Hybrid
# from hybrid_track import Track

"""
iRacing_Client_State:
Description: Class holds all information for local iRacing client
"""
class iRacing_Client_State:
    ir_connected = False
    track_trainer = None
    deploy_op = None
    out_lap = False
    started_laps = 0
    new_lap_started = False
    throttle_pcts = []
    on_straight = (False, 0, 0)

"""
check_iracing:
Description: Checks the current status of local iRacing client.
Parameters:
    ir - an irsdk object used to check for local iRacing client activity.
    state - an iRacing_Client_State object
"""
def check_iracing(ir, state):
    if state.ir_connected and not (ir.is_initialized and ir.is_connected):
        state.ir_connected = False
        # don't forget to reset your State variables
        state.track_trainer.shutdown()
        state.track_trainer = None
        state.deploy_op = None
        # we are shutting down ir library (clearing all internal variables)
        ir.shutdown()
        print('irsdk disconnected')
    elif not state.ir_connected and ir.startup() and ir.is_initialized and ir.is_connected:
        state.ir_connected = True
        # state.track_trainer = Track(ir['WeekendInfo']['TrackName'])
        # state.deploy_op = Deploy_Hybrid(state.track_trainer)
        state.started_laps = ir['Lap']
        print('irsdk connected')

if __name__ == '__main__':
    ## Iniialize Objects ##
    ir = irsdk.IRSDK()              # iRacing instance
    state = iRacing_Client_State()  # Information on iRacing Client

    ## Initialize Values ##
    # General Values #
    timout = 1 / 10
    # Epsilon Values #
    throttle_epsilon = 0.05

    # ---- Test Initialized Values ---- #
    # ---- End of Test Initialized Values ---- #

    try:
        while True:
            check_iracing(ir, state)

            print(state.ir_connected)

            # # Check if iRacing is connected
            # if state.ir_connected:
            #     # Freeze the data stream
            #     ir.freeze_var_buffer_latest()

            #     ## Check for Hybrid Deployment Amount ##
            #     if ir['IsOnTrack'] and state.track_trainer.get_lap_count() > 0:
            #         state.deploy_op.check_deploy(ir['LapDistPct'], ir['EnergyMGU_KLapDeployPct'])

            #     ## Gather Straight Information ##
            #     # Look for start of straight
            #     if 1 - ir['Throttle'] <= throttle_epsilon and not state.on_straight[0]:
            #         state.on_straight = (True, ir['LapDistPct'], ir['Speed'])
            #         # print("Started a Full Throttle Segment at: %1.3f with speed %3.1f" %(lap_pct, speed))

            #     # Look for end of straight
            #     if 1 - ir['Throttle'] > throttle_epsilon and state.on_straight[0]:
                    
            #         if ir['LapDistPct'] < state.on_straight[1]:
            #             full_throttle_pct = 1 - state.on_straight[1] + ir['LapDistPct']
            #         else:
            #             full_throttle_pct = ir['LapDistPct'] - state.on_straight[1]

            #         state.throttle_pcts.append((state.on_straight[1], full_throttle_pct, state.on_straight[2]))
                    
            #         state.on_straight = (False, 0, 0)

            #     ## Lap Information ##
            #     # Check if in pitlane (next lap is an out lap)
            #     if ir['IsOnTrack'] and ir['OnPitRoad']:
            #         state.out_lap = True
                
            #     # Check for driver crossing start finish line
            #     if ir['Lap'] != None and ir['IsOnTrack'] and state.started_laps < ir['Lap']:
            #         # ---- Test Print Statement ---- #
            #         print("Started New Lap")

            #         state.started_laps += 1
            #         state.new_lap_started = True
                
            #     # Check if out lap has finished
            #     if not ir['OnPitRoad'] and state.new_lap_started and state.out_lap:
            #         state.new_lap_started = False
            #         state.out_lap = False
                
            #     # Check if flying lap has finished
            #     if not ir['OnPitRoad'] and state.new_lap_started and not state.on_straight[0]:
            #         # Make sure that there was a straight
            #         if len(state.throttle_pcts) > 0:
            #             state.track_trainer.new_lap(state.throttle_pcts)

            #             # ---- Test Print Statement ---- #
            #             state.track_trainer.print_fracs()

            #             state.deploy_op.update()

            #         # Reset lap values
            #         state.new_lap_started = False
            #         state.throttle_pcts = []


            # Update 10 times per second (can go up to 60)
            time.sleep(timout)
    except KeyboardInterrupt:
        if state.ir_connected:
            state.track_trainer.shutdown()
        
        # press ctrl+c to exit
        pass
