from tkinter import messagebox

import model_builder as model
import irsdk as sdk
import tkinter as tk
import datetime as date
import numpy as np

import time

class iRacing_Client_State:
    ir_connected = False
    window = None
    canvas = None
    light_item = None
    off_colour = "#9b0000"
    on_colour="#00ff00"

def create_light_window(w, h, off_colour="#9b0000"):
    """
    Generate a tkinter window to show a red/green light to indiciate whether to use boost or not.

    Parameters:
        w : the desired width (x-axis) of the window
        h : the desired height (y-axis) of the window
    
    Returns:
        window : the generated tkinter window
        canvas : the widget containing the light rectangle
        canvas_item : the index of which item in the canvas is the light rectangle
    """
    print("%s ir_client: opening tkinter window" %date.datetime.now().strftime("%c"))
    window = tk.Tk()
    canvas = tk.Canvas(window, width=w, height=h)
    canvas.pack()
    canvas_item = canvas.create_rectangle(0, 0, w, h, fill=off_colour)

    # Need to change the close window opperation to do nothing to avoid errors
    # TODO: Update this to be something less janky
    def window_on_close():
        pass
    
    window.protocol("WM_DELETE_WINDOW", window_on_close)

    return window, canvas, canvas_item

def boost_off(client_state):
    """
    Changes the window in the iracing client state to be it's designated on colour.

    Paramters:
        client_state : an instance of the iRacing_Client_State() object
    """
    client_state.canvas.itemconfig(client_state.light_item, fill=client_state.off_colour)
    client_state.window.update()

def boost_on(client_state):
    """
    Changes the window in the iracing client state to be it's designated on colour.

    Paramters:
        client_state : an instance of the iRacing_Client_State() object
    """
    client_state.canvas.itemconfig(client_state.light_item, fill=client_state.on_colour)
    client_state.window.update()
    
def check_iracing(ir_client, client_state):
    """
    Function that updates the state object given an sdk.IRSDK() object.

    Parameters:
        ir_client : An instance of the pyirsdk.IRSDK() object
        client_state : an instance of the iRacing_Client_State() object
    """
    sdk_connected = ir_client.startup()

    if client_state.ir_connected and not (ir_client.is_initialized and ir_client.is_connected):
        client_state.ir_connected = False

        client_state.window.destroy()
        client_state.canvas = None
        client_state.light_item = None

        ir_client.shutdown()    # Clears all internal variables (doesn't close the listener)
        print("%s ir_client: iRacing Client Disconnected" %date.datetime.now().strftime("%c"))
    elif not client_state.ir_connected and sdk_connected and (ir_client.is_initialized and ir_client.is_connected):
        client_state.ir_connected = True
        client_state.window, client_state.canvas, client_state.light_item = create_light_window(500, 500)
        print("%s ir_client: iRacing Client Connected" %date.datetime.now().strftime("%c"))

def close_util(ir_client, client_state):
    """
    Function that destroys and shutsdown everything in the client_state object.

    Parameters:
        ir_client : An instance of the pyirsdk.IRSDK() object
        client_state : an instance of the iRacing_Client_State() object
    """
    client_state.window.destroy()
    ir_client.shutdown()

""" Start of Running Code """
if __name__ == "__main__":
    ## Iniialize Objects ##
    ir = sdk.IRSDK()              # iRacing instance
    state = iRacing_Client_State()  # Information on iRacing Client

    ## Initialize Values ##
    # General Values #
    timeout = 1 / 10     # How long before running the loop again
    features = ["Brake", "EnergyERSBatteryPct", "EnergyMGU_KLapDeployPct", "Speed", \
        "SteeringWheelAngle", "Throttle", "VelocityY", "dcMGUKDeployFixed", "dcMGUKDeployMode", "dcMGUKRegenGain"]

    ## Debug Mode Enables Some Command Line Output to Test Different Code Snippets ##
    debug_mode = True

    if debug_mode:
        state.window, state.canvas, state.light_item = create_light_window(500, 500)

    # Load the nearest neighbor model
    print("Attempting to load model")
    prediction_model = model.getNearestNeighborModel(features)
    print("Model Loaded")

    print("# ----- Utility Initialized ----- #")

    try:
        # Continuously loop until a keyboard interupt is detected
        while True:
            check_iracing(ir, state)    # Update the state object

            # If there is an iracing session predict the hybrid state
            if state.ir_connected and ir["IsOnTrack"]:
                extracted_features = np.reshape(np.array([ir[i] for i in features]), (1, -1))

                if prediction_model.predict(extracted_features)[0] == 1.0:
                    boost_on(state)
                else:
                    boost_off(state)

            ## Test Code ##
            if debug_mode:
                # extracted_features = np.reshape(np.array([ir[i] for i in features]), (1, -1))

                # print(extracted_features)

                # print(prediction_model.predict(extracted_features)[0])
            ## End Test Code ##
            
            # Wait the timeout duration before checking again
            time.sleep(timeout)

    except KeyboardInterrupt:
        if state.ir_connected:
            print("# ----- Shutting Down ----- #")
            close_util(ir, state)