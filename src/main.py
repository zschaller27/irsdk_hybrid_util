import irsdk as sdk
import time
import datetime as date

class iRacing_Client_State:
    ir_connected = False

def check_iracing(ir_client, client_state):
    if client_state.ir_connected and not (ir_client.is_initialized and ir_client.is_connected):
        client_state.ir_connected = False
        ir_client.shutdown()    # Clears all internal variables (doesn't close the listener)
        print("%s ir_client: iRacing Client Disconnected" %date.datetime.now().strftime("%c"))
    elif not client_state.ir_connected and (ir_client.is_initialized and ir_client.is_connected):
        client_state.ir_connected = True
        print("%s ir_client: iRacing Client Connected" %date.datetime.now().strftime("%c"))

