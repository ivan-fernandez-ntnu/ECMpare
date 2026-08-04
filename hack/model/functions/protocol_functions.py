import gc
import math
import re
import pybamm

import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pandas as pd

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from pathlib import Path

def generate_experiment(protocol_name, fixed_period = False, period_value = 0.01):
    """
    This function return the experiment protocol.
    
    :param protocol_name: Name of the protocol: HPPC, GITT, ICI, ICA, etc.
    :param fixed_period: Default is False. The period is define to each step to reduce the memory use, but it could be the same for all of them.
    :param period_value: The period used to all the steps in case fixed_period = True
    """
    if protocol_name == "HPPC":
        if fixed_period:
            experiment = pybamm.Experiment(
                [
                    pybamm.step.string("Rest for 30 minutes"),
                    pybamm.step.string("Discharge at 1C until 2.5 V"),
                    pybamm.step.string("Hold at 2.5 V until C/50", direction="discharge"),
                    pybamm.step.string("Rest for 30 minutes"),
                    pybamm.step.string("Charge at 1C until 4.2 V"),
                    pybamm.step.string("Hold at 4.2 V until C/50"),
                    pybamm.step.string("Rest for 30 minutes"),
                    pybamm.step.string("Hold at 4.2 V until C/50"),
                    pybamm.step.string("Rest for 30 minutes"),
                    pybamm.step.string("Discharge at 1C for 15 seconds"),
                    pybamm.step.string("Rest for 15 seconds"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Discharge at 0.5C for 0.2 hours"), # Discharge from 100.00 %SOC to 90.00
                    pybamm.step.string("Rest for 30 minutes"),
                    pybamm.step.string("Discharge at 1C for 15 seconds"),
                    pybamm.step.string("Rest for 15 seconds"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Charge at 1C for 15 seconds"),
                    pybamm.step.string("Rest for 15 seconds"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Discharge at 0.5C for 0.2 hours"), # Discharge from 90.00 %SOC to 80.00
                    pybamm.step.string("Rest for 30 minutes"),
                    pybamm.step.string("Discharge at 1C for 15 seconds"),
                    pybamm.step.string("Rest for 15 seconds"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Charge at 1C for 15 seconds"),
                    pybamm.step.string("Rest for 15 seconds"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Discharge at 0.5C for 0.2 hours"), # Discharge from 80.00 %SOC to 70.00
                    pybamm.step.string("Rest for 30 minutes"),
                    pybamm.step.string("Discharge at 1C for 15 seconds"),
                    pybamm.step.string("Rest for 15 seconds"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Charge at 1C for 15 seconds"),
                    pybamm.step.string("Rest for 15 seconds"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Discharge at 0.5C for 0.2 hours"), # Discharge from 70.00 %SOC to 60.00
                    pybamm.step.string("Rest for 30 minutes"),
                    pybamm.step.string("Discharge at 1C for 15 seconds"),
                    pybamm.step.string("Rest for 15 seconds"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Charge at 1C for 15 seconds"),
                    pybamm.step.string("Rest for 15 seconds"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Discharge at 0.5C for 0.2 hours"), # Discharge from 60.00 %SOC to 50.00
                    pybamm.step.string("Rest for 30 minutes"),
                    pybamm.step.string("Discharge at 1C for 15 seconds"),
                    pybamm.step.string("Rest for 15 seconds"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Charge at 1C for 15 seconds"),
                    pybamm.step.string("Rest for 15 seconds"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Discharge at 0.5C for 0.2 hours"), # Discharge from 50.00 %SOC to 40.00
                    pybamm.step.string("Rest for 30 minutes"),
                    pybamm.step.string("Discharge at 1C for 15 seconds"),
                    pybamm.step.string("Rest for 15 seconds"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Charge at 1C for 15 seconds"),
                    pybamm.step.string("Rest for 15 seconds"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Discharge at 0.5C for 0.2 hours"), # Discharge from 40.00 %SOC to 30.00
                    pybamm.step.string("Rest for 30 minutes"),
                    pybamm.step.string("Discharge at 1C for 15 seconds"),
                    pybamm.step.string("Rest for 15 seconds"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Charge at 1C for 15 seconds"),
                    pybamm.step.string("Rest for 15 seconds"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Discharge at 0.5C for 0.2 hours"), # Discharge from 30.00 %SOC to 20.00
                    pybamm.step.string("Rest for 30 minutes"),
                    pybamm.step.string("Discharge at 1C for 15 seconds"),
                    pybamm.step.string("Rest for 15 seconds"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Charge at 1C for 15 seconds"),
                    pybamm.step.string("Rest for 15 seconds"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Discharge at 0.5C for 0.2 hours or until 2.5 V"), # Discharge from 20.00 %SOC to 10.00
                    pybamm.step.string("Rest for 30 minutes"),
                    pybamm.step.string("Discharge at 1C for 15 seconds or until 2.5V"),
                    pybamm.step.string("Rest for 15 seconds"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Charge at 1C for 15 seconds"),
                    pybamm.step.string("Rest for 15 seconds"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Discharge at 0.5C until 2.5 V"),
                    pybamm.step.string("Hold at 2.5 V until C/20", direction="discharge"),
                    pybamm.step.string("Rest for 30 minutes"),
                    pybamm.step.string("Charge at 1C for 15 seconds"),
                    pybamm.step.string("Rest for 15 seconds"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Charge at 0.5C for 0.2 hours"), # Charge from 0 %SOC to 10 %SOC
                    pybamm.step.string("Rest for 30 minutes"),
                    pybamm.step.string("Discharge at 1C for 15 seconds"),
                    pybamm.step.string("Rest for 15 seconds"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Charge at 1C for 15 seconds"),
                    pybamm.step.string("Rest for 15 seconds"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Charge at 0.5C for 0.2 hours"), # Charge from 10 %SOC to 20 %SOC
                    pybamm.step.string("Rest for 30 minutes"),
                    pybamm.step.string("Discharge at 1C for 15 seconds"),
                    pybamm.step.string("Rest for 15 seconds"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Charge at 1C for 15 seconds"),
                    pybamm.step.string("Rest for 15 seconds"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Charge at 0.5C for 0.2 hours"), # Charge from 20 %SOC to 30 %SOC
                    pybamm.step.string("Rest for 30 minutes"),
                    pybamm.step.string("Discharge at 1C for 15 seconds"),
                    pybamm.step.string("Rest for 15 seconds"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Charge at 1C for 15 seconds"),
                    pybamm.step.string("Rest for 15 seconds"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Charge at 0.5C for 0.2 hours"), # Charge from 30 %SOC to 40 %SOC
                    pybamm.step.string("Rest for 30 minutes"),
                    pybamm.step.string("Discharge at 1C for 15 seconds"),
                    pybamm.step.string("Rest for 15 seconds"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Charge at 1C for 15 seconds"),
                    pybamm.step.string("Rest for 15 seconds"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Charge at 0.5C for 0.2 hours"), # Charge from 40 %SOC to 50 %SOC
                    pybamm.step.string("Rest for 30 minutes"),
                    pybamm.step.string("Discharge at 1C for 15 seconds"),
                    pybamm.step.string("Rest for 15 seconds"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Charge at 1C for 15 seconds"),
                    pybamm.step.string("Rest for 15 seconds"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Charge at 0.5C for 0.2 hours"), # Charge from 50 %SOC to 60 %SOC
                    pybamm.step.string("Rest for 30 minutes"),
                    pybamm.step.string("Discharge at 1C for 15 seconds"),
                    pybamm.step.string("Rest for 15 seconds"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Charge at 1C for 15 seconds"),
                    pybamm.step.string("Rest for 15 seconds"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Charge at 0.5C for 0.2 hours"), # Charge from 60 %SOC to 70 %SOC
                    pybamm.step.string("Rest for 30 minutes"),
                    pybamm.step.string("Discharge at 1C for 15 seconds"),
                    pybamm.step.string("Rest for 15 seconds"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Charge at 1C for 15 seconds or until 4.2 V"),
                    pybamm.step.string("Rest for 15 seconds"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Charge at 0.5C for 0.2 hours or until 4.2 V"), # Charge from 70 %SOC to 80 %SOC
                    pybamm.step.string("Rest for 30 minutes"),
                    pybamm.step.string("Discharge at 1C for 15 seconds"),
                    pybamm.step.string("Rest for 15 seconds"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Charge at 1C for 15 seconds or until 4.2 V"),
                    pybamm.step.string("Rest for 15 seconds"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Charge at 0.5C for 0.2 hours or until 4.2 V"), # Charge from 80 %SOC to 90 %SOC
                    pybamm.step.string("Rest for 30 minutes"),
                    pybamm.step.string("Discharge at 1C for 15 seconds"),
                    pybamm.step.string("Rest for 15 seconds"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Charge at 1C for 15 seconds or until 4.2 V"),
                    pybamm.step.string("Rest for 15 seconds"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Charge at 0.5C until 4.2 V"),
                    pybamm.step.string("Hold at 4.2 V until C/50"),
                    pybamm.step.string("Rest for 30 minutes"),
                    pybamm.step.string("Discharge at 1C for 15 seconds"),
                    pybamm.step.string("Rest for 15 seconds"),
                    pybamm.step.string("Rest for 10 minutes"),
                ],
                period=f"{period_value} seconds",   # output sampling
            )
        else:
            experiment = pybamm.Experiment(
                [
                    pybamm.step.string("Rest for 30 minutes", period="10 seconds"),
                    pybamm.step.string("Discharge at 1C until 2.5 V", period="1 seconds"),
                    pybamm.step.string("Hold at 2.5 V until C/50", direction="discharge", period="1 seconds"),
                    pybamm.step.string("Rest for 30 minutes", period="10 seconds"),
                    pybamm.step.string("Charge at 1C until 4.2 V", period="1 seconds"),
                    pybamm.step.string("Hold at 4.2 V until C/50", period="1 seconds"),
                    pybamm.step.string("Rest for 30 minutes", period="10 seconds"),
                    pybamm.step.string("Hold at 4.2 V until C/50", period="1 seconds"),
                    pybamm.step.string("Rest for 30 minutes", period="10 seconds"),                 # Resting at 100% SOC
                    pybamm.step.string("Discharge at 1C for 15 seconds", period="0.001 seconds"),   # Discharge Pulse
                    pybamm.step.string("Rest for 15 seconds", period="0.001 seconds"),
                    pybamm.step.string("Rest for 10 minutes", period="10 seconds"),   
                    pybamm.step.string("Discharge at 0.5C for 0.2 hours", period="0.5 seconds"),      # Discharge until 10% DOD
                    pybamm.step.string("Rest for 30 minutes", period="10 seconds"),                 # Resting at 90% SOC   
                    pybamm.step.string("Discharge at 1C for 15 seconds", period="0.001 seconds"),   # Discharge Pulse
                    pybamm.step.string("Rest for 15 seconds", period="0.001 seconds"),
                    pybamm.step.string("Rest for 10 minutes", period="10 seconds"),
                    pybamm.step.string("Charge at 1C for 15 seconds", period="0.001 seconds"),      # Charge pulse
                    pybamm.step.string("Rest for 15 seconds", period="0.001 seconds"),
                    pybamm.step.string("Rest for 10 minutes", period="10 seconds"),        
                    pybamm.step.string("Discharge at 0.5C for 0.2 hours", period="0.5 seconds"),      # Discharge until 10% DOD
                    pybamm.step.string("Rest for 30 minutes", period="10 seconds"),                 # Resting at 80% SOC   
                    pybamm.step.string("Discharge at 1C for 15 seconds", period="0.001 seconds"),   # Discharge Pulse
                    pybamm.step.string("Rest for 15 seconds", period="0.001 seconds"),
                    pybamm.step.string("Rest for 10 minutes", period="10 seconds"),
                    pybamm.step.string("Charge at 1C for 15 seconds", period="0.001 seconds"),      # Charge pulse
                    pybamm.step.string("Rest for 15 seconds", period="0.001 seconds"),
                    pybamm.step.string("Rest for 10 minutes", period="10 seconds"),        
                    pybamm.step.string("Discharge at 0.5C for 0.2 hours", period="0.5 seconds"),      # Discharge until 10% DOD
                    pybamm.step.string("Rest for 30 minutes", period="10 seconds"),                 # Resting at 70% SOC   
                    pybamm.step.string("Discharge at 1C for 15 seconds", period="0.001 seconds"),   # Discharge Pulse
                    pybamm.step.string("Rest for 15 seconds", period="0.001 seconds"),
                    pybamm.step.string("Rest for 10 minutes", period="10 seconds"),
                    pybamm.step.string("Charge at 1C for 15 seconds", period="0.001 seconds"),      # Charge pulse
                    pybamm.step.string("Rest for 15 seconds", period="0.001 seconds"),
                    pybamm.step.string("Rest for 10 minutes", period="10 seconds"),        
                    pybamm.step.string("Discharge at 0.5C for 0.2 hours", period="0.5 seconds"),      # Discharge until 10% DOD
                    pybamm.step.string("Rest for 30 minutes", period="10 seconds"),                 # Resting at 60% SOC   
                    pybamm.step.string("Discharge at 1C for 15 seconds", period="0.001 seconds"),   # Discharge Pulse
                    pybamm.step.string("Rest for 15 seconds", period="0.001 seconds"),
                    pybamm.step.string("Rest for 10 minutes", period="10 seconds"),
                    pybamm.step.string("Charge at 1C for 15 seconds", period="0.001 seconds"),      # Charge pulse
                    pybamm.step.string("Rest for 15 seconds", period="0.001 seconds"),
                    pybamm.step.string("Rest for 10 minutes", period="10 seconds"),        
                    pybamm.step.string("Discharge at 0.5C for 0.2 hours", period="0.5 seconds"),      # Discharge until 10% DOD
                    pybamm.step.string("Rest for 30 minutes", period="10 seconds"),                 # Resting at 50% SOC   
                    pybamm.step.string("Discharge at 1C for 15 seconds", period="0.001 seconds"),   # Discharge Pulse
                    pybamm.step.string("Rest for 15 seconds", period="0.001 seconds"),
                    pybamm.step.string("Rest for 10 minutes", period="10 seconds"),
                    pybamm.step.string("Charge at 1C for 15 seconds", period="0.001 seconds"),      # Charge pulse
                    pybamm.step.string("Rest for 15 seconds", period="0.001 seconds"),
                    pybamm.step.string("Rest for 10 minutes", period="10 seconds"),        
                    pybamm.step.string("Discharge at 0.5C for 0.2 hours", period="0.5 seconds"),      # Discharge until 10% DOD
                    pybamm.step.string("Rest for 30 minutes", period="10 seconds"),                 # Resting at 40% SOC   
                    pybamm.step.string("Discharge at 1C for 15 seconds", period="0.001 seconds"),   # Discharge Pulse
                    pybamm.step.string("Rest for 15 seconds", period="0.001 seconds"),
                    pybamm.step.string("Rest for 10 minutes", period="10 seconds"),
                    pybamm.step.string("Charge at 1C for 15 seconds", period="0.001 seconds"),      # Charge pulse
                    pybamm.step.string("Rest for 15 seconds", period="0.001 seconds"),
                    pybamm.step.string("Rest for 10 minutes", period="10 seconds"),        
                    pybamm.step.string("Discharge at 0.5C for 0.2 hours", period="0.5 seconds"),      # Discharge until 10% DOD
                    pybamm.step.string("Rest for 30 minutes", period="10 seconds"),                 # Resting at 30% SOC   
                    pybamm.step.string("Discharge at 1C for 15 seconds", period="0.001 seconds"),   # Discharge Pulse
                    pybamm.step.string("Rest for 15 seconds", period="0.001 seconds"),
                    pybamm.step.string("Rest for 10 minutes", period="10 seconds"),
                    pybamm.step.string("Charge at 1C for 15 seconds", period="0.001 seconds"),      # Charge pulse
                    pybamm.step.string("Rest for 15 seconds", period="0.001 seconds"),
                    pybamm.step.string("Rest for 10 minutes", period="10 seconds"),        
                    pybamm.step.string("Discharge at 0.5C for 0.2 hours", period="0.5 seconds"),      # Discharge until 10% DOD
                    pybamm.step.string("Rest for 30 minutes", period="10 seconds"),                 # Resting at 20% SOC   
                    pybamm.step.string("Discharge at 1C for 15 seconds", period="0.001 seconds"),   # Discharge Pulse
                    pybamm.step.string("Rest for 15 seconds", period="0.001 seconds"),
                    pybamm.step.string("Rest for 10 minutes", period="10 seconds"),
                    pybamm.step.string("Charge at 1C for 15 seconds", period="0.001 seconds"),      # Charge pulse
                    pybamm.step.string("Rest for 15 seconds", period="0.001 seconds"),
                    pybamm.step.string("Rest for 10 minutes", period="10 seconds"),        
                    pybamm.step.string("Discharge at 0.5C for 0.2 hours", period="0.5 seconds"),      # Discharge until 10% DOD
                    pybamm.step.string("Rest for 30 minutes", period="10 seconds"),                 # Resting at 10% SOC   
                    pybamm.step.string("Discharge at 1C for 15 seconds", period="0.001 seconds"),   # Discharge Pulse
                    pybamm.step.string("Rest for 15 seconds", period="0.001 seconds"),
                    pybamm.step.string("Rest for 10 minutes", period="10 seconds"),
                    pybamm.step.string("Charge at 1C for 15 seconds", period="0.001 seconds"),      # Charge pulse
                    pybamm.step.string("Rest for 15 seconds", period="0.001 seconds"),
                    pybamm.step.string("Rest for 10 minutes", period="10 seconds"),        
                    pybamm.step.string("Discharge at 0.5C until 2.5 V", period="0.5 seconds"),      # Discharge until 10% DOD
                    pybamm.step.string("Hold at 2.5 V until C/50", direction="discharge", period="1 seconds"),
                    pybamm.step.string("Rest for 30 minutes", period="10 seconds"),                 # Resting at 0% SOC   
                    pybamm.step.string("Charge at 1C for 15 seconds", period="0.001 seconds"),      # Charge pulse
                    pybamm.step.string("Rest for 15 seconds", period="0.001 seconds"),
                    pybamm.step.string("Rest for 10 minutes", period="10 seconds"),        
                    pybamm.step.string("Charge at 0.5C for 0.2 hours", period="0.5 seconds"),         # Charge until 10% DOD
                    pybamm.step.string("Rest for 30 minutes", period="10 seconds"),                 # Resting at 10% SOC 
                    pybamm.step.string("Discharge at 1C for 15 seconds", period="0.001 seconds"),   # Discharge Pulse
                    pybamm.step.string("Rest for 15 seconds", period="0.001 seconds"),
                    pybamm.step.string("Rest for 10 minutes", period="10 seconds"),  
                    pybamm.step.string("Charge at 1C for 15 seconds", period="0.001 seconds"),      # Charge pulse
                    pybamm.step.string("Rest for 15 seconds", period="0.001 seconds"),
                    pybamm.step.string("Rest for 10 minutes", period="10 seconds"),        
                    pybamm.step.string("Charge at 0.5C for 0.2 hours", period="0.5 seconds"),         # Charge until 10% DOD
                    pybamm.step.string("Rest for 30 minutes", period="10 seconds"),                 # Resting at 20% SOC   
                    pybamm.step.string("Discharge at 1C for 15 seconds", period="0.001 seconds"),   # Discharge Pulse
                    pybamm.step.string("Rest for 15 seconds", period="0.001 seconds"),
                    pybamm.step.string("Rest for 10 minutes", period="10 seconds"),  
                    pybamm.step.string("Charge at 1C for 15 seconds", period="0.001 seconds"),      # Charge pulse
                    pybamm.step.string("Rest for 15 seconds", period="0.001 seconds"),
                    pybamm.step.string("Rest for 10 minutes", period="10 seconds"),        
                    pybamm.step.string("Charge at 0.5C for 0.2 hours", period="0.5 seconds"),         # Charge until 10% DOD
                    pybamm.step.string("Rest for 30 minutes", period="10 seconds"),                 # Resting at 30% SOC   
                    pybamm.step.string("Discharge at 1C for 15 seconds", period="0.001 seconds"),   # Discharge Pulse
                    pybamm.step.string("Rest for 15 seconds", period="0.001 seconds"),
                    pybamm.step.string("Rest for 10 minutes", period="10 seconds"),  
                    pybamm.step.string("Charge at 1C for 15 seconds", period="0.001 seconds"),      # Charge pulse
                    pybamm.step.string("Rest for 15 seconds", period="0.001 seconds"),
                    pybamm.step.string("Rest for 10 minutes", period="10 seconds"),        
                    pybamm.step.string("Charge at 0.5C for 0.2 hours", period="0.5 seconds"),         # Charge until 10% DOD
                    pybamm.step.string("Rest for 30 minutes", period="10 seconds"),                 # Resting at 40% SOC   
                    pybamm.step.string("Discharge at 1C for 15 seconds", period="0.001 seconds"),   # Discharge Pulse
                    pybamm.step.string("Rest for 15 seconds", period="0.001 seconds"),
                    pybamm.step.string("Rest for 10 minutes", period="10 seconds"),  
                    pybamm.step.string("Charge at 1C for 15 seconds", period="0.001 seconds"),      # Charge pulse
                    pybamm.step.string("Rest for 15 seconds", period="0.001 seconds"),
                    pybamm.step.string("Rest for 10 minutes", period="10 seconds"),        
                    pybamm.step.string("Charge at 0.5C for 0.2 hours", period="0.5 seconds"),         # Charge until 10% DOD
                    pybamm.step.string("Rest for 30 minutes", period="10 seconds"),                 # Resting at 50% SOC   
                    pybamm.step.string("Discharge at 1C for 15 seconds", period="0.001 seconds"),   # Discharge Pulse
                    pybamm.step.string("Rest for 15 seconds", period="0.001 seconds"),
                    pybamm.step.string("Rest for 10 minutes", period="10 seconds"),  
                    pybamm.step.string("Charge at 1C for 15 seconds", period="0.001 seconds"),      # Charge pulse
                    pybamm.step.string("Rest for 15 seconds", period="0.001 seconds"),
                    pybamm.step.string("Rest for 10 minutes", period="10 seconds"),        
                    pybamm.step.string("Charge at 0.5C for 0.2 hours", period="0.5 seconds"),         # Charge until 10% DOD
                    pybamm.step.string("Rest for 30 minutes", period="10 seconds"),                 # Resting at 60% SOC   
                    pybamm.step.string("Discharge at 1C for 15 seconds", period="0.001 seconds"),   # Discharge Pulse
                    pybamm.step.string("Rest for 15 seconds", period="0.001 seconds"),
                    pybamm.step.string("Rest for 10 minutes", period="10 seconds"),  
                    pybamm.step.string("Charge at 1C for 15 seconds", period="0.001 seconds"),      # Charge pulse
                    pybamm.step.string("Rest for 15 seconds", period="0.001 seconds"),
                    pybamm.step.string("Rest for 10 minutes", period="10 seconds"),        
                    pybamm.step.string("Charge at 0.5C for 0.2 hours", period="0.5 seconds"),         # Charge until 10% DOD
                    pybamm.step.string("Rest for 30 minutes", period="10 seconds"),                 # Resting at 70% SOC   
                    pybamm.step.string("Discharge at 1C for 15 seconds", period="0.001 seconds"),   # Discharge Pulse
                    pybamm.step.string("Rest for 15 seconds", period="0.001 seconds"),
                    pybamm.step.string("Rest for 10 minutes", period="10 seconds"),  
                    pybamm.step.string("Charge at 1C for 15 seconds", period="0.001 seconds"),      # Charge pulse
                    pybamm.step.string("Rest for 15 seconds", period="0.001 seconds"),
                    pybamm.step.string("Rest for 10 minutes", period="10 seconds"),        
                    pybamm.step.string("Charge at 0.5C for 0.2 hours", period="0.5 seconds"),         # Charge until 10% DOD
                    pybamm.step.string("Rest for 30 minutes", period="10 seconds"),                 # Resting at 80% SOC   
                    pybamm.step.string("Discharge at 1C for 15 seconds", period="0.001 seconds"),   # Discharge Pulse
                    pybamm.step.string("Rest for 15 seconds", period="0.001 seconds"),
                    pybamm.step.string("Rest for 10 minutes", period="10 seconds"),  
                    pybamm.step.string("Charge at 1C for 15 seconds", period="0.001 seconds"),      # Charge pulse
                    pybamm.step.string("Rest for 15 seconds", period="0.001 seconds"),
                    pybamm.step.string("Rest for 10 minutes", period="10 seconds"),        
                    pybamm.step.string("Charge at 0.5C for 0.2 hours", period="0.5 seconds"),         # Charge until 10% DOD
                    pybamm.step.string("Rest for 30 minutes", period="10 seconds"),                 # Resting at 90% SOC   
                    pybamm.step.string("Discharge at 1C for 15 seconds", period="0.001 seconds"),   # Discharge Pulse
                    pybamm.step.string("Rest for 15 seconds", period="0.001 seconds"),
                    pybamm.step.string("Rest for 10 minutes", period="10 seconds"),  
                    pybamm.step.string("Charge at 1C for 15 seconds", period="0.001 seconds"),      # Charge pulse
                    pybamm.step.string("Rest for 15 seconds", period="0.001 seconds"),
                    pybamm.step.string("Rest for 10 minutes", period="10 seconds"),        
                    pybamm.step.string("Charge at 0.5C until 4.2 V", period="1 seconds"),
                    pybamm.step.string("Hold at 4.2 V until C/50", period="1 seconds"),
                    pybamm.step.string("Rest for 30 minutes", period="10 seconds"),                 # Resting at 100% SOC   
                    pybamm.step.string("Discharge at 1C for 15 seconds", period="0.001 seconds"),   # Discharge Pulse
                    pybamm.step.string("Rest for 15 seconds", period="0.001 seconds"),
                    pybamm.step.string("Rest for 10 minutes", period="10 seconds"),  
                ],
            )
    elif protocol_name == "HPPC_short_rest_100soc_backup":
        if fixed_period:
            experiment = pybamm.Experiment(
                [
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at 1C for 15 seconds"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Discharge at 0.5C for 0.2 hours"), # Discharge from 100 %SOC to 90 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at 1C for 15 seconds"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Charge at 1C for 15 seconds"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Discharge at 0.5C for 0.2 hours"), # Discharge from 90 %SOC to 80 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at 1C for 15 seconds"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Charge at 1C for 15 seconds"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Discharge at 0.5C for 0.2 hours"), # Discharge from 80 %SOC to 70 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at 1C for 15 seconds"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Charge at 1C for 15 seconds"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Discharge at 0.5C for 0.2 hours"), # Discharge from 70 %SOC to 60 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at 1C for 15 seconds"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Charge at 1C for 15 seconds"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Discharge at 0.5C for 0.2 hours"), # Discharge from 60 %SOC to 50 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at 1C for 15 seconds"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Charge at 1C for 15 seconds"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Discharge at 0.5C for 0.2 hours"), # Discharge from 50 %SOC to 40 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at 1C for 15 seconds"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Charge at 1C for 15 seconds"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Discharge at 0.5C for 0.2 hours"), # Discharge from 40 %SOC to 30 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at 1C for 15 seconds or until 2.5 V"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Charge at 1C for 15 seconds"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Discharge at 0.5C for 0.2 hours or until 2.5 V"), # Discharge from 30 %SOC to 20 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at 1C for 15 seconds or until 2.5 V"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Charge at 1C for 15 seconds"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Discharge at 0.5C for 0.2 hours or until 2.5 V"), # Discharge from 20 %SOC to 10 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at 1C for 15 seconds or until 2.5 V"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Charge at 1C for 15 seconds"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Discharge at 0.5C until 2.5 V"),
                    pybamm.step.string("Hold at 2.5 V until C/20", direction="discharge"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at 1C for 15 seconds"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Charge at 0.5C for 0.2 hours"), # Charge from 0 %SOC to 10 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at 1C for 15 seconds"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Charge at 1C for 15 seconds"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Charge at 0.5C for 0.2 hours"), # Charge from 10 %SOC to 20 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at 1C for 15 seconds"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Charge at 1C for 15 seconds"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Charge at 0.5C for 0.2 hours"), # Charge from 20 %SOC to 30 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at 1C for 15 seconds"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Charge at 1C for 15 seconds"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Charge at 0.5C for 0.2 hours"), # Charge from 30 %SOC to 40 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at 1C for 15 seconds"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Charge at 1C for 15 seconds"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Charge at 0.5C for 0.2 hours"), # Charge from 40 %SOC to 50 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at 1C for 15 seconds"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Charge at 1C for 15 seconds"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Charge at 0.5C for 0.2 hours"), # Charge from 50 %SOC to 60 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at 1C for 15 seconds"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Charge at 1C for 15 seconds"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Charge at 0.5C for 0.2 hours"), # Charge from 60 %SOC to 70 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at 1C for 15 seconds"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Charge at 1C for 15 seconds or until 4.2 V"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Charge at 0.5C for 0.2 hours or until 4.2 V"), # Charge from 70 %SOC to 80 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at 1C for 15 seconds"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Charge at 1C for 15 seconds or until 4.2 V"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Charge at 0.5C for 0.2 hours or until 4.2 V"), # Charge from 80 %SOC to 90 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at 1C for 15 seconds"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Charge at 1C for 15 seconds or until 4.2 V"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Charge at 0.5C until 4.2 V"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at 1C for 15 seconds"),
                    pybamm.step.string("Rest for 10 minutes"),
                ],
                period=f"{period_value} seconds",   # output sampling
            )
        else:
            experiment = pybamm.Experiment(
                [
                    
                ],
            )
    elif protocol_name == "HPPC_short_rest_100soc":
        if fixed_period:
            experiment = pybamm.Experiment(
                [   #* 10 normal steps of HPPC profile and discharge
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Discharge at 1C for 15 seconds or until 2.5V"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Charge at 1C for 15 seconds or until 4.2V"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Discharge at 0.5C for 0.2 hours or until 2.5V"), # Discharge aprox 10 %SOC
                ]*10
                +
                [   #* Try to ensure 0 %SOC
                    pybamm.step.string("Discharge at 0.5C until 2.5V"),
                    pybamm.step.string("Hold at 2.5 V until C/20", direction="discharge"),
                ]
                +
                [   #* Repeat profile at 0 %SOC
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Discharge at 1C for 15 seconds or until 2.5V"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Charge at 1C for 15 seconds or until 4.2V"),
                    pybamm.step.string("Rest for 10 minutes"),
                ]
                +
                [   #* 10 adapted steps of HPPC profile and charge
                    pybamm.step.string("Charge at 0.5C for 0.2 hours or until 4.2V"), # Charge aprox 10 %SOC
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Charge at 1C for 15 seconds or until 4.2V"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Discharge at 1C for 15 seconds or until 2.5V"),
                    pybamm.step.string("Rest for 10 minutes"),
                ]*10
                +
                [   #* Try to ensure 100 %SOC
                    pybamm.step.string("Charge at 0.5C until 4.2V"),
                    pybamm.step.string("Hold at 4.2 V until C/20", direction="charge"),
                ]
                +
                [   #* Repeat profile at 100%
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Charge at 1C for 15 seconds or until 4.2V"),
                    pybamm.step.string("Rest for 10 minutes"),
                    pybamm.step.string("Discharge at 1C for 15 seconds or until 2.5V"),
                    pybamm.step.string("Rest for 10 minutes"),
                ],                
                period=f"{period_value} seconds",   # output sampling
            )
        else:
            experiment = pybamm.Experiment(
                [
                    
                ],
            )
    elif  protocol_name == "short":
        if fixed_period:
            experiment = pybamm.Experiment(
                [
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Discharge at 1C for 10 seconds"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Charge at 1C until 4.2 V"),
                    pybamm.step.string("Hold at 4.2 V until C/10"),
                    pybamm.step.string("Rest for 1 minutes"),
                ],
                period=f"{period_value} seconds",   # output sampling
            )
        else:
            experiment = pybamm.Experiment(
                [
                    pybamm.step.string("Rest for 1 minutes", period="10 seconds"),
                    pybamm.step.string("Discharge at 1C for 10 seconds", period="0.01 seconds"),
                    pybamm.step.string("Rest for 1 minutes", period="1 seconds"),
                    pybamm.step.string("Charge at 1C until 4.2 V", period="0.01 seconds"),
                    pybamm.step.string("Hold at 4.2 V until C/10", period="0.01 seconds"),
                    pybamm.step.string("Rest for 1 minutes", period="1 seconds"),
                ]
            )
    elif  protocol_name == "GITT":
        if fixed_period:
            experiment = pybamm.Experiment(
                [
                    pybamm.step.string("Rest for 30 minutes"),
                    pybamm.step.string("Discharge at 1C until 2.5 V"),
                    pybamm.step.string("Rest for 30 minutes"),
                    pybamm.step.string("Charge at 1C until 4.2 V"),
                    pybamm.step.string("Hold at 4.2 V until C/50"),
                    pybamm.step.string("Rest for 30 minutes"),
                    pybamm.step.string("Hold at 4.2 V until C/50"),
                    pybamm.step.string("Rest for 30 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 until 2.5 V"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                    pybamm.step.string("Charge at C/20 until 4.2 V"),
                    pybamm.step.string("Hold at 4.2 V until C/50"),
                    pybamm.step.string("Rest for 1 minutes"),
                    pybamm.step.string("Rest for 60 minutes"),
                ],
                period=f"{period_value} seconds",   # output sampling
            )
   
        else:
            experiment = pybamm.Experiment(
                [
                    pybamm.step.string("Rest for 30 minutes", period="10 seconds"),
                    pybamm.step.string("Discharge at 1C until 2.5 V", period="1 seconds"),
                    pybamm.step.string("Rest for 30 minutes", period="10 seconds"),
                    pybamm.step.string("Charge at 1C until 4.2 V", period="1 seconds"),
                    pybamm.step.string("Hold at 4.2 V until C/50", period="1 seconds"),
                    pybamm.step.string("Rest for 30 minutes", period="10 seconds"),
                    pybamm.step.string("Hold at 4.2 V until C/50", period="1 seconds"),
                    pybamm.step.string("Rest for 30 minutes", period="10 seconds"),                 # Resting at 100% SOC
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 98.75% SOC
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 97.5% SOC
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 96.25% SOC
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 95% SOC
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 93.75% SOC
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 92.5% SOC
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 91.25% SOC
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 90% SOC
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 88.75% SOC
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 87.5% SOC
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 86.25% SOC
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 85% SOC
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 83.75% SOC
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 82.5% SOC
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 81.25% SOC
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 80% SOC
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 78.75% SOC
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 77.5% SOC
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 76.25% SOC
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 75% SOC
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 73.75% SOC
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 72.5% SOC
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 71.25% SOC
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 70% SOC
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 68.75% SOC
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 67.5% SOC
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 66.25% SOC   
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 65.00% SOC   
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 63.75% SOC   
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 62.50% SOC   
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 61.25% SOC   
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 60.00% SOC   
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 58.75% SOC   
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 57.50% SOC   
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 56.25% SOC   
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 55.00% SOC   
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 53.75% SOC   
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 52.50% SOC   
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 51.25% SOC   
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 50.00% SOC   
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 48.75% SOC   
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 47.50% SOC   
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 46.25% SOC   
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 45.00% SOC   
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 43.75% SOC   
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 42.50% SOC   
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 41.25% SOC   
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 40.00% SOC   
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 38.75% SOC   
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 37.50% SOC   
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 36.25% SOC   
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 35.00% SOC   
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 33.75% SOC   
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 32.50% SOC   
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 31.25% SOC   
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 30% SOC   
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 28.75% SOC   
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 27.5% SOC   
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 26.25% SOC   
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 25% SOC   
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 23.75% SOC   
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 22.5% SOC   
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 21.25% SOC   
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 20% SOC   
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 18.75% SOC   
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 17.5% SOC   
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 16.25% SOC   
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 15% SOC   
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 13.75% SOC   
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 12.50% SOC   
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 11.25% SOC   
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 10.00% SOC   
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 8.75% SOC   
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 7.50% SOC   
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 6.25% SOC   
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 5.00% SOC   
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 3.75% SOC   
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 2.50% SOC   
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 1.25% SOC   
                    pybamm.step.string("Discharge at C/20 until 2.5 V", period="0.001 seconds"),     # Discharge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 0.00% SOC   
                    pybamm.step.string("Discharge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 1.25% SOC
                    pybamm.step.string("Charge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 2.50% SOC
                    pybamm.step.string("Charge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 3.75% SOC
                    pybamm.step.string("Charge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 5.00% SOC
                    pybamm.step.string("Charge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 6.25% SOC
                    pybamm.step.string("Charge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 7.50% SOC
                    pybamm.step.string("Charge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 8.75% SOC
                    pybamm.step.string("Charge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 10.00% SOC
                    pybamm.step.string("Charge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 11.25% SOC
                    pybamm.step.string("Charge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 12.50% SOC
                    pybamm.step.string("Charge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 13.75% SOC
                    pybamm.step.string("Charge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 15.00% SOC
                    pybamm.step.string("Charge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 16.25% SOC
                    pybamm.step.string("Charge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 17.50% SOC
                    pybamm.step.string("Charge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 18.75% SOC
                    pybamm.step.string("Charge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 20.00% SOC
                    pybamm.step.string("Charge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 21.25% SOC
                    pybamm.step.string("Charge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 22.50% SOC
                    pybamm.step.string("Charge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 23.75% SOC
                    pybamm.step.string("Charge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 25.00% SOC
                    pybamm.step.string("Charge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 26.25% SOC
                    pybamm.step.string("Charge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 27.50% SOC
                    pybamm.step.string("Charge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 28.75% SOC
                    pybamm.step.string("Charge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 30.00% SOC
                    pybamm.step.string("Charge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 31.25% SOC
                    pybamm.step.string("Charge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 32.50% SOC
                    pybamm.step.string("Charge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 33.75% SOC
                    pybamm.step.string("Charge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 35.00% SOC
                    pybamm.step.string("Charge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 36.25% SOC
                    pybamm.step.string("Charge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 37.50% SOC
                    pybamm.step.string("Charge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 38.75% SOC
                    pybamm.step.string("Charge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 40.00% SOC
                    pybamm.step.string("Charge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 41.25% SOC
                    pybamm.step.string("Charge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 42.50% SOC
                    pybamm.step.string("Charge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 43.75% SOC
                    pybamm.step.string("Charge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 45.00% SOC
                    pybamm.step.string("Charge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 46.25% SOC
                    pybamm.step.string("Charge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 47.50% SOC
                    pybamm.step.string("Charge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 48.75% SOC
                    pybamm.step.string("Charge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 50.00% SOC
                    pybamm.step.string("Charge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 51.25% SOC
                    pybamm.step.string("Charge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 52.50% SOC
                    pybamm.step.string("Charge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 53.75% SOC
                    pybamm.step.string("Charge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 55.00% SOC
                    pybamm.step.string("Charge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 56.25% SOC
                    pybamm.step.string("Charge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 57.50% SOC
                    pybamm.step.string("Charge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 58.75% SOC
                    pybamm.step.string("Charge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 60.00% SOC
                    pybamm.step.string("Charge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 61.25% SOC
                    pybamm.step.string("Charge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 62.50% SOC
                    pybamm.step.string("Charge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 63.75% SOC
                    pybamm.step.string("Charge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 65.00% SOC
                    pybamm.step.string("Charge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 66.25% SOC
                    pybamm.step.string("Charge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 67.50% SOC
                    pybamm.step.string("Charge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 68.75% SOC
                    pybamm.step.string("Charge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 70.00% SOC
                    pybamm.step.string("Charge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 71.25% SOC
                    pybamm.step.string("Charge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 72.50% SOC
                    pybamm.step.string("Charge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 73.75% SOC
                    pybamm.step.string("Charge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 75.00% SOC
                    pybamm.step.string("Charge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 76.25% SOC
                    pybamm.step.string("Charge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 77.50% SOC
                    pybamm.step.string("Charge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 78.75% SOC
                    pybamm.step.string("Charge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 80.00% SOC
                    pybamm.step.string("Charge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 81.25% SOC
                    pybamm.step.string("Charge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 82.50% SOC
                    pybamm.step.string("Charge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 83.75% SOC
                    pybamm.step.string("Charge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 85.00% SOC
                    pybamm.step.string("Charge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 86.25% SOC
                    pybamm.step.string("Charge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 87.50% SOC
                    pybamm.step.string("Charge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 88.75% SOC
                    pybamm.step.string("Charge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 90.00% SOC
                    pybamm.step.string("Charge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 91.25% SOC
                    pybamm.step.string("Charge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 92.50% SOC
                    pybamm.step.string("Charge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 93.75% SOC
                    pybamm.step.string("Charge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 95.00% SOC
                    pybamm.step.string("Charge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 96.25% SOC
                    pybamm.step.string("Charge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 97.50% SOC
                    pybamm.step.string("Charge at C/20 for 15 min", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 98.75% SOC
                    pybamm.step.string("Charge at C/20 until 4.2 V", period="0.001 seconds"),        # Charge Pulse 1.25% DOD
                    pybamm.step.string("Hold at 4.2 V until C/50", period="1 seconds"),
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                 # Resting at 100.00% SOC
                ],
            )   
    elif  protocol_name == "GITT_short_rest":
        if fixed_period:
            experiment = pybamm.Experiment(
                [
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at 1C until 2.5 V"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at 1C until 4.2 V"),
                    pybamm.step.string("Hold at 4.2 V until C/50"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Hold at 4.2 V until C/50"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 until 2.5 V"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 for 15 min"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/20 until 4.2 V"),
                    pybamm.step.string("Hold at 4.2 V until C/50"),
                    pybamm.step.string("Rest for 5 minutes"),
                ],
                period=f"{period_value} seconds",   # output sampling
            )
   
        else:
            experiment = pybamm.Experiment(
                [
                 
                ],
            )   
    elif  protocol_name == "GITT_short_rest_100soc_backup":
        if fixed_period:
            experiment = pybamm.Experiment(
                [
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min"), # Discharge from 100.00 %SOC to 98.75 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min"), # Discharge from 98.75 %SOC to 97.50 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min"), # Discharge from 97.50 %SOC to 96.25 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min"), # Discharge from 96.25 %SOC to 95.00 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min"), # Discharge from 95.00 %SOC to 93.75 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min"), # Discharge from 93.75 %SOC to 92.50 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min"), # Discharge from 92.50 %SOC to 91.25 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min"), # Discharge from 91.25 %SOC to 90.00 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min"), # Discharge from 90.00 %SOC to 88.75 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min"), # Discharge from 88.75 %SOC to 87.50 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min"), # Discharge from 87.50 %SOC to 86.25 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min"), # Discharge from 86.25 %SOC to 85.00 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min"), # Discharge from 85.00 %SOC to 83.75 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min"), # Discharge from 83.75 %SOC to 82.50 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min"), # Discharge from 82.50 %SOC to 81.25 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min"), # Discharge from 81.25 %SOC to 80.00 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min"), # Discharge from 80.00 %SOC to 78.75 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min"), # Discharge from 78.75 %SOC to 77.50 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min"), # Discharge from 77.50 %SOC to 76.25 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min"), # Discharge from 76.25 %SOC to 75.00 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min"), # Discharge from 75.00 %SOC to 73.75 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min"), # Discharge from 73.75 %SOC to 72.50 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min"), # Discharge from 72.50 %SOC to 71.25 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min"), # Discharge from 71.25 %SOC to 70.00 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min"), # Discharge from 70.00 %SOC to 68.75 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min"), # Discharge from 68.75 %SOC to 67.50 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min"), # Discharge from 67.50 %SOC to 66.25 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min"), # Discharge from 66.25 %SOC to 65.00 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min"), # Discharge from 65.00 %SOC to 63.75 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min"), # Discharge from 63.75 %SOC to 62.50 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min"), # Discharge from 62.50 %SOC to 61.25 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min"), # Discharge from 61.25 %SOC to 60.00 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min"), # Discharge from 60.00 %SOC to 58.75 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min"), # Discharge from 58.75 %SOC to 57.50 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min"), # Discharge from 57.50 %SOC to 56.25 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min"), # Discharge from 56.25 %SOC to 55.00 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min"), # Discharge from 55.00 %SOC to 53.75 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min"), # Discharge from 53.75 %SOC to 52.50 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min"), # Discharge from 52.50 %SOC to 51.25 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min"), # Discharge from 51.25 %SOC to 50.00 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min"), # Discharge from 50.00 %SOC to 48.75 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min"), # Discharge from 48.75 %SOC to 47.50 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min"), # Discharge from 47.50 %SOC to 46.25 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min"), # Discharge from 46.25 %SOC to 45.00 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min"), # Discharge from 45.00 %SOC to 43.75 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min"), # Discharge from 43.75 %SOC to 42.50 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min"), # Discharge from 42.50 %SOC to 41.25 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min"), # Discharge from 41.25 %SOC to 40.00 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min"), # Discharge from 40.00 %SOC to 38.75 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min"), # Discharge from 38.75 %SOC to 37.50 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min"), # Discharge from 37.50 %SOC to 36.25 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min"), # Discharge from 36.25 %SOC to 35.00 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min"), # Discharge from 35.00 %SOC to 33.75 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min"), # Discharge from 33.75 %SOC to 32.50 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min"), # Discharge from 32.50 %SOC to 31.25 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min"), # Discharge from 31.25 %SOC to 30.00 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min"), # Discharge from 30.00 %SOC to 28.75 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min"), # Discharge from 28.75 %SOC to 27.50 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min"), # Discharge from 27.50 %SOC to 26.25 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min"), # Discharge from 26.25 %SOC to 25.00 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min"), # Discharge from 25.00 %SOC to 23.75 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min"), # Discharge from 23.75 %SOC to 22.50 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min"), # Discharge from 22.50 %SOC to 21.25 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min"), # Discharge from 21.25 %SOC to 20.00 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min"), # Discharge from 20.00 %SOC to 18.75 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min"), # Discharge from 18.75 %SOC to 17.50 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min or until 2.5V"), # Discharge from 17.50 %SOC to 16.25 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min or until 2.5V"), # Discharge from 16.25 %SOC to 15.00 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min or until 2.5V"), # Discharge from 15.00 %SOC to 13.75 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min or until 2.5V"), # Discharge from 13.75 %SOC to 12.50 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min or until 2.5V"), # Discharge from 12.50 %SOC to 11.25 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min or until 2.5V"), # Discharge from 11.25 %SOC to 10.00 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min or until 2.5V"), # Discharge from 10.00 %SOC to 8.75 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min or until 2.5V"), # Discharge from 8.75 %SOC to 7.50 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min or until 2.5V"), # Discharge from 7.50 %SOC to 6.25 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min or until 2.5V"), # Discharge from 6.25 %SOC to 5.00 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min or until 2.5V"), # Discharge from 5.00 %SOC to 3.75 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min or until 2.5V"), # Discharge from 3.75 %SOC to 2.50 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min or until 2.5V"), # Discharge from 2.50 %SOC to 1.25 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/20 until 2.5 V"),
                    pybamm.step.string("Hold at 2.5 V until C/20", direction="discharge"),
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min"), # Charge from 00 % SOC to 1.25 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min"), # Charge from 1.25 % SOC to 2.50 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min"), # Charge from 2.5 % SOC to 3.75 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min"), # Charge from 3.75 % SOC to 5.00 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min"), # Charge from 05 % SOC to 6.25 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min"), # Charge from 6.25 % SOC to 7.50 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min"), # Charge from 7.5 % SOC to 8.75 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min"), # Charge from 8.75 % SOC to 10.00 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min"), # Charge from 10 % SOC to 11.25 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min"), # Charge from 11.25 % SOC to 12.50 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min"), # Charge from 12.5 % SOC to 13.75 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min"), # Charge from 13.75 % SOC to 15.00 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min"), # Charge from 15 % SOC to 16.25 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min"), # Charge from 16.25 % SOC to 17.50 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min"), # Charge from 17.5 % SOC to 18.75 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min"), # Charge from 18.75 % SOC to 20.00 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min"), # Charge from 20 % SOC to 21.25 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min"), # Charge from 21.25 % SOC to 22.50 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min"), # Charge from 22.5 % SOC to 23.75 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min"), # Charge from 23.75 % SOC to 25.00 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min"), # Charge from 25 % SOC to 26.25 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min"), # Charge from 26.25 % SOC to 27.50 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min"), # Charge from 27.5 % SOC to 28.75 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min"), # Charge from 28.75 % SOC to 30.00 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min"), # Charge from 30 % SOC to 31.25 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min"), # Charge from 31.25 % SOC to 32.50 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min"), # Charge from 32.5 % SOC to 33.75 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min"), # Charge from 33.75 % SOC to 35.00 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min"), # Charge from 35 % SOC to 36.25 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min"), # Charge from 36.25 % SOC to 37.50 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min"), # Charge from 37.5 % SOC to 38.75 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min"), # Charge from 38.75 % SOC to 40.00 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min"), # Charge from 40 % SOC to 41.25 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min"), # Charge from 41.25 % SOC to 42.50 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min"), # Charge from 42.5 % SOC to 43.75 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min"), # Charge from 43.75 % SOC to 45.00 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min"), # Charge from 45 % SOC to 46.25 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min"), # Charge from 46.25 % SOC to 47.50 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min"), # Charge from 47.5 % SOC to 48.75 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min"), # Charge from 48.75 % SOC to 50.00 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min"), # Charge from 50 % SOC to 51.25 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min"), # Charge from 51.25 % SOC to 52.50 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min"), # Charge from 52.5 % SOC to 53.75 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min"), # Charge from 53.75 % SOC to 55.00 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min"), # Charge from 55 % SOC to 56.25 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min"), # Charge from 56.25 % SOC to 57.50 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min"), # Charge from 57.5 % SOC to 58.75 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min"), # Charge from 58.75 % SOC to 60.00 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min"), # Charge from 60 % SOC to 61.25 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min"), # Charge from 61.25 % SOC to 62.50 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min"), # Charge from 62.5 % SOC to 63.75 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min"), # Charge from 63.75 % SOC to 65.00 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min"), # Charge from 65 % SOC to 66.25 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min"), # Charge from 66.25 % SOC to 67.50 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min"), # Charge from 67.5 % SOC to 68.75 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min"), # Charge from 68.75 % SOC to 70.00 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min"), # Charge from 70 % SOC to 71.25 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min"), # Charge from 71.25 % SOC to 72.50 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min"), # Charge from 72.5 % SOC to 73.75 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min"), # Charge from 73.75 % SOC to 75.00 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min"), # Charge from 75 % SOC to 76.25 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min"), # Charge from 76.25 % SOC to 77.50 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min"), # Charge from 77.5 % SOC to 78.75 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min"), # Charge from 78.75 % SOC to 80.00 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min"), # Charge from 80 % SOC to 81.25 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min"), # Charge from 81.25 % SOC to 82.50 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min"), # Charge from 82.5 % SOC to 83.75 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min"), # Charge from 83.75 % SOC to 85.00 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min or until 4.2V"), # Charge from 85 % SOC to 86.25 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min or until 4.2V"), # Charge from 86.25 % SOC to 87.50 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min or until 4.2V"), # Charge from 87.5 % SOC to 88.75 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min or until 4.2V"), # Charge from 88.75 % SOC to 90.00 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min or until 4.2V"), # Charge from 90 % SOC to 91.25 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min or until 4.2V"), # Charge from 91.25 % SOC to 92.50 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min or until 4.2V"), # Charge from 92.5 % SOC to 93.75 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min or until 4.2V"), # Charge from 93.75 % SOC to 95.00 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min or until 4.2V"), # Charge from 95 % SOC to 96.25 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min or until 4.2V"), # Charge from 96.25 % SOC to 97.50 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 until 4.2V"), # Charge from 96.25 % SOC to 97.50 %SOC

                ],
                period=f"{period_value} seconds",   # output sampling
            )
   
        else:
            experiment = pybamm.Experiment(
                [
                 
                ],
            )   
    elif  protocol_name == "GITT_short_rest_100soc":
        if fixed_period:
            experiment = pybamm.Experiment(
                [   #* 80 normal steps of GITT profile with a discharge of 1.25 %SOC each
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min or until 2.5V"), # Discharge step of 1.25 %SOC
                ]*80
                +   
                [   #* Try to ensure 0 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 until 2.5V"),
                    pybamm.step.string("Hold at 2.5 V until C/20", direction="discharge"),
                ]
                +
                [   #* 80 normal steps of GITT profile with a charge of 1.25 %SOC each
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 for 7.5 min or until 4.2V"), # Discharge step of 1.25 %SOC
                ]*80
                +                  
                [   #* Try to ensure 100 %SOC
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Charge at C/10 until 4.2V"),
                    pybamm.step.string("Rest for 5 minutes"),
                ]
                ,
                period=f"{period_value} seconds",   # output sampling
            )
   
        else:
            experiment = pybamm.Experiment(
                [
                 
                ],
            )   
    elif  protocol_name == "GITT_short_rest_100soc_long_experiment":
        if fixed_period:
            experiment = pybamm.Experiment(
                [   #* 80 normal steps of GITT profile with a discharge of 1.25 %SOC each
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min or until 2.5V"), # Discharge step of 1.25 %SOC
                ]#*80
                # +   
                # [   #* Try to ensure 0 %SOC
                #     pybamm.step.string("Rest for 5 minutes"),
                #     pybamm.step.string("Discharge at C/10 until 2.5V"),
                #     pybamm.step.string("Hold at 2.5 V until C/20", direction="discharge"),
                # ]
                # +
                # [   #* 80 normal steps of GITT profile with a charge of 1.25 %SOC each
                #     pybamm.step.string("Rest for 5 minutes"),
                #     pybamm.step.string("Charge at C/10 for 7.5 min or until 4.2V"), # Discharge step of 1.25 %SOC
                # ]*80
                # +                  
                # [   #* Try to ensure 100 %SOC
                #     pybamm.step.string("Rest for 5 minutes"),
                #     pybamm.step.string("Charge at C/10 until 4.2V"),
                #     pybamm.step.string("Rest for 5 minutes"),
                # ]
                ,
                period=f"{period_value} seconds",   # output sampling
            )
   
        else:
            experiment = pybamm.Experiment(
                [
                 
                ],
            )   
    elif  protocol_name == "ICI":
        if fixed_period:
            experiment = pybamm.Experiment(
                [
                    pybamm.step.string("Rest for 30 minutes"),
                    pybamm.step.string("Discharge at 1C until 2.5 V"),
                    pybamm.step.string("Rest for 30 minutes"),
                    pybamm.step.string("Charge at 1C until 4.2 V"),
                    pybamm.step.string("Hold at 4.2 V until C/50"),
                    pybamm.step.string("Rest for 30 minutes"),
                    pybamm.step.string("Hold at 4.2 V until C/50"),
                    pybamm.step.string("Rest for 30 minutes"),                  # Resting at 100.00% SOC
                    pybamm.step.string("Discharge at C/20 for 60 minutes"),     # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 95% SOC
                    pybamm.step.string("Discharge at C/20 for 60 minutes"),     # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 90% SOC
                    pybamm.step.string("Discharge at C/20 for 60 minutes"),     # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 85% SOC
                    pybamm.step.string("Discharge at C/20 for 60 minutes"),     # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 80% SOC
                    pybamm.step.string("Discharge at C/20 for 60 minutes"),     # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 75% SOC
                    pybamm.step.string("Discharge at C/20 for 60 minutes"),     # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 70% SOC
                    pybamm.step.string("Discharge at C/20 for 60 minutes"),     # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 65% SOC
                    pybamm.step.string("Discharge at C/20 for 60 minutes"),     # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 60% SOC
                    pybamm.step.string("Discharge at C/20 for 60 minutes"),     # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 55% SOC
                    pybamm.step.string("Discharge at C/20 for 60 minutes"),     # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 50% SOC
                    pybamm.step.string("Discharge at C/20 for 60 minutes"),     # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 45% SOC
                    pybamm.step.string("Discharge at C/20 for 60 minutes"),     # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 40% SOC
                    pybamm.step.string("Discharge at C/20 for 60 minutes"),     # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 35% SOC
                    pybamm.step.string("Discharge at C/20 for 60 minutes"),     # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 30% SOC
                    pybamm.step.string("Discharge at C/20 for 60 minutes"),     # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 25% SOC
                    pybamm.step.string("Discharge at C/20 for 60 minutes"),     # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 20% SOC
                    pybamm.step.string("Discharge at C/20 for 60 minutes"),     # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 15% SOC
                    pybamm.step.string("Discharge at C/20 for 60 minutes"),     # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 10% SOC
                    pybamm.step.string("Discharge at C/20 for 60 minutes"),     # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 5% SOC
                    pybamm.step.string("Discharge at C/20 for 60 minutes"),     # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 0% SOC
                    pybamm.step.string("Rest for 30 minutes"),                  # Resting at 0.00% SOC
                    pybamm.step.string("Charge at C/20 for 60 minutes"),        # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 5% SOC
                    pybamm.step.string("Charge at C/20 for 60 minutes"),        # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 10% SOC
                    pybamm.step.string("Charge at C/20 for 60 minutes"),        # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 15% SOC
                    pybamm.step.string("Charge at C/20 for 60 minutes"),        # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 20% SOC
                    pybamm.step.string("Charge at C/20 for 60 minutes"),        # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 25% SOC
                    pybamm.step.string("Charge at C/20 for 60 minutes"),        # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 30% SOC
                    pybamm.step.string("Charge at C/20 for 60 minutes"),        # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 35% SOC
                    pybamm.step.string("Charge at C/20 for 60 minutes"),        # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 40% SOC
                    pybamm.step.string("Charge at C/20 for 60 minutes"),        # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 45% SOC
                    pybamm.step.string("Charge at C/20 for 60 minutes"),        # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 50% SOC
                    pybamm.step.string("Charge at C/20 for 60 minutes"),        # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 55% SOC
                    pybamm.step.string("Charge at C/20 for 60 minutes"),        # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 60% SOC
                    pybamm.step.string("Charge at C/20 for 60 minutes"),        # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 65% SOC
                    pybamm.step.string("Charge at C/20 for 60 minutes"),        # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 70% SOC
                    pybamm.step.string("Charge at C/20 for 60 minutes"),        # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 75% SOC
                    pybamm.step.string("Charge at C/20 for 60 minutes"),        # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 80% SOC
                    pybamm.step.string("Charge at C/20 for 60 minutes"),        # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 85% SOC
                    pybamm.step.string("Charge at C/20 for 60 minutes"),        # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 90% SOC
                    pybamm.step.string("Charge at C/20 for 60 minutes"),        # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 95% SOC
                    pybamm.step.string("Charge at C/20 for 60 minutes"),        # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 100% SOC
                    pybamm.step.string("Rest for 10 minutes"),                       # Resting at 100% SOC
                ],
                period=f"{period_value} seconds",   # output sampling
            )   
        else:
            experiment = pybamm.Experiment(
                [
                    pybamm.step.string("Rest for 30 minutes", period="10 seconds"),
                    pybamm.step.string("Discharge at 1C until 2.5 V", period="1 seconds"),
                    pybamm.step.string("Rest for 30 minutes", period="10 seconds"),
                    pybamm.step.string("Charge at 1C until 4.2 V", period="1 seconds"),
                    pybamm.step.string("Hold at 4.2 V until C/50", period="1 seconds"),
                    pybamm.step.string("Rest for 30 minutes", period="10 seconds"),
                    pybamm.step.string("Hold at 4.2 V until C/50", period="1 seconds"),
                    pybamm.step.string("Rest for 30 minutes", period="10 seconds"),                     # Resting at 100.00% SOC
                    pybamm.step.string("Discharge at C/20 for 60 minutes", period="0.01 seconds"),      # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),                   # "Rest pulse" during 60 seconds at 95% SOC
                    pybamm.step.string("Discharge at C/20 for 60 minutes", period="0.01 seconds"),      # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),                   # "Rest pulse" during 60 seconds at 90% SOC
                    pybamm.step.string("Discharge at C/20 for 60 minutes", period="0.01 seconds"),      # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),                   # "Rest pulse" during 60 seconds at 85% SOC
                    pybamm.step.string("Discharge at C/20 for 60 minutes", period="0.01 seconds"),      # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),                   # "Rest pulse" during 60 seconds at 80% SOC
                    pybamm.step.string("Discharge at C/20 for 60 minutes", period="0.01 seconds"),      # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),                   # "Rest pulse" during 60 seconds at 75% SOC
                    pybamm.step.string("Discharge at C/20 for 60 minutes", period="0.01 seconds"),      # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),                   # "Rest pulse" during 60 seconds at 70% SOC
                    pybamm.step.string("Discharge at C/20 for 60 minutes", period="0.01 seconds"),      # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),                   # "Rest pulse" during 60 seconds at 65% SOC
                    pybamm.step.string("Discharge at C/20 for 60 minutes", period="0.01 seconds"),      # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),                   # "Rest pulse" during 60 seconds at 60% SOC
                    pybamm.step.string("Discharge at C/20 for 60 minutes", period="0.01 seconds"),      # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),                   # "Rest pulse" during 60 seconds at 55% SOC
                    pybamm.step.string("Discharge at C/20 for 60 minutes", period="0.01 seconds"),      # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),                   # "Rest pulse" during 60 seconds at 50% SOC
                    pybamm.step.string("Discharge at C/20 for 60 minutes", period="0.01 seconds"),      # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),                   # "Rest pulse" during 60 seconds at 45% SOC
                    pybamm.step.string("Discharge at C/20 for 60 minutes", period="0.01 seconds"),      # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),                   # "Rest pulse" during 60 seconds at 40% SOC
                    pybamm.step.string("Discharge at C/20 for 60 minutes", period="0.01 seconds"),      # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),                   # "Rest pulse" during 60 seconds at 35% SOC
                    pybamm.step.string("Discharge at C/20 for 60 minutes", period="0.01 seconds"),      # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),                   # "Rest pulse" during 60 seconds at 30% SOC
                    pybamm.step.string("Discharge at C/20 for 60 minutes", period="0.01 seconds"),      # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),                   # "Rest pulse" during 60 seconds at 25% SOC
                    pybamm.step.string("Discharge at C/20 for 60 minutes", period="0.01 seconds"),      # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),                   # "Rest pulse" during 60 seconds at 20% SOC
                    pybamm.step.string("Discharge at C/20 for 60 minutes", period="0.01 seconds"),      # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),                   # "Rest pulse" during 60 seconds at 15% SOC
                    pybamm.step.string("Discharge at C/20 for 60 minutes", period="0.01 seconds"),      # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),                   # "Rest pulse" during 60 seconds at 10% SOC
                    pybamm.step.string("Discharge at C/20 for 60 minutes", period="0.01 seconds"),      # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),                   # "Rest pulse" during 60 seconds at 5% SOC
                    pybamm.step.string("Discharge at C/20 for 60 minutes", period="0.01 seconds"),      # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),                   # "Rest pulse" during 60 seconds at 0% SOC
                    pybamm.step.string("Rest for 30 minutes", period="10 seconds"),                     # Resting at 0.00% SOC
                    pybamm.step.string("Charge at C/20 for 60 minutes", period="0.01 seconds"),         # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),                   # "Rest pulse" during 60 seconds at 5% SOC
                    pybamm.step.string("Charge at C/20 for 60 minutes", period="0.01 seconds"),         # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),                   # "Rest pulse" during 60 seconds at 10% SOC
                    pybamm.step.string("Charge at C/20 for 60 minutes", period="0.01 seconds"),         # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),                   # "Rest pulse" during 60 seconds at 15% SOC
                    pybamm.step.string("Charge at C/20 for 60 minutes", period="0.01 seconds"),         # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),                   # "Rest pulse" during 60 seconds at 20% SOC
                    pybamm.step.string("Charge at C/20 for 60 minutes", period="0.01 seconds"),         # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),                   # "Rest pulse" during 60 seconds at 25% SOC
                    pybamm.step.string("Charge at C/20 for 60 minutes", period="0.01 seconds"),         # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),                   # "Rest pulse" during 60 seconds at 30% SOC
                    pybamm.step.string("Charge at C/20 for 60 minutes", period="0.01 seconds"),         # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),                   # "Rest pulse" during 60 seconds at 35% SOC
                    pybamm.step.string("Charge at C/20 for 60 minutes", period="0.01 seconds"),         # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),                   # "Rest pulse" during 60 seconds at 40% SOC
                    pybamm.step.string("Charge at C/20 for 60 minutes", period="0.01 seconds"),         # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),                   # "Rest pulse" during 60 seconds at 45% SOC
                    pybamm.step.string("Charge at C/20 for 60 minutes", period="0.01 seconds"),         # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),                   # "Rest pulse" during 60 seconds at 50% SOC
                    pybamm.step.string("Charge at C/20 for 60 minutes", period="0.01 seconds"),         # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),                   # "Rest pulse" during 60 seconds at 55% SOC
                    pybamm.step.string("Charge at C/20 for 60 minutes", period="0.01 seconds"),         # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),                   # "Rest pulse" during 60 seconds at 60% SOC
                    pybamm.step.string("Charge at C/20 for 60 minutes", period="0.01 seconds"),         # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),                   # "Rest pulse" during 60 seconds at 65% SOC
                    pybamm.step.string("Charge at C/20 for 60 minutes", period="0.01 seconds"),         # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),                   # "Rest pulse" during 60 seconds at 70% SOC
                    pybamm.step.string("Charge at C/20 for 60 minutes", period="0.01 seconds"),         # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),                   # "Rest pulse" during 60 seconds at 75% SOC
                    pybamm.step.string("Charge at C/20 for 60 minutes", period="0.01 seconds"),         # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),                   # "Rest pulse" during 60 seconds at 80% SOC
                    pybamm.step.string("Charge at C/20 for 60 minutes", period="0.01 seconds"),         # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),                   # "Rest pulse" during 60 seconds at 85% SOC
                    pybamm.step.string("Charge at C/20 for 60 minutes", period="0.01 seconds"),         # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),                   # "Rest pulse" during 60 seconds at 90% SOC
                    pybamm.step.string("Charge at C/20 for 60 minutes", period="0.01 seconds"),         # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),                   # "Rest pulse" during 60 seconds at 95% SOC
                    pybamm.step.string("Charge at C/20 for 60 minutes", period="0.01 seconds"),         # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes", period="0.001 seconds"),                   # "Rest pulse" during 60 seconds at 100% SOC
                    pybamm.step.string("Rest for 10 minutes", period="10 seconds"),                     # Resting at 100% SOC
                ],
            )  
    elif  protocol_name == "ICI_short_rest_100soc_backup":
        if fixed_period:
            experiment = pybamm.Experiment(
                [
                    pybamm.step.string("Rest for 5 minutes"),                  # Resting at 100.00% SOC
                    pybamm.step.string("Discharge at C/2.5 for 7.5 minutes"),     # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 95% SOC
                    pybamm.step.string("Discharge at C/2.5 for 7.5 minutes"),     # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 90% SOC
                    pybamm.step.string("Discharge at C/2.5 for 7.5 minutes"),     # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 85% SOC
                    pybamm.step.string("Discharge at C/2.5 for 7.5 minutes"),     # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 80% SOC
                    pybamm.step.string("Discharge at C/2.5 for 7.5 minutes"),     # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 75% SOC
                    pybamm.step.string("Discharge at C/2.5 for 7.5 minutes"),     # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 70% SOC
                    pybamm.step.string("Discharge at C/2.5 for 7.5 minutes"),     # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 65% SOC
                    pybamm.step.string("Discharge at C/2.5 for 7.5 minutes"),     # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 60% SOC
                    pybamm.step.string("Discharge at C/2.5 for 7.5 minutes"),     # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 55% SOC
                    pybamm.step.string("Discharge at C/2.5 for 7.5 minutes"),     # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 50% SOC
                    pybamm.step.string("Discharge at C/2.5 for 7.5 minutes"),     # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 45% SOC
                    pybamm.step.string("Discharge at C/2.5 for 7.5 minutes"),     # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 40% SOC
                    pybamm.step.string("Discharge at C/2.5 for 7.5 minutes"),     # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 35% SOC
                    pybamm.step.string("Discharge at C/2.5 for 7.5 minutes"),     # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 30% SOC
                    pybamm.step.string("Discharge at C/2.5 for 7.5 minutes"),     # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 25% SOC
                    pybamm.step.string("Discharge at C/2.5 for 7.5 minutes"),     # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 20% SOC
                    pybamm.step.string("Discharge at C/2.5 for 7.5 minutes"),     # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 15% SOC
                    pybamm.step.string("Discharge at C/2.5 for 7.5 minutes"),     # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 10% SOC
                    pybamm.step.string("Discharge at C/2.5 for 7.5 minutes"),     # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 5% SOC
                    pybamm.step.string("Discharge at C/2.5 for 7.5 minutes"),     # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 0% SOC
                    pybamm.step.string("Rest for 5 minutes"),                  # Resting at 0.00% SOC
                    pybamm.step.string("Charge at C/2.5 for 7.5 minutes"),        # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 5% SOC
                    pybamm.step.string("Charge at C/2.5 for 7.5 minutes"),        # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 10% SOC
                    pybamm.step.string("Charge at C/2.5 for 7.5 minutes"),        # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 15% SOC
                    pybamm.step.string("Charge at C/2.5 for 7.5 minutes"),        # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 20% SOC
                    pybamm.step.string("Charge at C/2.5 for 7.5 minutes"),        # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 25% SOC
                    pybamm.step.string("Charge at C/2.5 for 7.5 minutes"),        # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 30% SOC
                    pybamm.step.string("Charge at C/2.5 for 7.5 minutes"),        # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 35% SOC
                    pybamm.step.string("Charge at C/2.5 for 7.5 minutes"),        # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 40% SOC
                    pybamm.step.string("Charge at C/2.5 for 7.5 minutes"),        # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 45% SOC
                    pybamm.step.string("Charge at C/2.5 for 7.5 minutes"),        # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 50% SOC
                    pybamm.step.string("Charge at C/2.5 for 7.5 minutes"),        # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 55% SOC
                    pybamm.step.string("Charge at C/2.5 for 7.5 minutes"),        # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 60% SOC
                    pybamm.step.string("Charge at C/2.5 for 7.5 minutes"),        # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 65% SOC
                    pybamm.step.string("Charge at C/2.5 for 7.5 minutes"),        # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 70% SOC
                    pybamm.step.string("Charge at C/2.5 for 7.5 minutes"),        # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 75% SOC
                    pybamm.step.string("Charge at C/2.5 for 7.5 minutes"),        # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 80% SOC
                    pybamm.step.string("Charge at C/2.5 for 7.5 minutes"),        # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 85% SOC
                    pybamm.step.string("Charge at C/2.5 for 7.5 minutes"),        # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 90% SOC
                    pybamm.step.string("Charge at C/2.5 for 7.5 minutes"),        # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 95% SOC
                    pybamm.step.string("Charge at C/2.5 for 7.5 minutes"),        # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 1 minutes"),                   # "Rest pulse" during 60 seconds at 100% SOC
                ],
                period=f"{period_value} seconds",   # output sampling
            )   
        else:
            experiment = pybamm.Experiment(
                [
                    
                ],
            )  
    elif  protocol_name == "ICI_shorter_rest_100soc_backup":
        if fixed_period:
            experiment = pybamm.Experiment(
                [
                    pybamm.step.string("Rest for 5 minutes"),                  # Resting at 100.00% SOC
                    pybamm.step.string("Discharge at C/5 for 15 minutes or until 2.5V"),     # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 10 seconds"),                   # "Rest pulse" during 60 seconds at 95% SOC
                    pybamm.step.string("Discharge at C/5 for 15 minutes or until 2.5V"),     # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 10 seconds"),                   # "Rest pulse" during 60 seconds at 90% SOC
                    pybamm.step.string("Discharge at C/5 for 15 minutes or until 2.5V"),     # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 10 seconds"),                   # "Rest pulse" during 60 seconds at 85% SOC
                    pybamm.step.string("Discharge at C/5 for 15 minutes or until 2.5V"),     # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 10 seconds"),                   # "Rest pulse" during 60 seconds at 80% SOC
                    pybamm.step.string("Discharge at C/5 for 15 minutes or until 2.5V"),     # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 10 seconds"),                   # "Rest pulse" during 60 seconds at 75% SOC
                    pybamm.step.string("Discharge at C/5 for 15 minutes or until 2.5V"),     # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 10 seconds"),                   # "Rest pulse" during 60 seconds at 70% SOC
                    pybamm.step.string("Discharge at C/5 for 15 minutes or until 2.5V"),     # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 10 seconds"),                   # "Rest pulse" during 60 seconds at 65% SOC
                    pybamm.step.string("Discharge at C/5 for 15 minutes or until 2.5V"),     # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 10 seconds"),                   # "Rest pulse" during 60 seconds at 60% SOC
                    pybamm.step.string("Discharge at C/5 for 15 minutes or until 2.5V"),     # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 10 seconds"),                   # "Rest pulse" during 60 seconds at 55% SOC
                    pybamm.step.string("Discharge at C/5 for 15 minutes or until 2.5V"),     # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 10 seconds"),                   # "Rest pulse" during 60 seconds at 50% SOC
                    pybamm.step.string("Discharge at C/5 for 15 minutes or until 2.5V"),     # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 10 seconds"),                   # "Rest pulse" during 60 seconds at 45% SOC
                    pybamm.step.string("Discharge at C/5 for 15 minutes or until 2.5V"),     # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 10 seconds"),                   # "Rest pulse" during 60 seconds at 40% SOC
                    pybamm.step.string("Discharge at C/5 for 15 minutes or until 2.5V"),     # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 10 seconds"),                   # "Rest pulse" during 60 seconds at 35% SOC
                    pybamm.step.string("Discharge at C/5 for 15 minutes or until 2.5V"),     # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 10 seconds"),                   # "Rest pulse" during 60 seconds at 30% SOC
                    pybamm.step.string("Discharge at C/5 for 15 minutes or until 2.5V"),     # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 10 seconds"),                   # "Rest pulse" during 60 seconds at 25% SOC
                    pybamm.step.string("Discharge at C/5 for 15 minutes or until 2.5V"),     # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 10 seconds"),                   # "Rest pulse" during 60 seconds at 20% SOC
                    pybamm.step.string("Discharge at C/5 for 15 minutes or until 2.5V"),     # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 10 seconds"),                   # "Rest pulse" during 60 seconds at 15% SOC
                    pybamm.step.string("Discharge at C/5 for 15 minutes or until 2.5V"),     # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 10 seconds"),                   # "Rest pulse" during 60 seconds at 10% SOC
                    pybamm.step.string("Discharge at C/5 for 15 minutes or until 2.5V"),     # Discharge period 5.00% DOD
                    pybamm.step.string("Rest for 10 seconds"),                   # "Rest pulse" during 60 seconds at 5% SOC
                    pybamm.step.string("Discharge at C/5 until 2.5V"),     # Discharge period 5.00% DOD
                    pybamm.step.string("Hold at 2.5 V until C/20", direction="discharge"),
                    pybamm.step.string("Rest for 10 seconds"),                   # "Rest pulse" during 60 seconds at 0% SOC
                    pybamm.step.string("Rest for 10 minutes"),                  # Resting at 0.00% SOC
                    pybamm.step.string("Charge at C/5 for 15 minutes or until 4.2V"),        # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 10 seconds"),                   # "Rest pulse" during 60 seconds at 5% SOC
                    pybamm.step.string("Charge at C/5 for 15 minutes or until 4.2V"),        # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 10 seconds"),                   # "Rest pulse" during 60 seconds at 10% SOC
                    pybamm.step.string("Charge at C/5 for 15 minutes or until 4.2V"),        # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 10 seconds"),                   # "Rest pulse" during 60 seconds at 15% SOC
                    pybamm.step.string("Charge at C/5 for 15 minutes or until 4.2V"),        # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 10 seconds"),                   # "Rest pulse" during 60 seconds at 20% SOC
                    pybamm.step.string("Charge at C/5 for 15 minutes or until 4.2V"),        # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 10 seconds"),                   # "Rest pulse" during 60 seconds at 25% SOC
                    pybamm.step.string("Charge at C/5 for 15 minutes or until 4.2V"),        # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 10 seconds"),                   # "Rest pulse" during 60 seconds at 30% SOC
                    pybamm.step.string("Charge at C/5 for 15 minutes or until 4.2V"),        # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 10 seconds"),                   # "Rest pulse" during 60 seconds at 35% SOC
                    pybamm.step.string("Charge at C/5 for 15 minutes or until 4.2V"),        # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 10 seconds"),                   # "Rest pulse" during 60 seconds at 40% SOC
                    pybamm.step.string("Charge at C/5 for 15 minutes or until 4.2V"),        # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 10 seconds"),                   # "Rest pulse" during 60 seconds at 45% SOC
                    pybamm.step.string("Charge at C/5 for 15 minutes or until 4.2V"),        # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 10 seconds"),                   # "Rest pulse" during 60 seconds at 50% SOC
                    pybamm.step.string("Charge at C/5 for 15 minutes or until 4.2V"),        # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 10 seconds"),                   # "Rest pulse" during 60 seconds at 55% SOC
                    pybamm.step.string("Charge at C/5 for 15 minutes or until 4.2V"),        # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 10 seconds"),                   # "Rest pulse" during 60 seconds at 60% SOC
                    pybamm.step.string("Charge at C/5 for 15 minutes or until 4.2V"),        # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 10 seconds"),                   # "Rest pulse" during 60 seconds at 65% SOC
                    pybamm.step.string("Charge at C/5 for 15 minutes or until 4.2V"),        # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 10 seconds"),                   # "Rest pulse" during 60 seconds at 70% SOC
                    pybamm.step.string("Charge at C/5 for 15 minutes or until 4.2V"),        # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 10 seconds"),                   # "Rest pulse" during 60 seconds at 75% SOC
                    pybamm.step.string("Charge at C/5 for 15 minutes or until 4.2V"),        # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 10 seconds"),                   # "Rest pulse" during 60 seconds at 80% SOC
                    pybamm.step.string("Charge at C/5 for 15 minutes or until 4.2V"),        # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 10 seconds"),                   # "Rest pulse" during 60 seconds at 85% SOC
                    pybamm.step.string("Charge at C/5 for 15 minutes or until 4.2V"),        # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 10 seconds"),                   # "Rest pulse" during 60 seconds at 90% SOC
                    pybamm.step.string("Charge at C/5 for 15 minutes or until 4.2V"),        # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 10 seconds"),                   # "Rest pulse" during 60 seconds at 95% SOC
                    pybamm.step.string("Charge at C/5 until 4.2V"),        # Charge period 5.00% DOD
                    pybamm.step.string("Rest for 10 seconds"),                   # "Rest pulse" during 60 seconds at 100% SOC
                ],
                period=f"{period_value} seconds",   # output sampling
            )   
        else:
            experiment = pybamm.Experiment(
                [
                    
                ],
            )  
    elif  protocol_name == "ICI_short_rest_100soc":
        if fixed_period:
            experiment = pybamm.Experiment(
                [   #* Initial rest
                    pybamm.step.string("Rest for 5 minutes"),
                ]
                +
                [   #* 20 normal ICI steps during discharge with aprox 5% SOC variation each                 
                    pybamm.step.string("Discharge at C/5 for 15 minutes or until 2.5V"),     
                    pybamm.step.string("Rest for 10 seconds"),  
                ]*20
                +
                [   #* Try to ensure 0 %SOC
                    pybamm.step.string("Discharge at 0.5C until 2.5V"),
                    pybamm.step.string("Hold at 2.5 V until C/20", direction="discharge"),
                    pybamm.step.string("Rest for 5 minutes"),
                ]
                +
                [   #* 20 normal ICI steps during charge with aprox 5% SOC variation each
                    pybamm.step.string("Charge at C/5 for 15 minutes or until 4.2V"),        
                    pybamm.step.string("Rest for 10 seconds"),                   
                ]*20
                +
                [   #* Try to ensure one step at 100 %SOC   
                    pybamm.step.string("Charge at C/5 until 4.2V"),       
                    pybamm.step.string("Rest for 10 seconds"),                   
                ],
                period=f"{period_value} seconds",   # output sampling
            )   
        else:
            experiment = pybamm.Experiment(
                [
                    
                ],
            )  
    elif  protocol_name == "ICA":
        if fixed_period:
            experiment = pybamm.Experiment(
                [
                    pybamm.step.string("Rest for 30 minutes"),
                    pybamm.step.string("Discharge at 1C until 2.5 V"),
                    pybamm.step.string("Rest for 30 minutes"),
                    pybamm.step.string("Charge at 1C until 4.2 V"),
                    pybamm.step.string("Hold at 4.2 V until C/50"),
                    pybamm.step.string("Rest for 30 minutes"),
                    pybamm.step.string("Hold at 4.2 V until C/50"),
                    pybamm.step.string("Rest for 60 minutes"),                  # Resting at 100.00% SOC
                    pybamm.step.string("Discharge at C/40 until 2.5V"),         # Discharge period at C/40
                    pybamm.step.string("Rest for 60 minutes"),                  # Resting at 0.00% SOC
                    pybamm.step.string("Charge at C/40 until 4.2V"),            # Charge period at C/40
                    pybamm.step.string("Rest for 1 minutes"),                   # Resting at 100.00% SOC
                ],
                period=f"{period_value} seconds",   # output sampling
            )   
        else:
            experiment = pybamm.Experiment(
                [
                    pybamm.step.string("Rest for 30 minutes", period="10 seconds"),
                    pybamm.step.string("Discharge at 1C until 2.5 V", period="1 seconds"),
                    pybamm.step.string("Rest for 30 minutes", period="10 seconds"),
                    pybamm.step.string("Charge at 1C until 4.2 V", period="1 seconds"),
                    pybamm.step.string("Hold at 4.2 V until C/50", period="1 seconds"),
                    pybamm.step.string("Rest for 30 minutes", period="10 seconds"),
                    pybamm.step.string("Hold at 4.2 V until C/50", period="1 seconds"),
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                  # Resting at 100.00% SOC
                    pybamm.step.string("Discharge at C/40 until 2.5V", period="0.01 seconds"),         # Discharge period at C/40
                    pybamm.step.string("Rest for 60 minutes", period="10 seconds"),                  # Resting at 0.00% SOC
                    pybamm.step.string("Charge at C/40 until 4.2V", period="0.01 seconds"),            # Charge period at C/40
                    pybamm.step.string("Rest for 1 minutes", period="10 seconds"),                   # Resting at 100.00% SOC
                ],
            )
    elif  protocol_name == "ICA_short_rest_100soc":
        if fixed_period:
            experiment = pybamm.Experiment(
                [
                    pybamm.step.string("Rest for 5 minutes"),                  # Resting at 100.00% SOC
                    pybamm.step.string("Discharge at C/20 until 2.5V"),         # Discharge period at C/40
                    pybamm.step.string("Rest for 5 minutes"),                  # Resting at 0.00% SOC
                    pybamm.step.string("Charge at C/20 until 4.2V"),            # Charge period at C/40
                    pybamm.step.string("Rest for 1 minutes"),                   # Resting at 100.00% SOC
                ],
                period=f"{period_value} seconds",   # output sampling
            )   
        else:
            experiment = pybamm.Experiment(
                [
                ],
            )
    elif  protocol_name == "FUDS":
        # 1) Load USABC Table 5A-1 (time [s] vs power fraction of peak)
        df = pd.read_csv(f"hack\\model\\drive_cycles\\usabc_fuds_test_cycle.csv")  # columns: time_s, power_frac
        t_s = df["time_s"].to_numpy(dtype=float)
        frac = df["power_frac"].to_numpy(dtype=float)

        # Safety: PyBaMM drive cycle must start at t=0
        assert t_s[0] == 0.0, "Drive cycle must start at t=0 s"

        # 2) Scale to absolute power [W] using your test-plan peak power
        P_peak_W = 80.0  # <-- set this for your cell/module/pack
        # If USABC convention is negative=discharge, flip to PyBaMM (positive=discharge)
        P_cmd_W = -frac * P_peak_W

        drive_cycle_power = np.column_stack([t_s, P_cmd_W])

        if fixed_period:
            # 3) Create a power-controlled drive-cycle step
            fuds_step = pybamm.step.power(
                drive_cycle_power,
                duration=f"{t_s[-1]} seconds",   # explicit is safer
                period=f"{period_value} second",              # matches USABC 1 s steps / logging
                termination=[pybamm.step.VoltageTermination(2.51, operator="<"), pybamm.step.VoltageTermination(4.2, operator=">")],            # optional: add your cutoff
            )

            # 4) Repeat the profile N times (as cycles)
            experiment = pybamm.Experiment([
                # pybamm.step.string("Rest for 30 minutes"),
                # pybamm.step.string("Discharge at 1C until 2.5 V"),
                # pybamm.step.string("Hold at 2.5 V until C/50", direction="discharge"),
                # pybamm.step.string("Rest for 30 minutes"),
                pybamm.step.string("Charge at 1C until 4.2 V"),
                # pybamm.step.string("Hold at 4.2 V until C/50"),
                # pybamm.step.string("Rest for 30 minutes"),
                pybamm.step.string("Hold at 4.2 V until C/50", direction = "charge"),
                fuds_step.copy(),
                fuds_step.copy(),
                fuds_step.copy(),
                fuds_step.copy(),
                fuds_step.copy(),
                # pybamm.step.string("Charge at 1C for 0.5 hour"),
                pybamm.step.string("Rest for 30 seconds"),
                ],
                period=f"{period_value} seconds",
                )
        else:
            # 3) Create a power-controlled drive-cycle step
            fuds_step = pybamm.step.power(
                drive_cycle_power,
                duration=f"{t_s[-1]} seconds",   # explicit is safer
                period="0.001 second",              # matches USABC 1 s steps / logging
                termination=[pybamm.step.VoltageTermination(2.51, operator="<"), pybamm.step.VoltageTermination(4.2, operator=">")],            # optional: add your cutoff
            )

            # 4) Repeat the profile N times (as cycles)
            experiment = pybamm.Experiment([
                # pybamm.step.string("Rest for 30 minutes", period="10 seconds"),
                # pybamm.step.string("Discharge at 1C until 2.5 V", period="1 seconds"),
                # pybamm.step.string("Hold at 2.5 V until C/50", direction="discharge", period="1 seconds"),
                # pybamm.step.string("Rest for 30 minutes", period="10 seconds"),
                pybamm.step.string("Charge at 1C until 4.2 V", period="1 seconds"),
                # pybamm.step.string("Hold at 4.2 V until C/50", period="1 seconds"),
                # pybamm.step.string("Rest for 30 minutes", period="10 seconds"),
                pybamm.step.string("Hold at 4.2 V until C/50", direction = "charge", period="1 seconds"),
                fuds_step.copy(),
                fuds_step.copy(),
                fuds_step.copy(),
                fuds_step.copy(),
                fuds_step.copy(),
                # pybamm.step.string("Charge at 1C for 0.5 hour", period="1 seconds"),
                pybamm.step.string("Rest for 30 seconds", period="1 seconds"),                
                ])
    elif  protocol_name == "FUDS_short_rest_100soc":
        # 1) Load USABC Table 5A-1 (time [s] vs power fraction of peak)
        df = pd.read_csv(f"hack\\model\\drive_cycles\\usabc_fuds_test_cycle.csv")  # columns: time_s, power_frac
        t_s = df["time_s"].to_numpy(dtype=float)
        frac = df["power_frac"].to_numpy(dtype=float)

        # Safety: PyBaMM drive cycle must start at t=0
        assert t_s[0] == 0.0, "Drive cycle must start at t=0 s"

        # 2) Scale to absolute power [W] using your test-plan peak power
        P_peak_W = 80.0  # <-- set this for your cell/module/pack
        # If USABC convention is negative=discharge, flip to PyBaMM (positive=discharge)
        P_cmd_W = -frac * P_peak_W

        drive_cycle_power = np.column_stack([t_s, P_cmd_W])

        if fixed_period:
            # 3) Create a power-controlled drive-cycle step
            fuds_step = pybamm.step.power(
                drive_cycle_power,
                duration=f"{t_s[-1]} seconds",   # explicit is safer
                period=f"{period_value} second",              # matches USABC 1 s steps / logging
                termination=[pybamm.step.VoltageTermination(2.51, operator="<"), pybamm.step.VoltageTermination(4.2, operator=">")],            # optional: add your cutoff
            )

            # 4) Repeat the profile N times (as cycles)
            experiment = pybamm.Experiment([
                pybamm.step.string("Rest for 5 minutes"),
                fuds_step.copy(),
                fuds_step.copy(),
                fuds_step.copy(),
                fuds_step.copy(),
                fuds_step.copy(),
                # pybamm.step.string("Charge at 1C for 0.5 hour"),
                pybamm.step.string("Rest for 60 seconds"),
                ],
                period=f"{period_value} seconds",
                )
        else:
            True
    elif  protocol_name == "FUDS_short_rest_100soc_eval":
        # 1) Load USABC Table 5A-1 (time [s] vs power fraction of peak)
        df = pd.read_csv(f"hack\\model\\drive_cycles\\usabc_fuds_test_cycle.csv")  # columns: time_s, power_frac
        t_s = df["time_s"].to_numpy(dtype=float)
        frac = df["power_frac"].to_numpy(dtype=float)

        # Safety: PyBaMM drive cycle must start at t=0
        assert t_s[0] == 0.0, "Drive cycle must start at t=0 s"

        # 2) Scale to absolute power [W] using your test-plan peak power
        P_peak_W = 80.0  # <-- set this for your cell/module/pack
        # If USABC convention is negative=discharge, flip to PyBaMM (positive=discharge)
        P_cmd_W = -frac * P_peak_W

        drive_cycle_power = np.column_stack([t_s, P_cmd_W])

        if fixed_period:
            # 3) Create a power-controlled drive-cycle step
            fuds_step = pybamm.step.power(
                drive_cycle_power,
                duration=f"{t_s[-1]} seconds",   # explicit is safer
                period=f"{period_value} second",              # matches USABC 1 s steps / logging
                termination=[pybamm.step.VoltageTermination(2.51, operator="<"), pybamm.step.VoltageTermination(4.2, operator=">")],            # optional: add your cutoff
            )

            # 4) Repeat the profile N times (as cycles)
            experiment = pybamm.Experiment([
                fuds_step.copy(),
                ],
                period=f"{period_value} seconds",
                )
        else:
            True
    elif  protocol_name == "FUDS_short_rest_090soc_eval":
        # 1) Load USABC Table 5A-1 (time [s] vs power fraction of peak)
        df = pd.read_csv(f"hack\\model\\drive_cycles\\usabc_fuds_test_cycle.csv")  # columns: time_s, power_frac
        t_s = df["time_s"].to_numpy(dtype=float)
        frac = df["power_frac"].to_numpy(dtype=float)

        # Safety: PyBaMM drive cycle must start at t=0
        assert t_s[0] == 0.0, "Drive cycle must start at t=0 s"

        # 2) Scale to absolute power [W] using your test-plan peak power
        P_peak_W = 80.0  # <-- set this for your cell/module/pack
        # If USABC convention is negative=discharge, flip to PyBaMM (positive=discharge)
        P_cmd_W = -frac * P_peak_W

        drive_cycle_power = np.column_stack([t_s, P_cmd_W])

        if fixed_period:
            # 3) Create a power-controlled drive-cycle step
            fuds_step = pybamm.step.power(
                drive_cycle_power,
                duration=f"{t_s[-1]} seconds",   # explicit is safer
                period=f"{period_value} second",              # matches USABC 1 s steps / logging
                termination=[pybamm.step.VoltageTermination(2.51, operator="<"), pybamm.step.VoltageTermination(4.2, operator=">")],            # optional: add your cutoff
            )

            # 4) Repeat the profile N times (as cycles)
            experiment = pybamm.Experiment([
                fuds_step.copy(),
                ],
                period=f"{period_value} seconds",
                )
        else:
            True
    elif  protocol_name == "FUDS_short_rest_080soc_eval":
        # 1) Load USABC Table 5A-1 (time [s] vs power fraction of peak)
        df = pd.read_csv(f"hack\\model\\drive_cycles\\usabc_fuds_test_cycle.csv")  # columns: time_s, power_frac
        t_s = df["time_s"].to_numpy(dtype=float)
        frac = df["power_frac"].to_numpy(dtype=float)

        # Safety: PyBaMM drive cycle must start at t=0
        assert t_s[0] == 0.0, "Drive cycle must start at t=0 s"

        # 2) Scale to absolute power [W] using your test-plan peak power
        P_peak_W = 80.0  # <-- set this for your cell/module/pack
        # If USABC convention is negative=discharge, flip to PyBaMM (positive=discharge)
        P_cmd_W = -frac * P_peak_W

        drive_cycle_power = np.column_stack([t_s, P_cmd_W])

        if fixed_period:
            # 3) Create a power-controlled drive-cycle step
            fuds_step = pybamm.step.power(
                drive_cycle_power,
                duration=f"{t_s[-1]} seconds",   # explicit is safer
                period=f"{period_value} second",              # matches USABC 1 s steps / logging
                termination=[pybamm.step.VoltageTermination(2.51, operator="<"), pybamm.step.VoltageTermination(4.2, operator=">")],            # optional: add your cutoff
            )

            # 4) Repeat the profile N times (as cycles)
            experiment = pybamm.Experiment([
                fuds_step.copy(),
                ],
                period=f"{period_value} seconds",
                )
        else:
            True
    elif  protocol_name == "FUDS_short_rest_070soc_eval":
        # 1) Load USABC Table 5A-1 (time [s] vs power fraction of peak)
        df = pd.read_csv(f"hack\\model\\drive_cycles\\usabc_fuds_test_cycle.csv")  # columns: time_s, power_frac
        t_s = df["time_s"].to_numpy(dtype=float)
        frac = df["power_frac"].to_numpy(dtype=float)

        # Safety: PyBaMM drive cycle must start at t=0
        assert t_s[0] == 0.0, "Drive cycle must start at t=0 s"

        # 2) Scale to absolute power [W] using your test-plan peak power
        P_peak_W = 80.0  # <-- set this for your cell/module/pack
        # If USABC convention is negative=discharge, flip to PyBaMM (positive=discharge)
        P_cmd_W = -frac * P_peak_W

        drive_cycle_power = np.column_stack([t_s, P_cmd_W])

        if fixed_period:
            # 3) Create a power-controlled drive-cycle step
            fuds_step = pybamm.step.power(
                drive_cycle_power,
                duration=f"{t_s[-1]} seconds",   # explicit is safer
                period=f"{period_value} second",              # matches USABC 1 s steps / logging
                #termination=[pybamm.step.VoltageTermination(2.51, operator="<"), pybamm.step.VoltageTermination(4.2, operator=">")],            # optional: add your cutoff
            )

            # 4) Repeat the profile N times (as cycles)
            experiment = pybamm.Experiment([
                fuds_step.copy(),
                ],
                period=f"{period_value} seconds",
                )
        else:
            True
    elif  protocol_name == "FUDS_short_rest_060soc_eval":
        # 1) Load USABC Table 5A-1 (time [s] vs power fraction of peak)
        df = pd.read_csv(f"hack\\model\\drive_cycles\\usabc_fuds_test_cycle.csv")  # columns: time_s, power_frac
        t_s = df["time_s"].to_numpy(dtype=float)
        frac = df["power_frac"].to_numpy(dtype=float)

        # Safety: PyBaMM drive cycle must start at t=0
        assert t_s[0] == 0.0, "Drive cycle must start at t=0 s"

        # 2) Scale to absolute power [W] using your test-plan peak power
        P_peak_W = 80.0  # <-- set this for your cell/module/pack
        # If USABC convention is negative=discharge, flip to PyBaMM (positive=discharge)
        P_cmd_W = -frac * P_peak_W

        drive_cycle_power = np.column_stack([t_s, P_cmd_W])

        if fixed_period:
            # 3) Create a power-controlled drive-cycle step
            fuds_step = pybamm.step.power(
                drive_cycle_power,
                duration=f"{t_s[-1]} seconds",   # explicit is safer
                period=f"{period_value} second",              # matches USABC 1 s steps / logging
                termination=[pybamm.step.VoltageTermination(2.51, operator="<"), pybamm.step.VoltageTermination(4.2, operator=">")],            # optional: add your cutoff
            )

            # 4) Repeat the profile N times (as cycles)
            experiment = pybamm.Experiment([
                fuds_step.copy(),
                ],
                period=f"{period_value} seconds",
                )
        else:
            True
    elif  protocol_name == "FUDS_short_rest_050soc_eval":
        # 1) Load USABC Table 5A-1 (time [s] vs power fraction of peak)
        df = pd.read_csv(f"hack\\model\\drive_cycles\\usabc_fuds_test_cycle.csv")  # columns: time_s, power_frac
        t_s = df["time_s"].to_numpy(dtype=float)
        frac = df["power_frac"].to_numpy(dtype=float)

        # Safety: PyBaMM drive cycle must start at t=0
        assert t_s[0] == 0.0, "Drive cycle must start at t=0 s"

        # 2) Scale to absolute power [W] using your test-plan peak power
        P_peak_W = 80.0  # <-- set this for your cell/module/pack
        # If USABC convention is negative=discharge, flip to PyBaMM (positive=discharge)
        P_cmd_W = -frac * P_peak_W

        drive_cycle_power = np.column_stack([t_s, P_cmd_W])

        if fixed_period:
            # 3) Create a power-controlled drive-cycle step
            fuds_step = pybamm.step.power(
                drive_cycle_power,
                duration=f"{t_s[-1]} seconds",   # explicit is safer
                period=f"{period_value} second",              # matches USABC 1 s steps / logging
                #termination=[pybamm.step.VoltageTermination(2.51, operator="<"), pybamm.step.VoltageTermination(4.2, operator=">")],            # optional: add your cutoff
            )

            # 4) Repeat the profile N times (as cycles)
            experiment = pybamm.Experiment([
                fuds_step.copy(),
                ],
                period=f"{period_value} seconds",
                )
        else:
            True
    elif  protocol_name == "FUDS_short_rest_040soc_eval":
        # 1) Load USABC Table 5A-1 (time [s] vs power fraction of peak)
        df = pd.read_csv(f"hack\\model\\drive_cycles\\usabc_fuds_test_cycle.csv")  # columns: time_s, power_frac
        t_s = df["time_s"].to_numpy(dtype=float)
        frac = df["power_frac"].to_numpy(dtype=float)

        # Safety: PyBaMM drive cycle must start at t=0
        assert t_s[0] == 0.0, "Drive cycle must start at t=0 s"

        # 2) Scale to absolute power [W] using your test-plan peak power
        P_peak_W = 80.0  # <-- set this for your cell/module/pack
        # If USABC convention is negative=discharge, flip to PyBaMM (positive=discharge)
        P_cmd_W = -frac * P_peak_W

        drive_cycle_power = np.column_stack([t_s, P_cmd_W])

        if fixed_period:
            # 3) Create a power-controlled drive-cycle step
            fuds_step = pybamm.step.power(
                drive_cycle_power,
                duration=f"{t_s[-1]} seconds",   # explicit is safer
                period=f"{period_value} second",              # matches USABC 1 s steps / logging
                #termination=[pybamm.step.VoltageTermination(2.51, operator="<"), pybamm.step.VoltageTermination(4.2, operator=">")],            # optional: add your cutoff
            )

            # 4) Repeat the profile N times (as cycles)
            experiment = pybamm.Experiment([
                fuds_step.copy(),
                ],
                period=f"{period_value} seconds",
                )
        else:
            True
    elif  protocol_name == "FUDS_short_rest_030soc_eval":
        # 1) Load USABC Table 5A-1 (time [s] vs power fraction of peak)
        df = pd.read_csv(f"hack\\model\\drive_cycles\\usabc_fuds_test_cycle.csv")  # columns: time_s, power_frac
        t_s = df["time_s"].to_numpy(dtype=float)
        frac = df["power_frac"].to_numpy(dtype=float)

        # Safety: PyBaMM drive cycle must start at t=0
        assert t_s[0] == 0.0, "Drive cycle must start at t=0 s"

        # 2) Scale to absolute power [W] using your test-plan peak power
        P_peak_W = 80.0  # <-- set this for your cell/module/pack
        # If USABC convention is negative=discharge, flip to PyBaMM (positive=discharge)
        P_cmd_W = -frac * P_peak_W

        drive_cycle_power = np.column_stack([t_s, P_cmd_W])

        if fixed_period:
            # 3) Create a power-controlled drive-cycle step
            fuds_step = pybamm.step.power(
                drive_cycle_power,
                duration=f"{t_s[-1]} seconds",   # explicit is safer
                period=f"{period_value} second",              # matches USABC 1 s steps / logging
                termination=[pybamm.step.VoltageTermination(2.51, operator="<"), pybamm.step.VoltageTermination(4.2, operator=">")],            # optional: add your cutoff
            )

            # 4) Repeat the profile N times (as cycles)
            experiment = pybamm.Experiment([
                fuds_step.copy(),
                ],
                period=f"{period_value} seconds",
                )
        else:
            True
    elif  protocol_name == "FUDS_short_rest_020soc_eval":
        # 1) Load USABC Table 5A-1 (time [s] vs power fraction of peak)
        df = pd.read_csv(f"hack\\model\\drive_cycles\\usabc_fuds_test_cycle.csv")  # columns: time_s, power_frac
        t_s = df["time_s"].to_numpy(dtype=float)
        frac = df["power_frac"].to_numpy(dtype=float)

        # Safety: PyBaMM drive cycle must start at t=0
        assert t_s[0] == 0.0, "Drive cycle must start at t=0 s"

        # 2) Scale to absolute power [W] using your test-plan peak power
        P_peak_W = 80.0  # <-- set this for your cell/module/pack
        # If USABC convention is negative=discharge, flip to PyBaMM (positive=discharge)
        P_cmd_W = -frac * P_peak_W

        drive_cycle_power = np.column_stack([t_s, P_cmd_W])

        if fixed_period:
            # 3) Create a power-controlled drive-cycle step
            fuds_step = pybamm.step.power(
                drive_cycle_power,
                duration=f"{t_s[-1]} seconds",   # explicit is safer
                period=f"{period_value} second",              # matches USABC 1 s steps / logging
                termination=[pybamm.step.VoltageTermination(2.51, operator="<"), pybamm.step.VoltageTermination(4.2, operator=">")],            # optional: add your cutoff
            )

            # 4) Repeat the profile N times (as cycles)
            experiment = pybamm.Experiment([
                fuds_step.copy(),
                ],
                period=f"{period_value} seconds",
                )
        else:
            True
    elif  protocol_name == "FUDS_short_rest_010soc_eval":
        # 1) Load USABC Table 5A-1 (time [s] vs power fraction of peak)
        df = pd.read_csv(f"hack\\model\\drive_cycles\\usabc_fuds_test_cycle.csv")  # columns: time_s, power_frac
        t_s = df["time_s"].to_numpy(dtype=float)
        frac = df["power_frac"].to_numpy(dtype=float)

        # Safety: PyBaMM drive cycle must start at t=0
        assert t_s[0] == 0.0, "Drive cycle must start at t=0 s"

        # 2) Scale to absolute power [W] using your test-plan peak power
        P_peak_W = 80.0  # <-- set this for your cell/module/pack
        # If USABC convention is negative=discharge, flip to PyBaMM (positive=discharge)
        P_cmd_W = -frac * P_peak_W

        drive_cycle_power = np.column_stack([t_s, P_cmd_W])

        if fixed_period:
            # 3) Create a power-controlled drive-cycle step
            fuds_step = pybamm.step.power(
                drive_cycle_power,
                duration=f"{t_s[-1]} seconds",   # explicit is safer
                period=f"{period_value} second",              # matches USABC 1 s steps / logging
                termination=[pybamm.step.VoltageTermination(2.51, operator="<"), pybamm.step.VoltageTermination(4.2, operator=">")],            # optional: add your cutoff
            )

            # 4) Repeat the profile N times (as cycles)
            experiment = pybamm.Experiment([
                fuds_step.copy(),
                ],
                period=f"{period_value} seconds",
                )
        else:
            True
    elif  protocol_name == "FUDS_short_rest_000soc_eval":
        # 1) Load USABC Table 5A-1 (time [s] vs power fraction of peak)
        df = pd.read_csv(f"hack\\model\\drive_cycles\\usabc_fuds_test_cycle.csv")  # columns: time_s, power_frac
        t_s = df["time_s"].to_numpy(dtype=float)
        frac = df["power_frac"].to_numpy(dtype=float)

        # Safety: PyBaMM drive cycle must start at t=0
        assert t_s[0] == 0.0, "Drive cycle must start at t=0 s"

        # 2) Scale to absolute power [W] using your test-plan peak power
        P_peak_W = 80.0  # <-- set this for your cell/module/pack
        # If USABC convention is negative=discharge, flip to PyBaMM (positive=discharge)
        P_cmd_W = -frac * P_peak_W

        drive_cycle_power = np.column_stack([t_s, P_cmd_W])

        if fixed_period:
            # 3) Create a power-controlled drive-cycle step
            fuds_step = pybamm.step.power(
                drive_cycle_power,
                duration=f"{t_s[-1]} seconds",   # explicit is safer
                period=f"{period_value} second",              # matches USABC 1 s steps / logging
                termination=[pybamm.step.VoltageTermination(2.51, operator="<"), pybamm.step.VoltageTermination(4.2, operator=">")],            # optional: add your cutoff
            )

            # 4) Repeat the profile N times (as cycles)
            experiment = pybamm.Experiment([
                fuds_step.copy(),
                ],
                period=f"{period_value} seconds",
                )
        else:
            True
    elif  protocol_name == "DST":
        # 1) Load USABC Table 5A-1 (time [s] vs power fraction of peak)
        df = pd.read_csv(f"hack\\model\\drive_cycles\\usabc_dst_test_cycle.csv")  # columns: time_s, power_frac

        if fixed_period:
            # 3) Create a power-controlled drive-cycle step
            drive_cycle_power = build_drive_cycle_power_from_dst_table(
                df,
                p_peak_w=80.0,
                dt=1,  # DST total is 360 s, so 1 s is usually fine; use 0.1 for sharper steps
            )

            # TODO: comprobar la forma del dst_step vs fuds_step porque me da la impresion de que puede ser porque empieza por cero
            decimals = len(str(period_value).split(".")[1]) if "." in str(period_value) else 0
            drive_cycle_power = drive_cycle_power.round(decimals)

            dst_step = pybamm.step.power(
                drive_cycle_power,
                duration=f"{360} seconds",
                period=f"{period_value} second",              # matches USABC 1 s steps / logging
                termination=[pybamm.step.VoltageTermination(2.51, operator="<"), pybamm.step.VoltageTermination(4.2, operator=">")],            # optional: add your cutoff
            )

            # 4) Repeat the profile N times (as cycles)
            experiment = pybamm.Experiment([
                pybamm.step.string("Charge at 1C until 4.2 V"),
                pybamm.step.string("Hold at 4.2 V until C/50", direction = "charge"),
                pybamm.step.string("Rest for 30 minutes"),
                dst_step.copy(), # 1
                dst_step.copy(), # 2
                dst_step.copy(), # 3
                dst_step.copy(), # 4
                dst_step.copy(), # 5
                dst_step.copy(), # 6
                dst_step.copy(), # 7
                dst_step.copy(), # 8
                dst_step.copy(), # 9
                dst_step.copy(), # 10
                dst_step.copy(), # 11
                dst_step.copy(), # 12
                dst_step.copy(), # 13
                dst_step.copy(), # 14
                pybamm.step.string("Charge at 1C until 4.2 V"),
                pybamm.step.string("Hold at 4.2 V until C/50", direction = "charge"),
                pybamm.step.string("Rest for 30 minutes"),
                ],
                period=f"{period_value} seconds",
                )
        else:
            # 3) Create a power-controlled drive-cycle step
            drive_cycle_power = build_drive_cycle_power_from_dst_table(
                df,
                p_peak_w=80.0,
                dt=0.001,  # DST total is 360 s, so 1 s is usually fine; use 0.1 for sharper steps
            )

            dst_step = pybamm.step.power(
                drive_cycle_power,
                period=f"{0.001} second",              # matches USABC 1 s steps / logging
                termination=[pybamm.step.VoltageTermination(2.51, operator="<"), pybamm.step.VoltageTermination(4.2, operator=">")],            # optional: add your cutoff
            )

            # 4) Repeat the profile N times (as cycles)
            experiment = pybamm.Experiment([
                pybamm.step.string("Charge at 1C until 4.2 V", period="1 seconds"),
                pybamm.step.string("Hold at 4.2 V until C/50", direction = "charge", period="1 seconds"),
                pybamm.step.string("Rest for 30 minutes", period="10 seconds"),
                dst_step.copy(), # 1
                dst_step.copy(), # 2
                dst_step.copy(), # 3
                dst_step.copy(), # 4
                dst_step.copy(), # 5
                dst_step.copy(), # 6
                dst_step.copy(), # 7
                dst_step.copy(), # 8
                dst_step.copy(), # 9
                dst_step.copy(), # 10
                dst_step.copy(), # 11
                dst_step.copy(), # 12
                dst_step.copy(), # 13
                dst_step.copy(), # 14
                pybamm.step.string("Charge at 1C until 4.2 V", period="1 seconds"),
                pybamm.step.string("Hold at 4.2 V until C/50", direction = "charge", period="1 seconds"),
                pybamm.step.string("Rest for 30 minutes", period="10 seconds"),            
                ])
    elif  protocol_name == "DST_short_rest_100soc":
        # 1) Load USABC Table 5A-1 (time [s] vs power fraction of peak)
        df = pd.read_csv(f"hack\\model\\drive_cycles\\usabc_dst_test_cycle.csv")  # columns: time_s, power_frac

        if fixed_period:
            # 3) Create a power-controlled drive-cycle step
            drive_cycle_power = build_drive_cycle_power_from_dst_table(
                df,
                p_peak_w=80.0,
                dt=1,  # DST total is 360 s, so 1 s is usually fine; use 0.1 for sharper steps
            )

            # TODO: comprobar la forma del dst_step vs fuds_step porque me da la impresion de que puede ser porque empieza por cero
            decimals = len(str(period_value).split(".")[1]) if "." in str(period_value) else 0
            drive_cycle_power = drive_cycle_power.round(decimals)

            dst_step = pybamm.step.power(
                drive_cycle_power,
                duration=f"{360} seconds",
                period=f"{period_value} second",              # matches USABC 1 s steps / logging
                termination=[pybamm.step.VoltageTermination(2.51, operator="<"), pybamm.step.VoltageTermination(4.2, operator=">")],            
            )

            # 4) Repeat the profile N times (as cycles)
            experiment = pybamm.Experiment([
                pybamm.step.string("Rest for 5 minutes"),
                dst_step.copy(), # 1
                dst_step.copy(), # 2
                dst_step.copy(), # 3
                dst_step.copy(), # 4
                dst_step.copy(), # 5
                dst_step.copy(), # 6
                dst_step.copy(), # 7
                dst_step.copy(), # 8
                dst_step.copy(), # 9
                dst_step.copy(), # 10
                dst_step.copy(), # 11
                dst_step.copy(), # 12
                dst_step.copy(), # 13
                dst_step.copy(), # 14
                pybamm.step.string("Rest for 5 minutes"),
                ],
                period=f"{period_value} seconds",
                )
        else:
            # 3) Create a power-controlled drive-cycle step
            drive_cycle_power = build_drive_cycle_power_from_dst_table(
                df,
                p_peak_w=80.0,
                dt=0.001,  # DST total is 360 s, so 1 s is usually fine; use 0.1 for sharper steps
            )

            dst_step = pybamm.step.power(
                drive_cycle_power,
                period=f"{0.001} second",              # matches USABC 1 s steps / logging
                termination=[pybamm.step.VoltageTermination(2.51, operator="<"), pybamm.step.VoltageTermination(4.2, operator=">")],            # optional: add your cutoff
            )

            # 4) Repeat the profile N times (as cycles)
            experiment = pybamm.Experiment([
        
                ])
    elif  protocol_name == "DST_short_rest_100soc_eval":
        # 1) Load USABC Table 5A-1 (time [s] vs power fraction of peak)
        df = pd.read_csv(f"hack\\model\\drive_cycles\\usabc_dst_test_cycle.csv")  # columns: time_s, power_frac

        if fixed_period:
            # 3) Create a power-controlled drive-cycle step
            drive_cycle_power = build_drive_cycle_power_from_dst_table(
                df,
                p_peak_w=80.0,
                dt=1,  # DST total is 360 s, so 1 s is usually fine; use 0.1 for sharper steps
            )

            # TODO: comprobar la forma del dst_step vs fuds_step porque me da la impresion de que puede ser porque empieza por cero
            decimals = len(str(period_value).split(".")[1]) if "." in str(period_value) else 0
            drive_cycle_power = drive_cycle_power.round(decimals)

            dst_step = pybamm.step.power(
                drive_cycle_power,
                duration=f"{360} seconds",
                period=f"{period_value} second",              # matches USABC 1 s steps / logging
                termination=[pybamm.step.VoltageTermination(2.51, operator="<"), pybamm.step.VoltageTermination(4.2, operator=">")],            
            )

            # 4) Repeat the profile N times (as cycles)
            experiment = pybamm.Experiment([
                dst_step.copy(), # 1
                ],
                period=f"{period_value} seconds",
                )
        else:
            True
    elif  protocol_name == "DST_short_rest_090soc_eval":
        # 1) Load USABC Table 5A-1 (time [s] vs power fraction of peak)
        df = pd.read_csv(f"hack\\model\\drive_cycles\\usabc_dst_test_cycle.csv")  # columns: time_s, power_frac

        if fixed_period:
            # 3) Create a power-controlled drive-cycle step
            drive_cycle_power = build_drive_cycle_power_from_dst_table(
                df,
                p_peak_w=80.0,
                dt=1,  # DST total is 360 s, so 1 s is usually fine; use 0.1 for sharper steps
            )

            # TODO: comprobar la forma del dst_step vs fuds_step porque me da la impresion de que puede ser porque empieza por cero
            decimals = len(str(period_value).split(".")[1]) if "." in str(period_value) else 0
            drive_cycle_power = drive_cycle_power.round(decimals)

            dst_step = pybamm.step.power(
                drive_cycle_power,
                duration=f"{360} seconds",
                period=f"{period_value} second",              # matches USABC 1 s steps / logging
                termination=[pybamm.step.VoltageTermination(2.51, operator="<"), pybamm.step.VoltageTermination(4.2, operator=">")],            
            )

            # 4) Repeat the profile N times (as cycles)
            experiment = pybamm.Experiment([
                dst_step.copy(), # 1
                ],
                period=f"{period_value} seconds",
                )
        else:
            True
    elif  protocol_name == "DST_short_rest_080soc_eval":
        # 1) Load USABC Table 5A-1 (time [s] vs power fraction of peak)
        df = pd.read_csv(f"hack\\model\\drive_cycles\\usabc_dst_test_cycle.csv")  # columns: time_s, power_frac

        if fixed_period:
            # 3) Create a power-controlled drive-cycle step
            drive_cycle_power = build_drive_cycle_power_from_dst_table(
                df,
                p_peak_w=80.0,
                dt=1,  # DST total is 360 s, so 1 s is usually fine; use 0.1 for sharper steps
            )

            # TODO: comprobar la forma del dst_step vs fuds_step porque me da la impresion de que puede ser porque empieza por cero
            decimals = len(str(period_value).split(".")[1]) if "." in str(period_value) else 0
            drive_cycle_power = drive_cycle_power.round(decimals)

            dst_step = pybamm.step.power(
                drive_cycle_power,
                duration=f"{360} seconds",
                period=f"{period_value} second",              # matches USABC 1 s steps / logging
                termination=[pybamm.step.VoltageTermination(2.51, operator="<"), pybamm.step.VoltageTermination(4.2, operator=">")],            
            )

            # 4) Repeat the profile N times (as cycles)
            experiment = pybamm.Experiment([
                dst_step.copy(), # 1
                ],
                period=f"{period_value} seconds",
                )
        else:
            True
    elif  protocol_name == "DST_short_rest_070soc_eval":
        # 1) Load USABC Table 5A-1 (time [s] vs power fraction of peak)
        df = pd.read_csv(f"hack\\model\\drive_cycles\\usabc_dst_test_cycle.csv")  # columns: time_s, power_frac

        if fixed_period:
            # 3) Create a power-controlled drive-cycle step
            drive_cycle_power = build_drive_cycle_power_from_dst_table(
                df,
                p_peak_w=80.0,
                dt=1,  # DST total is 360 s, so 1 s is usually fine; use 0.1 for sharper steps
            )

            # TODO: comprobar la forma del dst_step vs fuds_step porque me da la impresion de que puede ser porque empieza por cero
            decimals = len(str(period_value).split(".")[1]) if "." in str(period_value) else 0
            drive_cycle_power = drive_cycle_power.round(decimals)

            dst_step = pybamm.step.power(
                drive_cycle_power,
                duration=f"{360} seconds",
                period=f"{period_value} second",              # matches USABC 1 s steps / logging
            #    termination=[pybamm.step.VoltageTermination(2.51, operator="<"), pybamm.step.VoltageTermination(4.2, operator=">")],            
            )

            # 4) Repeat the profile N times (as cycles)
            experiment = pybamm.Experiment([
                dst_step.copy(), # 1
                ],
                period=f"{period_value} seconds",
                )
        else:
            True
    elif  protocol_name == "DST_short_rest_060soc_eval":
        # 1) Load USABC Table 5A-1 (time [s] vs power fraction of peak)
        df = pd.read_csv(f"hack\\model\\drive_cycles\\usabc_dst_test_cycle.csv")  # columns: time_s, power_frac

        if fixed_period:
            # 3) Create a power-controlled drive-cycle step
            drive_cycle_power = build_drive_cycle_power_from_dst_table(
                df,
                p_peak_w=80.0,
                dt=1,  # DST total is 360 s, so 1 s is usually fine; use 0.1 for sharper steps
            )

            # TODO: comprobar la forma del dst_step vs fuds_step porque me da la impresion de que puede ser porque empieza por cero
            decimals = len(str(period_value).split(".")[1]) if "." in str(period_value) else 0
            drive_cycle_power = drive_cycle_power.round(decimals)

            dst_step = pybamm.step.power(
                drive_cycle_power,
                duration=f"{360} seconds",
                period=f"{period_value} second",              # matches USABC 1 s steps / logging
                termination=[pybamm.step.VoltageTermination(2.51, operator="<"), pybamm.step.VoltageTermination(4.2, operator=">")],            
            )

            # 4) Repeat the profile N times (as cycles)
            experiment = pybamm.Experiment([
                dst_step.copy(), # 1
                ],
                period=f"{period_value} seconds",
                )
        else:
            True
    elif  protocol_name == "DST_short_rest_050soc_eval":
        # 1) Load USABC Table 5A-1 (time [s] vs power fraction of peak)
        df = pd.read_csv(f"hack\\model\\drive_cycles\\usabc_dst_test_cycle.csv")  # columns: time_s, power_frac

        if fixed_period:
            # 3) Create a power-controlled drive-cycle step
            drive_cycle_power = build_drive_cycle_power_from_dst_table(
                df,
                p_peak_w=80.0,
                dt=1,  # DST total is 360 s, so 1 s is usually fine; use 0.1 for sharper steps
            )

            # TODO: comprobar la forma del dst_step vs fuds_step porque me da la impresion de que puede ser porque empieza por cero
            decimals = len(str(period_value).split(".")[1]) if "." in str(period_value) else 0
            drive_cycle_power = drive_cycle_power.round(decimals)

            dst_step = pybamm.step.power(
                drive_cycle_power,
                duration=f"{360} seconds",
                period=f"{period_value} second",              # matches USABC 1 s steps / logging
                #termination=[pybamm.step.VoltageTermination(2.51, operator="<"), pybamm.step.VoltageTermination(4.2, operator=">")],            
            )

            # 4) Repeat the profile N times (as cycles)
            experiment = pybamm.Experiment([
                dst_step.copy(), # 1
                ],
                period=f"{period_value} seconds",
                )
        else:
            True
    elif  protocol_name == "DST_short_rest_040soc_eval":
        # 1) Load USABC Table 5A-1 (time [s] vs power fraction of peak)
        df = pd.read_csv(f"hack\\model\\drive_cycles\\usabc_dst_test_cycle.csv")  # columns: time_s, power_frac

        if fixed_period:
            # 3) Create a power-controlled drive-cycle step
            drive_cycle_power = build_drive_cycle_power_from_dst_table(
                df,
                p_peak_w=80.0,
                dt=1,  # DST total is 360 s, so 1 s is usually fine; use 0.1 for sharper steps
            )

            # TODO: comprobar la forma del dst_step vs fuds_step porque me da la impresion de que puede ser porque empieza por cero
            decimals = len(str(period_value).split(".")[1]) if "." in str(period_value) else 0
            drive_cycle_power = drive_cycle_power.round(decimals)

            dst_step = pybamm.step.power(
                drive_cycle_power,
                duration=f"{360} seconds",
                period=f"{period_value} second",              # matches USABC 1 s steps / logging
                termination=[pybamm.step.VoltageTermination(2.51, operator="<"), pybamm.step.VoltageTermination(4.2, operator=">")],            
            )

            # 4) Repeat the profile N times (as cycles)
            experiment = pybamm.Experiment([
                dst_step.copy(), # 1
                ],
                period=f"{period_value} seconds",
                )
        else:
            True
    elif  protocol_name == "DST_short_rest_030soc_eval":
        # 1) Load USABC Table 5A-1 (time [s] vs power fraction of peak)
        df = pd.read_csv(f"hack\\model\\drive_cycles\\usabc_dst_test_cycle.csv")  # columns: time_s, power_frac

        if fixed_period:
            # 3) Create a power-controlled drive-cycle step
            drive_cycle_power = build_drive_cycle_power_from_dst_table(
                df,
                p_peak_w=80.0,
                dt=1,  # DST total is 360 s, so 1 s is usually fine; use 0.1 for sharper steps
            )

            # TODO: comprobar la forma del dst_step vs fuds_step porque me da la impresion de que puede ser porque empieza por cero
            decimals = len(str(period_value).split(".")[1]) if "." in str(period_value) else 0
            drive_cycle_power = drive_cycle_power.round(decimals)

            dst_step = pybamm.step.power(
                drive_cycle_power,
                duration=f"{360} seconds",
                period=f"{period_value} second",              # matches USABC 1 s steps / logging
                #termination=[pybamm.step.VoltageTermination(2.51, operator="<"), pybamm.step.VoltageTermination(4.2, operator=">")],            
            )

            # 4) Repeat the profile N times (as cycles)
            experiment = pybamm.Experiment([
                dst_step.copy(), # 1
                ],
                period=f"{period_value} seconds",
                )
        else:
            True
    elif  protocol_name == "DST_short_rest_020soc_eval":
        # 1) Load USABC Table 5A-1 (time [s] vs power fraction of peak)
        df = pd.read_csv(f"hack\\model\\drive_cycles\\usabc_dst_test_cycle.csv")  # columns: time_s, power_frac

        if fixed_period:
            # 3) Create a power-controlled drive-cycle step
            drive_cycle_power = build_drive_cycle_power_from_dst_table(
                df,
                p_peak_w=80.0,
                dt=1,  # DST total is 360 s, so 1 s is usually fine; use 0.1 for sharper steps
            )

            # TODO: comprobar la forma del dst_step vs fuds_step porque me da la impresion de que puede ser porque empieza por cero
            decimals = len(str(period_value).split(".")[1]) if "." in str(period_value) else 0
            drive_cycle_power = drive_cycle_power.round(decimals)

            dst_step = pybamm.step.power(
                drive_cycle_power,
                duration=f"{360} seconds",
                period=f"{period_value} second",              # matches USABC 1 s steps / logging
                termination=[pybamm.step.VoltageTermination(2.51, operator="<"), pybamm.step.VoltageTermination(4.2, operator=">")],            
            )

            # 4) Repeat the profile N times (as cycles)
            experiment = pybamm.Experiment([
                dst_step.copy(), # 1
                ],
                period=f"{period_value} seconds",
                )
        else:
            True
    elif  protocol_name == "DST_short_rest_010soc_eval":
        # 1) Load USABC Table 5A-1 (time [s] vs power fraction of peak)
        df = pd.read_csv(f"hack\\model\\drive_cycles\\usabc_dst_test_cycle.csv")  # columns: time_s, power_frac

        if fixed_period:
            # 3) Create a power-controlled drive-cycle step
            drive_cycle_power = build_drive_cycle_power_from_dst_table(
                df,
                p_peak_w=80.0,
                dt=1,  # DST total is 360 s, so 1 s is usually fine; use 0.1 for sharper steps
            )

            # TODO: comprobar la forma del dst_step vs fuds_step porque me da la impresion de que puede ser porque empieza por cero
            decimals = len(str(period_value).split(".")[1]) if "." in str(period_value) else 0
            drive_cycle_power = drive_cycle_power.round(decimals)

            dst_step = pybamm.step.power(
                drive_cycle_power,
                duration=f"{360} seconds",
                period=f"{period_value} second",              # matches USABC 1 s steps / logging
                termination=[pybamm.step.VoltageTermination(2.51, operator="<"), pybamm.step.VoltageTermination(4.2, operator=">")],            
            )

            # 4) Repeat the profile N times (as cycles)
            experiment = pybamm.Experiment([
                dst_step.copy(), # 1
                ],
                period=f"{period_value} seconds",
                )
        else:
            True
    elif  protocol_name == "DST_short_rest_000soc_eval":
        # 1) Load USABC Table 5A-1 (time [s] vs power fraction of peak)
        df = pd.read_csv(f"hack\\model\\drive_cycles\\usabc_dst_test_cycle.csv")  # columns: time_s, power_frac

        if fixed_period:
            # 3) Create a power-controlled drive-cycle step
            drive_cycle_power = build_drive_cycle_power_from_dst_table(
                df,
                p_peak_w=80.0,
                dt=1,  # DST total is 360 s, so 1 s is usually fine; use 0.1 for sharper steps
            )

            # TODO: comprobar la forma del dst_step vs fuds_step porque me da la impresion de que puede ser porque empieza por cero
            decimals = len(str(period_value).split(".")[1]) if "." in str(period_value) else 0
            drive_cycle_power = drive_cycle_power.round(decimals)

            dst_step = pybamm.step.power(
                drive_cycle_power,
                duration=f"{360} seconds",
                period=f"{period_value} second",              # matches USABC 1 s steps / logging
                termination=[pybamm.step.VoltageTermination(2.51, operator="<"), pybamm.step.VoltageTermination(4.2, operator=">")],            
            )

            # 4) Repeat the profile N times (as cycles)
            experiment = pybamm.Experiment([
                dst_step.copy(), # 1
                ],
                period=f"{period_value} seconds",
                )
        else:
            True
    elif protocol_name == "APP_short_rest_100soc_backup":
        """
        Build the 3-cycle APP experiment starting from a fully discharged cell.

        Cycle 1:
            charge block [0.1C, 0.25C, 0.5C] until 4.2 V
            rest 10 min
            discharge block [0.1C, 0.25C, 0.5C] until 2.5 V

        Cycle 2:
            charge block [0.75C, 1.0C, 0.5C] until 4.2 V
            rest 10 min
            discharge block [0.75C, 1.0C, 0.5C] until 2.5 V

        Cycle 3:
            charge block [1.5C, 2.0C, 0.5C] until 4.2 V
            rest 10 min
            discharge block [1.5C, 2.0C, 0.5C] until 2.5 V
        """
        rate_pairs = [
            (0.1, 0.25),
            (0.75, 1.0),
            (1.5, 2.0),
        ]

        experiment = pybamm.Experiment(
            [
                "Rest for 10 minutes",
                "Discharge at 0.10C for 180 seconds",
                "Rest for 10 minutes",
                "Discharge at 0.25C for 72 seconds",
                "Rest for 10 minutes",
                "Discharge at 0.50C for 72 seconds",
                "Rest for 10 minutes",
                "Discharge at 0.75C for 24 seconds",
                "Rest for 10 minutes",
                "Discharge at 1.00C for 18 seconds",
                "Rest for 10 minutes",
                "Discharge at 0.50C for 72 seconds",
                "Rest for 10 minutes",
                "Discharge at 1.50C for 12 seconds",
                "Rest for 10 minutes",
                "Discharge at 2.00C for 9 seconds",
                "Rest for 10 minutes",
                "Discharge at 0.50C for 72 seconds",
                "Rest for 10 minutes",
            ]*16
            +[
                "Discharge at C/10 until 2.5V",
                "Rest for 10 minutes"
            ]            
            +[
                "Charge at 0.10C for 180 seconds or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 0.25C for 72 seconds or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 0.50C for 72 seconds or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 0.75C for 24 seconds or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 1.00C for 18 seconds or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 0.50C for 72 seconds or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 1.50C for 12 seconds or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 2.00C for 9 seconds or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 0.50C for 72 seconds or until 4.2V",
                "Rest for 10 minutes",
            ]*11
            +[
                "Charge at C/10 until 4.2V",
                "Rest for 10 minutes"
            ]   
        )
    elif protocol_name == "APP_short_rest_100soc":
        """
        Build the 3-cycle APP experiment starting from a fully discharged cell.

        Cycle 1:
            charge block [0.1C, 0.25C, 0.5C] until 4.2 V
            rest 10 min
            discharge block [0.1C, 0.25C, 0.5C] until 2.5 V

        Cycle 2:
            charge block [0.75C, 1.0C, 0.5C] until 4.2 V
            rest 10 min
            discharge block [0.75C, 1.0C, 0.5C] until 2.5 V

        Cycle 3:
            charge block [1.5C, 2.0C, 0.5C] until 4.2 V
            rest 10 min
            discharge block [1.5C, 2.0C, 0.5C] until 2.5 V
        """

        experiment = pybamm.Experiment(
            [
                # Cycle 1 - Charge
                "Charge at 0.10C for 180 seconds or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 0.25C for 72 seconds or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 0.50C for 72 seconds or until 4.2V",
                "Rest for 10 minutes",
                # Cycle 1 - Dicharge
                "Discharge at 0.10C for 180 seconds or until 2.5V",
                "Rest for 10 minutes",
                "Discharge at 0.25C for 72 seconds or until 2.5V",
                "Rest for 10 minutes",
                "Discharge at 0.50C for 72 seconds or until 2.5V",
                "Rest for 10 minutes",
                # Cycle 2 - Charge
                "Charge at 0.75C for 24 seconds or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 1.00C for 18 seconds or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 0.50C for 72 seconds or until 4.2V",
                "Rest for 10 minutes",
                # Cycle 2 - Discharge
                "Discharge at 0.75C for 24 seconds or until 2.5V",
                "Rest for 10 minutes",
                "Discharge at 1.00C for 18 seconds or until 2.5V",
                "Rest for 10 minutes",
                "Discharge at 0.50C for 72 seconds or until 2.5V",
                "Rest for 10 minutes",
                # Cycle 3 - Charge
                "Charge at 1.50C for 12 seconds or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 2.00C for 9 seconds or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 0.50C for 72 seconds or until 4.2V",
                "Rest for 10 minutes",
                # Cycle 3 - Discharge
                "Discharge at 1.50C for 12 seconds or until 2.5V",
                "Rest for 10 minutes",
                "Discharge at 2.00C for 9 seconds or until 2.5V",
                "Rest for 10 minutes",
                "Discharge at 0.50C for 72 seconds or until 2.5V",
                "Rest for 10 minutes",
                # Discharge aprox 10%SOC
                "Discharge at C/2.5 for 15 minutes or until 2.5V",
                "Rest for 10 minutes"
            ]*10
            +
            [   #* Try to ensure 0 %SOC
                pybamm.step.string("Discharge at 0.5C until 2.5V"),
                pybamm.step.string("Hold at 2.5 V until C/20", direction="discharge"),
            ]
            +
            [   #* Repeat the cycle block at 0 %SOC
                # Cycle 1 - Charge
                "Charge at 0.10C for 180 seconds or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 0.25C for 72 seconds or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 0.50C for 72 seconds or until 4.2V",
                "Rest for 10 minutes",
                # Cycle 1 - Dicharge
                "Discharge at 0.10C for 180 seconds or until 2.5V",
                "Rest for 10 minutes",
                "Discharge at 0.25C for 72 seconds or until 2.5V",
                "Rest for 10 minutes",
                "Discharge at 0.50C for 72 seconds or until 2.5V",
                "Rest for 10 minutes",
                # Cycle 2 - Charge
                "Charge at 0.75C for 24 seconds or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 1.00C for 18 seconds or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 0.50C for 72 seconds or until 4.2V",
                "Rest for 10 minutes",
                # Cycle 2 - Discharge
                "Discharge at 0.75C for 24 seconds or until 2.5V",
                "Rest for 10 minutes",
                "Discharge at 1.00C for 18 seconds or until 2.5V",
                "Rest for 10 minutes",
                "Discharge at 0.50C for 72 seconds or until 2.5V",
                "Rest for 10 minutes",
                # Cycle 3 - Charge
                "Charge at 1.50C for 12 seconds or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 2.00C for 9 seconds or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 0.50C for 72 seconds or until 4.2V",
                "Rest for 10 minutes",
                # Cycle 3 - Discharge
                "Discharge at 1.50C for 12 seconds or until 2.5V",
                "Rest for 10 minutes",
                "Discharge at 2.00C for 9 seconds or until 2.5V",
                "Rest for 10 minutes",
                "Discharge at 0.50C for 72 seconds or until 2.5V",
                "Rest for 10 minutes",
            ],
            period=f"{period_value} seconds"
        )
    elif protocol_name == "APP_short_rest_100soc_eval":
        """
        Build the 3-cycle APP experiment starting from a fully discharged cell.

        Cycle 1:
            charge block [0.1C, 0.25C, 0.5C] until 4.2 V
            rest 10 min
            discharge block [0.1C, 0.25C, 0.5C] until 2.5 V

        Cycle 2:
            charge block [0.75C, 1.0C, 0.5C] until 4.2 V
            rest 10 min
            discharge block [0.75C, 1.0C, 0.5C] until 2.5 V

        Cycle 3:
            charge block [1.5C, 2.0C, 0.5C] until 4.2 V
            rest 10 min
            discharge block [1.5C, 2.0C, 0.5C] until 2.5 V
        """

        experiment = pybamm.Experiment(
            [
                # Cycle 1 - Charge
                "Charge at 0.10C for 180 seconds or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 0.25C for 72 seconds or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 0.50C for 72 seconds or until 4.2V",
                "Rest for 10 minutes",
                # Cycle 1 - Dicharge
                "Discharge at 0.10C for 180 seconds or until 2.5V",
                "Rest for 10 minutes",
                "Discharge at 0.25C for 72 seconds or until 2.5V",
                "Rest for 10 minutes",
                "Discharge at 0.50C for 72 seconds or until 2.5V",
                "Rest for 10 minutes",
                # Cycle 2 - Charge
                "Charge at 0.75C for 24 seconds or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 1.00C for 18 seconds or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 0.50C for 72 seconds or until 4.2V",
                "Rest for 10 minutes",
                # Cycle 2 - Discharge
                "Discharge at 0.75C for 24 seconds or until 2.5V",
                "Rest for 10 minutes",
                "Discharge at 1.00C for 18 seconds or until 2.5V",
                "Rest for 10 minutes",
                "Discharge at 0.50C for 72 seconds or until 2.5V",
                "Rest for 10 minutes",
                # Cycle 3 - Charge
                "Charge at 1.50C for 12 seconds or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 2.00C for 9 seconds or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 0.50C for 72 seconds or until 4.2V",
                "Rest for 10 minutes",
                # Cycle 3 - Discharge
                "Discharge at 1.50C for 12 seconds or until 2.5V",
                "Rest for 10 minutes",
                "Discharge at 2.00C for 9 seconds or until 2.5V",
                "Rest for 10 minutes",
                "Discharge at 0.50C for 72 seconds or until 2.5V",
                "Rest for 10 minutes",
                # Discharge aprox 10%SOC
                "Discharge at C/2.5 for 15 minutes or until 2.5V",
                "Rest for 10 minutes"
            ],
            period=f"{period_value} seconds"
            
        )
    elif protocol_name == "APP_short_rest_090soc_eval":
        """
        Build the 3-cycle APP experiment starting from a fully discharged cell.

        Cycle 1:
            charge block [0.1C, 0.25C, 0.5C] until 4.2 V
            rest 10 min
            discharge block [0.1C, 0.25C, 0.5C] until 2.5 V

        Cycle 2:
            charge block [0.75C, 1.0C, 0.5C] until 4.2 V
            rest 10 min
            discharge block [0.75C, 1.0C, 0.5C] until 2.5 V

        Cycle 3:
            charge block [1.5C, 2.0C, 0.5C] until 4.2 V
            rest 10 min
            discharge block [1.5C, 2.0C, 0.5C] until 2.5 V
        """

        experiment = pybamm.Experiment(
            [
                # Cycle 1 - Charge
                "Charge at 0.10C for 180 seconds or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 0.25C for 72 seconds or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 0.50C for 72 seconds or until 4.2V",
                "Rest for 10 minutes",
                # Cycle 1 - Dicharge
                "Discharge at 0.10C for 180 seconds or until 2.5V",
                "Rest for 10 minutes",
                "Discharge at 0.25C for 72 seconds or until 2.5V",
                "Rest for 10 minutes",
                "Discharge at 0.50C for 72 seconds or until 2.5V",
                "Rest for 10 minutes",
                # Cycle 2 - Charge
                "Charge at 0.75C for 24 seconds or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 1.00C for 18 seconds or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 0.50C for 72 seconds or until 4.2V",
                "Rest for 10 minutes",
                # Cycle 2 - Discharge
                "Discharge at 0.75C for 24 seconds or until 2.5V",
                "Rest for 10 minutes",
                "Discharge at 1.00C for 18 seconds or until 2.5V",
                "Rest for 10 minutes",
                "Discharge at 0.50C for 72 seconds or until 2.5V",
                "Rest for 10 minutes",
                # Cycle 3 - Charge
                "Charge at 1.50C for 12 seconds or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 2.00C for 9 seconds or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 0.50C for 72 seconds or until 4.2V",
                "Rest for 10 minutes",
                # Cycle 3 - Discharge
                "Discharge at 1.50C for 12 seconds or until 2.5V",
                "Rest for 10 minutes",
                "Discharge at 2.00C for 9 seconds or until 2.5V",
                "Rest for 10 minutes",
                "Discharge at 0.50C for 72 seconds or until 2.5V",
                "Rest for 10 minutes",
                # Discharge aprox 10%SOC
                "Discharge at C/2.5 for 15 minutes or until 2.5V",
                "Rest for 10 minutes"
            ],
            period=f"{period_value} seconds"
            
        )
    elif protocol_name == "APP_short_rest_080soc_eval":
        """
        Build the 3-cycle APP experiment starting from a fully discharged cell.

        Cycle 1:
            charge block [0.1C, 0.25C, 0.5C] until 4.2 V
            rest 10 min
            discharge block [0.1C, 0.25C, 0.5C] until 2.5 V

        Cycle 2:
            charge block [0.75C, 1.0C, 0.5C] until 4.2 V
            rest 10 min
            discharge block [0.75C, 1.0C, 0.5C] until 2.5 V

        Cycle 3:
            charge block [1.5C, 2.0C, 0.5C] until 4.2 V
            rest 10 min
            discharge block [1.5C, 2.0C, 0.5C] until 2.5 V
        """

        experiment = pybamm.Experiment(
            [
                # Cycle 1 - Charge
                "Charge at 0.10C for 180 seconds or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 0.25C for 72 seconds or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 0.50C for 72 seconds or until 4.2V",
                "Rest for 10 minutes",
                # Cycle 1 - Dicharge
                "Discharge at 0.10C for 180 seconds or until 2.5V",
                "Rest for 10 minutes",
                "Discharge at 0.25C for 72 seconds or until 2.5V",
                "Rest for 10 minutes",
                "Discharge at 0.50C for 72 seconds or until 2.5V",
                "Rest for 10 minutes",
                # Cycle 2 - Charge
                "Charge at 0.75C for 24 seconds or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 1.00C for 18 seconds or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 0.50C for 72 seconds or until 4.2V",
                "Rest for 10 minutes",
                # Cycle 2 - Discharge
                "Discharge at 0.75C for 24 seconds or until 2.5V",
                "Rest for 10 minutes",
                "Discharge at 1.00C for 18 seconds or until 2.5V",
                "Rest for 10 minutes",
                "Discharge at 0.50C for 72 seconds or until 2.5V",
                "Rest for 10 minutes",
                # Cycle 3 - Charge
                "Charge at 1.50C for 12 seconds or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 2.00C for 9 seconds or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 0.50C for 72 seconds or until 4.2V",
                "Rest for 10 minutes",
                # Cycle 3 - Discharge
                "Discharge at 1.50C for 12 seconds or until 2.5V",
                "Rest for 10 minutes",
                "Discharge at 2.00C for 9 seconds or until 2.5V",
                "Rest for 10 minutes",
                "Discharge at 0.50C for 72 seconds or until 2.5V",
                "Rest for 10 minutes",
                # Discharge aprox 10%SOC
                "Discharge at C/2.5 for 15 minutes or until 2.5V",
                "Rest for 10 minutes"
            ],
            period=f"{period_value} seconds"
            
        )
    elif protocol_name == "APP_short_rest_070soc_eval":
        """
        Build the 3-cycle APP experiment starting from a fully discharged cell.

        Cycle 1:
            charge block [0.1C, 0.25C, 0.5C] until 4.2 V
            rest 10 min
            discharge block [0.1C, 0.25C, 0.5C] until 2.5 V

        Cycle 2:
            charge block [0.75C, 1.0C, 0.5C] until 4.2 V
            rest 10 min
            discharge block [0.75C, 1.0C, 0.5C] until 2.5 V

        Cycle 3:
            charge block [1.5C, 2.0C, 0.5C] until 4.2 V
            rest 10 min
            discharge block [1.5C, 2.0C, 0.5C] until 2.5 V
        """

        experiment = pybamm.Experiment(
            [
                # Cycle 1 - Charge
                "Charge at 0.10C for 180 seconds", #or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 0.25C for 72 seconds", #or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 0.50C for 72 seconds", #or until 4.2V",
                "Rest for 10 minutes",
                # Cycle 1 - Dicharge
                "Discharge at 0.10C for 180 seconds", #or until 2.5V",
                "Rest for 10 minutes",
                "Discharge at 0.25C for 72 seconds", #or until 2.5V",
                "Rest for 10 minutes",
                "Discharge at 0.50C for 72 seconds", #or until 2.5V",
                "Rest for 10 minutes",
                # Cycle 2 - Charge
                "Charge at 0.75C for 24 seconds", #or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 1.00C for 18 seconds", #or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 0.50C for 72 seconds", #or until 4.2V",
                "Rest for 10 minutes",
                # Cycle 2 - Discharge
                "Discharge at 0.75C for 24 seconds", #or until 2.5V",
                "Rest for 10 minutes",
                "Discharge at 1.00C for 18 seconds", #or until 2.5V",
                "Rest for 10 minutes",
                "Discharge at 0.50C for 72 seconds", #or until 2.5V",
                "Rest for 10 minutes",
                # Cycle 3 - Charge
                "Charge at 1.50C for 12 seconds", #or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 2.00C for 9 seconds", #or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 0.50C for 72 seconds", #or until 4.2V",
                "Rest for 10 minutes",
                # Cycle 3 - Discharge
                "Discharge at 1.50C for 12 seconds", #or until 2.5V",
                "Rest for 10 minutes",
                "Discharge at 2.00C for 9 seconds", #or until 2.5V",
                "Rest for 10 minutes",
                "Discharge at 0.50C for 72 seconds", #or until 2.5V",
                "Rest for 10 minutes",
                # Discharge aprox 10%SOC
                "Discharge at C/2.5 for 15 minutes", #or until 2.5V",
                "Rest for 10 minutes"
            ],
            
        )
    elif protocol_name == "APP_short_rest_060soc_eval":
        """
        Build the 3-cycle APP experiment starting from a fully discharged cell.

        Cycle 1:
            charge block [0.1C, 0.25C, 0.5C] until 4.2 V
            rest 10 min
            discharge block [0.1C, 0.25C, 0.5C] until 2.5 V

        Cycle 2:
            charge block [0.75C, 1.0C, 0.5C] until 4.2 V
            rest 10 min
            discharge block [0.75C, 1.0C, 0.5C] until 2.5 V

        Cycle 3:
            charge block [1.5C, 2.0C, 0.5C] until 4.2 V
            rest 10 min
            discharge block [1.5C, 2.0C, 0.5C] until 2.5 V
        """

        experiment = pybamm.Experiment(
            [
                # Cycle 1 - Charge
                "Charge at 0.10C for 180 seconds or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 0.25C for 72 seconds or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 0.50C for 72 seconds or until 4.2V",
                "Rest for 10 minutes",
                # Cycle 1 - Dicharge
                "Discharge at 0.10C for 180 seconds or until 2.5V",
                "Rest for 10 minutes",
                "Discharge at 0.25C for 72 seconds or until 2.5V",
                "Rest for 10 minutes",
                "Discharge at 0.50C for 72 seconds or until 2.5V",
                "Rest for 10 minutes",
                # Cycle 2 - Charge
                "Charge at 0.75C for 24 seconds or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 1.00C for 18 seconds or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 0.50C for 72 seconds or until 4.2V",
                "Rest for 10 minutes",
                # Cycle 2 - Discharge
                "Discharge at 0.75C for 24 seconds or until 2.5V",
                "Rest for 10 minutes",
                "Discharge at 1.00C for 18 seconds or until 2.5V",
                "Rest for 10 minutes",
                "Discharge at 0.50C for 72 seconds or until 2.5V",
                "Rest for 10 minutes",
                # Cycle 3 - Charge
                "Charge at 1.50C for 12 seconds or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 2.00C for 9 seconds or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 0.50C for 72 seconds or until 4.2V",
                "Rest for 10 minutes",
                # Cycle 3 - Discharge
                "Discharge at 1.50C for 12 seconds or until 2.5V",
                "Rest for 10 minutes",
                "Discharge at 2.00C for 9 seconds or until 2.5V",
                "Rest for 10 minutes",
                "Discharge at 0.50C for 72 seconds or until 2.5V",
                "Rest for 10 minutes",
                # Discharge aprox 10%SOC
                "Discharge at C/2.5 for 15 minutes or until 2.5V",
                "Rest for 10 minutes"
            ],
            period=f"{period_value} seconds"
            
        )
    elif protocol_name == "APP_short_rest_050soc_eval":
        """
        Build the 3-cycle APP experiment starting from a fully discharged cell.

        Cycle 1:
            charge block [0.1C, 0.25C, 0.5C] until 4.2 V
            rest 10 min
            discharge block [0.1C, 0.25C, 0.5C] until 2.5 V

        Cycle 2:
            charge block [0.75C, 1.0C, 0.5C] until 4.2 V
            rest 10 min
            discharge block [0.75C, 1.0C, 0.5C] until 2.5 V

        Cycle 3:
            charge block [1.5C, 2.0C, 0.5C] until 4.2 V
            rest 10 min
            discharge block [1.5C, 2.0C, 0.5C] until 2.5 V
        """

        experiment = pybamm.Experiment(
            [
                # Cycle 1 - Charge
                "Charge at 0.10C for 180 seconds", #or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 0.25C for 72 seconds", #or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 0.50C for 72 seconds", #or until 4.2V",
                "Rest for 10 minutes",
                # Cycle 1 - Dicharge
                "Discharge at 0.10C for 180 seconds", #or until 2.5V",
                "Rest for 10 minutes",
                "Discharge at 0.25C for 72 seconds", #or until 2.5V",
                "Rest for 10 minutes",
                "Discharge at 0.50C for 72 seconds", #or until 2.5V",
                "Rest for 10 minutes",
                # Cycle 2 - Charge
                "Charge at 0.75C for 24 seconds", #or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 1.00C for 18 seconds", #or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 0.50C for 72 seconds", #or until 4.2V",
                "Rest for 10 minutes",
                # Cycle 2 - Discharge
                "Discharge at 0.75C for 24 seconds", #or until 2.5V",
                "Rest for 10 minutes",
                "Discharge at 1.00C for 18 seconds", #or until 2.5V",
                "Rest for 10 minutes",
                "Discharge at 0.50C for 72 seconds", #or until 2.5V",
                "Rest for 10 minutes",
                # Cycle 3 - Charge
                "Charge at 1.50C for 12 seconds", #or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 2.00C for 9 seconds", #or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 0.50C for 72 seconds", #or until 4.2V",
                "Rest for 10 minutes",
                # Cycle 3 - Discharge
                "Discharge at 1.50C for 12 seconds", #or until 2.5V",
                "Rest for 10 minutes",
                "Discharge at 2.00C for 9 seconds", #or until 2.5V",
                "Rest for 10 minutes",
                "Discharge at 0.50C for 72 seconds", #or until 2.5V",
                "Rest for 10 minutes",
                # Discharge aprox 10%SOC
                "Discharge at C/2.5 for 15 minutes", #or until 2.5V",
                "Rest for 10 minutes"
            ],
            
        )
    elif protocol_name == "APP_short_rest_040soc_eval":
        """
        Build the 3-cycle APP experiment starting from a fully discharged cell.

        Cycle 1:
            charge block [0.1C, 0.25C, 0.5C] until 4.2 V
            rest 10 min
            discharge block [0.1C, 0.25C, 0.5C] until 2.5 V

        Cycle 2:
            charge block [0.75C, 1.0C, 0.5C] until 4.2 V
            rest 10 min
            discharge block [0.75C, 1.0C, 0.5C] until 2.5 V

        Cycle 3:
            charge block [1.5C, 2.0C, 0.5C] until 4.2 V
            rest 10 min
            discharge block [1.5C, 2.0C, 0.5C] until 2.5 V
        """

        experiment = pybamm.Experiment(
            [
                # Cycle 1 - Charge
                "Charge at 0.10C for 180 seconds or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 0.25C for 72 seconds or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 0.50C for 72 seconds or until 4.2V",
                "Rest for 10 minutes",
                # Cycle 1 - Dicharge
                "Discharge at 0.10C for 180 seconds or until 2.5V",
                "Rest for 10 minutes",
                "Discharge at 0.25C for 72 seconds or until 2.5V",
                "Rest for 10 minutes",
                "Discharge at 0.50C for 72 seconds or until 2.5V",
                "Rest for 10 minutes",
                # Cycle 2 - Charge
                "Charge at 0.75C for 24 seconds or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 1.00C for 18 seconds or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 0.50C for 72 seconds or until 4.2V",
                "Rest for 10 minutes",
                # Cycle 2 - Discharge
                "Discharge at 0.75C for 24 seconds or until 2.5V",
                "Rest for 10 minutes",
                "Discharge at 1.00C for 18 seconds or until 2.5V",
                "Rest for 10 minutes",
                "Discharge at 0.50C for 72 seconds or until 2.5V",
                "Rest for 10 minutes",
                # Cycle 3 - Charge
                "Charge at 1.50C for 12 seconds or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 2.00C for 9 seconds or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 0.50C for 72 seconds or until 4.2V",
                "Rest for 10 minutes",
                # Cycle 3 - Discharge
                "Discharge at 1.50C for 12 seconds or until 2.5V",
                "Rest for 10 minutes",
                "Discharge at 2.00C for 9 seconds or until 2.5V",
                "Rest for 10 minutes",
                "Discharge at 0.50C for 72 seconds or until 2.5V",
                "Rest for 10 minutes",
                # Discharge aprox 10%SOC
                "Discharge at C/2.5 for 15 minutes or until 2.5V",
                "Rest for 10 minutes"
            ],
            period=f"{period_value} seconds"
            
        )
    elif protocol_name == "APP_short_rest_030soc_eval":
        """
        Build the 3-cycle APP experiment starting from a fully discharged cell.

        Cycle 1:
            charge block [0.1C, 0.25C, 0.5C] until 4.2 V
            rest 10 min
            discharge block [0.1C, 0.25C, 0.5C] until 2.5 V

        Cycle 2:
            charge block [0.75C, 1.0C, 0.5C] until 4.2 V
            rest 10 min
            discharge block [0.75C, 1.0C, 0.5C] until 2.5 V

        Cycle 3:
            charge block [1.5C, 2.0C, 0.5C] until 4.2 V
            rest 10 min
            discharge block [1.5C, 2.0C, 0.5C] until 2.5 V
        """

        experiment = pybamm.Experiment(
            [
                # Cycle 1 - Charge
                "Charge at 0.10C for 180 seconds", #or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 0.25C for 72 seconds", #or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 0.50C for 72 seconds", #or until 4.2V",
                "Rest for 10 minutes",
                # Cycle 1 - Dicharge
                "Discharge at 0.10C for 180 seconds", #or until 2.5V",
                "Rest for 10 minutes",
                "Discharge at 0.25C for 72 seconds", #or until 2.5V",
                "Rest for 10 minutes",
                "Discharge at 0.50C for 72 seconds", #or until 2.5V",
                "Rest for 10 minutes",
                # Cycle 2 - Charge
                "Charge at 0.75C for 24 seconds", #or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 1.00C for 18 seconds", #or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 0.50C for 72 seconds", #or until 4.2V",
                "Rest for 10 minutes",
                # Cycle 2 - Discharge
                "Discharge at 0.75C for 24 seconds", #or until 2.5V",
                "Rest for 10 minutes",
                "Discharge at 1.00C for 18 seconds", #or until 2.5V",
                "Rest for 10 minutes",
                "Discharge at 0.50C for 72 seconds", #or until 2.5V",
                "Rest for 10 minutes",
                # Cycle 3 - Charge
                "Charge at 1.50C for 12 seconds", #or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 2.00C for 9 seconds", #or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 0.50C for 72 seconds", #or until 4.2V",
                "Rest for 10 minutes",
                # Cycle 3 - Discharge
                "Discharge at 1.50C for 12 seconds", #or until 2.5V",
                "Rest for 10 minutes",
                "Discharge at 2.00C for 9 seconds", #or until 2.5V",
                "Rest for 10 minutes",
                "Discharge at 0.50C for 72 seconds", #or until 2.5V",
                "Rest for 10 minutes",
                # Discharge aprox 10%SOC
                "Discharge at C/2.5 for 15 minutes", #or until 2.5V",
                "Rest for 10 minutes"
            ],
            
        )
    elif protocol_name == "APP_short_rest_020soc_eval":
        """
        Build the 3-cycle APP experiment starting from a fully discharged cell.

        Cycle 1:
            charge block [0.1C, 0.25C, 0.5C] until 4.2 V
            rest 10 min
            discharge block [0.1C, 0.25C, 0.5C] until 2.5 V

        Cycle 2:
            charge block [0.75C, 1.0C, 0.5C] until 4.2 V
            rest 10 min
            discharge block [0.75C, 1.0C, 0.5C] until 2.5 V

        Cycle 3:
            charge block [1.5C, 2.0C, 0.5C] until 4.2 V
            rest 10 min
            discharge block [1.5C, 2.0C, 0.5C] until 2.5 V
        """

        experiment = pybamm.Experiment(
            [
                # Cycle 1 - Charge
                "Charge at 0.10C for 180 seconds or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 0.25C for 72 seconds or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 0.50C for 72 seconds or until 4.2V",
                "Rest for 10 minutes",
                # Cycle 1 - Dicharge
                "Discharge at 0.10C for 180 seconds or until 2.5V",
                "Rest for 10 minutes",
                "Discharge at 0.25C for 72 seconds or until 2.5V",
                "Rest for 10 minutes",
                "Discharge at 0.50C for 72 seconds or until 2.5V",
                "Rest for 10 minutes",
                # Cycle 2 - Charge
                "Charge at 0.75C for 24 seconds or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 1.00C for 18 seconds or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 0.50C for 72 seconds or until 4.2V",
                "Rest for 10 minutes",
                # Cycle 2 - Discharge
                "Discharge at 0.75C for 24 seconds or until 2.5V",
                "Rest for 10 minutes",
                "Discharge at 1.00C for 18 seconds or until 2.5V",
                "Rest for 10 minutes",
                "Discharge at 0.50C for 72 seconds or until 2.5V",
                "Rest for 10 minutes",
                # Cycle 3 - Charge
                "Charge at 1.50C for 12 seconds or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 2.00C for 9 seconds or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 0.50C for 72 seconds or until 4.2V",
                "Rest for 10 minutes",
                # Cycle 3 - Discharge
                "Discharge at 1.50C for 12 seconds or until 2.5V",
                "Rest for 10 minutes",
                "Discharge at 2.00C for 9 seconds or until 2.5V",
                "Rest for 10 minutes",
                "Discharge at 0.50C for 72 seconds or until 2.5V",
                "Rest for 10 minutes",
                # Discharge aprox 10%SOC
                "Discharge at C/2.5 for 15 minutes or until 2.5V",
                "Rest for 10 minutes"
            ],
            period=f"{period_value} seconds"
            
        )
    elif protocol_name == "APP_short_rest_010soc_eval":
        """
        Build the 3-cycle APP experiment starting from a fully discharged cell.

        Cycle 1:
            charge block [0.1C, 0.25C, 0.5C] until 4.2 V
            rest 10 min
            discharge block [0.1C, 0.25C, 0.5C] until 2.5 V

        Cycle 2:
            charge block [0.75C, 1.0C, 0.5C] until 4.2 V
            rest 10 min
            discharge block [0.75C, 1.0C, 0.5C] until 2.5 V

        Cycle 3:
            charge block [1.5C, 2.0C, 0.5C] until 4.2 V
            rest 10 min
            discharge block [1.5C, 2.0C, 0.5C] until 2.5 V
        """

        experiment = pybamm.Experiment(
            [
                # Rest 10 seconds to ensure no error in the OCV value
                "Rest for 10 seconds",
                # Cycle 1 - Charge
                "Charge at 0.10C for 180 seconds or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 0.25C for 72 seconds or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 0.50C for 72 seconds or until 4.2V",
                "Rest for 10 minutes",
                # Cycle 1 - Dicharge
                "Discharge at 0.10C for 180 seconds or until 2.5V",
                "Rest for 10 minutes",
                "Discharge at 0.25C for 72 seconds or until 2.5V",
                "Rest for 10 minutes",
                "Discharge at 0.50C for 72 seconds or until 2.5V",
                "Rest for 10 minutes",
                # Cycle 2 - Charge
                "Charge at 0.75C for 24 seconds or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 1.00C for 18 seconds or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 0.50C for 72 seconds or until 4.2V",
                "Rest for 10 minutes",
                # Cycle 2 - Discharge
                "Discharge at 0.75C for 24 seconds or until 2.5V",
                "Rest for 10 minutes",
                "Discharge at 1.00C for 18 seconds or until 2.5V",
                "Rest for 10 minutes",
                "Discharge at 0.50C for 72 seconds or until 2.5V",
                "Rest for 10 minutes",
                # Cycle 3 - Charge
                "Charge at 1.50C for 12 seconds or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 2.00C for 9 seconds or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 0.50C for 72 seconds or until 4.2V",
                "Rest for 10 minutes",
                # Cycle 3 - Discharge
                "Discharge at 1.50C for 12 seconds or until 2.5V",
                "Rest for 10 minutes",
                "Discharge at 2.00C for 9 seconds or until 2.5V",
                "Rest for 10 minutes",
                "Discharge at 0.50C for 72 seconds or until 2.5V",
                "Rest for 10 minutes",
                # Discharge aprox 10%SOC
                "Discharge at C/2.5 for 15 minutes or until 2.5V",
                "Rest for 10 minutes"
            ],
            period=f"{period_value} seconds"
            
        )
    elif protocol_name == "APP_short_rest_000soc_eval":
        """
        Build the 3-cycle APP experiment starting from a fully discharged cell.

        Cycle 1:
            charge block [0.1C, 0.25C, 0.5C] until 4.2 V
            rest 10 min
            discharge block [0.1C, 0.25C, 0.5C] until 2.5 V

        Cycle 2:
            charge block [0.75C, 1.0C, 0.5C] until 4.2 V
            rest 10 min
            discharge block [0.75C, 1.0C, 0.5C] until 2.5 V

        Cycle 3:
            charge block [1.5C, 2.0C, 0.5C] until 4.2 V
            rest 10 min
            discharge block [1.5C, 2.0C, 0.5C] until 2.5 V
        """

        experiment = pybamm.Experiment(
            [
                # Rest 10 seconds to ensure no error in the OCV value
                "Rest for 10 seconds",
                # Cycle 1 - Charge
                "Charge at 0.10C for 180 seconds or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 0.25C for 72 seconds or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 0.50C for 72 seconds or until 4.2V",
                "Rest for 10 minutes",
                # Cycle 1 - Dicharge
                "Discharge at 0.10C for 180 seconds or until 2.5V",
                "Rest for 10 minutes",
                "Discharge at 0.25C for 72 seconds or until 2.5V",
                "Rest for 10 minutes",
                "Discharge at 0.50C for 72 seconds or until 2.5V",
                "Rest for 10 minutes",
                # Cycle 2 - Charge
                "Charge at 0.75C for 24 seconds or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 1.00C for 18 seconds or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 0.50C for 72 seconds or until 4.2V",
                "Rest for 10 minutes",
                # Cycle 2 - Discharge
                "Discharge at 0.75C for 24 seconds or until 2.5V",
                "Rest for 10 minutes",
                "Discharge at 1.00C for 18 seconds or until 2.5V",
                "Rest for 10 minutes",
                "Discharge at 0.50C for 72 seconds or until 2.5V",
                "Rest for 10 minutes",
                # Cycle 3 - Charge
                "Charge at 1.50C for 12 seconds or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 2.00C for 9 seconds or until 4.2V",
                "Rest for 10 minutes",
                "Charge at 0.50C for 72 seconds or until 4.2V",
                "Rest for 10 minutes",
                # Cycle 3 - Discharge
                "Discharge at 1.50C for 12 seconds or until 2.5V",
                "Rest for 10 minutes",
                "Discharge at 2.00C for 9 seconds or until 2.5V",
                "Rest for 10 minutes",
                "Discharge at 0.50C for 72 seconds or until 2.5V",
                "Rest for 10 minutes",
                # Discharge aprox 10%SOC
                "Discharge at C/2.5 for 15 minutes or until 2.5V",
                "Rest for 10 minutes"
            ],
            period=f"{period_value} seconds"
            
        )
    elif protocol_name == "STEP_short_rest_050soc":
        experiment = pybamm.Experiment(
            [
                "Rest for 10 seconds",
                "Discharge at 0.2C for 1000 seconds or until 2.5V",
                "Rest for 10 seconds",
            ],
            period=f"{period_value} seconds",
        )
    return experiment


def simulate_long_experiment(intial_soc_pu, test_protocol, model, params, solver, model_type, param_set, solver_name, max_time_step, initial_cell_temp_k, output_vars, plot_interm = False, generate_csv = False):

    max_time_step_text = str(max_time_step).replace(".","p")
    initial_cell_temp_k_text = str(initial_cell_temp_k).replace(".","p")

    if test_protocol == "GITT_short_rest_100soc_long_experiment":      

        # SECTION DESCRIPTION : DISCHARGE STEPS

        for step in range(0, 200):

            try:

                experiment = pybamm.Experiment(
                        [   #* 80 normal steps of GITT profile with a discharge of 1.25 %SOC each
                            pybamm.step.string("Rest for 5 minutes"),
                            pybamm.step.string("Discharge at C/10 for 7.5 min or until 2.501V"), # Discharge step of 1.25 %SOC
                        ],
                        period=f"{max_time_step} seconds"
                        )
                
                sim = pybamm.Simulation(
                    model,
                    parameter_values=params,
                    experiment=experiment,
                    solver=solver,
                )

                if step == 0:
                    solution = sim.solve(showprogress=True, initial_soc=intial_soc_pu)
                else:
                    solution = sim.solve(showprogress=True, starting_solution = last_state)
                last_state = solution.last_state


                if plot_interm:
                    sim.plot(
                        output_variables = output_vars,
                        show_plot = False
                    )

                    plt.savefig(fname = f"hack\\model\\output\\figures\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.svg")
                    plt.close("all")

                data_output = pd.DataFrame(solution.get_data_dict())
                data_output["time [s]"] = solution.t
                data_output.to_parquet(f"hack\\model\\output\\parquet\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.parquet", compression="gzip")
                if generate_csv: data_output.to_csv(f"hack\\model\\output\\csv\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.csv")

                last_state = solution.last_state # This step is after data_output generation because there is where the step fails, so it does not affect ot the last valid state.

                last_voltage = float(
                    last_state["Voltage [V]"].entries[-1]
                )

                if last_voltage < 2.501:
                    break

            except ValueError as error:

                    if "Solution does not have any data" in str(error):
                        print(f"No data available for step {step}. Skipping it.")
                        break
                    raise

        # SECTION DESCRIPTION : ENSURE 0% SOC

        step = 200

        experiment = pybamm.Experiment([   #* Try to ensure 0 %SOC
                pybamm.step.string("Rest for 5 minutes"),
                pybamm.step.string("Discharge at C/10 until 2.5V"),
                pybamm.step.string("Hold at 2.5 V until C/20", direction="discharge"),
            ],
            period=f"{max_time_step} seconds"
        )

        sim = pybamm.Simulation(
            model,
            parameter_values=params,
            experiment=experiment,
            solver=solver,
        )

        solution = sim.solve(showprogress=True, starting_solution = last_state )

        last_state = solution.last_state

        if plot_interm:
            sim.plot(
                output_variables = output_vars,
                show_plot = False
            )

            plt.savefig(fname = f"hack\\model\\output\\figures\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.svg")
            plt.close("all")

        data_output = pd.DataFrame(solution.get_data_dict())
        data_output["time [s]"] = solution.t
        data_output.to_parquet(f"hack\\model\\output\\parquet\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.parquet", compression="gzip")
        if generate_csv: data_output.to_csv(f"hack\\model\\output\\csv\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.csv")

        # SECTION DESCRIPTION : CHARGE STEPS 

        for step in range(201, 400):
                
            try:
                experiment = pybamm.Experiment(
                        [   #* 80 normal steps of GITT profile with a discharge of 1.25 %SOC each
                            pybamm.step.string("Rest for 5 minutes"),
                            pybamm.step.string("Charge at C/10 for 7.5 min or until 4.2V"), # Discharge step of 1.25 %SOC
                        ],
                        period=f"{max_time_step} seconds"
                        )
                
                sim = pybamm.Simulation(
                    model,
                    parameter_values=params,
                    experiment=experiment,
                    solver=solver,
                )

                solution = sim.solve(showprogress=True, starting_solution = last_state )

                if plot_interm:
                    sim.plot(
                        output_variables = output_vars,
                        show_plot = False
                    )

                    plt.savefig(fname = f"hack\\model\\output\\figures\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.svg")
                    plt.close("all")

                data_output = pd.DataFrame(solution.get_data_dict())
                data_output["time [s]"] = solution.t
                data_output.to_parquet(f"hack\\model\\output\\parquet\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.parquet", compression="gzip")
                if generate_csv: data_output.to_csv(f"hack\\model\\output\\csv\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.csv")

                last_state = solution.last_state # This step is after data_output generation because there is where the step fails, so it does not affect ot the last valid state.

                last_voltage = float(
                    last_state["Voltage [V]"].entries[-1]
                )

                if last_voltage > 4.199:
                    break

            except ValueError as error:

                    if "Solution does not have any data" in str(error):
                        print(f"No data available for step {step}. Skipping it.")
                        break
                    raise

        # SECTION DESCRIPTION : ENSURE 100% SOC

        step = 400

        experiment = pybamm.Experiment([   #* Try to ensure 100 %SOC
                pybamm.step.string("Rest for 5 minutes"),
                pybamm.step.string("Charge at C/10 until 4.2V"),
                pybamm.step.string("Rest for 5 minutes"),
                ],
                period=f"{max_time_step} seconds"
        )

        sim = pybamm.Simulation(
            model,
            parameter_values=params,
            experiment=experiment,
            solver=solver,
        )

        solution = sim.solve(showprogress=True, starting_solution = last_state )

        if plot_interm:
            sim.plot(
                output_variables = output_vars,
                show_plot = False
            )

            plt.savefig(fname = f"hack\\model\\output\\figures\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.svg")
            plt.close("all")

        data_output = pd.DataFrame(solution.get_data_dict())
        data_output["time [s]"] = solution.t
        data_output.to_parquet(f"hack\\model\\output\\parquet\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.parquet", compression="gzip")
        if generate_csv: data_output.to_csv(f"hack\\model\\output\\csv\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.csv")

        last_state = solution.last_state

    elif test_protocol == "ICI_short_rest_100soc_long_experiment":      

        # SECTION DESCRIPTION : DISCHARGE STEPS

        for step in range(0, 200):

            try:

                experiment = pybamm.Experiment(
                        [   #* 80 normal steps of GITT profile with a discharge of 1.25 %SOC each
                            pybamm.step.string("Rest for 10 seconds"),  
                            pybamm.step.string("Discharge at C/5 for 15 minutes or until 2.5V"),     
                        ]
                        )
                
                sim = pybamm.Simulation(
                    model,
                    parameter_values=params,
                    experiment=experiment,
                    solver=solver,
                )

                if step == 0:
                    solution = sim.solve(showprogress=True, initial_soc=intial_soc_pu)
                else:
                    solution = sim.solve(showprogress=True, starting_solution = last_state)
                last_state = solution.last_state


                if plot_interm:
                    sim.plot(
                        output_variables = output_vars,
                        show_plot = False
                    )

                    plt.savefig(fname = f"hack\\model\\output\\figures\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.svg")
                    plt.close("all")

                data_output = pd.DataFrame(solution.get_data_dict())
                data_output["time [s]"] = solution.t
                data_output.to_parquet(f"hack\\model\\output\\parquet\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.parquet", compression="gzip")
                if generate_csv: data_output.to_csv(f"hack\\model\\output\\csv\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.csv")

                last_state = solution.last_state # This step is after data_output generation because there is where the step fails, so it does not affect ot the last valid state.

                last_voltage = float(
                    last_state["Voltage [V]"].entries[-1]
                )

                if last_voltage < 2.501:
                    break

            except ValueError as error:

                    if "Solution does not have any data" in str(error):
                        print(f"No data available for step {step}. Skipping it.")
                        break
                    raise

        # SECTION DESCRIPTION : ENSURE 0% SOC

        step = 200

        experiment = [   #* Try to ensure 0 %SOC
                pybamm.step.string("Hold at 2.5 V until C/20", direction="discharge"),
            ]

        sim = pybamm.Simulation(
            model,
            parameter_values=params,
            experiment=experiment,
            solver=solver,
        )

        solution = sim.solve(showprogress=True, starting_solution = last_state )

        last_state = solution.last_state

        if plot_interm:
            sim.plot(
                output_variables = output_vars,
                show_plot = False
            )

            plt.savefig(fname = f"hack\\model\\output\\figures\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.svg")
            plt.close("all")

        data_output = pd.DataFrame(solution.get_data_dict())
        data_output["time [s]"] = solution.t
        data_output.to_parquet(f"hack\\model\\output\\parquet\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.parquet", compression="gzip")
        if generate_csv: data_output.to_csv(f"hack\\model\\output\\csv\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.csv")

        # SECTION DESCRIPTION : CHARGE STEPS 

        for step in range(201, 400):
                
            try:
                experiment = pybamm.Experiment(
                        [   #* 80 normal steps of GITT profile with a discharge of 1.25 %SOC each
                            pybamm.step.string("Rest for 10 seconds"),   
                            pybamm.step.string("Charge at C/5 for 15 minutes or until 4.2V"),        
                        ]
                        )
                
                sim = pybamm.Simulation(
                    model,
                    parameter_values=params,
                    experiment=experiment,
                    solver=solver,
                )

                solution = sim.solve(showprogress=True, starting_solution = last_state )

                if plot_interm:
                    sim.plot(
                        output_variables = output_vars,
                        show_plot = False
                    )

                    plt.savefig(fname = f"hack\\model\\output\\figures\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.svg")
                    plt.close("all")

                data_output = pd.DataFrame(solution.get_data_dict())
                data_output["time [s]"] = solution.t
                data_output.to_parquet(f"hack\\model\\output\\parquet\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.parquet", compression="gzip")
                if generate_csv: data_output.to_csv(f"hack\\model\\output\\csv\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.csv")

                last_state = solution.last_state # This step is after data_output generation because there is where the step fails, so it does not affect ot the last valid state.

                last_voltage = float(
                    last_state["Voltage [V]"].entries[-1]
                )

                if last_voltage > 4.199:
                    break

            except ValueError as error:

                    if "Solution does not have any data" in str(error):
                        print(f"No data available for step {step}. Skipping it.")
                        break
                    raise

        # SECTION DESCRIPTION : ENSURE 100% SOC

        step = 400

        experiment = [   #* Try to ensure 100 %SOC
                pybamm.step.string("Hold at 4.2 V until C/20", direction = "charge"),
                pybamm.step.string("Rest for 5 minutes"),
                ]

        sim = pybamm.Simulation(
            model,
            parameter_values=params,
            experiment=experiment,
            solver=solver,
        )

        solution = sim.solve(showprogress=True, starting_solution = last_state )

        if plot_interm:
            sim.plot(
                output_variables = output_vars,
                show_plot = False
            )

            plt.savefig(fname = f"hack\\model\\output\\figures\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.svg")
            plt.close("all")

        data_output = pd.DataFrame(solution.get_data_dict())
        data_output["time [s]"] = solution.t
        data_output.to_parquet(f"hack\\model\\output\\parquet\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.parquet", compression="gzip")
        if generate_csv: data_output.to_csv(f"hack\\model\\output\\csv\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.csv")

        last_state = solution.last_state

    elif test_protocol == "HPPC_short_rest_100soc_long_experiment":      

        step = 0

        # SECTION DESCRIPTION : DISCHARGE STEPS

        for step in range(0, 200):

            try:

                experiment = pybamm.Experiment(
                        [   
                            pybamm.step.string("Rest for 10 minutes"),
                            pybamm.step.string("Discharge at 1C for 15 seconds or until 2.5V"),
                            pybamm.step.string("Rest for 10 minutes"),
                            pybamm.step.string("Charge at 1C for 15 seconds or until 4.2V"),
                            pybamm.step.string("Rest for 10 minutes"),
                            pybamm.step.string("Discharge at 0.5C for 0.2 hours or until 2.5V"), # Discharge aprox 10 %SOC
                        ],
                        period=f"{max_time_step} seconds"
                        )
                
                sim = pybamm.Simulation(
                    model,
                    parameter_values=params,
                    experiment=experiment,
                    solver=solver,
                )

                if step == 0:
                    solution = sim.solve(showprogress=True, initial_soc=intial_soc_pu)
                else:
                    solution = sim.solve(showprogress=True, starting_solution = last_state)

                if plot_interm:
                    sim.plot(
                        output_variables = output_vars,
                        show_plot = False
                    )

                    plt.savefig(fname = f"hack\\model\\output\\figures\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.svg")
                    plt.close("all")

                data_output = pd.DataFrame(solution.get_data_dict())
                data_output["time [s]"] = solution.t
                data_output.to_parquet(f"hack\\model\\output\\parquet\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.parquet", compression="gzip")
                if generate_csv: data_output.to_csv(f"hack\\model\\output\\csv\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.csv")

                last_state = solution.last_state # This step is after data_output generation because there is where the step fails, so it does not affect ot the last valid state.

                last_voltage = float(
                    last_state["Voltage [V]"].entries[-1]
                )

                if last_voltage < 2.501:
                    break

            except ValueError as error:

                    if "Solution does not have any data" in str(error):
                        print(f"No data available for step {step}. Skipping it.")
                        break
                    raise

        # SECTION DESCRIPTION : ENSURE 0% SOC

        step = 200

        experiment = pybamm.Experiment([   #* Try to ensure 0 %SOC
                pybamm.step.string("Rest for 5 minutes"),
                pybamm.step.string("Discharge at C/10 until 2.5V"),
                pybamm.step.string("Hold at 2.5 V until C/20", direction="discharge"),
            ],
            period=f"{max_time_step} seconds"
        )

        sim = pybamm.Simulation(
            model,
            parameter_values=params,
            experiment=experiment,
            solver=solver,
        )

        solution = sim.solve(showprogress=True, starting_solution = last_state )

        last_state = solution.last_state

        if plot_interm:
            sim.plot(
                output_variables = output_vars,
                show_plot = False
            )

            plt.savefig(fname = f"hack\\model\\output\\figures\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.svg")
            plt.close("all")

        data_output = pd.DataFrame(solution.get_data_dict())
        data_output["time [s]"] = solution.t
        data_output.to_parquet(f"hack\\model\\output\\parquet\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.parquet", compression="gzip")
        if generate_csv: data_output.to_csv(f"hack\\model\\output\\csv\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.csv")

        # SECTION DESCRIPTION : CHARGE STEPS 

        for step in range(201, 400):
                
            try:
                experiment = pybamm.Experiment(
                        [                      
                            pybamm.step.string("Rest for 10 minutes"),
                            pybamm.step.string("Charge at 1C for 15 seconds or until 4.2V"),
                            pybamm.step.string("Rest for 10 minutes"),
                            pybamm.step.string("Discharge at 1C for 15 seconds or until 2.5V"),
                            pybamm.step.string("Charge at 0.5C for 0.2 hours or until 4.2V"), # Charge aprox 10 %SOC                            
                        ],
                        period=f"{max_time_step} seconds"
                        )
                
                sim = pybamm.Simulation(
                    model,
                    parameter_values=params,
                    experiment=experiment,
                    solver=solver,
                )

                solution = sim.solve(showprogress=True, starting_solution = last_state )

                if plot_interm:
                    sim.plot(
                        output_variables = output_vars,
                        show_plot = False
                    )

                    plt.savefig(fname = f"hack\\model\\output\\figures\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.svg")
                    plt.close("all")

                data_output = pd.DataFrame(solution.get_data_dict())
                data_output["time [s]"] = solution.t
                data_output.to_parquet(f"hack\\model\\output\\parquet\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.parquet", compression="gzip")
                if generate_csv: data_output.to_csv(f"hack\\model\\output\\csv\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.csv")

                last_state = solution.last_state # This step is after data_output generation because there is where the step fails, so it does not affect ot the last valid state.

                last_voltage = float(
                    last_state["Voltage [V]"].entries[-1]
                )

                if last_voltage > 4.199:
                    break

            except ValueError as error:

                    if "Solution does not have any data" in str(error):
                        print(f"No data available for step {step}. Skipping it.")
                        break
                    raise

        # SECTION DESCRIPTION : ENSURE 100% SOC

        step = 400

        experiment = pybamm.Experiment([   #* Try to ensure 100 %SOC
                pybamm.step.string("Rest for 10 minutes"),
                pybamm.step.string("Charge at C/10 until 4.2V"),
                pybamm.step.string("Hold at 4.2V until C/20", direction="charge"), #! Start with this is dangerous because if it is not at 4.2V it can demand a high current suddenly
                pybamm.step.string("Rest for 5 minutes"),
                ],
                period=f"{max_time_step} seconds"
        )

        sim = pybamm.Simulation(
            model,
            parameter_values=params,
            experiment=experiment,
            solver=solver,
        )

        solution = sim.solve(showprogress=True, starting_solution = last_state )

        if plot_interm:
            sim.plot(
                output_variables = output_vars,
                show_plot = False
            )

            plt.savefig(fname = f"hack\\model\\output\\figures\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.svg")
            plt.close("all")

        data_output = pd.DataFrame(solution.get_data_dict())
        data_output["time [s]"] = solution.t
        data_output.to_parquet(f"hack\\model\\output\\parquet\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.parquet", compression="gzip")
        if generate_csv: data_output.to_csv(f"hack\\model\\output\\csv\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.csv")

        last_state = solution.last_state

    elif test_protocol == "fix_fail_long_experiment":      

        step = 0

        experiment = pybamm.Experiment(
                [   #* 80 normal steps of GITT profile with a discharge of 1.25 %SOC each
                    pybamm.step.string("Rest for 5 minutes"),
                    pybamm.step.string("Discharge at C/10 for 7.5 min or until 2.5V"), # Discharge step of 1.25 %SOC
                ]
                )
        
        sim = pybamm.Simulation(
            model,
            parameter_values=params,
            experiment=experiment,
            solver=solver,
        )

        intial_soc_pu = 0.1

        solution = sim.solve(showprogress=True, initial_soc=intial_soc_pu)
        
        last_state = solution.last_state

        if plot_interm:
            sim.plot(
                output_variables = output_vars,
                show_plot = False
            )

            plt.savefig(fname = f"hack\\model\\output\\figures\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.svg")
            plt.close("all")
        data_output = pd.DataFrame(solution.get_data_dict())
        data_output["time [s]"] = solution.t
        data_output.to_parquet(f"hack\\model\\output\\parquet\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.parquet", compression="gzip")
        if generate_csv: data_output.to_csv(f"hack\\model\\output\\csv\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.csv")

        # SECTION DESCRIPTION : DISCHARGE STEPS

        for step in range(1, 10):

            try:

                experiment = pybamm.Experiment(
                        [   #* 80 normal steps of GITT profile with a discharge of 1.25 %SOC each
                            pybamm.step.string("Rest for 5 minutes"),
                            pybamm.step.string("Discharge at C/10 for 7.5 min or until 2.5V"), # Discharge step of 1.25 %SOC
                        ]
                        )
                
                sim = pybamm.Simulation(
                    model,
                    parameter_values=params,
                    experiment=experiment,
                    solver=solver,
                )

                solution = sim.solve(showprogress=True, starting_solution = last_state )            

                if plot_interm:
                    sim.plot(
                        output_variables = output_vars,
                        show_plot = False
                    )

                    plt.savefig(fname = f"hack\\model\\output\\figures\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.svg")
                    plt.close("all")

                data_output = pd.DataFrame(solution.get_data_dict())



                data_output["time [s]"] = solution.t
                data_output.to_parquet(f"hack\\model\\output\\parquet\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.parquet", compression="gzip")
                if generate_csv: data_output.to_csv(f"hack\\model\\output\\csv\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.csv")

                last_state = solution.last_state # This step is after data_output generation because there is where the step fails, so it does not affect ot the last valid state.

                last_voltage = float(
                    last_state["Voltage [V]"].entries[-1]
                )

                if last_voltage < 2.501:
                    break

            except ValueError as error:

                    if "Solution does not have any data" in str(error):
                        print(f"No data available for step {step}. Skipping it.")
                        break
                    raise

        # SECTION DESCRIPTION : ENSURE 0% SOC

        step = 80

        experiment = [   #* Try to ensure 0 %SOC                
                pybamm.step.string("Hold at 2.5 V until C/40", direction="discharge"),
                pybamm.step.string("Rest for 5 minutes"),
            ]

        sim = pybamm.Simulation(
            model,
            parameter_values=params,
            experiment=experiment,
            solver=solver,
        )

        solution = sim.solve(showprogress=True, starting_solution = last_state )


        if plot_interm:
            sim.plot(
                output_variables = output_vars,
                show_plot = False
            )

            plt.savefig(fname = f"hack\\model\\output\\figures\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.svg")
            plt.close("all")

        data_output = pd.DataFrame(solution.get_data_dict())

        last_state = solution.last_state

        data_output["time [s]"] = solution.t
        data_output.to_parquet(f"hack\\model\\output\\parquet\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.parquet", compression="gzip")
        if generate_csv: data_output.to_csv(f"hack\\model\\output\\csv\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.csv")

        # SECTION DESCRIPTION : CHARGE STEPS 

        for step in range(81, 83):

            experiment = pybamm.Experiment(
                    [   #* 80 normal steps of GITT profile with a discharge of 1.25 %SOC each
                        pybamm.step.string("Rest for 5 minutes"),
                        pybamm.step.string("Charge at C/10 for 7.5 min or until 4.199V"), # Discharge step of 1.25 %SOC
                    ]
                    )
            
            sim = pybamm.Simulation(
                model,
                parameter_values=params,
                experiment=experiment,
                solver=solver,
            )

            solution = sim.solve(showprogress=True, starting_solution = last_state )


            if plot_interm:
                sim.plot(
                    output_variables = output_vars,
                    show_plot = False
                )

                plt.savefig(fname = f"hack\\model\\output\\figures\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.svg")
                plt.close("all")

            data_output = pd.DataFrame(solution.get_data_dict())

            last_state = solution.last_state

            data_output["time [s]"] = solution.t
            data_output.to_parquet(f"hack\\model\\output\\parquet\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.parquet", compression="gzip")
            if generate_csv: data_output.to_csv(f"hack\\model\\output\\csv\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.csv")

        # SECTION DESCRIPTION : ENSURE 100% SOC

        step = 160

        experiment = [   #* Try to ensure 0 %SOC
                pybamm.step.string("Rest for 5 minutes"),
                pybamm.step.string("Charge at C/10 for 1 minute"),
                pybamm.step.string("Rest for 5 minutes"),
                ]

        sim = pybamm.Simulation(
            model,
            parameter_values=params,
            experiment=experiment,
            solver=solver,
        )

        solution = sim.solve(showprogress=True, starting_solution = last_state )

        last_state = solution.last_state

        if plot_interm:
            sim.plot(
                output_variables = output_vars,
                show_plot = False
            )

            plt.savefig(fname = f"hack\\model\\output\\figures\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.svg")
            plt.close("all")

        data_output = pd.DataFrame(solution.get_data_dict())
        data_output["time [s]"] = solution.t
        data_output.to_parquet(f"hack\\model\\output\\parquet\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.parquet", compression="gzip")
        if generate_csv: data_output.to_csv(f"hack\\model\\output\\csv\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.csv")

    elif test_protocol == "APP_short_rest_100soc_long_experiment":      

        # SECTION DESCRIPTION : DISCHARGE STEPS

        for step in range(0, 200):

            try:

                experiment = pybamm.Experiment(
                    [
                        # Initial rest (to make the long experiment logic feasible)
                        "Rest for 10 minutes",
                        # Cycle 1 - Charge
                        "Charge at 0.10C for 180 seconds or until 4.2V",
                        "Rest for 10 minutes",
                        "Charge at 0.25C for 72 seconds or until 4.2V",
                        "Rest for 10 minutes",
                        "Charge at 0.50C for 72 seconds or until 4.2V",
                        "Rest for 10 minutes",
                        # Cycle 1 - Dicharge
                        "Discharge at 0.10C for 180 seconds or until 2.5V",
                        "Rest for 10 minutes",
                        "Discharge at 0.25C for 72 seconds or until 2.5V",
                        "Rest for 10 minutes",
                        "Discharge at 0.50C for 72 seconds or until 2.5V",
                        "Rest for 10 minutes",
                        # Cycle 2 - Charge
                        "Charge at 0.75C for 24 seconds or until 4.2V",
                        "Rest for 10 minutes",
                        "Charge at 1.00C for 18 seconds or until 4.2V",
                        "Rest for 10 minutes",
                        "Charge at 0.50C for 72 seconds or until 4.2V",
                        "Rest for 10 minutes",
                        # Cycle 2 - Discharge
                        "Discharge at 0.75C for 24 seconds or until 2.5V",
                        "Rest for 10 minutes",
                        "Discharge at 1.00C for 18 seconds or until 2.5V",
                        "Rest for 10 minutes",
                        "Discharge at 0.50C for 72 seconds or until 2.5V",
                        "Rest for 10 minutes",
                        # Cycle 3 - Charge
                        "Charge at 1.50C for 12 seconds or until 4.2V",
                        "Rest for 10 minutes",
                        "Charge at 2.00C for 9 seconds or until 4.2V",
                        "Rest for 10 minutes",
                        "Charge at 0.50C for 72 seconds or until 4.2V",
                        "Rest for 10 minutes",
                        # Cycle 3 - Discharge
                        "Discharge at 1.50C for 12 seconds or until 2.5V",
                        "Rest for 10 minutes",
                        "Discharge at 2.00C for 9 seconds or until 2.5V",
                        "Rest for 10 minutes",
                        "Discharge at 0.50C for 72 seconds or until 2.5V",
                        "Rest for 10 minutes",
                        # Discharge aprox 10%SOC
                        "Discharge at C/2.5 for 15 minutes or until 2.5V"
                        
                    ],
                        period=f"{max_time_step} seconds"
                        )
                
                sim = pybamm.Simulation(
                    model,
                    parameter_values=params,
                    experiment=experiment,
                    solver=solver,
                )

                if step == 0:
                    solution = sim.solve(showprogress=True, initial_soc=intial_soc_pu)
                else:
                    solution = sim.solve(showprogress=True, starting_solution = last_state)
                
                if plot_interm:
                    sim.plot(
                        output_variables = output_vars,
                        show_plot = False
                    )

                    plt.savefig(fname = f"hack\\model\\output\\figures\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.svg")
                    plt.close("all")
                data_output = pd.DataFrame(solution.get_data_dict())
                data_output["time [s]"] = solution.t
                data_output.to_parquet(f"hack\\model\\output\\parquet\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.parquet", compression="gzip")
                if generate_csv: data_output.to_csv(f"hack\\model\\output\\csv\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.csv")

                last_state = solution.last_state # This step is after data_output generation because there is where the step fails, so it does not affect ot the last valid state.

                last_voltage = float(
                    last_state["Voltage [V]"].entries[-1]
                )

                if last_voltage < 2.501:
                    break

            except ValueError as error:

                    if "Solution does not have any data" in str(error):
                        print(f"No data available for step {step}. Skipping it.")
                        break
                    raise

        # SECTION DESCRIPTION : ENSURE 0% SOC

        step = 200

        experiment = pybamm.Experiment([   #* Try to ensure 0 %SOC
                pybamm.step.string("Hold at 2.5 V until C/40", direction="discharge"),
            ],
            period=f"{max_time_step} seconds"
        )
            

        sim = pybamm.Simulation(
            model,
            parameter_values=params,
            experiment=experiment,
            solver=solver,
        )

        solution = sim.solve(showprogress=True, starting_solution = last_state )

        last_state = solution.last_state

        if plot_interm:
            sim.plot(
                output_variables = output_vars,
                show_plot = False
            )

            plt.savefig(fname = f"hack\\model\\output\\figures\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.svg")
            plt.close("all")

        data_output = pd.DataFrame(solution.get_data_dict())
        data_output["time [s]"] = solution.t
        data_output.to_parquet(f"hack\\model\\output\\parquet\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.parquet", compression="gzip")
        if generate_csv: data_output.to_csv(f"hack\\model\\output\\csv\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.csv")

        # SECTION DESCRIPTION : REPEAT AT 0% SOC

        step = 201

        experiment = pybamm.Experiment([   
                        # Initial rest (to make the long experiment logic feasible)
                        "Rest for 10 minutes",
                        #* Repeat the cycle block at 0 %SOC
                        # Cycle 1 - Charge
                        "Charge at 0.10C for 180 seconds or until 4.2V",
                        "Rest for 10 minutes",
                        "Charge at 0.25C for 72 seconds or until 4.2V",
                        "Rest for 10 minutes",
                        "Charge at 0.50C for 72 seconds or until 4.2V",
                        "Rest for 10 minutes",
                        # Cycle 1 - Dicharge
                        "Discharge at 0.10C for 180 seconds or until 2.5V",
                        "Rest for 10 minutes",
                        "Discharge at 0.25C for 72 seconds or until 2.5V",
                        "Rest for 10 minutes",
                        "Discharge at 0.50C for 72 seconds or until 2.5V",
                        "Rest for 10 minutes",
                        # Cycle 2 - Charge
                        "Charge at 0.75C for 24 seconds or until 4.2V",
                        "Rest for 10 minutes",
                        "Charge at 1.00C for 18 seconds or until 4.2V",
                        "Rest for 10 minutes",
                        "Charge at 0.50C for 72 seconds or until 4.2V",
                        "Rest for 10 minutes",
                        # Cycle 2 - Discharge
                        "Discharge at 0.75C for 24 seconds or until 2.5V",
                        "Rest for 10 minutes",
                        "Discharge at 1.00C for 18 seconds or until 2.5V",
                        "Rest for 10 minutes",
                        "Discharge at 0.50C for 72 seconds or until 2.5V",
                        "Rest for 10 minutes",
                        # Cycle 3 - Charge
                        "Charge at 1.50C for 12 seconds or until 4.2V",
                        "Rest for 10 minutes",
                        "Charge at 2.00C for 9 seconds or until 4.2V",
                        "Rest for 10 minutes",
                        "Charge at 0.50C for 72 seconds or until 4.2V",
                        "Rest for 10 minutes",
                        # Cycle 3 - Discharge
                        "Discharge at 1.50C for 12 seconds or until 2.5V",
                        "Rest for 10 minutes",
                        "Discharge at 2.00C for 9 seconds or until 2.5V",
                        "Rest for 10 minutes",
                        "Discharge at 0.50C for 72 seconds or until 2.5V",
                        "Rest for 10 minutes",
                    ],
                    period=f"{max_time_step} seconds"
        )

        sim = pybamm.Simulation(
            model,
            parameter_values=params,
            experiment=experiment,
            solver=solver,
        )

        solution = sim.solve(showprogress=True, starting_solution = last_state )

        last_state = solution.last_state

        if plot_interm:
            sim.plot(
                output_variables = output_vars,
                show_plot = False
            )

            plt.savefig(fname = f"hack\\model\\output\\figures\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.svg")
            plt.close("all")

        data_output = pd.DataFrame(solution.get_data_dict())
        data_output["time [s]"] = solution.t
        data_output.to_parquet(f"hack\\model\\output\\parquet\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.parquet", compression="gzip")
        if generate_csv: data_output.to_csv(f"hack\\model\\output\\csv\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.csv")

    elif test_protocol == "ICA_short_rest_100soc_long_experiment_alternative":      

        step = 0

        # SECTION DESCRIPTION : DISCHARGE STEPS       

        experiment = pybamm.Experiment(
                [
                    pybamm.step.string("Rest for 5 minutes"),                  # Resting at 100.00% SOC
                    pybamm.step.string("Discharge at C/20 until 2.5V"),         # Discharge period at C/40
                ],
                        period=f"{max_time_step} seconds"
                )
        
        sim = pybamm.Simulation(
            model,
            parameter_values=params,
            experiment=experiment,
            solver=solver,
        )

        if step == 0:
            solution = sim.solve(showprogress=True, initial_soc=intial_soc_pu)
        else:
            solution = sim.solve(showprogress=True, starting_solution = last_state)
        
        last_state = solution.last_state

        if plot_interm:
            sim.plot(
                output_variables = output_vars,
                show_plot = False
            )

            plt.savefig(fname = f"hack\\model\\output\\figures\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.svg")
            plt.close("all")
        data_output = pd.DataFrame(solution.get_data_dict())
        data_output["time [s]"] = solution.t
        data_output.to_parquet(f"hack\\model\\output\\parquet\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.parquet", compression="gzip")
        if generate_csv: data_output.to_csv(f"hack\\model\\output\\csv\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.csv")

        step = 1

        experiment = pybamm.Experiment(
                [
                    pybamm.step.string("Rest for 5 minutes"),                  # Resting at 0.00% SOC
                    pybamm.step.string("Charge at C/20 until 4.2V"),            # Charge period at C/40
                ],
                        period=f"{max_time_step} seconds"
                )
        
        sim = pybamm.Simulation(
            model,
            parameter_values=params,
            experiment=experiment,
            solver=solver,
        )

        solution = sim.solve(showprogress=True, starting_solution = last_state)
        
        last_state = solution.last_state

        if plot_interm:
            sim.plot(
                output_variables = output_vars,
                show_plot = False
            )

            plt.savefig(fname = f"hack\\model\\output\\figures\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.svg")
            plt.close("all")
        data_output = pd.DataFrame(solution.get_data_dict())
        data_output["time [s]"] = solution.t
        data_output.to_parquet(f"hack\\model\\output\\parquet\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.parquet", compression="gzip")
        if generate_csv: data_output.to_csv(f"hack\\model\\output\\csv\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.csv")
  
        step = 2

        experiment = pybamm.Experiment(
                [
                    pybamm.step.string("Rest for 5 minutes"),                   # Resting at 100.00% SOC
                ],
                        period=f"{max_time_step} seconds"
                )
        
        sim = pybamm.Simulation(
            model,
            parameter_values=params,
            experiment=experiment,
            solver=solver,
        )

        solution = sim.solve(showprogress=True, starting_solution = last_state)
        
        last_state = solution.last_state

        if plot_interm:
            sim.plot(
                output_variables = output_vars,
                show_plot = False
            )

            plt.savefig(fname = f"hack\\model\\output\\figures\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.svg")
            plt.close("all")
        data_output = pd.DataFrame(solution.get_data_dict())
        data_output["time [s]"] = solution.t
        data_output.to_parquet(f"hack\\model\\output\\parquet\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.parquet", compression="gzip")
        if generate_csv: data_output.to_csv(f"hack\\model\\output\\csv\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.csv")
  
    elif test_protocol == "ICA_short_rest_100soc_long_experiment":      

        # SECTION DESCRIPTION : DISCHARGE STEPS

        for step in range(0,200):

            try:                

                experiment = pybamm.Experiment(
                        [
                            pybamm.step.string("Discharge at C/20 for 0.5 hours or until 2.5V"),         # Discharge period at C/40
                        ],
                        period=f"{max_time_step} seconds"
                        )
                
                sim = pybamm.Simulation(
                    model,
                    parameter_values=params,
                    experiment=experiment,
                    solver=solver,
                )

                if step == 0:
                    solution = sim.solve(showprogress=True, initial_soc=intial_soc_pu)
                else:
                    solution = sim.solve(showprogress=True, starting_solution = last_state)

                if plot_interm:
                    sim.plot(
                        output_variables = output_vars,
                        show_plot = False
                    )

                    plt.savefig(fname = f"hack\\model\\output\\figures\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.svg")
                    plt.close("all")
                data_output = pd.DataFrame(solution.get_data_dict())
                data_output["time [s]"] = solution.t
                data_output.to_parquet(f"hack\\model\\output\\parquet\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.parquet", compression="gzip")
                if generate_csv: data_output.to_csv(f"hack\\model\\output\\csv\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.csv")

                last_state = solution.last_state # This step is after data_output generation because there is where the step fails, so it does not affect ot the last valid state.

                last_voltage = float(
                    last_state["Voltage [V]"].entries[-1]
                )

                if last_voltage < 2.501:
                    break

            except ValueError as error:

                    if "Solution does not have any data" in str(error):
                        print(f"No data available for step {step}. Skipping it.")
                        break
                    raise

        step = 200

        experiment = pybamm.Experiment(
                [
                    pybamm.step.string("Rest for 5 minutes"),                  # Resting at 0.00% SOC
                ],
                        period=f"{max_time_step} seconds"
                )
        
        sim = pybamm.Simulation(
            model,
            parameter_values=params,
            experiment=experiment,
            solver=solver,
        )

        solution = sim.solve(showprogress=True, starting_solution = last_state)
        
        last_state = solution.last_state

        if plot_interm:
            sim.plot(
                output_variables = output_vars,
                show_plot = False
            )

            plt.savefig(fname = f"hack\\model\\output\\figures\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.svg")
            plt.close("all")
        data_output = pd.DataFrame(solution.get_data_dict())
        data_output["time [s]"] = solution.t
        data_output.to_parquet(f"hack\\model\\output\\parquet\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.parquet", compression="gzip")
        if generate_csv: data_output.to_csv(f"hack\\model\\output\\csv\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.csv")

        for step in range(201,400):
            try:
                experiment = pybamm.Experiment(
                        [
                            pybamm.step.string("Charge at C/20 for 0.5 hours or until 4.2V"),            # Charge period at C/40
                        ],
                        period=f"{max_time_step} seconds"
                        )
                
                sim = pybamm.Simulation(
                    model,
                    parameter_values=params,
                    experiment=experiment,
                    solver=solver,
                )

                solution = sim.solve(showprogress=True, starting_solution = last_state)

                if plot_interm:
                    sim.plot(
                        output_variables = output_vars,
                        show_plot = False
                    )

                    plt.savefig(fname = f"hack\\model\\output\\figures\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.svg")
                    plt.close("all")
                data_output = pd.DataFrame(solution.get_data_dict())
                data_output["time [s]"] = solution.t
                data_output.to_parquet(f"hack\\model\\output\\parquet\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.parquet", compression="gzip")
                if generate_csv: data_output.to_csv(f"hack\\model\\output\\csv\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.csv")
    
                last_state = solution.last_state # This step is after data_output generation because there is where the step fails, so it does not affect ot the last valid state.

                last_voltage = float(
                    last_state["Voltage [V]"].entries[-1]
                )

                if last_voltage > 4.199:
                    break

            except ValueError as error:

                    if "Solution does not have any data" in str(error):
                        print(f"No data available for step {step}. Skipping it.")
                        break
                    raise

        step = 400

        experiment = pybamm.Experiment(
                [
                    pybamm.step.string("Rest for 5 minutes"),                   # Resting at 100.00% SOC
                ],
                        period=f"{max_time_step} seconds"
                )
        
        sim = pybamm.Simulation(
            model,
            parameter_values=params,
            experiment=experiment,
            solver=solver,
        )

        solution = sim.solve(showprogress=True, starting_solution = last_state)
        
        last_state = solution.last_state

        if plot_interm:
            sim.plot(
                output_variables = output_vars,
                show_plot = False
            )

            plt.savefig(fname = f"hack\\model\\output\\figures\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.svg")
            plt.close("all")
        data_output = pd.DataFrame(solution.get_data_dict())
        data_output["time [s]"] = solution.t
        data_output.to_parquet(f"hack\\model\\output\\parquet\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.parquet", compression="gzip")
        if generate_csv: data_output.to_csv(f"hack\\model\\output\\csv\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.csv")
          
    elif test_protocol == "cycling_A_HPPC_long_experiment":      

        loop_max = 5

        for loop in range(0,loop_max):            

            step = 0 + loop*100
            
            experiment = generate_experiment("HPPC_short_rest_100soc", fixed_period = True, period_value = max_time_step)
            
            sim = pybamm.Simulation(
                model,
                parameter_values=params,
                experiment=experiment,
                solver=solver,
            )

            if step == 0:
                solution = sim.solve(showprogress=True, initial_soc=intial_soc_pu)
            else:
                solution = sim.solve(showprogress=True, starting_solution = last_state)
            last_state = solution.last_state

            if plot_interm:
                sim.plot(
                    output_variables = output_vars,
                    show_plot = False
                )

                plt.savefig(fname = f"hack\\model\\output\\figures\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.svg")
                plt.close("all")
            data_output = pd.DataFrame(solution.get_data_dict())
            data_output["time [s]"] = solution.t
            data_output.to_parquet(f"hack\\model\\output\\parquet\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.parquet", compression="gzip")
            if generate_csv: data_output.to_csv(f"hack\\model\\output\\csv\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.csv")

            # SECTION DESCRIPTION : DISCHARGE STEPS

            for step in range(1, 100):

                step += loop*100

                experiment = pybamm.Experiment(
                        [   
                            pybamm.step.string("Discharge at C/2 until 2.501V"),
                            pybamm.step.string("Charge at C/2 until 4.199V"),
                        ],
                        period=f"{max_time_step} seconds"
                        )
                
                sim = pybamm.Simulation(
                    model,
                    parameter_values=params,
                    experiment=experiment,
                    solver=solver,
                )

                solution = sim.solve(showprogress=True, starting_solution = last_state )

                last_state = solution.last_state

                if plot_interm:
                    sim.plot(
                        output_variables = output_vars,
                        show_plot = False
                    )

                    plt.savefig(fname = f"hack\\model\\output\\figures\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.svg")
                    plt.close("all")

                data_output = pd.DataFrame(solution.get_data_dict())
                data_output["time [s]"] = solution.t
                data_output.to_parquet(f"hack\\model\\output\\parquet\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.parquet", compression="gzip")
                if generate_csv: data_output.to_csv(f"hack\\model\\output\\csv\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.csv")

            

        # SECTION DESCRIPTION : ENSURE 0% SOC

        step = loop_max*100

        experiment = generate_experiment("HPPC_short_rest_100soc", fixed_period = True, period_value = max_time_step)

        sim = pybamm.Simulation(
            model,
            parameter_values=params,
            experiment=experiment,
            solver=solver,
        )

        solution = sim.solve(showprogress=True, starting_solution = last_state )

        last_state = solution.last_state

        if plot_interm:
            sim.plot(
                output_variables = output_vars,
                show_plot = False
            )

            plt.savefig(fname = f"hack\\model\\output\\figures\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.svg")
            plt.close("all")

        data_output = pd.DataFrame(solution.get_data_dict())
        data_output["time [s]"] = solution.t
        data_output.to_parquet(f"hack\\model\\output\\parquet\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.parquet", compression="gzip")
        if generate_csv: data_output.to_csv(f"hack\\model\\output\\csv\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_step_{step}.csv")
           
    # SECTION DESCRIPTION: MERGE ALL GENERATED PARQUET FILES

    parquet_directory = Path(
        "hack/model/output/parquet"
    )

    figure_directory = Path(
        "hack/model/output/figures"
    )

    file_prefix = (
        f"simulation_{model_type}_{param_set}_{solver_name}_"
        f"{test_protocol}_T{initial_cell_temp_k_text}_"
        f"{max_time_step_text}"
    )

    step_parquet_files = sorted(
        parquet_directory.glob(
            f"{file_prefix}_step_*.parquet"
        ),
        key=extract_step_number,
    )

    if not step_parquet_files:
        raise FileNotFoundError(
            f"No step Parquet files were found for prefix: {file_prefix}"
        )

    merged_parquet_file = (
        parquet_directory
        / f"{file_prefix}.parquet"
    )

    merged_figure_file = (
        figure_directory
        / f"{file_prefix}.png"
    )

    merged_rows = merge_parquet_files_streaming(
        input_files=step_parquet_files,
        output_file=merged_parquet_file,
        time_column="time [s]",
        batch_size=250_000,
        compression="gzip",
    )

    plot_merged_parquet(
        parquet_file=merged_parquet_file,
        output_file=merged_figure_file,
        output_variables=output_vars,
        time_column="time [s]",
        time_unit="hours",
        max_points=None,
        # batch_size=250_000,
        line_width=1.0,
        show_titles=True,
    )

    delete_step_parquet_files(
        step_parquet_files=step_parquet_files,
        merged_parquet_file=merged_parquet_file,
        expected_rows=merged_rows,
    )

    return False, False

def build_drive_cycle_power_from_dst_table(
    df: pd.DataFrame,
    duration_col: str = "Duration [s]",
    pct_col: str = "Discharge Power [%]",
    p_peak_w: float = 80.0,
    dt: float = 1.0,
    usabc_negative_is_discharge: bool = True,
) -> np.ndarray:
    """
    Convert a DST table (durations + % of peak power) into a PyBaMM drive-cycle array.

    Returns
    -------
    drive_cycle_power : np.ndarray, shape (N, 2)
        Column 0: time [s] starting at 0 (monotonic increasing)
        Column 1: power command [W], with PyBaMM convention:
                  + discharge, - charge. (pybamm.step.power)
    """
    df = df.copy()
    df.columns = df.columns.str.strip()

    durations = df[duration_col].to_numpy(dtype=float)
    pct = df[pct_col].to_numpy(dtype=float)

    if np.any(durations <= 0):
        raise ValueError("All durations must be > 0 seconds.")

    t_end = np.cumsum(durations)
    total_t = float(t_end[-1])

    # Constant time step grid (recommended for drive cycles) :contentReference[oaicite:1]{index=1}
    n = int(np.round(total_t / dt))
    if not np.isclose(n * dt, total_t, rtol=0, atol=1e-9):
        raise ValueError(
            f"Total duration {total_t} s is not an integer multiple of dt={dt} s."
        )

    t = np.arange(0.0, total_t + dt, dt)  # include endpoint (0 ... total_t)

    # Map each time to its step index
    # side="right": at exact boundaries we switch to the next step
    idx = np.searchsorted(t_end, t, side="right")
    idx = np.clip(idx, 0, len(pct) - 1)

    pct_series = pct[idx]  # [% of peak] at each time point

    # Convert % -> W. Your table uses negative values for discharge.
    # PyBaMM expects positive discharge. :contentReference[oaicite:2]{index=2}
    sign = -1.0 if usabc_negative_is_discharge else 1.0
    p_cmd_w = sign * (pct_series / 100.0) * p_peak_w

    drive_cycle_power = np.column_stack([t, p_cmd_w])

    # Safety: PyBaMM drive cycle must start at t=0
    assert drive_cycle_power[0, 0] == 0.0

    return drive_cycle_power

def extract_step_number(file_path: Path) -> int:
    """
    Extract the numerical step from a filename ending in '_step_<number>.parquet'.
    """
    match = re.search(r"_step_(\d+)\.parquet$", file_path.name)

    if match is None:
        raise ValueError(
            f"Could not extract the step number from: {file_path.name}"
        )

    return int(match.group(1))

def get_common_parquet_schema(
    input_files: list[Path],
) -> pa.Schema:
    """
    Determine a common schema for all input Parquet files.

    Column names must be identical in every file. Compatible datatype
    differences, such as float32 versus float64, are promoted.
    """
    schemas = []
    reference_columns = None

    for input_file in input_files:
        schema = pq.ParquetFile(input_file).schema_arrow.remove_metadata()
        column_names = schema.names

        if reference_columns is None:
            reference_columns = column_names
        else:
            missing_columns = set(reference_columns) - set(column_names)
            additional_columns = set(column_names) - set(reference_columns)

            if missing_columns or additional_columns:
                raise ValueError(
                    f"Column mismatch in '{input_file.name}'.\n"
                    f"Missing columns: {sorted(missing_columns)}\n"
                    f"Additional columns: {sorted(additional_columns)}"
                )

        # Reorder every schema according to the first file.
        reordered_schema = pa.schema(
            [
                schema.field(column_name)
                for column_name in reference_columns
            ]
        )

        schemas.append(reordered_schema)

    try:
        common_schema = pa.unify_schemas(
            schemas,
            promote_options="permissive",
        )
    except TypeError:
        raise RuntimeError(
            "Your PyArrow version does not support "
            "promote_options='permissive'. Update PyArrow or use the "
            "diagnostic function below to identify the differing columns."
        )

    return common_schema.remove_metadata()

def merge_parquet_files_streaming(
    input_files: list[Path],
    output_file: Path,
    time_column: str = "time [s]",
    batch_size: int = 250_000,
    compression: str = "gzip",
) -> int:
    """
    Merge Parquet files incrementally without loading all data into RAM.

    Compatible schema differences are normalised before writing. At every
    file boundary, rows whose time is not greater than the last written time
    are removed.
    """
    if not input_files:
        raise ValueError("No input Parquet files were provided.")

    input_files = [Path(file) for file in input_files]
    output_file = Path(output_file)

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if output_file.exists():
        output_file.unlink()

    # Reading schemas only reads Parquet metadata, not the complete datasets.
    output_schema = get_common_parquet_schema(input_files)

    if time_column not in output_schema.names:
        raise KeyError(
            f"Time column '{time_column}' was not found.\n"
            f"Available columns: {output_schema.names}"
        )

    writer = pq.ParquetWriter(
        where=output_file,
        schema=output_schema,
        compression=compression,
    )

    last_written_time = None
    total_rows_written = 0

    try:
        for file_number, input_file in enumerate(
            input_files,
            start=1,
        ):
            print(
                f"Merging file {file_number}/{len(input_files)}: "
                f"{input_file.name}"
            )

            parquet_file = pq.ParquetFile(input_file)

            for record_batch in parquet_file.iter_batches(
                batch_size=batch_size,
            ):
                table = pa.Table.from_batches([record_batch])

                # Ensure the same column order in every file.
                table = table.select(output_schema.names)

                # Convert compatible datatype differences to the common schema.
                try:
                    table = table.cast(
                        output_schema,
                        safe=True,
                    )
                except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as error:
                    raise ValueError(
                        f"Could not convert the schema of "
                        f"'{input_file.name}' to the common schema.\n\n"
                        f"File schema:\n{table.schema}\n\n"
                        f"Common schema:\n{output_schema}"
                    ) from error

                # Remove the copied final state from the previous simulation.
                if last_written_time is not None:
                    time_array = table[time_column]

                    valid_time_mask = pc.greater(
                        time_array,
                        pa.scalar(
                            last_written_time,
                            type=time_array.type,
                        ),
                    )

                    table = table.filter(valid_time_mask)

                if table.num_rows == 0:
                    continue

                writer.write_table(
                    table,
                    row_group_size=table.num_rows,
                )

                last_written_time = table[
                    time_column
                ][table.num_rows - 1].as_py()

                total_rows_written += table.num_rows

            del parquet_file
            gc.collect()

    finally:
        writer.close()

    print(
        f"Merged Parquet created: {output_file}\n"
        f"Rows written: {total_rows_written:,}\n"
        f"Final time: {last_written_time}"
    )

    return total_rows_written

def normalise_output_variable_groups(
    output_variables: list,
) -> list[list[str]]:
    """
    Convert PyBaMM-style output-variable definitions into plot groups.

    A string creates an individual subplot. A list or tuple causes its
    variables to be plotted together in the same subplot.
    """
    groups = []

    for item in output_variables:
        if isinstance(item, str):
            groups.append([item])

        elif isinstance(item, (list, tuple)):
            group = [variable for variable in item if isinstance(variable, str)]

            if group:
                groups.append(group)

        else:
            raise TypeError(
                "Every output variable must be a string, list, or tuple. "
                f"Received: {type(item)}"
            )

    return groups

def read_decimated_parquet_data(
    parquet_file: Path,
    columns: list[str],
    max_points: int = 50_000,
    batch_size: int = 250_000,
) -> dict[str, np.ndarray]:
    """
    Read selected Parquet columns with bounded memory consumption.

    At most approximately `max_points` rows are retained for plotting.
    """
    parquet_reader = pq.ParquetFile(parquet_file)

    available_columns = set(parquet_reader.schema_arrow.names)
    missing_columns = [
        column for column in columns
        if column not in available_columns
    ]

    if missing_columns:
        raise KeyError(
            "The following plot columns were not found in the merged "
            f"Parquet file:\n{missing_columns}"
        )

    total_rows = parquet_reader.metadata.num_rows

    if total_rows == 0:
        raise ValueError(f"The Parquet file is empty: {parquet_file}")

    sampling_stride = max(
        1,
        math.ceil(total_rows / max_points),
    )

    sampled_chunks = {
        column: []
        for column in columns
    }

    global_row_offset = 0
    final_row = None

    for record_batch in parquet_reader.iter_batches(
        batch_size=batch_size,
        columns=columns,
    ):
        number_of_rows = record_batch.num_rows

        # Select rows based on their global position in the full Parquet file.
        first_local_index = (
            -global_row_offset
        ) % sampling_stride

        local_indices = np.arange(
            first_local_index,
            number_of_rows,
            sampling_stride,
            dtype=np.int64,
        )

        if local_indices.size > 0:
            sampled_batch = record_batch.take(
                pa.array(local_indices)
            )

            for column in columns:
                column_index = sampled_batch.schema.get_field_index(column)

                sampled_chunks[column].append(
                    sampled_batch.column(column_index).to_numpy(
                        zero_copy_only=False
                    )
                )

        # Retain the actual final row, even when it is not selected by stride.
        final_row = record_batch.slice(
            number_of_rows - 1,
            1,
        )

        global_row_offset += number_of_rows

    sampled_data = {
        column: np.concatenate(sampled_chunks[column])
        if sampled_chunks[column]
        else np.empty(0)
        for column in columns
    }

    # Ensure that the final simulation point appears in the plot.
    if (total_rows - 1) % sampling_stride != 0:
        for column in columns:
            column_index = final_row.schema.get_field_index(column)

            final_value = final_row.column(
                column_index
            ).to_numpy(
                zero_copy_only=False
            )

            sampled_data[column] = np.concatenate(
                [
                    sampled_data[column],
                    final_value,
                ]
            )

    print(
        f"Plotting {len(sampled_data[columns[0]]):,} sampled rows "
        f"from {total_rows:,} total rows "
        f"(sampling stride: {sampling_stride})."
    )

    return sampled_data

def plot_merged_parquet(
    parquet_file: Path,
    output_file: Path,
    output_variables: list,
    time_column: str = "time [s]",
    time_unit: str = "hours",
    max_points: int | None = None,
    line_width: float = 1.0,
    show_titles: bool = True,
    dpi: int = 600,
) -> Path:
    """
    Plot selected columns from a merged Parquet file and save them as PNG.

    Each element of ``output_variables`` can be:
        - A string: one variable in one subplot.
        - A list/tuple: several variables in the same subplot.

    Parameters
    ----------
    max_points
        Maximum number of uniformly selected rows. Use None to plot all rows.
    dpi
        PNG resolution. Use 300 for normal documents or 600 for publication.
    """
    parquet_file = Path(parquet_file)
    output_file = Path(output_file).with_suffix(".png")

    if not parquet_file.is_file():
        raise FileNotFoundError(f"Parquet file not found: {parquet_file}")

    groups = [
        [item] if isinstance(item, str) else list(item)
        for item in output_variables
    ]

    if not groups:
        raise ValueError("No output variables were provided.")

    variables = list(
        dict.fromkeys(
            variable
            for group in groups
            for variable in group
        )
    )

    data = pd.read_parquet(
        parquet_file,
        columns=[time_column, *variables],
    )

    if data.empty:
        raise ValueError("The Parquet file contains no rows.")

    original_rows = len(data)

    # Uniformly reduce the data only when requested.
    if max_points is not None and original_rows > max_points:
        if max_points <= 1:
            raise ValueError("max_points must be greater than 1.")

        indices = np.linspace(
            0,
            original_rows - 1,
            max_points,
            dtype=int,
        )
        data = data.iloc[indices]

    time_units = {
        "seconds": (1.0, "Time [s]"),
        "minutes": (60.0, "Time [min]"),
        "hours": (3600.0, "Time [h]"),
    }

    if time_unit not in time_units:
        raise ValueError(
            "time_unit must be 'seconds', 'minutes', or 'hours'."
        )

    divisor, time_label = time_units[time_unit]

    time = (
        pd.to_numeric(data[time_column], errors="coerce")
        .to_numpy(dtype=float)
        / divisor
    )

    number_of_plots = len(groups)
    number_of_columns = math.ceil(math.sqrt(number_of_plots))
    number_of_rows = math.ceil(
        number_of_plots / number_of_columns
    )

    # Local figure with a non-interactive PNG renderer.
    figure = Figure(
        figsize=(
            5.5 * number_of_columns,
            3.5 * number_of_rows,
        ),
        facecolor="white",
    )
    FigureCanvasAgg(figure)

    lines_plotted = 0

    for index, group in enumerate(groups, start=1):
        axis = figure.add_subplot(
            number_of_rows,
            number_of_columns,
            index,
        )

        for variable in group:
            values = (
                pd.to_numeric(data[variable], errors="coerce")
                .to_numpy(dtype=float)
            )

            valid = np.isfinite(time) & np.isfinite(values)

            if not valid.any():
                print(f"Warning: no finite data for {variable!r}.")
                continue

            axis.plot(
                time[valid],
                values[valid],
                linewidth=line_width,
                label=variable,
            )
            lines_plotted += 1

        axis.set_xlabel(time_label)
        axis.set_ylabel(group[0] if len(group) == 1 else "Value")
        axis.grid(True, alpha=0.3)

        if len(group) > 1:
            axis.legend()

        if show_titles:
            axis.set_title(" / ".join(group))

    if lines_plotted == 0:
        figure.clear()
        raise ValueError("No finite data could be plotted.")

    figure.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    figure.savefig(
        output_file,
        format="png",
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )

    figure.clear()

    if not output_file.is_file() or output_file.stat().st_size == 0:
        raise RuntimeError(f"PNG export failed: {output_file}")

    print(
        f"PNG created using {len(data):,} of {original_rows:,} rows: "
        f"{output_file} "
        f"({output_file.stat().st_size / 1_048_576:.2f} MB)"
    )

    return output_file

def delete_step_parquet_files(
    step_parquet_files: list[Path],
    merged_parquet_file: Path,
    expected_rows: int | None = None,
) -> int:
    """
    Delete individual step Parquet files after validating the merged file.

    Parameters
    ----------
    step_parquet_files
        Exact list of individual step Parquet files to delete.
    merged_parquet_file
        Path to the merged Parquet file.
    expected_rows
        Expected number of rows in the merged file. When provided, deletion
        only proceeds if the merged file contains exactly this number of rows.

    Returns
    -------
    int
        Number of step files successfully deleted.

    Raises
    ------
    FileNotFoundError
        If the merged file does not exist.
    ValueError
        If the merged file is empty or its row count does not match.
    RuntimeError
        If one or more step files could not be deleted.
    """
    merged_parquet_file = Path(merged_parquet_file)
    step_parquet_files = [
        Path(file_path)
        for file_path in step_parquet_files
    ]

    if not merged_parquet_file.exists():
        raise FileNotFoundError(
            f"Merged Parquet file does not exist: {merged_parquet_file}"
        )

    # Confirm that the merged file can be opened and contains data.
    try:
        parquet_metadata = pq.read_metadata(merged_parquet_file)
        merged_rows = parquet_metadata.num_rows
    except Exception as error:
        raise ValueError(
            f"The merged Parquet file could not be validated: "
            f"{merged_parquet_file}"
        ) from error

    if merged_rows == 0:
        raise ValueError(
            "The merged Parquet file contains no rows. "
            "The step files will not be deleted."
        )

    if expected_rows is not None and merged_rows != expected_rows:
        raise ValueError(
            "Merged Parquet row count does not match the merge result.\n"
            f"Expected rows: {expected_rows:,}\n"
            f"Merged rows:   {merged_rows:,}\n"
            "The step files will not be deleted."
        )

    # Ensure that the merged file is never included in the deletion list.
    merged_resolved = merged_parquet_file.resolve()

    files_to_delete = [
        file_path
        for file_path in step_parquet_files
        if file_path.resolve() != merged_resolved
    ]

    deleted_files = 0
    failed_files = []

    for file_path in files_to_delete:
        if not file_path.exists():
            print(f"Step file already absent: {file_path.name}")
            continue

        try:
            file_path.unlink()
            deleted_files += 1
        except OSError as error:
            failed_files.append((file_path, error))

    if failed_files:
        failure_description = "\n".join(
            f"- {file_path}: {error}"
            for file_path, error in failed_files
        )

        raise RuntimeError(
            f"Deleted {deleted_files} step files, but could not delete "
            f"{len(failed_files)} files:\n{failure_description}"
        )

    print(
        f"Successfully validated merged Parquet file:\n"
        f"  {merged_parquet_file}\n"
        f"Merged rows: {merged_rows:,}\n"
        f"Deleted step files: {deleted_files}"
    )

    return deleted_files