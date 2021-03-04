# iRacing LMP1H Hybrid Deployment Utility
This is the beginning of a small utility to calculate when and how much hybrid energy in real time while driving one of iRacing's LMP1H (Audi R18 and Porsche 919) cars.

# Dependencies
This utility's code relies on the pyirsdk which can be found, with installation instructions, here: https://github.com/kutu/pyirsdk

# Usage
To start the utility run the main.py file using python3 such as: ```python main.py```  
The utility will output when a new lap is completed along with the information gathered that lap about the straights on the circuit. Once the driver has run lap that didn't start in the pit lane it will start printing hybrid deploy information to the command line.
