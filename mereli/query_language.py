import numpy as np

"""

: -> route 
@ -> final asset or attribute
(1,2,3) -> indexing
t=200 -> time indexing (if time is involved)
agg:
    - sum[]
    - diff[]
    - mean[]

Examples:
robotA:controller:neural_network@weights{i=(1:10; 1:10)}
robotA:sensors:distance_sensor@reading{i=(0,7),t=[1:100]}
"""


