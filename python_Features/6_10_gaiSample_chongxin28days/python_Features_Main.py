# -*- coding: utf-8 -*-
"""
Created on Wed Jun 07 19:13:46 2017

@author: NEU001
"""

from python_Features_Sample2 import Sample2_Features_i_days
from python_Features_Sample1 import Sample1_Features_i_days
from python_Features_Online import Online_Features_i_days
 
for i in [100]:
    Sample2_Features_i_days(i)
    Sample1_Features_i_days(i)
    Online_Features_i_days(i)
    