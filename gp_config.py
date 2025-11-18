# gp_config.py
import operator
import math
import random
import functools

import numpy as np

# ---- FEATURE DEFINITIONS ----
FEATURES = [
    "delta_net_rating_3yr",
    "delta_TS",
    "delta_last10_ORtg",
    "delta_last10_DRtg",
    "delta_last10_TOV",
]
