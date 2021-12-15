#!/usr/bin/env python
# -*- coding: utf-8 -*-
from enum import Enum
from typing import Tuple


# Colors
class GameColors(Enum):
    WHITE: Tuple = (255, 255, 255)
    RED: Tuple = (150, 0, 0)
    DARK_GREEN: Tuple = (0, 60, 10)
    LIGHT_GREEN: Tuple = (50, 160, 80)
    BLACK: Tuple = (0, 0, 0)
