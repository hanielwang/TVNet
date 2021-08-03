# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os
from lib.TEM_train_L15 import main as main_L15
from lib.TEM_train_L5 import main as main_L5


if __name__ == "__main__":
	main_L15()
	main_L5()


        