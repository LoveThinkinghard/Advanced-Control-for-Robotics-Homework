# -*- coding: utf-8 -*-

"""
This code compute the angle between matrix A and B
"""

import numpy as np

A = np.array([[1, 0, 1], [0, 1, 0]])
B = np.array([[-1, 2, 1], [-1, 0, 1]])

THETA = np.arccos(np.matmul(A.T, B).trace() / (np.linalg.norm(A, 2) * np.linalg.norm(B, 2)))

print(np.rad2deg(THETA))
