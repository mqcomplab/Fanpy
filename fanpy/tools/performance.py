r"""Functions for management of computational perfomance.

Functions
---------
current_memory()
    Obtain memory in use by the current Fanpy computational task.
"""

import os
import psutil


# Memory Management Tools
def current_memory():
    """
    Obtain memory in use by the current Fanpy computational task.

    Return
    ----------
    current : Float
        Memory in use by Fanpy in Megabytes.
    """

    pid = os.getpid()
    process = psutil.Process(pid)
    current = process.memory_info().rss / 1024**2

    return current
