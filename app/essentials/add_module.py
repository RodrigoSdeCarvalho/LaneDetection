def set_working_directory():
    import os
    import sys
    import re

    script_dir = os.path.abspath(__file__)
    script_dir = re.sub(pattern="lane_detection.*", repl="lane_detection/", string=script_dir)
    script_dir = os.path.abspath(script_dir)
    os.chdir(script_dir)
    sys.path.append(os.path.join(script_dir))
