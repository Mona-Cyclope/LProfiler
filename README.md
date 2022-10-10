# Requirements


## install the profiler (IN DEV ENV)

The line profiler must be in the same environment where you are running the code you want to profile.

```
conda install line_profiler=3.3.1
```

## install extra for visualization 

to run the comand line and export the visuals and the xlsx install the following.
These packages do not need to be installed in your dev environment.

```
conda install pandas networkx seaborn xlsxwriter pydot
```

# Profiling

## add guard for non profiling environment

```
try:
    profile  # throws an exception when profile isn't defined
except NameError:
    profile = lambda x: x   # if it's not defined simply ignore the decorator.
```

## add the profile decorator
```
@profile
def routine(args)
```

## add tag comment
to add a tag to the function: just add this in the body
```
##### PROFILERTAG: YOUR TAG
```

## launch the application 

At the end of the execution (must stop the application) the profile data will be stored in the `run_prof.dat`
this file must be interpreted by the line_profiler to get readeable output to the `run_prof.prof`
```
kernprof -l -z -o run_prof.dat ( entrypoint file e.g. main.py );
python -m line_profiler run_prof.dat > run_prof.prof
```

# Visualization of the results

You can either look directly at the .prof file or run it thourgh this visualization code.

## Example of use cmdline

You can use it as command line to output 2 images and the excel table.
```
python lprofiler.py --profile_file_path=<.prof file>
```

### Arguments
* --profile_file_path'
* --graph_image_h', action="store", default=20, type=int
* --graph_image_w', action="store", default=100, type=int
* --graph_font_size', action="store", default=30, type=int
* --graph_legendHandlesSize', action="store", default=500, type=int
* --graph_max_node_size', action="store", default=30000, type=int
* --line_image_w', action="store", default=10, type=int
* --line_image_h_ligne', action="store", default=2, type=int
* --line_min_perc_time', action="store", default=0.1, type=float

## Example of use on a notebook

```
from Profiler.lprofiler import LineProfile
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pprint as pp

sns.set_theme(context='notebook', style='darkgrid', palette='deep', font='sans-serif', font_scale=1, color_codes=True, rc=None)
profile_file_path = "run_prof_entries_fmatch.prof"
lprofile = LineProfile(profile_file_path)
tag2node = lprofile.draw_graph_calls(figsize=(60,20), min_node_size=0, max_node_size=30000, legendHandlesSize=500, font_size=30)
```

![image](https://user-images.githubusercontent.com/110400029/193823831-2de83a6d-c45f-497f-954e-13247656f477.png)
