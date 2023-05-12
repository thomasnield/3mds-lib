import re, os, sys

def load_file(filepath):
    file = open(filepath, mode='r')
    text = file.read()
    file.close()
    return text

def findall(regex, str=None, file=None):
    if str is None:
        str = load_file(file)

    return re.findall(regex, str, re.MULTILINE)

def render_scenes(q="l"):
    regex = r"(?<=^class )[A-Za-z]+(?=\(Scene\))"
    f = sys.argv[0]
    for i,scene_name in enumerate(findall(regex,file=f)):
       print(f"Rendering: {scene_name}")
       os.system(f"manim -q{q} -v WARNING --disable_caching -o {i+1:02d}_{scene_name} {sys.argv[0]} {scene_name}")