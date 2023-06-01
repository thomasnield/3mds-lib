import re, os, sys

scene_regex = r"(?<=^class )[A-Za-z]+(?=\([A-Za-z]*Scene\))"
def load_file(filepath):
    file = open(filepath, mode='r')
    text = file.read()
    file.close()
    return text

def findall(regex, str=None, file=None):
    if str is None:
        str = load_file(file)

    return re.findall(regex, str, re.MULTILINE)

def render_scenes(q="l", play=False, gif=False, scene_names=None):
    f = sys.argv[0]
    for i,scene_name in enumerate(findall(scene_regex,file=f)):
        print(scene_name)
        if scene_names is None or scene_name in scene_names:
           print(f"Rendering: {scene_name}")
           os.system(f"manim -q{q} -v WARNING --disable_caching {'-p' if play else ''} {'--format=gif' if gif else ''} -o {i+1:02d}_{scene_name} {sys.argv[0]} {scene_name}")

def render_slides():
    regex = r"(?<=^class )[A-Za-z]+(?=\(Slide\))"
    f = sys.argv[0]

    os.system(f"manim-slides {' '.join(findall(regex,file=f))}")
