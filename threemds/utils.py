import base64
import re, os, sys

from manim import config, tempconfig, VGroup
from threemds import create_svg_from_vgroup, create_svg_from_vmobject

scene_regex = r"(?<=^class )[A-Za-z0-9]+(?=\([A-Za-z]*Scene.*\))"

def mobj_to_png(mob, filename: str):
    padding = 0
    m_width = mob.width + padding
    m_height = mob.height + padding
    p_width = int(m_width * config.pixel_width / config.frame_width)
    with tempconfig({
        "frame_width": m_width,
        "frame_height": m_height,
        "pixel_width": p_width,
        "pixel_height": int(p_width * m_height / m_width)
    }):
        img = mob.get_image()
        img.save(filename)
        print(r"![img](data:image/png;base64," + base64.b64encode(open(filename,"rb").read()).decode('ascii') + ")")

def mobj_to_svg(mob, filename: str,  trim=True, padding=0):
    m_width = mob.width + padding
    m_height = mob.height + padding
    p_width = int(m_width * config.pixel_width / config.frame_width)
    with tempconfig({
        "frame_width": m_width,
        "frame_height": m_height,
        "pixel_width": p_width,
        "pixel_height": int(p_width * m_height / m_width)
    }):
        if isinstance(mob, VGroup):
            create_svg_from_vgroup(mob, filename)
        else:
            create_svg_from_vmobject(mob, filename)

        print(r"![img](data:image/svg+xml;base64," + base64.b64encode(open(filename,"rb").read()).decode('ascii') + ")")

def load_file(filepath):
    file = open(filepath, mode='r')
    text = file.read()
    file.close()
    return text

def findall(regex, str=None, file=None):
    if str is None:
        str = load_file(file)

    return re.findall(regex, str, re.MULTILINE)

def render_scenes(q="l",
                  play=False,
                  gif=False,
                  last_scene=False,
                  frames_only=False,
                  fps=None,
                  scene_names=None):
    f = sys.argv[0]
    for i,scene_name in enumerate(findall(scene_regex,file=f)):
        print(scene_name)
        if scene_names is None or scene_name in scene_names:
           print(f"Rendering: {scene_name}")
           os.system(f"manim -q{q} -v WARNING --disable_caching "
                     f"{' --fps ' + str(fps) if fps else ''} "
                     f"{' -p' if play else ''} "
                     f"{' -g' if frames_only else ''} "
                     f"{' -s' if last_scene else ''} "
                     f"{' --format=gif' if gif else ''} "
                     f" -o {i+1:02d}_{scene_name} {sys.argv[0]} {scene_name}")

def render_slides():
    regex = r"(?<=^class )[A-Za-z]+(?=\(Slide\))"
    f = sys.argv[0]

    os.system(f"manim-slides {' '.join(findall(regex,file=f))}")
