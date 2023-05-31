from pathlib import Path

from manim import VMobject, VGroup, Scene, UL
from .svg import *
import os

__all__ = ["create_svg_from_vgroup", "create_svg_from_vmobject"]

VMobject.to_svg = create_svg_from_vmobject
VGroup.to_svg = create_svg_from_vgroup

def create_svg_from_scene(scene: Scene, file_name: str | Path = None):
    return VGroup(*scene.mobjects).to_corner(UL).to_svg(file_name)

Scene.to_svg = create_svg_from_scene

if not os.path.exists("media"):
    os.mkdir("media")