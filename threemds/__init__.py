from manim import VMobject, VGroup, Scene, UL
from .svg import *
import os

__all__ = ["create_svg_from_vgroup", "create_svg_from_vmobject"]
VMobject.to_svg = create_svg_from_vmobject
VGroup.to_svg = create_svg_from_vgroup
Scene.to_svg = lambda scene: VGroup(*scene.mobjects).to_corner(UL).to_svg()

if not os.path.exists("media"):
    os.mkdir("media")