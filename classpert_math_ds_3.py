from manim import *
import numpy as np

from threemds.utils import render_scenes, file_to_base_64, mobj_to_svg, mobj_to_png
import sympy as sp




class CodeRender(Scene):
    def construct(self):
        raw_code = """from sympy import *

# Declare 'x' to SymPy
x = symbols('x')

# Now just use Python syntax to declare function
f = (x - 2)**2 + 1

# Calculate the integral of the function with respect to x
# for the area between x = 0 and 2
area = integrate(f, (x, 0, 2))

print(area) # prints 14/3
"""

        code = Code(code=raw_code, language="Python", font="Monospace", style="monokai", background="window")
        self.add(code)
        mobj_to_svg(VGroup(code), "out.svg")

class TexRender(Scene):
    def construct(self):

        x,y,z = sp.symbols('x y z')
        f = (x-2)**2 + 1

        tex = MathTex(r"A^{-1}AX = A^{-1}B")

        mobj_to_svg(tex, 'out.svg')

if __name__ == "__main__":
    render_scenes(q="l", play=True, scene_names=['TexRender'])
