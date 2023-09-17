import urllib

from manim import *

from threemds.utils import render_scenes
import sympy as sp
import numpy as np
import scipy

TITLE_SCALE=1.4
SUBTITLE_SCALE=1.0
BULLET_BUFF=.75
BULLET_TEX_SCALE=.8

class ProbabilityTitle(Scene):
    def construct(self):
        title = Title("Probability", color=BLUE).scale(TITLE_SCALE)

        self.play(FadeIn(title))
        self.wait()

        ax = Axes(x_range=(-3,3,1), y_range=(-.1))

        v = Vector([2, 1, 0], color=YELLOW) \
            .put_start_and_end_on(np.c2p(0,0), np.c2p(2,1))

        VGroup(np,v).scale(.7).next_to(title, DOWN, buff=1)

        self.play(Write(np))
        self.wait()
        self.play(GrowArrow(v))
        self.wait()

        self.play(Unwrite(np), Unwrite(v), FadeOut(title))
        self.wait()

class WhatIsLinearAlgebra(Scene):
    def construct(self):
        title = Title("What is Linear Algebra?", color=BLUE)
        tex = VGroup(
                Tex(r"\textbf{Linear algebra} ", "models data and operations as vectors and matrices."),
                Tex(r"Data and operations can be thought of in vectors and matrices.")
             ).arrange_in_grid(cols=1, cell_alignment=LEFT, buff=BULLET_BUFF) \
            .scale(.8)  \
            .next_to(title.get_corner(DL), DOWN, aligned_edge=LEFT, buff=1)

        tex[0][0].set_color(RED)


        v = MathTex(r"\vec{v} = \begin{bmatrix} 2 \\ 3 \end{bmatrix}")
        m = MathTex(r"A = \begin{bmatrix} 1 & 0 \\ 2 & -1 \end{bmatrix}")
        v_and_m = VGroup(v, m).arrange(RIGHT,buff=2).next_to(tex, DOWN, buff=2)

        self.play(*[Write(m) for m in (title, tex, v_and_m)], lag_ratio=.5)
        self.wait()



if __name__ == "__main__":
    render_scenes(q='k',play=True, scene_names=["TransformedThreeDVectorScene"])