from manim import *
from threemds.utils import render_scenes
import sympy as sp
import numpy as np

TITLE_SCALE=1.4
SUBTITLE_SCALE=1.0
BULLET_BUFF=.75
BULLET_TEX_SCALE=.8

class TitleScene(Scene):
    def construct(self):
        title = Tex("Introduction to Math for Data Science", color=BLUE).scale(TITLE_SCALE)
        subtitle = Tex("ODSC West 2023", color=WHITE).scale(SUBTITLE_SCALE).next_to(title, DOWN)
        speaker = Tex("Thomas Nield",color=YELLOW).scale(SUBTITLE_SCALE).to_edge(DOWN)
        self.play(*(FadeIn(m) for m in (title, subtitle, speaker)))
        self.wait()

class OutlineScene(Scene):
    def construct(self):
        title = Title("Outline", color=BLUE)
        tex = Tex("Linear Algebra", "Calculus", "Probability", "Statistics", "Machine Learning") \
            .arrange(DOWN)

        self.play(Write(title), Write(tex))
        self.wait()

# ================================
# LINEAR ALGEBRA
# ================================
class LinearAlgebraTitle(Scene):
    def construct(self):
        title = Title("Linear Algebra", color=BLUE).scale(TITLE_SCALE)

        self.play(FadeIn(title))
        self.wait()
        self.play(FadeOut(title))
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

        self.add(title, tex, v_and_m)

class SimpleVectorScene(Scene):
    def construct(self):

        np = NumberPlane(x_range=(-.5,4.5,1),
                         y_range=(-1.5,3.5,1)
        )

        v = Vector([2,1,0], color=YELLOW).move_to(np.get_origin(), aligned_edge=DL)
        v_lbl = MathTex(r"\vec{v} = \begin{bmatrix} 2 \\ 1 \end{bmatrix}", color=YELLOW) \
            .next_to(v.get_tip(), RIGHT, aligned_edge=DOWN)

        self.play(Write(np))
        self.wait()
        self.play(GrowArrow(v))
        self.wait()
        self.play(Write(v_lbl))
        self.wait()

        i_brace = Brace(v, direction=LEFT, color=GREEN)
        i_txt = Tex("2", color=GREEN).next_to(i_brace, LEFT).scale(1.2)

        j_brace = Brace(v, direction=DOWN, color=RED)
        j_txt = Tex("1", color=RED).next_to(j_brace, DOWN).scale(1.2)

        self.play(LaggedStart(*(Write(m) for m in (i_brace, i_txt, j_brace, j_txt))))
        self.wait(1)

        self.play(*(FadeOut(m) for m in (i_brace, i_txt, j_brace, j_txt)))

        # NOW SHOW CODE NEXT TO IT
        raw_code = """import numpy as np
v = np.array([2, 1])
print(v) # [2. 1.]
"""

        left_pane = VGroup(np, v_lbl, v)
        code = Code(code=raw_code, language="Python", font="Monospace", style="monokai", background="window")

        VGroup(left_pane.generate_target(), code).arrange(RIGHT, buff=1)
        self.play(MoveToTarget(left_pane))
        self.wait()
        self.play(Write(code))
        self.wait()

class VectorExamplesScene(Scene):
    def construct(self):

        np = NumberPlane(x_range=(-4.5,4.5,1),
                         y_range=(-3.5,3.5,1)
        )

        v1 = Vector([3,2,0], color=BLUE).move_to(np.get_origin(), aligned_edge=DL)
        v2 = Vector([2,-1,0], color=ORANGE).move_to(np.get_origin(), aligned_edge=UL)
        v3 = Vector([-2,-1.5,0], color=GREEN).move_to(np.get_origin(), aligned_edge=UR)
        v4 = Vector([-1,2,0], color=YELLOW).move_to(np.get_origin(), aligned_edge=DR)

        v1_lbl = MathTex(r"\vec{a} = \begin{bmatrix} 3 \\ 2 \end{bmatrix}", color=BLUE) \
            .next_to(v1.get_end(), UL)

        v2_lbl = MathTex(r"\vec{b} = \begin{bmatrix} 2 \\ -1 \end{bmatrix}", color=ORANGE)  \
            .next_to(v2.get_end(), DR)

        v3_lbl = MathTex(r"\vec{c} = \begin{bmatrix} -2 \\ -1.5 \end{bmatrix}", color=GREEN) \
            .next_to(v3.get_end(), DL + UP + LEFT)

        v4_lbl = MathTex(r"\vec{d} = \begin{bmatrix} -1 \\ 2 \end{bmatrix}", color=YELLOW)  \
            .next_to(v4.get_end(), UL + LEFT)

        grp = VGroup(v1, v1_lbl, v2 ,v2_lbl, v3, v3_lbl, v4, v4_lbl)

        self.play(Write(np))
        self.wait()

        self.play(LaggedStart(*(Write(m) for m in grp),lag_ratio=1))

class ThreeDVectorScene(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=75 * DEGREES, theta=-30 * DEGREES)

        ax = ThreeDAxes()
        _v = np.array([1,4,2])
        v = Arrow3D(start=ax.c2p(0,0,0), end=ax.c2p(*_v), color=YELLOW)

        i_walk = Line(start=ax.c2p(0,0,0), end=ax.c2p(_v[0],0,0), color=GREEN)
        j_walk = Line(start=ax.c2p(_v[0],0,0), end=ax.c2p(_v[0],_v[1],0), color=RED)
        k_walk = Line(start=ax.c2p(_v[0],_v[1],0), end=ax.c2p(_v[0],_v[1],_v[2]), color=PURPLE)

        self.begin_ambient_camera_rotation()
        self.play(Write(ax))
        self.wait()
        self.play(GrowFromPoint(v,point=ORIGIN))
        self.wait()
        self.play(LaggedStart(*(GrowFromPoint(m, m.get_start()) for m in (i_walk, j_walk, k_walk)),lag_ratio=1))
        self.wait()

        v_lbl = MathTex(r"\vec{v} = \begin{bmatrix} 1 \\ 4 \\ 2 \end{bmatrix}", color=YELLOW) \
            .scale(.6) \
            .next_to(v.get_end(), RIGHT, buff=2) \

        self.add_fixed_orientation_mobjects(v_lbl)
        self.wait()

        raw_code = """import numpy as np
v = np.array([1,4,2])
print(v) # [1. 4. 2.]
"""
        code = Code(code=raw_code, language="Python", font="Monospace", style="monokai", background="window") \
            .to_edge(DL)

        self.add_fixed_in_frame_mobjects(code)
        self.wait(6)

        self.stop_ambient_camera_rotation()

class ScaleVectorScene(Scene):
    def construct(self):

        np = NumberPlane(x_range=(-.5,4.5,1),
                         y_range=(-1.5,3.5,1)
        )

        v = Vector([2,1,0], color=YELLOW).move_to(np.get_origin(), aligned_edge=DL)
        v_lbl = MathTex(r"\vec{v} = \begin{bmatrix} 2 \\ 1 \end{bmatrix}", color=YELLOW) \
            .next_to(v.get_tip(), RIGHT, aligned_edge=DOWN)

        self.play(Write(np))
        self.wait()
        self.play(GrowArrow(v))
        self.wait()
        self.play(Write(v_lbl))
        self.wait()

        w = Vector([3,1.5,0], color=RED).move_to(np.get_origin(), aligned_edge=DL)
        w_lbl = MathTex(r"\vec{w} = \begin{bmatrix} 3 \\ 1.5 \end{bmatrix}", color=RED) \
            .next_to(w.get_end(), UP, aligned_edge=DOWN)


        self.play(FadeTransform(v, w), FadeTransform(v_lbl, w_lbl))
        self.wait()

        # NOW SHOW CODE NEXT TO IT
        raw_code = """import numpy as np
v = np.array([2, 1])
w = 1.5*v
print(w) # [3. 1.5]
"""

        left_pane = VGroup(np, w_lbl, w)
        code = Code(code=raw_code, language="Python", font="Monospace", style="monokai", background="window")

        VGroup(left_pane.generate_target(), code).arrange(RIGHT, buff=1)
        self.play(MoveToTarget(left_pane))
        self.wait()
        self.play(Write(code))
        self.wait()

class AddVectorScene(Scene):
    def construct(self):
        np = NumberPlane(x_range=(-.5, 4.5, 1),
                         y_range=(-1.5, 3.5, 1)
                         )

        v = Vector([1, 3, 0], color=YELLOW).move_to(np.get_origin(), aligned_edge=DL)
        v_lbl = MathTex(r"\vec{v}", color=YELLOW) \
            .move_to(v.copy().rotate(-90*DEGREES).set_length(.8).get_end())

        w = Vector([2, -1, 0], color=ORANGE).move_to(np.get_origin(), aligned_edge=UL)

        w_lbl = always_redraw(lambda: MathTex(r"\vec{w}", color=ORANGE) \
                              .move_to(w.copy().rotate(90*DEGREES).set_length(.8).get_end()))

        self.play(Write(np))
        self.wait()
        self.play(LaggedStart(GrowArrow(v), Write(v_lbl), lag_ratio=1))
        self.wait()
        self.play(LaggedStart(GrowArrow(w), Write(w_lbl), lag_ratio=1))
        self.wait()
        self.play(w.animate.put_start_and_end_on(v.get_end(), np.c2p(3,2)))
        self.wait()

        v_plus_w = Line(start=np.get_origin(), end=np.c2p(3,2), color=PURPLE, tip_style=ArrowTriangleFilledTip)

        self.play(GrowFromPoint(v_plus_w, np.get_origin()))
        self.wait()

        plus = MathTex("+")
        VGroup(v_lbl.generate_target(), plus, w_lbl.generate_target()) \
            .arrange(RIGHT) \
            .move_to(v_plus_w.copy().rotate(-90 * DEGREES).set_length(1.5).get_end()) \
            .set_color(PURPLE)

        self.play(FadeIn(plus), MoveToTarget(v_lbl), MoveToTarget(w_lbl))

        self.wait()
        self.remove(w_lbl)
        w_lbl = w_lbl.copy()
        self.add(w_lbl)

        # NOW SHOW CODE NEXT TO IT
        raw_code = """import numpy as np
v = np.array([1, 3])
w = np.array([2, -1])
print(v+w) # [3, 2]
"""

        left_pane = VGroup(np, v, w, v_lbl, w_lbl, v_plus_w, plus)
        code = Code(code=raw_code, language="Python", font="Monospace", style="monokai", background="window")

        VGroup(left_pane.generate_target(), code).arrange(RIGHT, buff=1)
        self.play(MoveToTarget(left_pane))
        self.wait()
        self.play(Write(code))
        self.wait()

class MatrixVectorMultiplication(Scene):
    def construct(self):

        class Matrix2x2Tex(MathTex):
            def __init__(self, a, b, c, d, colors=None, **kwargs):
                self.matrix_tex = sp.latex(sp.Matrix([[a,b],[c,d]]))
                super().__init__(self.matrix_tex, **kwargs)

                self.left_bracket = self[0][0]

                global i
                i=1
                def incr(j):
                    global i
                    if type(j) == str:
                        i += len(j)
                    else:
                        i += j
                    return i

                self.a = self[0][i:incr(a)]
                self.b = self[0][i:incr(b)]
                self.c = self[0][i:incr(c)]
                self.d = self[0][i:incr(d)]
                self.right_bracket = self[0][-1]

                if colors:
                    for t,c in zip((self.a,self.b,self.c,self.d), colors): t.set_color(c)


        class Vector2x1Tex(MathTex):
            def __init__(self, x, y, colors=None, **kwargs):
                self.matrix_tex = sp.latex(sp.Matrix([[x],[y]]))
                super().__init__(self.matrix_tex, **kwargs)

                self.left_bracket = self[0][0]

                global i
                i=1
                def incr(j):
                    global i
                    if type(j) == str:
                        i += len(j)
                    else:
                        i += j
                    return i

                self.x = self[0][i:incr(x)]
                self.y = self[0][i:incr(y)]
                self.right_bracket = self[0][-1]

                if colors:
                    for t,c in zip((self.x, self.y), colors): t.set_color(c)

        class MatrixVector2x1Result(MathTex):
            def __init__(self, a,b,c,d,x,y, colors = (WHITE,)*6, **kwargs):

                self.matrix_tex = sp.latex(sp.Matrix([[a,b],[c,d]]) @ sp.Matrix([[x],[y]]))
                super().__init__(self.matrix_tex,**kwargs)

                self.left_bracket = self[0][0]
                global i
                i=1
                def incr(j):
                    global i
                    if type(j) == str:
                        i += len(j)
                    else:
                        i += j
                    return i

                self.a = self[0][1: incr(a)].set_color(colors[0])
                self.x1 = self[0][i: incr(x)].set_color(colors[4])
                self.plus_top = self[0][i:incr(1)]
                self.b = self[0][i: incr(b)].set_color(colors[1])
                self.y1 = self[0][i: incr(y)].set_color(colors[5])
                self.c = self[0][i:incr(c)].set_color(colors[2])
                self.x2 = self[0][i:incr(x)].set_color(colors[4])
                self.plus_bottom = self[0][i:incr(1)]
                self.d = self[0][i:incr(d)].set_color(colors[3])
                self.y2 = self[0][i:incr(y)].set_color(colors[5])
                self.right_bracket = self[0][-1]


        A = Matrix2x2Tex("a","b","c","d", (BLUE,BLUE,RED,RED))
        X = Vector2x1Tex("x", "y", (YELLOW,)*2)
        B = MatrixVector2x1Result("a","b","c","d","x","y", (BLUE,BLUE,RED,RED,YELLOW,YELLOW))
        eq = Tex("=")

        VGroup(A,X,eq, B).arrange(RIGHT).scale(2)

        self.add(A,X, eq, B.left_bracket, B.right_bracket)

        self.play(
            ReplacementTransform(A.a.copy(), B.a),
            ReplacementTransform(A.b.copy(), B.b)
        )
        self.wait()
        self.play(
            ReplacementTransform(X.x.copy(), B.x1),
            ReplacementTransform(X.y.copy(), B.y1),
        )
        self.play(FadeIn(B.plus_top))
        self.wait()
        self.play(
            ReplacementTransform(A.c.copy(), B.c),
            ReplacementTransform(A.d.copy(), B.d)
        )
        self.wait()
        self.play(
            ReplacementTransform(X.x.copy(), B.x2),
            ReplacementTransform(X.y.copy(), B.y2)
        )
        self.play(FadeIn(B.plus_bottom))
        self.wait()


if __name__ == "__main__":
    render_scenes(q='l', play=True, scene_names=['MatrixVectorMultiplication'])