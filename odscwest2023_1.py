import urllib

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
        self.play(*(Write(m) for m in (title, subtitle, speaker)),lag_ratio=.3)
        self.wait()
        self.play(*(Unwrite(m) for m in (title, subtitle, speaker)))
        self.wait()

class BioScene(Scene):
    def construct(self):
        title = Title("About the Speaker", color=BLUE)

        tex = VGroup(
                Tex(r"Thomas Nield").set_color(YELLOW),
                Tex(r"Founder at Yawman Flight").scale(.8),
                Tex(r"Nield Consulting Group").scale(.8),
                Tex(r"AI System Safety at USC").scale(.8)
        ).arrange_in_grid(cols=1, cell_alignment=LEFT, buff=BULLET_BUFF) \
        .next_to(title.get_corner(DL), DOWN, aligned_edge=LEFT, buff=1)

        urllib.request.urlretrieve(r"https://images-na.ssl-images-amazon.com/images/I/51yHtuQ9wAL._SX379_BO1,204,203,200_.jpg", "image3.jpg")
        urllib.request.urlretrieve(r"https://images-na.ssl-images-amazon.com/images/I/41khDop3M4L._SX379_BO1,204,203,200_.jpg", "image4.jpg")

        image3, image4 =  ImageMobject(r"image3.jpg"), ImageMobject(r"image4.jpg")

        books = Group(
            image3,image4
        ).scale(1) \
        .arrange(LEFT, buff=.5) \
        .next_to(title, DOWN, buff=1, aligned_edge=RIGHT)

        self.play(Write(title), Write(tex))
        self.play(FadeIn(books))

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

        np = NumberPlane(x_range=(-.5, 4.5, 1),
                         y_range=(-1.5, 3.5, 1)
                         )

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
        i_txt = Tex("1", color=GREEN).next_to(i_brace, LEFT).scale(1.2)

        j_brace = Brace(v, direction=DOWN, color=RED)
        j_txt = Tex("2", color=RED).next_to(j_brace, DOWN).scale(1.2)

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
        self.wait(6)
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
        self.wait(18)

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

        w_lbl = MathTex(r"\vec{w}", color=ORANGE) \
                              .move_to(w.copy().rotate(90*DEGREES).set_length(.8).get_end())

        self.play(Write(np))
        self.wait()
        self.play(LaggedStart(GrowArrow(v), Write(v_lbl), lag_ratio=1))
        self.wait()
        self.play(LaggedStart(GrowArrow(w), Write(w_lbl), lag_ratio=1))
        self.wait()

        v_plus_w = Line(start=np.get_origin(), end=np.c2p(3,2), color=PURPLE, tip_style=ArrowTriangleFilledTip)

        self.play(w.animate.put_start_and_end_on(v.get_end(), np.c2p(3,2)),
                  w_lbl.animate.move_to(w.copy().put_start_and_end_on(v.get_end(), np.c2p(3,2)) \
                                        .rotate(90*DEGREES).set_length(.8).get_end()
                                        )
                  )

        self.wait()

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

        self.wait()
        self.add(A,X, eq, B.left_bracket, B.right_bracket)
        self.wait()
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

        # Write braces

        A_brace = Brace(A, UP)
        A_brace_txt = A_brace.get_text("A")

        X_brace = Brace(X, UP)
        X_brace_txt = X_brace.get_text("X")

        B_brace = Brace(B, UP)
        B_brace_txt = B_brace.get_text("B")

        self.play(
            LaggedStart(Write(A_brace), Write(A_brace_txt), lag_ratio=.1)
        )
        self.wait()
        self.play(
            LaggedStart(Write(X_brace), Write(X_brace_txt), lag_ratio=.1)
        )
        self.wait()
        self.play(
            LaggedStart(Write(B_brace), Write(B_brace_txt), lag_ratio=.1)
        )
        self.wait()
        self.play(
            *[Unwrite(m) for m in (A_brace, X_brace, A_brace_txt,
                                   X_brace_txt, B_brace, B_brace_txt)]
        )
        self.wait()

class MatrixVectorMultiplication2(Scene):
    def construct(self):

        _A = np.array([[2,1],
                      [-1,3]])

        _X = np.array([2,1])

        _B = _A @ _X

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

                self.matrix_tex = r"\begin{bmatrix}" + \
                                  f"({a})({x}) + ({b})({y})" + \
                                  r"\\" + f"({c})({x}) + ({d})({y})" + \
                                  r"\end{bmatrix}"
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

                self.a = self[0][1: incr(len(a)+2)].set_color(colors[0])
                self.x1 = self[0][i: incr(len(x)+2)].set_color(colors[4])
                self.plus_top = self[0][i:incr(1)]
                self.b = self[0][i: incr(len(b)+2)].set_color(colors[1])
                self.y1 = self[0][i: incr(len(y)+2)].set_color(colors[5])
                self.c = self[0][i:incr(len(c)+2)].set_color(colors[2])
                self.x2 = self[0][i:incr(len(x)+2)].set_color(colors[4])
                self.plus_bottom = self[0][i:incr(1)]
                self.d = self[0][i:incr(len(d)+2)].set_color(colors[3])
                self.y2 = self[0][i:incr(len(y)+2)].set_color(colors[5])
                self.right_bracket = self[0][-1]


        A = Matrix2x2Tex(*[str(c) for c in _A.flatten()], (BLUE,BLUE,RED,RED))
        X = Vector2x1Tex(*[str(c) for c in _X.flatten()], (YELLOW,)*2)
        B = MatrixVector2x1Result(*[str(c) for c in _A.flatten()], *[str(c) for c in _X.flatten()],
                                  colors=(BLUE,BLUE,RED,RED,YELLOW,YELLOW))
        eq = Tex("=")

        VGroup(A,X,eq, B).arrange(RIGHT).scale(1.5)
        B_final = MathTex(sp.latex(sp.Matrix(_B))).scale(1.5).next_to(eq,RIGHT)

        self.add(A,X, eq, B.left_bracket, B.right_bracket)

        self.wait()

        A_brace = Brace(A, UP)
        A_brace_txt = A_brace.get_text("A")

        X_brace = Brace(X, UP)
        X_brace_txt = X_brace.get_text("X")

        B_brace = Brace(B, UP)
        B_brace_txt = B_brace.get_text("B")

        self.play(
            LaggedStart(Write(A_brace), Write(A_brace_txt), lag_ratio=.1)
        )
        self.wait()
        self.play(
            LaggedStart(Write(X_brace), Write(X_brace_txt), lag_ratio=.1)
        )
        self.wait()
        self.play(
            LaggedStart(Write(B_brace), Write(B_brace_txt), lag_ratio=.1)
        )
        self.wait()
        self.play(
            *[Unwrite(m) for m in (A_brace, X_brace, A_brace_txt,
                                   X_brace_txt, B_brace, B_brace_txt)]
        )
        self.wait()

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
        self.play(
            TransformMatchingShapes(B, B_final)
        )
        self.wait()

        raw_code = """import numpy as np

A = np.array([[2, 1],
                  [-1, 3]])

X = np.array([2, 1])
B = A @ X

print(B) # [5 1]"""

        code = Code(code=raw_code, language="Python", font="Monospace", style="monokai", background="window") \
            .to_edge(DR)

        VGroup(VGroup(*[m.generate_target() for m in (A,X,eq,B_final)]),
               code
               ).arrange(RIGHT,buff=1)\
                .move_to(ORIGIN)

        self.play(*[MoveToTarget(m) for m in (A,X,eq,B_final)])
        self.play(Write(code))
        self.wait()

class TransformationScene(LinearTransformationScene):

    def __init__(self):
        LinearTransformationScene.__init__(
            self,
            show_coordinates=True,
            leave_ghost_vectors=False,
            show_basis_vectors=True
        )

    def construct(self):

        # begin transformation
        _A = np.array([[2,1],
                      [-1,3]])

        _X = np.array([2,1])

        _B = _A @ _X
        """
        self.add_foreground_mobjects(
            VGroup(
            MathTex(sp.latex(sp.Matrix(_A))),
            MathTex(sp.latex(sp.Matrix(_X))),
            MathTex("="),
            MathTex(sp.latex(sp.Matrix(_B)))
            ).arrange(RIGHT).scale(1.5).to_edge(DL)
        )
        """
        self.add_vector(_X, animate=False, color=YELLOW)
        self.apply_matrix(_A)
        self.wait()

        A = MathTex("A = ", sp.latex(sp.Matrix(_A))).to_edge(DL).scale(1.3)

        i_hat = lambda mtx: VGroup(*[m for i,m in enumerate(mtx[1]) if i in (1,3,4)])
        j_hat = lambda mtx: VGroup(*[m for i,m in enumerate(mtx[1]) if i in (2,5)])
        i_hat(A).set_color(GREEN)
        j_hat(A).set_color(RED)

        self.play(
            Write(A)
        )
        self.wait()



class MatrixVectorTransformationScene(LinearTransformationScene):

    def __init__(self):
        LinearTransformationScene.__init__(
            self,
            show_coordinates=True,
            leave_ghost_vectors=False,
            show_basis_vectors=True
        )

    def construct(self):

        # begin transformation
        _A = np.array([[2,1],
                      [-1,3]])

        _X = np.array([2,1])

        _B = _A @ _X

        A = MathTex(sp.latex(sp.Matrix(_A)))
        i_hat = lambda mtx: VGroup(*[m for i,m in enumerate(mtx[0]) if i in (1,3,4)])
        j_hat = lambda mtx: VGroup(*[m for i,m in enumerate(mtx[0]) if i in (2,5)])
        i_hat(A).set_color(GREEN)
        j_hat(A).set_color(RED)

        X = MathTex(sp.latex(sp.Matrix(_X))).set_color(YELLOW)
        B = MathTex(sp.latex(sp.Matrix(_B))).set_color(YELLOW)

        self.add(
            VGroup(
                A,
                X,
                MathTex("="),
                B
            ).arrange(RIGHT).scale(1.5).to_edge(DL)
        )

        self.add_vector(_X, animate=False, color=YELLOW)
        self.apply_matrix(_A)

        self.wait()


class FourTypesTransformation(Scene):

    def construct(self):
        title = Title("Types of Transformations", color=BLUE)
        tex = VGroup(
            Tex("Rotate"),
            Tex("Scale"),
            Tex("Shear"),
            Tex("Inversion")
        ).arrange(DOWN)

        self.play(*[Write(m) for m in (title, tex)], lag_ratio=.5)
        self.wait()

class FourTransformations(LinearTransformationScene):

    def __init__(self):
        LinearTransformationScene.__init__(
            self,
            show_coordinates=True,
            leave_ghost_vectors=False,
            show_basis_vectors=True
        )

    def construct(self):

        # setup
        self.wait()
        self.add_vector([1, 1], color=YELLOW)

        A = np.array([[0,1],[-1,0]])
        tex = Tex("", color=YELLOW).to_edge(DR)
        matrix_tex = Tex("", color=WHITE).to_edge(DR)

        #code = Code("")
        self.add_foreground_mobjects(tex)

        # rotate
        self.wait()
        self.play(tex.animate.become(Tex("Rotate", color=YELLOW).scale(1.3).to_edge(DR)))
        self.play(matrix_tex.animate.become(MathTex(sp.latex(sp.Matrix(A))).scale(1.3).to_edge(DL)))

        self.moving_mobjects = []
        self.apply_matrix(A)

        # scale
        B = np.array([[1,0],[0,2]])
        self.wait()
        self.play(tex.animate.become(Tex("Scale", color=YELLOW).scale(1.3).to_edge(DR)))
        self.play(matrix_tex.animate.become(MathTex(sp.latex(sp.Matrix(B))).scale(1.3).to_edge(DL)))
        self.moving_mobjects = []
        self.apply_matrix(B)

        # shear
        C = np.array([[1,1],[0,1]])
        self.wait()
        self.play(tex.animate.become(Tex("Shear", color=YELLOW).scale(1.3).to_edge(DR)))
        self.play(matrix_tex.animate.become(MathTex(sp.latex(sp.Matrix(C))).scale(1.3).to_edge(DL)))
        self.moving_mobjects = []
        self.apply_matrix(C)

        # inversion
        D = np.array([[0,1],[1,0]])
        self.wait()
        self.play(tex.animate.become(Tex("Inversion", color=YELLOW).scale(1.3).to_edge(DR)))
        self.play(matrix_tex.animate.become(MathTex(sp.latex(sp.Matrix(D))).scale(1.3).to_edge(DL)))
        self.moving_mobjects = []
        self.apply_matrix(D)

class IHatJHat(Scene):
    def construct(self):
        I = MathTex("I = ", sp.latex(sp.Matrix([[1,0], [0,1]]))).scale(1.2)
        A = MathTex("A = ", sp.latex(sp.Matrix([[2,1],[0,3]]))).scale(1.2)

        i_hat = lambda mtx: VGroup(*[m for i,m in enumerate(mtx[1]) if i in (1,3)])

        j_hat = lambda mtx: VGroup(*[m for i,m in enumerate(mtx[1]) if i in (2,4)])

        for mtx in (I,A): i_hat(mtx).set_color(GREEN)
        for mtx in (I,A): j_hat(mtx).set_color(RED)

        self.play(Write(I))

        i_hat_lbl = MathTex(r"\hat{i}", color=GREEN).next_to(i_hat(I), DOWN)
        j_hat_lbl = MathTex(r"\hat{j}", color=RED).next_to(j_hat(I), DOWN)

        self.play(*[Write(m) for m in (i_hat_lbl, j_hat_lbl)])
        self.wait()
        self.play(ReplacementTransform(I,A),
                  i_hat_lbl.animate.next_to(i_hat(A), DOWN),
                  j_hat_lbl.animate.next_to(j_hat(A), DOWN)
                  )
        self.wait()


        raw_code = """import numpy as np
A = np.array([[2, 1],
                  [-1, 3]])

print(A)"""

        code = Code(code=raw_code, language="Python", font="Monospace", style="monokai", background="window") \
            .to_edge(DR)

        VGroup(VGroup(*[m.generate_target() for m in (A,  i_hat_lbl, j_hat_lbl)]),
               code
               ).arrange(RIGHT, buff=1) \
            .move_to(ORIGIN)

        self.play(*[MoveToTarget(m) for m in (A, i_hat_lbl, j_hat_lbl)])
        self.play(Write(code))
        self.wait()

class TransformationScene2(LinearTransformationScene):

    def __init__(self):
        LinearTransformationScene.__init__(
            self,
            show_coordinates=True,
            leave_ghost_vectors=False,
            show_basis_vectors=True
        )

    def construct(self):

        # begin transformation
        _A = np.array([[2,1],
                      [1,3]])
        _X = np.array([1,1])
        _B = _A @ _X

        matrix_tex = MathTex("A = ", sp.latex(sp.Matrix(_A)))
        i_hat_col = VGroup(*[m for i,m in enumerate(matrix_tex[1]) if i in (1,3)]).set_color(GREEN)
        j_hat_col = VGroup(*[m for i,m in enumerate(matrix_tex[1]) if i in (2,4)]).set_color(RED)

        i_hat_lbl = MathTex(r"\hat{i}", color=GREEN).next_to(i_hat_col, DOWN)
        j_hat_lbl = MathTex(r"\hat{j}", color=RED).next_to(j_hat_col, DOWN)

        self.add_foreground_mobjects(
            VGroup(matrix_tex, i_hat_lbl, j_hat_lbl).to_edge(DL)
        )


        self.add_vector(_X, animate=False, color=YELLOW)
        self.apply_matrix(_A)
        self.wait()

class FourSpecialMatrices(Scene):

    def construct(self):
        title = Title("Special Matrices", color=BLUE)
        self.play(Write(title))

        matrices = VGroup(
            MathTex(sp.latex(sp.Matrix([[1,0,0],[0,1,0],[0,0,1]]))),
            MathTex(sp.latex(sp.Matrix([[4,0,0],[0,2,0],[0,0,5]]))),
            MathTex(sp.latex(sp.Matrix([[4,2,9],[0,1,6],[0,0,5]]))),
            MathTex(sp.latex(sp.Matrix([[0,0,0],[0,0,1],[0,0,0]])))
        ).arrange(RIGHT, buff=1)

        names = (
            Tex("Identity"),
            Tex("Diagonal"),
            Tex("Triangular"),
            Tex("Sparse")
        )
        highlights = [
            lambda m: VGroup(*[n for i,n in enumerate(m) if i in (1,5,9)]),
            lambda m: VGroup(*[n for i,n in enumerate(m) if i in (1, 5, 9)]),
            lambda m: VGroup(*[n for i,n in enumerate(m) if i in (1,2,3,5,6,9)]),
            lambda m: VGroup(*[n for i,n in enumerate(m) if i in (6,)])
        ]
        self.wait()

        for m,n,h in zip(matrices, names, highlights):
            n.next_to(m, DOWN)
            self.play(Write(m), Write(n), lag_ratio=.5)
            self.wait()
            self.play(*[Indicate(m) for m in h(m[0])])
            self.wait()


class TransformedThreeDVectorScene(ThreeDScene):
    def construct(self):

        ax = ThreeDAxes()
        X = np.array([1,4,3])
        A = np.array([
            [1.5,-1,2.5],
            [0,-2,1],
            [-2,1,-0.5]
        ])
        self.set_camera_orientation(phi=75 * DEGREES, theta=-30 * DEGREES)

        self.add_fixed_orientation_mobjects(
            MathTex(sp.latex(sp.Matrix(A)), sp.latex(sp.Matrix(X)), "=" , sp.latex(sp.Matrix(A@X))) \
                .to_edge(UL)
        )

        vector = Arrow3D(start=ax.c2p(0,0,0), end=ax.c2p(*(X)), color=YELLOW)

        i_hat_start = Line(start=ax.c2p(0,0,0), end=ax.c2p(1,0,0), color=GREEN)
        j_hat_start = Line(start=ax.c2p(0,0,0), end=ax.c2p(0,1,0), color=RED)
        k_hat_start = Line(start=ax.c2p(0,0,0), end=ax.c2p(0,0,1), color=PURPLE)

        i_hat_end = Line(start=ax.c2p(0,0,0), end=ax.c2p(*A[:,0]), color=GREEN)
        j_hat_end = Line(start=ax.c2p(0,0,0), end=ax.c2p(*A[:,1]), color=RED)
        k_hat_end = Line(start=ax.c2p(0,0,0), end=ax.c2p(*A[:,2]), color=PURPLE)

        self.add(ax)
        self.add(vector)
        self.add(i_hat_start, j_hat_start, k_hat_start)
        self.wait()
        self.play(
            vector.animate.become(Arrow3D(start=ax.c2p(0,0,0), end=ax.c2p(*(A @ X)), color=YELLOW)),
            i_hat_start.animate.become(i_hat_end),
            j_hat_start.animate.become(j_hat_end),
            k_hat_start.animate.become(k_hat_end),
            run_time=3
        )
        self.wait()


class ZeroDeterminantScene(LinearTransformationScene):

    def __init__(self):
        LinearTransformationScene.__init__(
            self,
            show_coordinates=True,
            leave_ghost_vectors=False,
            show_basis_vectors=True
        )

    def construct(self):
        sq = Square(side_length=1, fill_opacity=.4, fill_color=YELLOW, stroke_opacity=0) \
            .move_to(self.plane.c2p(0,0), aligned_edge=DL)

        self.add_transformable_mobject(sq)
        self.wait()
        self.apply_matrix(np.array([[-1,1],[2,-2]]))

        raw_code = """import numpy as np
from numpy.linalg import det
A = np.array([[-1, 1],
                  [2, -2]])

print(det(A)) # 0,0 """

        code = Code(code=raw_code, language="Python", font="Monospace", style="monokai", background="window") \
            .to_edge(DL)

        self.play(
            Write(code)
        )
        self.wait()


class ThreeDZeroDeterminantScene(ThreeDScene):
    def construct(self):
        ax = ThreeDAxes()

        i_hat = np.array([1,0,0])
        j_hat = np.array([0,1,0])
        k_hat = np.array([0,0,1])

        i_hat_start = Line(start=ax.c2p(0, 0, 0), end=ax.c2p(*i_hat), color=GREEN)
        j_hat_start = Line(start=ax.c2p(0, 0, 0), end=ax.c2p(*j_hat), color=RED)
        k_hat_start = Line(start=ax.c2p(0, 0, 0), end=ax.c2p(*k_hat), color=PURPLE)
        basis_start = VGroup(i_hat_start, j_hat_start, k_hat_start)

        A = np.array([[1, 0, 0], [-1, 2, 0], [0, 0, 0]])

        i_hat_end = i_hat_start.copy().apply_matrix(A)
        j_hat_end = j_hat_start.copy().apply_matrix(A)
        k_hat_end = k_hat_start.copy().apply_matrix(A)

        cube = Cube(side_length=1, fill_color=YELLOW) \
            .move_to(basis_start)

        self.set_camera_orientation(phi=75 * DEGREES, theta=-30 * DEGREES)
        self.add(ax, cube, basis_start)

        self.wait()
        self.play(
            cube.animate.become(cube.copy().apply_matrix(A)),
            i_hat_start.animate.become(i_hat_end),
            j_hat_start.animate.become(j_hat_end),
            k_hat_start.animate.become(k_hat_end),
            run_time=3
        )
        self.wait()

if __name__ == "__main__":
    render_scenes(q='k',play=True, scene_names=["TransformedThreeDVectorScene"])