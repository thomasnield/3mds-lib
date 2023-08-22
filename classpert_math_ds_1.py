from manim import *
import numpy as np
from threemds.utils import render_scenes, file_to_base_64, mobj_to_svg, mobj_to_png
import pandas as pd
import sympy as sp

#config.frame_rate = 60

CLASSPERT_ORANGE="#DF6437" #dark orange
CLASSPERT_NAVY="#1E1E27" #darky navy
text_color="#FFFFFF" #white

config.background_color = CLASSPERT_NAVY

#config.frame_width = 7
#config.frame_height = 7
#config.frame_width = 12
#config.pixel_width = 420
#config.pixel_height = 420

config.background_color=CLASSPERT_NAVY

def sanitize_code(raw_code):
    import re
    cleaned= re.sub(r"^\s+", "", raw_code)
    cleaned= cleaned.strip()

    text_file = open("out.py", "w")
    text_file.write(cleaned)
    text_file.close()
    return raw_code

class SimpleVectorScene(Scene):
    def construct(self):

        np = NumberPlane(x_range=(-.5,4.5,1),
                         y_range=(-1.5,3.5,1)
        ).add_background_rectangle(color=CLASSPERT_NAVY)

        v = Vector([3,2,0], color=CLASSPERT_ORANGE).move_to(np.get_origin(), aligned_edge=DL)
        grp = VGroup(np,v)
        self.add(grp)
        #mobj_to_svg(grp, filename="out.png")

class SystemOfEquationsScene(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=75 * DEGREES, theta=-30 * DEGREES)

        ax = ThreeDAxes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            z_range=[-3, 3, 1]
        )

        matrix = np.array([
            [.25, 0, -1],
            [-2, -.5, 1],
            [-.01, 2, .2]
        ])
        v_end = np.array([-1,-1, 2])
        v_start = np.linalg.inv(matrix) @ v_end

        vector_start = Arrow3D(start=ax.c2p(0, 0, 0), end=ax.c2p(*(v_start)), color=YELLOW)
        vector_end = Arrow3D(start=ax.c2p(0, 0, 0), end=ax.c2p(*(v_end)), color=YELLOW)

        i_hat_start = Line(start=ax.c2p(0, 0, 0), end=ax.c2p(*matrix[:, 0]), color=GREEN)
        j_hat_start = Line(start=ax.c2p(0, 0, 0), end=ax.c2p(*matrix[:, 1]), color=RED)
        k_hat_start = Line(start=ax.c2p(0, 0, 0), end=ax.c2p(*matrix[:, 2]), color=PURPLE)

        i_hat_end = Line(start=ax.c2p(0, 0, 0), stroke_width=5, end=ax.c2p(1, 0, 0), color=GREEN)
        j_hat_end = Line(start=ax.c2p(0, 0, 0), stroke_width=5, end=ax.c2p(0, 1, 0), color=RED)
        k_hat_end = Line(start=ax.c2p(0, 0, 0), stroke_width=5, end=ax.c2p(0, 0, 1), color=PURPLE)

        self.add(ax, vector_start)
        self.add(i_hat_start, j_hat_start, k_hat_start)
        self.wait()

        self.play(
            vector_start.animate.become(vector_end),
            i_hat_start.animate.become(i_hat_end),
            j_hat_start.animate.become(j_hat_end),
            k_hat_start.animate.become(k_hat_end),
            run_time=3
        )
        self.wait(1)

class MNISTSample(Scene):
    def construct(self):
        # Load the MNIST dataset
        url = r"https://github.com/thomasnield/machine-learning-demo-data/raw/master/classification/mnist_784.zip"
        df = pd.read_csv(url, compression='zip', delimiter=",", nrows=1)

        # Extract input variables (all rows, all columns but last column)
        X = df.values[:, :-1]

        # Extract output column (all rows, last column)
        Y = df.values[:, -1]

        pixels = np.array(X, dtype='uint8')

        # Print the pixels
        print(pixels)

        # Reshape the array into 28 x 28 array (2-dimensional array)
        pixels = pixels.reshape((28, 28))

        # Plot number 5
        matrix_latex = MathTex(sp.latex(sp.Matrix(pixels)).replace("pmatrix","bmatrix"), color=WHITE) \
            .scale_to_fit_height(8) \
            .scale_to_fit_width(14)

        matrix_latex.add_background_rectangle(color=CLASSPERT_NAVY)

        self.add(matrix_latex)

        #mobj_to_svg(matrix_latex, "out.svg")

class VectorComponentScene(VectorScene):

    def construct(self):

        number_plane = NumberPlane(x_range=(-.5,4.5,1),
                         y_range=(-1.5,4.5,1)
        ).add_background_rectangle(color=CLASSPERT_NAVY)

        v = Vector([2,3,0], color=YELLOW) \
            .move_to(number_plane.get_origin(), aligned_edge=DL)

        i_brace = Brace(v,direction=DOWN,color=GREEN)
        i_txt = Tex("2", color=GREEN).next_to(i_brace, DOWN)

        j_brace = Brace(v, direction=RIGHT, color=RED)
        j_txt = Tex("3", color=RED).next_to(j_brace, RIGHT)

        grp = VGroup(number_plane,v, i_brace, i_txt, j_brace,j_txt)
        mobj_to_svg(grp, "out.svg")
        self.add(grp)

class VectorNotation(Scene):
    def construct(self):
        tex = MathTex(r"\vec{v} = \begin{bmatrix} x \\ y \\ z \end{bmatrix} = \begin{bmatrix} 2 \\ 1 \\ 1.5 \end{bmatrix}")
        tex[0][5].set_color(GREEN)
        tex[0][13].set_color(GREEN)
        tex[0][6].set_color(RED)
        tex[0][14].set_color(RED)
        tex[0][7].set_color(PURPLE)
        tex[0][15:18].set_color(PURPLE)
        self.add(tex)

        mobj_to_svg(tex, "out.svg")

class CodeRender(Scene):

    def construct(self):

        raw_code="""import numpy as np

A = np.array([[2, 1],
              [-1, -2]])

A_inv = np.linalg.inv(A)
B = np.array([5,-4])

print(A_inv @ B) # [2. 1.]
"""

        code=Code(code=raw_code, language="Python", font="Monospace", style="monokai", background="window")
        self.add(code)
        mobj_to_svg(VGroup(code), "out.svg")


class ThreeDVectorScene(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=75 * DEGREES, theta=-30 * DEGREES)

        ax = ThreeDAxes(x_range=[-3,3,1],y_range=[-3,3,1],z_range=[-3,3,1])
        v = np.array([2,1,1.5])
        vector = Arrow3D(start=ax.c2p(0,0,0), end=ax.c2p(*v), color=YELLOW)

        i_walk = Line(start=ax.c2p(0,0,0), end=ax.c2p(v[0],0,0), color=GREEN)
        j_walk = Line(start=ax.c2p(v[0],0,0), end=ax.c2p(v[0],v[1],0), color=RED)
        k_walk = Line(start=ax.c2p(v[0],v[1],0), end=ax.c2p(v[0],v[1],v[2]), color=PURPLE)

        self.add(ax)
        self.add(vector)
        self.add(i_walk, j_walk, k_walk)



class VectorExamplesScene(Scene):
    def construct(self):

        number_plane = NumberPlane(x_range=(-4.5,4.5,1),
                         y_range=(-4.5,4.5,1)
        )

        _v1,_v2,_v3,_v4 = np.array([3,1,0]), np.array([-2,1,0]), np.array([-3,-2,0]), np.array([1.5,-2.5,0])

        v1 = Vector(_v1, color=BLUE)
        v2 = Vector(_v2, color=ORANGE)
        v3 = Vector(_v3, color=GREEN)
        v4 = Vector(_v4, color=PURPLE)

        v1_lbl = MathTex(r"\vec{a} = " + sp.latex(sp.Matrix(_v1[:2])).replace("pmatrix","bmatrix"), color=BLUE) \
            .next_to(v1.get_end(), UL)

        v2_lbl = MathTex(r"\vec{b} = " + sp.latex(sp.Matrix(_v2[:2])).replace("pmatrix","bmatrix"), color=ORANGE)  \
            .next_to(v2.get_end(), UL)

        v3_lbl = MathTex(r"\vec{c} = " + sp.latex(sp.Matrix(_v3[:2])).replace("pmatrix","bmatrix"),  color=GREEN) \
            .next_to(v3.get_end(), DOWN)

        v4_lbl = MathTex(r"\vec{d} = " + sp.latex(sp.Matrix(_v4[:2])).replace("pmatrix","bmatrix"),  color=PURPLE)  \
            .next_to(v4.get_end(), RIGHT)

        grp = VGroup(number_plane,v1, v2, v3, v4, v1_lbl, v2_lbl, v3_lbl, v4_lbl)

        self.add(grp)
        # mobj_to_svg(grp, filename="out.svg")

class AddVectorScene(Scene):
    def construct(self):

        number_plane = NumberPlane(x_range=(-4.5,4.5,1),y_range=(-4.5,4.5,1))

        _v1 = np.array([-1,2,0])
        _v2 = np.array([2,1,0])
        _v3 = _v1+_v2

        v1 = Vector(_v1, color=BLUE)
        v1_lbl = MathTex(r"\vec{v}", color=BLUE).next_to(v1.get_midpoint(), UL + RIGHT*1.5)

        v2 = Vector(_v2, color=ORANGE)
        v2_lbl = MathTex(r"\vec{w}", color=ORANGE).next_to(v2.get_midpoint(), UP)

        v3 = Vector(_v3, color=YELLOW)
        v3_lbl = MathTex(r"\vec{v} + \vec{w} = ", sp.latex(sp.Matrix(_v3[:2])).replace("pmatrix","bmatrix"), color=YELLOW) \
            .next_to(v3.get_tip(), UP)

        grp = VGroup(number_plane,v1, v2, v1_lbl, v2_lbl, v3, v3_lbl)

        self.add(grp)
        mobj_to_svg(grp, filename="out.svg")

class AddVectorFormula(Scene):

    def construct(self):
        _v1 = np.array([-1,2,0])
        _v2 = np.array([2,1,0])
        _v3 = _v1+_v2

        math_tex = MathTex(
            r"\vec{v} + \vec{w} &= ",
            sp.latex(sp.Matrix(_v1[:2])).replace("pmatrix","bmatrix"),
            "+",
            sp.latex(sp.Matrix(_v2[:2])).replace("pmatrix", "bmatrix"),
            r"\\ &=",
            sp.latex(sp.Matrix(_v3[:2])).replace("pmatrix", "bmatrix"),
        )
        self.add(math_tex)
        #mobj_to_svg(math_tex, "out.svg")

class ScaleVector1(Scene):
    def construct(self):

        ax = NumberPlane(x_range=(-1.5,4.5,1),
                         y_range=(-1.5,4.5,1)
        )
        _v = np.array([1,2,0])
        _w = 2*_v
        v = Vector(_v, color=RED) \
            .put_start_and_end_on(ax.get_origin(), ax.c2p(*_v[:2]))

        v_lbl = MathTex(r"\vec{v}", color=RED) \
            .next_to(v.get_midpoint(), RIGHT)

        w = Vector(_w, color=BLUE) \
            .put_start_and_end_on(ax.get_origin(), ax.c2p(*_w[:2]))

        w_lbl = MathTex(r"\vec{2v}", color=BLUE) \
            .next_to(w.get_midpoint(), RIGHT)

        grp1 = VGroup(ax,v, v_lbl)
        grp2 = VGroup(ax.copy(), w, w_lbl)

        dual_pane = VGroup(grp1, grp2) \
            .arrange(DOWN,buff=1) \
            .scale_to_fit_height(8)

        self.add(dual_pane)
        mobj_to_svg(dual_pane, filename="out.svg")

class ScaleVector2(Scene):
    def construct(self):

        ax = NumberPlane(x_range=(-.5,3.5,1),
                         y_range=(-.5,3.5,1)
        )
        _v = np.array([1,2,0])
        _w = .5*_v
        v = Vector(_v, color=RED) \
            .put_start_and_end_on(ax.get_origin(), ax.c2p(*_v[:2]))

        v_lbl = MathTex(r"\vec{v}", color=RED) \
            .next_to(v.get_midpoint(), RIGHT)

        w = Vector(_w, color=BLUE) \
            .put_start_and_end_on(ax.get_origin(), ax.c2p(*_w[:2]))

        w_lbl = MathTex(r".5\vec{v}", color=BLUE) \
            .next_to(w.get_midpoint(), RIGHT)

        grp1 = VGroup(ax,v, v_lbl)
        grp2 = VGroup(ax.copy(), w, w_lbl)

        dual_pane = VGroup(grp1, grp2) \
            .arrange(RIGHT,buff=1) \
            .scale_to_fit_height(8)

        self.add(dual_pane)
        mobj_to_svg(dual_pane, filename="out.svg")

class ScaleVector3(Scene):
    def construct(self):

        ax = NumberPlane(x_range=(-3.5,3.5,1),
                         y_range=(-3.5,3.5,1)
        )
        _v = np.array([1,2,0])
        _w = -1.5*_v
        v = Vector(_v, color=RED) \
            .put_start_and_end_on(ax.get_origin(), ax.c2p(*_v[:2]))

        v_lbl = MathTex(r"\vec{v}", color=RED) \
            .next_to(v.get_midpoint(), RIGHT)

        w = Vector(_w, color=BLUE) \
            .put_start_and_end_on(ax.get_origin(), ax.c2p(*_w[:2]))

        w_lbl = MathTex(r"-1.5\vec{v}", color=BLUE) \
            .next_to(w.get_midpoint(), LEFT)

        grp1 = VGroup(ax,v, v_lbl)
        grp2 = VGroup(ax.copy(), w, w_lbl)

        dual_pane = VGroup(grp1, grp2) \
            .arrange(RIGHT,buff=1) \
            .scale_to_fit_height(8)

        self.add(dual_pane)
        mobj_to_svg(dual_pane, filename="out.svg")

class ScaleVectorFormula(Scene):
    def construct(self):
        _v = np.array([1,2,0])
        _w = 2*_v
        tex = MathTex(r"2\vec{v} &= 2", sp.latex(sp.Matrix(_v[:2])).replace("pmatrix", "bmatrix"),
                      r"\\ &= ", sp.latex(sp.Matrix(_w[:2])).replace("pmatrix", "bmatrix"))
        self.add(tex)
        mobj_to_svg(tex, 'out.svg')


class MatrixTransformation(LinearTransformationScene):
    def construct(self):

        A = np.array([[2,1],[-1,-2]])
        self.add_plane(animate=False)
        self.add_vector([2,1], animate=False, color=YELLOW)
        self.get_basis_vectors()
        self.apply_matrix(A)

class MatrixNotation(Scene):
    def construct(self):
        I = np.array([[1,0],[0,1]])
        A = np.array([[2,1],[-1,-2]])

        I_tex = MathTex("I=", sp.latex(sp.Matrix(I)))
        A_tex = MathTex("A=", sp.latex(sp.Matrix(A)))

        i_col_I = VGroup(*[mobj for i, mobj in enumerate(I_tex[1]) if i in (1,3)])
        j_col_I = VGroup(*[mobj for i, mobj in enumerate(I_tex[1]) if i in (2,4)])
        i_col_I.set_color(GREEN)
        j_col_I.set_color(RED)

        i_col_A = VGroup(*[mobj for i, mobj in enumerate(A_tex[1]) if i in (1,3,4)])
        j_col_A = VGroup(*[mobj for i, mobj in enumerate(A_tex[1]) if i in (2,5,6)])
        i_col_A.set_color(GREEN)
        j_col_A.set_color(RED)

        i_hat = MathTex(r"\hat{i}", color=GREEN)
        j_hat = MathTex(r"\hat{j}", color=RED)

        i_hat.next_to(i_col_A, DOWN)
        j_hat.next_to(j_col_A, DOWN)


        mobj_to_svg(VGroup(I_tex), 'out.svg')

class MatrixVectorNotation(Scene):
    def construct(self):
        A = np.array([[2,1],
                      [-1,-2]]
                     )
        v = np.array([2,1])

        tex = MathTex(r"A\vec{v} &= ", sp.latex(sp.Matrix(A)), r"\cdot", sp.latex(sp.Matrix(v)),
                      r"\\ &=", r"\begin{bmatrix} (2)(2) + (1)(1) \\ (-1)(2) + (1)(-2) \end{bmatrix}",
                      r"\\ &=", sp.latex(sp.Matrix(A @ v))
                      )
        self.add(tex)
        mobj_to_svg(tex, 'out.svg')

class LinearDependence(LinearTransformationScene):
    def construct(self):
        self.add_plane(animate=False)
        self.add_vector([2,1], animate=False, color=YELLOW)
        self.get_basis_vectors()
        self.apply_matrix(np.array([[-1,1],[2,-2]]))

class ZeroDeterminantScene(LinearTransformationScene):
    """
    config.frame_width = 7
    config.frame_height = 7
    config.pixel_width = 420
    config.pixel_height = 420
    """
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
        self.apply_matrix(np.array([[-1,1],[2,-2]]))

class CombinedMatrixTransformationScene(LinearTransformationScene):

    def __init__(self):
        LinearTransformationScene.__init__(
            self,
            show_coordinates=True,
            leave_ghost_vectors=False,
            show_basis_vectors=True
        )

    def construct(self):
        A = np.array([[2,0],
                      [0,2]])

        B = np.array([[1,0.5],
                      [0,-1]])

        self.apply_matrix(A@B)
        self.wait()

class MatrixMultiplicationFormula(Scene):
    def construct(self):

        a,b,c,d,e,f,g,h = sp.symbols('a b c d e f g h')
        A = sp.Matrix([[a,b], [c,d]])
        B = sp.Matrix([[e,f],[g,h]])

        A_latex = sp.latex(A)
        B_latex = sp.latex(B)
        AB_latex = sp.latex(A.multiply(B))

        mathtex=MathTex("AB &= ", A_latex, B_latex, r"\\&=", AB_latex)
        mobj_to_svg(mathtex, 'out.svg')

        subs_mathex= ("AB &= " + A_latex + B_latex + r"\\&=" + AB_latex)

        mobj_to_svg(MathTex(subs_mathex), 'out_subs.svg')


class MatrixVectorMultiplicationFormula(Scene):
    def construct(self):

        a,b,c,d,x,y = sp.symbols('a b c d x y')
        A = sp.Matrix([[a,b], [c,d]])
        v = sp.Matrix([[x],[y]])

        A_latex = sp.latex(A)
        v_latex = sp.latex(v)

        Av_latex = sp.latex(A.multiply(v))

        mathtex=MathTex(r"A\vec{v} &= ", A_latex, v_latex, r"\\&=", Av_latex)
        mobj_to_svg(mathtex, 'out.svg')

        subs_mathex= (r"A\vec{v} &= " + A_latex + v_latex + r"\\&=" + Av_latex)

        mobj_to_svg(MathTex(subs_mathex), 'out.svg')

class InverseMatrixFormula(Scene):
    def construct(self):
        A = sp.Matrix([[2,1],
                      [-1,-2]])

        v = sp.Matrix([5,-4,0])

        A_inv = sp.inv_quick(A)

        tex = VGroup(
            MathTex(r"A =", sp.latex(A)),
            MathTex(r"A^{-1} =", sp.latex(A_inv)),
            MathTex(r"A^{-1} A =", sp.latex(A_inv@A))
        ).arrange(DOWN, aligned_edge=LEFT)

        mobj_to_svg(tex, 'out.svg')

class SimpleSystemEquationsFormula(Scene):
    def construct(self):

        x,y = sp.symbols('x y')
        sos = r"2x + 1y &= 5 \\-1x - 2y &= -4"

        A = sp.Matrix([[2,1],
                      [-1,-2]])

        x = sp.Matrix([[x],[y]])
        b = sp.Matrix([[5], [-4]])

        axb_latex = r"AX &= B\\",
        axb_subs_latex = sp.latex(sp.inv_quick(A) @ b) + r"&=" + sp.latex(x)

        tex = MathTex(axb_subs_latex) #sp.latex(sp.inv_quick(A)))

        mobj_to_svg(tex, 'out.svg')


if __name__ == "__main__":
    render_scenes(q="l", transparent=False, scene_names=['CodeRender'])
