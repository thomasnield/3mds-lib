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

class VectorInNumpy(Scene):

    def construct(self):

        raw_code="""import numpy as np
v = np.array([2,3])
print(v) # [2 3]"""

        code=Code(code=raw_code, language="Python", font="Monospace", style="monokai", background="window")
        self.add(code)
        #mobj_to_svg(VGroup(code), "out.svg")


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

if __name__ == "__main__":
    render_scenes(q="l", transparent=True, scene_names=['AddVectorScene', 'AddVectorFormula'])
