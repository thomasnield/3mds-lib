from manim import *
from threemds.utils import *
import numpy as np

# configure the grid size

def resize_grid(m_width, m_height):
    config.frame_width = m_width
    config.frame_height = m_height
    config.pixel_width = int(config.frame_width * config.pixel_width / config.frame_width)
    config.pixel_height = int(config.pixel_width * config.frame_height / config.frame_width)

#resize_grid(6,6)

class SimpleVectorScene(VectorScene):
    def construct(self):
        plane = self.add_plane(animate=False)
        vector = self.add_vector([3,2], animate=False)
        #basis = self.get_basis_vectors()
        self.vector_to_coords(vector=vector)

        self.add(plane, vector)

class ThreeDVectorScene(ThreeDScene):
    def construct(self):
        ax = ThreeDAxes(x_range=[-3,3,1],y_range=[-3,3,1],z_range=[-2,2,1])
        line = Arrow3D(start=ax.c2p(0, 0, 0), end=ax.c2p(2, 2, 2), color=YELLOW)

        basis_x = Arrow3D(start=ax.c2p(0, 0, 0),
                         end=ax.c2p(1, 0, 0),
                         color=GREEN,
                         resolution=8)

        basis_y = Arrow3D(start=ax.c2p(0, 0, 0),
                         end=ax.c2p(0, 1, 0),
                         color=RED,
                         resolution=8)

        basis_z = Arrow3D(start=ax.c2p(0, 0, 0),
                         end=ax.c2p(0, 0, 1),
                         color=PURPLE,
                         resolution=8)

        self.set_camera_orientation(phi=75 * DEGREES, theta=25 * DEGREES)
        self.add(ax, line, basis_x, basis_y, basis_z)

class ScaledVector(VectorScene):
    def construct(self):
        plane = self.add_plane(animate=False)
        v = self.add_vector([.75, 1.5], animate=False, color=RED)
        w = self.add_vector([1.5, 3], animate=False, color=GREEN)

        v_label = MathTex(r"\vec{v}", fill_color=RED).scale(1.5).next_to(v.get_midpoint(), RIGHT)
        w_label = MathTex(r"\vec{w}", fill_color=GREEN).scale(1.5).next_to(w.get_midpoint(), UR + RIGHT)
        # basis = self.get_basis_vectors()
        self.add(plane, w, v_label, w_label)
        self.add_foreground_mobjects(v)

class LinearIndependent(VectorScene):
    def construct(self):
        plane = self.add_plane(animate=False)
        v = self.add_vector([3,2], animate=False, color=RED)
        w = self.add_vector([2,-1], animate=False, color=GREEN)

        v_label = MathTex(r"\vec{v}", fill_color=RED).scale(1.5).next_to(v.get_midpoint(), UP)
        w_label = MathTex(r"\vec{w}", fill_color=GREEN).scale(1.5).next_to(w.get_midpoint(), DOWN)

        self.add(plane, v, w, v_label, w_label)

class LinearDependent(VectorScene):
    def construct(self):
        plane = self.add_plane(animate=False)
        v = self.add_vector([3,2], animate=False, color=RED)
        w = self.add_vector([-1.5,-1], animate=False, color=GREEN)
        v_label = MathTex(r"\vec{v}", fill_color=RED).scale(1.5).next_to(v.get_midpoint(), UP)
        w_label = MathTex(r"\vec{w}", fill_color=GREEN).scale(1.5).next_to(w.get_midpoint(), DOWN)
        # basis = self.get_basis_vectors()
        self.add(plane, v, w, v_label, w_label)

class TransformationVectorOne(LinearTransformationScene):

    def __init__(self):
        LinearTransformationScene.__init__(self,
                                           show_coordinates=True,
                                           leave_ghost_vectors=False,
                                           show_basis_vectors=True
                       )
    def construct(self):

        i_hat = np.array([-2, 0])
        j_hat = np.array([1, 1])

        matrix = np.array([i_hat, j_hat]).transpose()

        v = self.add_vector([1,2], animate=False, color=YELLOW)
        v_label = always_redraw(lambda:
                    MathTex(r"\vec{v}", fill_color=YELLOW) \
                                .scale(1.5) \
                                .next_to(v.get_tip())
        )
        self.add_transformable_mobject(v)
        self.add_background_mobject(v_label)
        self.apply_matrix(matrix)

class TransformationVectorTwo(LinearTransformationScene):

    def __init__(self):
        LinearTransformationScene.__init__(self,
                                           show_coordinates=True,
                                           leave_ghost_vectors=False,
                                           show_basis_vectors=True
                       )
    def construct(self):

        i_hat = np.array([2, -1])
        j_hat = np.array([-1, 1])

        matrix = np.array([i_hat, j_hat]).transpose()

        v = self.add_vector([1,2], animate=False, color=YELLOW)
        v_label = always_redraw(lambda:
                    MathTex(r"\vec{v}", fill_color=YELLOW) \
                                .scale(1.5) \
                                .next_to(v.get_tip())
        )
        self.add_transformable_mobject(v)
        self.add_background_mobject(v_label)
        self.apply_matrix(matrix)


class Determinant(LinearTransformationScene):

    def __init__(self):
        LinearTransformationScene.__init__(self,
                                           show_coordinates=True,
                                           leave_ghost_vectors=False,
                                           show_basis_vectors=True
                       )
    def construct(self):

        i_hat = np.array([1, 1])
        j_hat = np.array([-2, 1])
        matrix = np.array([i_hat, j_hat]).transpose()

        sq = self.get_unit_square()
        self.add_transformable_mobject(sq)
        self.apply_matrix(matrix)

class MatrixMultiplication(Scene):
    def construct(self):
        from sympy import Matrix, latex
        transform1 = Matrix([[1, 2], [1,-1]]).transpose()
        transform2 = Matrix([[-2, 0], [1, -1]]).transpose()
        combined = transform2 @ transform1

        tex1 = MathTex(
            latex(transform2),
            r"\cdot",
            latex(transform1),
            "=",
            latex(combined),
            fill_color=BLACK
        ).add_background_rectangle(color=WHITE, opacity=1.0)
        mobj_to_png(tex1, "matrix_mult2.png")

class SystemOfEquations(Scene):
    def construct(self):
        from sympy import Matrix, latex

        A = Matrix([[3, 1, 0], [2, 4, 1],[3, 1, 8]])
        B = Matrix([54,12,6])

        X = (A.inv() @ B).evalf()

        tex = MathTex(
            r"A^{-1}B &= X \\",
            latex(A) + "^{-1}",
            latex(B),
            "&=",
            latex(X),
            fill_color=BLACK
        ).add_background_rectangle(color=WHITE, opacity=1.0)
        mobj_to_png(tex, "system_of_equations.png")


class LinearDependentDeterminant(Scene):
    def construct(self):
        from sympy import Matrix, latex
        A = Matrix([
            [2, 1],
            [6, 3]
        ])
        tex = MathTex(
            "A = ",
            latex(A),
            fill_color=BLACK
        ).add_background_rectangle(color=WHITE, opacity=1.0)

        mobj_to_png(tex, "LinearDependentDeterminant.png")



class EigenvectorDecompose(Scene):
    def construct(self):
        from sympy import Matrix, latex
        A = Matrix([
            [1, 3],
            [2, 6]
        ])
        tex = MathTex(
            "A = ",
            latex(A),
            fill_color=BLACK
        ).add_background_rectangle(color=WHITE, opacity=1.0)

        mobj_to_png(tex, "EigenvectorDecompose.png")

if __name__ == "__main__":
    render_scenes(q="k", frames_only=False, scene_names="EigenvectorDecompose")