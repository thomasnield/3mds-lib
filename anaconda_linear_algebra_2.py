from manim import *
from numpy.linalg import inv

from threemds.utils import render_scenes, mobj_to_svg, mobj_to_png, scene_fit_configs
from numpy import array

config.background_color = "WHITE"
config.frame_rate = 60

# config.frame_width = 7
# config.frame_height = 7
# config.pixel_width = config.pixel_width
# config.pixel_height = config.pixel_height / 2

light_background_line_style = {
                "stroke_color": LIGHT_GRAY,
                "stroke_width": 2,
                "stroke_opacity": 1,
                "axis_config" : {
                   "stroke_width": 4,
                   "include_tip": False,
                   "line_to_number_buff": SMALL_BUFF,
                   "label_direction": DR,
                   "font_size": 24,
                   "color" : BLACK
               }
            }
light_axis_config = {
               "stroke_width": 4,
               "include_tip": False,
               "line_to_number_buff": SMALL_BUFF,
               "label_direction": DR,
               "font_size": 24,
               "color" : BLACK,
           }

class MatrixTransformationScene(LinearTransformationScene):
    def construct(self):
        plane = self.add_plane(animate=False,
                               background_line_style = light_background_line_style)

        vector = self.add_vector([3,2], animate=False, color=BLACK)
        basis = self.get_basis_vectors()
        #self.vector_to_coords(vector=vector)
        self.apply_matrix(array([[-1,1],[0.5,-2]]))

class IHatJHatFormulaScene1(Scene):
    def construct(self):
        mathtex = MathTex(r"A = \begin{bmatrix} 1 & 0  \\ 0 & 1 \end{bmatrix}", color=BLACK)

        i_hat = VGroup(*[mobj for i,mobj in enumerate(mathtex[0]) if i in {3, 5}])
        j_hat = VGroup(*[mobj for i,mobj in enumerate(mathtex[0]) if i in {4, 6}])
        remaining = VGroup(*[mobj for i,mobj in enumerate(mathtex[0]) if i not in {3, 4, 5, 6}])

        for mobj in i_hat:
            mobj.color = GREEN

        for mobj in j_hat:
            mobj.color = RED

        i_hat_lbl = MathTex(r"\hat{i}", color=GREEN).next_to(i_hat, DOWN)
        j_hat_lbl = MathTex(r"\hat{j}", color=RED).next_to(j_hat, DOWN)

        grp = VGroup(remaining, i_hat, j_hat, i_hat_lbl, j_hat_lbl) \
            .add_background_rectangle(WHITE, opacity=1)

        self.add(grp)
        mobj_to_svg(grp, "out.svg", padding=1)

class BasisVectorsStartScene(LinearTransformationScene):
    def construct(self):
        plane = self.add_plane(animate=False,
                               background_line_style = light_background_line_style,
                               axis_config= light_axis_config)

        #vector = self.add_vector([3,2], animate=False, color=BLACK)
        i_hat, j_hat = self.get_basis_vectors()
        i_hat_lbl, j_hat_lbl = self.get_basis_vector_labels()

        i_hat_lbl.next_to(i_hat, DOWN)
        j_hat_lbl.next_to(j_hat, LEFT)

        grp = VGroup(plane, i_hat,j_hat, i_hat_lbl, j_hat_lbl)
        mobj_to_svg(grp, "out.svg", w_padding=-7, h_padding=-3)

        self.add(i_hat_lbl, j_hat_lbl)
        #self.vector_to_coords(vector=vector)
        #self.apply_matrix(array([[-1,1],[0.5,-2]]))


class IHatJHatFormulaScene2(Scene):
    def construct(self):
        mathtex = MathTex(r"A = \begin{bmatrix} 1 & 2  \\ -1 & 1 \end{bmatrix}", color=BLACK)

        i_hat = VGroup(*[mobj for i,mobj in enumerate(mathtex[0]) if i in {3, 5, 6}])
        j_hat = VGroup(*[mobj for i,mobj in enumerate(mathtex[0]) if i in {4, 7}])
        remaining = VGroup(*[mobj for i,mobj in enumerate(mathtex[0]) if i not in {3, 4, 5, 6, 7}])

        for mobj in i_hat:
            mobj.color = GREEN

        for mobj in j_hat:
            mobj.color = RED

        i_hat_lbl = MathTex(r"\hat{i}", color=GREEN).next_to(i_hat, DOWN)
        j_hat_lbl = MathTex(r"\hat{j}", color=RED).next_to(j_hat, DOWN)

        grp = VGroup(remaining, i_hat, j_hat, i_hat_lbl, j_hat_lbl) \
            .add_background_rectangle(WHITE, opacity=1)

        self.add(grp)
        mobj_to_svg(grp, "out.svg", padding=1)

class BasisVectorsEndScene(LinearTransformationScene):
    def construct(self):
        plane = self.add_plane(animate=False,
                               background_line_style = light_background_line_style,
                               axis_config= light_axis_config)

        #vector = self.add_vector([3,2], animate=False, color=BLACK)
        self.get_basis_vectors()
        self.apply_matrix(array([[1, -1], [2, 1]]).transpose())

class MatrixTransformationScene2(LinearTransformationScene):
    def construct(self):
        plane = self.add_plane(animate=False,
                               background_line_style = light_background_line_style,
                               axis_config= light_axis_config)

        self.add_vector([0.5, 1.5], animate=False, color=BLACK)
        self.get_basis_vectors()

        self.apply_matrix(array([[1, -1], [2, 1]]).transpose())

class MatrixVectorMultiplicationScene(Scene):

    def construct(self):
        #form1 = MathTex(r"A\vec{v}", color=BLACK)
        form1 = MathTex(r"\begin{bmatrix} 1 & 2 \\ -1 & 1 \end{bmatrix} \begin{bmatrix} 0.5 \\ 1.5 \end{bmatrix} &= ",
                        r"\begin{bmatrix} (1)(0.5) + (2)(1.5) \\ (-1)(0.5) + (1)(1.5) \end{bmatrix}",
                        r"= \begin{bmatrix} 3.5 \\ 1 \end{bmatrix}"
                        , color=BLACK)

        movements = [
            (
                ((1,), (2,), (1,3)),
                ((2,), (11,), (10,12))
            ), # animation 1
            (
                ((8, 9, 10), ((5,6,7)), (4,8)),
                ((11, 12, 13), (14,15,16), (13,17,9))
            ), # animation 2
            (
                ((3,4), (19,20), (18,21)),
                ((5,), (29,), (28,30))
            ),
            (
                ((8,9,10), (23,24,25), (22,26)),
                ((11,12,13), (32,33,34), (31,35, 27))
            )
        ]

        self.add(form1[0])
        self.play(
            FadeIn(VGroup(*[mobj for i,mobj in enumerate(form1[1]) if i in (0,36)])),
            FadeIn(VGroup(*[mobj for i, mobj in enumerate(form1[0]) if i in (15,)]))

        )

        for movement in movements:
            animations = []
            for start_i, end_i, fadein_i in movement:
                start_grp = VGroup(*[mobj.copy() for i,mobj in enumerate(form1[0]) if i in start_i])
                end_grp = VGroup(*[mobj.copy() for i,mobj in enumerate(form1[1]) if i in end_i])
                animations.append(ReplacementTransform(start_grp, end_grp))

                fadein_grp = VGroup(*[mobj for i,mobj in enumerate(form1[1]) if i in fadein_i])
                animations.append(FadeIn(fadein_grp))

            self.play(*animations)

        #self.add(form1, index_labels(form1[0]), index_labels(form1[1]))
        self.wait()
        self.play(FadeIn(form1[2]))
        self.wait()

        scene_fit_configs(self, w_padding=.5, h_padding=.5)

class RotateScene(LinearTransformationScene):
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
            show_basis_vectors=True,
            foreground_plane_kwargs = light_background_line_style | { "x_range" : [-5.5,5.5,1],
                                                                      "y_range" : [-5.5, 5.5, 1]},
            background_plane_kwargs = { "x_range" : [-5.5,5.5,1], "y_range" : [-5.5, 5.5, 1]}
        )

    def construct(self):
        self.add_vector([1,1], animate=False, color=BLACK)
        self.apply_matrix(array([[0,1],[-1,0]]))

class ScaleScene(LinearTransformationScene):

    def __init__(self):
        LinearTransformationScene.__init__(
            self,
            show_coordinates=True,
            leave_ghost_vectors=False,
            show_basis_vectors=True,
            foreground_plane_kwargs = light_background_line_style | { "x_range" : [-5.5,5.5,1],
                                                                      "y_range" : [-5.5, 5.5, 1]},
            background_plane_kwargs = { "x_range" : [-5.5,5.5,1], "y_range" : [-5.5, 5.5, 1]}
        )

    def construct(self):
        self.add_vector([1,1], animate=False, color=BLACK)
        self.apply_matrix(array([[1,0],[0,2]]))


class ShearScene(LinearTransformationScene):

    def __init__(self):
        LinearTransformationScene.__init__(
            self,
            show_coordinates=True,
            leave_ghost_vectors=False,
            show_basis_vectors=True,
            foreground_plane_kwargs = light_background_line_style | { "x_range" : [-5.5,5.5,1],
                                                                      "y_range" : [-5.5, 5.5, 1]},
            background_plane_kwargs = { "x_range" : [-5.5,5.5,1], "y_range" : [-5.5, 5.5, 1]}
        )

    def construct(self):
        self.add_vector([1,1], animate=False, color=BLACK)
        self.apply_matrix(array([[1,1],[0,1]]))


class InversionScene(LinearTransformationScene):

    def __init__(self):
        LinearTransformationScene.__init__(
            self,
            show_coordinates=True,
            leave_ghost_vectors=False,
            show_basis_vectors=True,
            foreground_plane_kwargs = light_background_line_style | { "x_range" : [-5.5,5.5,1],
                                                                      "y_range" : [-5.5, 5.5, 1]},
            background_plane_kwargs = { "x_range" : [-5.5,5.5,1], "y_range" : [-5.5, 5.5, 1]}
        )

    def construct(self):
        self.add_vector([1,1], animate=False, color=BLACK)
        self.apply_matrix(array([[0,1],[1,0]]))



if __name__ == "__main__":
    render_scenes(q='l', scene_names=['InverseMatrixTransformationScene'])