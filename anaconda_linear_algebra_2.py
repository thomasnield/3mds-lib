from manim import *
from threemds.utils import render_scenes, mobj_to_svg, mobj_to_png
from numpy import array

config.background_color = "WHITE"
config.frame_rate = 60

light_background_line_style = {
                "stroke_color": LIGHT_GRAY,
                "stroke_width": 2,
                "stroke_opacity": 1,
            }
light_axis_config = {
               "stroke_width": 4,
               "include_ticks": False,
               "include_tip": False,
               "line_to_number_buff": SMALL_BUFF,
               "label_direction": DR,
               "font_size": 24,
               "color" : BLACK,
           }

class MatrixTransformationScene(LinearTransformationScene):
    def construct(self):
        plane = self.add_plane(animate=False,
                               background_line_style = light_background_line_style,
                               axis_config= light_axis_config)

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


class MatrixTransformationScene2(LinearTransformationScene):
    def construct(self):
        plane = self.add_plane(animate=False,
                               background_line_style = light_background_line_style,
                               axis_config= light_axis_config)

        vector = self.add_vector([2,1], animate=False, color=BLACK)
        basis = self.get_basis_vectors()
        #self.vector_to_coords(vector=vector)
        self.apply_matrix(array([[1,2],[-1,1]]))

if __name__ == "__main__":
    render_scenes(last_scene=True, scene_names=['BasisVectorsStartScene'])