from manim import *
from numpy import array
from threemds.utils import render_scenes, file_to_base_64, mobj_to_svg
import sympy as sp

config.background_color = "WHITE"
config.frame_rate = 60
#config.frame_width = 7
#config.frame_height = 7
#config.frame_width = 12
#config.pixel_width = 420
#config.pixel_height = 420


light_background_line_style = {
                "stroke_color": LIGHT_GRAY,
                "stroke_width": 2,
                "stroke_opacity": 1
            }
light_axis_config = {
               "stroke_width": 4,
               "include_tip": False,
               "line_to_number_buff": SMALL_BUFF,
               "label_direction": DR,
               "font_size": 24,
               "color" : BLACK,
           }


class CombinedMatrixTransformationScene(LinearTransformationScene):

    def __init__(self):
        LinearTransformationScene.__init__(
            self,
            show_coordinates=True,
            leave_ghost_vectors=False,
            show_basis_vectors=True,
            foreground_plane_kwargs = light_background_line_style
        )


    def construct(self):
        transform1 = array([[0,1],[1,0]])
        transform2 = array([[2,1],[0,1]])

        self.add_vector([1,1],animate=False,color=BLACK)
        self.apply_matrix(transform2 @ transform1)

class VectorDotProduct(Scene):
    def construct(self):
        import numpy as np

        number_plane = NumberPlane(x_range=(-.5,4.5,1),
                         y_range=(-1.5,3.5,1),
                         background_line_style = light_background_line_style,
                         axis_config= light_axis_config
        ).add_background_rectangle(color=WHITE)

        # v = Vector([3,2,0], color=BLUE).move_to(np.get_origin(), aligned_edge=DL)
        # w = Vector([2,-1,0], color=ORANGE).move_to(np.get_origin(), aligned_edge=UL)
        w = array([3,2,0])
        v = array([2,-1,0])

        # finding norm of the vector v
        v_norm = np.sqrt(sum(w ** 2))

        # Apply the formula as mentioned above
        # for projecting a vector onto another vector
        # find dot product using np.dot()
        proj_of_u_on_v = Line(start=number_plane.get_origin(),
                                    end=number_plane.c2p(*(np.dot(v, w) / v_norm ** 2) * w),
                                    color=PURPLE)

        v = Vector([3, 2, 0], color=BLUE) \
            .move_to(number_plane.get_origin(), aligned_edge=DL)

        w = Vector([2, -1, 0], color=ORANGE) \
            .move_to(number_plane.get_origin(), aligned_edge=UL)

        projection_line = DashedLine(start=proj_of_u_on_v.get_end(),
                                     end=w.get_end(),
                                     color=PURPLE)

        v_label = MathTex(r"\vec{v}", color=BLUE) \
            .move_to(v.get_midpoint(), UL).shift(RIGHT*0.2)

        w_label = MathTex(r"\vec{w}", color=ORANGE) \
            .move_to(w.get_midpoint(), UR).shift(DOWN*0.2)

        proj_label = MathTex(r"proj(\vec{w})", color=PURPLE) \
            .scale(.6) \
            .move_to(proj_of_u_on_v.get_midpoint(), aligned_edge=DR) \
            .shift(UR *.75)

        grp = VGroup(number_plane,v,w, v_label, w_label, proj_label,proj_of_u_on_v, projection_line)

        self.add(grp)

        mobj_to_svg(grp, filename="out.svg")

class MatrixDotProduct(Scene):
    def construct(self):
        import numpy as np

        number_plane = NumberPlane(x_range=(-1.5,4.5,1),
                         y_range=(-1.5,3.5,1),
                         background_line_style = light_background_line_style,
                         axis_config= light_axis_config
        ).add_background_rectangle(color=WHITE)

        v = array([1,1])

        def projected(projected_vector, target):
            n = np.sqrt(sum(target ** 2))
            return (np.dot(projected_vector, target) / n ** 2) * target

        def norm(projected_vector, target):
            n = np.sqrt(sum(target ** 2))
            proj = (np.dot(projected_vector, target) / n ** 2) * target
            return np.linalg.norm(proj)

        i_hat = np.array([3, 1])
        j_hat = np.array([-1, 2])

        proj_of_v_on_i = DashedLine(start=number_plane.get_origin(),
                                    end=number_plane.c2p(*projected(v,i_hat)),
                                    color=YELLOW)

        proj_of_v_on_j = DashedLine(start=number_plane.get_origin(),
                                    end=number_plane.c2p(*projected(v,j_hat)),
                                    color=YELLOW)

        print(norm(v, i_hat) * np.linalg.norm(i_hat))

        v_mobj = Vector([*v,0], color=BLUE) \
            .move_to(number_plane.get_origin(), aligned_edge=DL)

        i_hat_mobj = Vector([*i_hat,0], color=GREEN) \
            .move_to(number_plane.get_origin(), aligned_edge=DL)

        j_hat_mobj = Vector([*j_hat,0], color=RED) \
            .move_to(number_plane.get_origin(), aligned_edge=DR)


        grp = VGroup(number_plane, v_mobj,
                     i_hat_mobj, j_hat_mobj,
                     proj_of_v_on_j, proj_of_v_on_i)

        self.add(grp)

        mobj_to_svg(grp, filename="out.svg")

if __name__ == "__main__":
    render_scenes(q='l', scene_names=['MatrixDotProduct'])