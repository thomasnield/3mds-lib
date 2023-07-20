from manim import *
from threemds.utils import render_scenes, mobj_to_svg, mobj_to_png

config.background_color = "WHITE"

light_background_line_style = {
                "stroke_color": BLACK,
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

class SimpleVectorScene(Scene):
    def construct(self):

        np = NumberPlane(x_range=(-.5,4.5,1),
                         y_range=(-1.5,3.5,1),
                         background_line_style = light_background_line_style,
                         axis_config= light_axis_config
        ).add_background_rectangle(color=WHITE)

        v = Vector([3,2,0], color=BLACK).move_to(np.get_origin(), aligned_edge=DL)
        grp = VGroup(np,v)
        self.add(grp)
        mobj_to_svg(grp, filename="out.svg")


class BasisVectorsScene(VectorScene):
    def construct(self):

        np = NumberPlane(x_range=(-.5,4.5,1),
                         y_range=(-1.5,3.5,1),
                         background_line_style = light_background_line_style,
                         axis_config= light_axis_config
        ).add_background_rectangle(color=WHITE)

        v = Vector([3,2,0], color=BLACK).move_to(np.get_origin(), aligned_edge=DL)

        i_brace = Brace(v,direction=RIGHT,color=GREEN)
        i_txt = Tex("2", color=GREEN).next_to(i_brace, RIGHT)

        j_brace = Brace(v, direction=DOWN, color=RED)
        j_txt = Tex("3", color=RED).next_to(j_brace, DOWN)

        grp = VGroup(np,v, i_brace, i_txt, j_brace,j_txt)
        mobj_to_svg(grp, "out.svg")

        self.add(grp)

if __name__ == "__main__":
    render_scenes(q="l", last_scene=True, scene_names=["SimpleVectorScene"])