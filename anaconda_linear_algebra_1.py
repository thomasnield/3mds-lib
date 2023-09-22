from manim import *
from threemds.utils import render_scenes, mobj_to_svg, mobj_to_png, file_to_base_64
from numpy import array

config.background_color = "WHITE"

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


class VectorComponentScene(VectorScene):

    def construct(self):

        np = NumberPlane(x_range=(-.5,4.5,1),
                         y_range=(-1.5,3.5,1),
                         background_line_style = light_background_line_style,
                         axis_config= light_axis_config
        ).add_background_rectangle(color=WHITE)

        v = Vector([3,2,0], color=BLACK) \
            .move_to(np.get_origin(), aligned_edge=DL)

        i_brace = Brace(v,direction=RIGHT,color=GREEN)
        i_txt = Tex("2", color=GREEN).next_to(i_brace, RIGHT)

        j_brace = Brace(v, direction=DOWN, color=RED)
        j_txt = Tex("3", color=RED).next_to(j_brace, DOWN)

        grp = VGroup(np,v, i_brace, i_txt, j_brace,j_txt)
        mobj_to_svg(grp, "out.svg")
        self.add(grp)


class VectorExamplesScene(Scene):
    def construct(self):

        np = NumberPlane(x_range=(-4.5,4.5,1),
                         y_range=(-3.5,3.5,1),
                         background_line_style = light_background_line_style,
                         axis_config= light_axis_config
        ).add_background_rectangle(color=WHITE)

        v1 = Vector([3,2,0], color=BLUE).move_to(np.get_origin(), aligned_edge=DL)
        v2 = Vector([2,-1,0], color=ORANGE).move_to(np.get_origin(), aligned_edge=UL)
        v3 = Vector([-2,-1.5,0], color=GREEN).move_to(np.get_origin(), aligned_edge=UR)
        v4 = Vector([-1,2,0], color=PURPLE).move_to(np.get_origin(), aligned_edge=DR)

        v1_lbl = MathTex(r"\vec{a} = \begin{bmatrix} 3 \\ 2 \end{bmatrix}", color=BLUE) \
            .next_to(v1.get_end(), UL)

        v2_lbl = MathTex(r"\vec{b} = \begin{bmatrix} 2 \\ -1 \end{bmatrix}", color=ORANGE)  \
            .next_to(v2.get_end(), DR)

        v3_lbl = MathTex(r"\vec{c} = \begin{bmatrix} -2 \\ -1.5 \end{bmatrix}", color=GREEN) \
            .next_to(v3.get_end(), DL + UP + LEFT)

        v4_lbl = MathTex(r"\vec{d} = \begin{bmatrix} -1 \\ 2 \end{bmatrix}", color=PURPLE)  \
            .next_to(v4.get_end(), UL + LEFT)

        grp = VGroup(np,v1, v2, v3, v4, v1_lbl, v2_lbl, v3_lbl, v4_lbl)

        self.add(grp)
        mobj_to_svg(grp, filename="out.svg")


class VectorExamplesDotsScene(Scene):
    def construct(self):

        np = NumberPlane(x_range=(-4.5,4.5,1),
                         y_range=(-3.5,3.5,1),
                         background_line_style = light_background_line_style,
                         axis_config= light_axis_config
        ).add_background_rectangle(color=WHITE)

        v1 = Dot([3,2,0], color=BLUE).move_to(np.c2p(3,2))
        v2 = Dot([2,-1,0], color=ORANGE).move_to(np.c2p(2,-1))
        v3 = Dot([-2,-1.5,0], color=GREEN).move_to(np.c2p(-2,-1.5))
        v4 = Dot([-1,2,0], color=PURPLE).move_to(np.c2p(-1,2))

        v1_lbl = MathTex(r"\vec{a} = \begin{bmatrix} 3 \\ 2 \end{bmatrix}", color=BLUE) \
            .next_to(v1.get_end(), UL)

        v2_lbl = MathTex(r"\vec{b} = \begin{bmatrix} 2 \\ -1 \end{bmatrix}", color=ORANGE)  \
            .next_to(v2.get_end(), DR)

        v3_lbl = MathTex(r"\vec{c} = \begin{bmatrix} -2 \\ -1.5 \end{bmatrix}", color=GREEN) \
            .next_to(v3.get_end(), DL + UP + LEFT)

        v4_lbl = MathTex(r"\vec{d} = \begin{bmatrix} -1 \\ 2 \end{bmatrix}", color=PURPLE)  \
            .next_to(v4.get_end(), UL + LEFT)

        grp = VGroup(np,v1, v2, v3, v4, v1_lbl, v2_lbl, v3_lbl, v4_lbl)

        self.add(grp)
        mobj_to_svg(grp, filename="out.svg")

class AddVectorScene1(Scene):
    def construct(self):

        np = NumberPlane(x_range=(-.5,4.5,1),
                         y_range=(-1.5,3.5,1),
                         background_line_style = light_background_line_style,
                         axis_config= light_axis_config
        ).add_background_rectangle(color=WHITE)

        v1 = Vector([3,2,0], color=BLUE).move_to(np.get_origin(), aligned_edge=DL)
        v2 = Vector([2,-1,0], color=ORANGE).move_to(np.get_origin(), aligned_edge=UL)
        v1_lbl = MathTex(r"\vec{v}", color=BLUE).next_to(v1.tip, UR, aligned_edge=DL)
        v2_lbl = MathTex(r"\vec{w}", color=ORANGE).next_to(v2.tip, UR, aligned_edge=DL)

        grp = VGroup(np,v1, v2, v1_lbl, v2_lbl)

        self.add(grp)
        mobj_to_svg(grp, filename="out.svg")


class AddVectorScene2(Scene):
    def construct(self):

        np = NumberPlane(x_range=(-.5,5.5,1),
                         y_range=(-1.5,3.5,1),
                         background_line_style = light_background_line_style,
                         axis_config= light_axis_config
        ).add_background_rectangle(color=WHITE)

        v1 = Vector([3,2,0], color=BLUE).move_to(np.get_origin(), aligned_edge=DL)
        v1_lbl = MathTex(r"\vec{v}", color=BLUE).next_to(v1.get_midpoint(), UL + RIGHT)

        v2 = Vector([2,-1,0], color=ORANGE).move_to(v1.get_corner(UR), aligned_edge=UL)
        v2_lbl = MathTex(r"\vec{w}", color=ORANGE).next_to(v2.get_midpoint(), UR + DOWN *.5)

        grp = VGroup(np,v1, v2, v1_lbl, v2_lbl)

        self.add(grp)
        mobj_to_svg(grp, filename="out.svg")


class AddVectorScene3(Scene):
    def construct(self):

        np = NumberPlane(x_range=(-.5,5.5,1),
                         y_range=(-1.5,3.5,1),
                         background_line_style = light_background_line_style,
                         axis_config= light_axis_config
        ).add_background_rectangle(color=WHITE)

        v1 = Vector([3,2,0], color=BLUE).move_to(np.get_origin(), aligned_edge=DL)
        v1_lbl = MathTex(r"\vec{v}", color=BLUE).next_to(v1.get_midpoint(), UL + RIGHT)

        v2 = Vector([2,-1,0], color=ORANGE).move_to(v1.get_corner(UR), aligned_edge=UL)
        v2_lbl = MathTex(r"\vec{w}", color=ORANGE).next_to(v2.get_midpoint(), UR + DOWN *.5)

        v3 = Vector([5,1], color=PURPLE).move_to(np.get_origin(), aligned_edge=DL)
        v3_lbl = MathTex(r"\vec{v} + \vec{w} = \begin{bmatrix} 5 \\ 1 \end{bmatrix}", color=PURPLE) \
            .scale(.5) \
            .next_to(v3.get_tip().get_right(), DOWN + LEFT*.25, aligned_edge=RIGHT)

        grp = VGroup(np,v1, v2, v1_lbl, v2_lbl, v3, v3_lbl)

        self.add(grp)
        mobj_to_svg(grp, filename="out.svg")


class AddVectorScene4(Scene):
    def construct(self):

        np = NumberPlane(x_range=(-.5,5.5,1),
                         y_range=(-1.5,3.5,1),
                         background_line_style = light_background_line_style,
                         axis_config= light_axis_config
        ).add_background_rectangle(color=WHITE)

        v2 = Vector([2,-1,0], color=ORANGE).move_to(np.get_origin(), aligned_edge=UL)
        v2_lbl = MathTex(r"\vec{w}", color=ORANGE).next_to(v2.get_midpoint(), DL * .5)

        v1 = Vector([3,2,0], color=BLUE).move_to(v2.get_corner(DR), aligned_edge=DL)
        v1_lbl = MathTex(r"\vec{v}", color=BLUE).next_to(v1.get_midpoint(), UL + RIGHT)

        v3 = Vector([5,1], color=PURPLE).move_to(np.get_origin(), aligned_edge=DL)
        v3_lbl = MathTex(r"\vec{v} + \vec{w} = \begin{bmatrix} 5 \\ 1 \end{bmatrix}", color=PURPLE) \
            .scale(.5) \
            .next_to(v3.get_center(), UP + LEFT*.25, aligned_edge=RIGHT)

        grp = VGroup(np,v1, v2, v1_lbl, v2_lbl, v3, v3_lbl)

        self.add(grp)
        mobj_to_svg(grp, filename="out.svg")

class ScaleVectorScene1(Scene):
    def construct(self):

        np = NumberPlane(x_range=(-.5,4.5,1),
                         y_range=(-1.5,3.5,1),
                         background_line_style = light_background_line_style,
                         axis_config= light_axis_config
        ).add_background_rectangle(color=WHITE)

        v = Vector([2,1,0], color=RED) \
            .move_to(np.get_origin(), aligned_edge=DL)

        v_lbl = MathTex(r"\vec{v}", color=RED) \
            .move_to(v.get_midpoint(), UL * .5)

        w = Vector([4,2,0], color=BLUE) \
            .move_to(np.get_origin(), aligned_edge=DL)

        w_lbl = MathTex(r"\vec{2v}", color=BLUE) \
            .move_to(w.point_from_proportion(.6), aligned_edge=UL)

        grp1 = VGroup(np,v, v_lbl)
        grp2 = VGroup(np.copy(), w, w_lbl)

        dual_pane = VGroup(grp1, grp2).arrange(RIGHT,buff=1)

        self.add(dual_pane)
        mobj_to_svg(dual_pane, filename="out.svg")

class ScaleVectorScene1(Scene):
    def construct(self):

        np = NumberPlane(x_range=(-.5,4.5,1),
                         y_range=(-1.5,3.5,1),
                         background_line_style = light_background_line_style,
                         axis_config= light_axis_config
        ).add_background_rectangle(color=WHITE)

        v = Vector([2,1,0], color=RED) \
            .move_to(np.get_origin(), aligned_edge=DL)

        v_lbl = MathTex(r"\vec{v}", color=RED) \
            .move_to(v.point_from_proportion(.5) + [0,.5,0], UP)

        w = Vector([1,0.5,0], color=BLUE) \
            .move_to(np.get_origin(), aligned_edge=DL)

        w_lbl = MathTex(r"\vec{.5v}", color=BLUE) \
            .move_to(w.point_from_proportion(.5) + [-.3, .7, 0], UL)

        grp1 = VGroup(np,v, v_lbl)
        grp2 = VGroup(np.copy(), w, w_lbl)

        dual_pane = VGroup(grp1, grp2).arrange(RIGHT,buff=1)

        self.add(dual_pane)
        mobj_to_svg(dual_pane, filename="out.svg")



class ScaleVectorScene2(Scene):
    def construct(self):

        np = NumberPlane(x_range=(-4.5,4.5,1),
                         y_range=(-3.5,3.5,1),
                         background_line_style = light_background_line_style,
                         axis_config= light_axis_config
        ).add_background_rectangle(color=WHITE)

        v = Vector([2,1,0], color=RED) \
            .move_to(np.get_origin(), aligned_edge=DL)

        v_lbl = MathTex(r"\vec{v}", color=RED) \
            .move_to(v.point_from_proportion(.5) + [0,.5,0], UP)

        w = Vector([-3,-1.5,0], color=BLUE) \
            .move_to(np.get_origin(), aligned_edge=UR)

        w_lbl = MathTex(r"\vec{-1.5v}", color=BLUE) \
            .move_to(w.point_from_proportion(.5) + DOWN * .45)

        underlying_line = DashedLine(start=[-4.5, -2.25, 0], end=[4.5,2.25,0], color=BLACK)

        grp1 = VGroup(np,underlying_line, v, v_lbl)
        grp2 = VGroup(np.copy(), underlying_line.copy(), w, w_lbl)

        dual_pane = VGroup(grp1, grp2).arrange(RIGHT,buff=1)

        self.add(dual_pane)
        mobj_to_svg(dual_pane, filename="out.svg")

class VectorScaleAndAddScene1(Scene):
    def construct(self):

        np = NumberPlane(x_range=(-4.5,4.5,1),
                         y_range=(-3.5,3.5,1),
                         background_line_style = light_background_line_style,
                         axis_config= light_axis_config
        ).add_background_rectangle(color=WHITE)

        v1_raw = array(np.c2p(1.5,0.5,0))
        v2_raw = array(np.c2p(0.75,-1,0))

        v1 = Vector(v1_raw, color=BLUE)
        v2 = Vector(v2_raw, color=ORANGE)
        v1_lbl = MathTex(r"\vec{v}", color=BLUE).next_to(v1.tip, UR, aligned_edge=DL)
        v2_lbl = MathTex(r"\vec{w}", color=ORANGE).next_to(v2.tip, UR, aligned_edge=DL)

        scalars = [(1,2), (-.5, 1.5), (-1.5, 1), (-3, .5), (-1, -1.8), (-2, -1.6), (1, -1.6)]
        combined_vectors = []

        for s1,s2 in scalars:
            combined_v = v1_raw*s1 + v2_raw*s2
            combined_vectors.append(
                Vector(combined_v, color=GREEN)
            )
        grp = VGroup(*combined_vectors, np,v1, v2, v1_lbl, v2_lbl)

        self.add(grp)
        mobj_to_svg(grp, filename="out.svg")

if __name__ == "__main__":
    render_scenes(q="l", last_frame=True, scene_names=['VectorExamplesDotsScene'])
    file_to_base_64('/Users/thomasnield/git/3mds-lib/media/images/anaconda_linear_algebra_1/04_VectorExamplesDotsScene.png')