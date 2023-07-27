from manim import *
from numpy import array
from threemds.utils import render_scenes, file_to_base_64, mobj_to_svg

#config.background_color = "WHITE"

config.frame_width = 12
config.frame_height = 7

class ThreeDVectorScene(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=75 * DEGREES, theta=-30 * DEGREES)

        ax = ThreeDAxes(axis_config={"color" : BLACK})
        v = array([1,4,3])
        vector = Arrow3D(start=ax.c2p(0,0,0), end=ax.c2p(*v), color=YELLOW)

        i_hat = Line(start=ax.c2p(0,0,0), end=ax.c2p(1,0,0), color=GREEN)
        j_hat = Line(start=ax.c2p(0,0,0), end=ax.c2p(0,1,0), color=RED)
        k_hat = Line(start=ax.c2p(0,0,0), end=ax.c2p(0,0,1), color=PURPLE)

        i_hat_lbl = MathTex(r"\hat{i}", color=GREEN).next_to(i_hat.get_midpoint(), DL + RIGHT + DOWN*.5)
        j_hat_lbl = MathTex(r"\hat{j}", color=RED).next_to(j_hat.get_midpoint(), RIGHT*3 )
        k_hat_lbl = MathTex(r"\hat{k}", color=PURPLE).next_to(k_hat.get_top(),  UL + UP)

        i_walk = Line(start=ax.c2p(0,0,0), end=ax.c2p(v[0],0,0), color=GREEN)

        j_walk = Line(start=ax.c2p(v[0],0,0), end=ax.c2p(v[0],v[1],0), color=RED)
        k_walk = Line(start=ax.c2p(v[0],v[1],0), end=ax.c2p(v[0],v[1],v[2]), color=PURPLE)
        """
        i_walk_lbl = MathTex("1", color=GREEN).next_to(ax.c2p(*i_walk.get_midpoint()), DL - LEFT * .75) \
            .rotate(90 * DEGREES, about_point=ax.c2p(*i_walk.get_midpoint()), axis=RIGHT) \
            .rotate(45 * DEGREES, axis=OUT)

        j_walk_lbl = MathTex("4", color=RED).next_to(ax.c2p(*j_walk.get_midpoint()), DOWN) \
            .rotate(90 * DEGREES, about_point=ax.c2p(*j_walk.get_midpoint()), axis=RIGHT) \
            .rotate(45 * DEGREES, axis=OUT)

        k_walk_lbl = MathTex("3", color=PURPLE).next_to(ax.c2p(*k_walk.get_midpoint())) \
            .rotate(90 * DEGREES, about_point=ax.c2p(*k_walk.get_midpoint()), axis=RIGHT) \
            .rotate(45 * DEGREES, axis=OUT)
        """
        self.add(ax)
        self.add(vector)
        self.add(i_hat, j_hat, k_hat)
        #self.add_fixed_in_frame_mobjects(i_hat_lbl, j_hat_lbl, k_hat_lbl)

        #self.add(i_walk, j_walk, k_walk)
        #self.add(i_walk_lbl, j_walk_lbl, k_walk_lbl)

class BasisVectorMatrixScene(Scene):
    def construct(self):
        mathtex = MathTex(r"A = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}", color=BLACK)

        i_hat = VGroup(*[mobj for i,mobj in enumerate(mathtex[0]) if i in {4,7,10}])
        j_hat = VGroup(*[mobj for i,mobj in enumerate(mathtex[0]) if i in {5,8,11}])
        k_hat = VGroup(*[mobj for i,mobj in enumerate(mathtex[0]) if i in {6,9,12}])

        remaining = VGroup(*[mobj for i,mobj in enumerate(mathtex[0]) if i not in {4,7,10,5,8,11,6,9,12}])

        for mobj in i_hat:
            mobj.color = GREEN
        for mobj in j_hat:
            mobj.color = RED
        for mobj in k_hat:
            mobj.color = PURPLE

        i_hat_lbl = MathTex(r"\hat{i}", color=GREEN).next_to(i_hat, DOWN)
        j_hat_lbl = MathTex(r"\hat{j}", color=RED).next_to(j_hat, DOWN)
        k_hat_lbl = MathTex(r"\hat{k}", color=PURPLE).next_to(k_hat, DOWN)

        grp = VGroup(remaining, i_hat, j_hat, k_hat, i_hat_lbl, j_hat_lbl, k_hat_lbl) \
            .add_background_rectangle(WHITE, opacity=1)

        #self.add(mathtex, index_labels(mathtex[0]))
        self.add(grp)
        mobj_to_svg(grp, "out.svg", h_padding=1,w_padding=1)


class TransformedThreeDVectorScene(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=75 * DEGREES, theta=-30 * DEGREES)

        ax = ThreeDAxes(axis_config={"color" : BLACK})
        v = array([1,4,3])
        matrix = array([
            [1.5,-1,2.5],
            [0,-2,1],
            [-2,1,-0.5]
        ])

        vector = Arrow3D(start=ax.c2p(0,0,0), end=ax.c2p(*(v)), color=YELLOW)

        i_hat_start = Line(start=ax.c2p(0,0,0), end=ax.c2p(1,0,0), color=GREEN)
        j_hat_start = Line(start=ax.c2p(0,0,0), end=ax.c2p(0,1,0), color=RED)
        k_hat_start = Line(start=ax.c2p(0,0,0), end=ax.c2p(0,0,1), color=PURPLE)

        i_hat_end = Line(start=ax.c2p(0,0,0), end=ax.c2p(*matrix[:,0]), color=GREEN)
        j_hat_end = Line(start=ax.c2p(0,0,0), end=ax.c2p(*matrix[:,1]), color=RED)
        k_hat_end = Line(start=ax.c2p(0,0,0), end=ax.c2p(*matrix[:,2]), color=PURPLE)

        self.add(ax)
        self.add(vector)
        self.add(i_hat_start, j_hat_start, k_hat_start)
        self.wait()
        self.play(
            vector.animate.become(Arrow3D(start=ax.c2p(0,0,0), end=ax.c2p(*(matrix @ v)), color=YELLOW)),
            i_hat_start.animate.become(i_hat_end),
            j_hat_start.animate.become(j_hat_end),
            k_hat_start.animate.become(k_hat_end),
            run_time=3
        )
        self.wait()

class PixelTensorScene(ThreeDScene):

    def construct(self):
        self.set_camera_orientation(phi=75 * DEGREES, theta=-30 * DEGREES)
        grp = VGroup()
        for i in range(0,5):
            for j in range(0,5):
                for k,c in enumerate([BLUE,GREEN,RED]):
                    cube = Cube(side_length=.5, fill_color=c).move_to([i,j,k])
                    grp.add(cube)

        grp.move_to(ORIGIN)
        self.add(grp)

class PixelVideoTensorScene(ThreeDScene):

    def construct(self):

        all_frames = VGroup()
        for z in range (0,3):
            grp = VGroup()
            for i in range(0,5):
                for j in range(0,5):
                    for k,c in enumerate([BLUE,GREEN,RED]):
                        cube = Cube(side_length=.5, fill_color=c).move_to([i,j,k])
                        grp.add(cube)

            all_frames.add(grp)

        all_frames.arrange(IN, buff=2).scale(.5).move_to(ORIGIN)
        self.add(all_frames)

        text = VGroup(*[Tex(f"Frame {i}") for i,frm in enumerate(all_frames)])
        text.arrange(DOWN, buff=2).move_to(ORIGIN).shift(UR*2)
        self.add_fixed_orientation_mobjects(text)

        self.set_camera_orientation(phi=75 * DEGREES, theta=-30 * DEGREES)


if __name__ == "__main__":
    render_scenes(q='l', scene_names=['PixelVideoTensorScene'])
    file_to_base_64(r"/Users/thomasnield/git/3mds-lib/media/images/anaconda_linear_algebra_3/05_PixelVideoTensorScene.png")