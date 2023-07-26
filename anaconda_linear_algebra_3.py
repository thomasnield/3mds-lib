from manim import *
from numpy import array
from threemds.utils import render_scenes


class ThreeDVectorScene(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=75 * DEGREES, theta=-30 * DEGREES)

        ax = ThreeDAxes()
        v = array([1,4,3])
        vector = Arrow3D(start=ax.c2p(0,0,0), end=ax.c2p(*v), color=YELLOW)

        i_hat = Line(start=ax.c2p(0,0,0), end=ax.c2p(1,0,0), color=GREEN)
        j_hat = Line(start=ax.c2p(0,0,0), end=ax.c2p(0,1,0), color=RED)
        k_hat = Line(start=ax.c2p(0,0,0), end=ax.c2p(0,0,1), color=PURPLE)

        i_hat_lbl = MathTex(r"\hat{i}", color=GREEN).next_to(i_hat.get_midpoint(), DL)
        j_hat_lbl = MathTex(r"\hat{j}", color=RED).next_to(j_hat.get_midpoint(), RIGHT + DR *2)
        k_hat_lbl = MathTex(r"\hat{k}", color=PURPLE).next_to(k_hat.get_top(),  UL + UP)

        i_walk = Line(start=ax.c2p(0,0,0), end=ax.c2p(v[0],0,0), color=GREEN)
        j_walk = Line(start=ax.c2p(v[0],0,0), end=ax.c2p(v[0],v[1],0), color=RED)
        k_walk = Line(start=ax.c2p(v[0],v[1],0), end=ax.c2p(v[0],v[1],v[2]), color=PURPLE)

        self.add(ax,vector, i_hat, j_hat, k_hat)
        self.add_fixed_in_frame_mobjects(i_hat_lbl, j_hat_lbl, k_hat_lbl)

if __name__ == "__main__":
    render_scenes(q='l', scene_names=['ThreeDVectorScene'])
