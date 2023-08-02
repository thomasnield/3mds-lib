from manim import *
from numpy import array
from threemds.utils import render_scenes, mobj_to_png

config.background_color = "WHITE"
config.frame_rate = 60

class SystemOfEquationsScene(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=75 * DEGREES, theta=-30 * DEGREES)

        ax = ThreeDAxes(
            x_range=[-1, 13, 1],
            y_range=[-1, 6, 1],
            z_range=[-1, 10, 1],
            axis_config={"color": BLACK}
        )
        v_start = array([12, 5, 6])
        v_end = array([7.65, -0.45, -0.25])

        matrix = array([
            [2, 9, -3],
            [1, 2, 7],
            [1, 2, 3]
        ])

        vector_start = Arrow3D(start=ax.c2p(0, 0, 0), end=ax.c2p(*(v_start)), color=YELLOW)
        vector_end = Arrow3D(start=ax.c2p(0, 0, 0), end=ax.c2p(*(v_end)), color=YELLOW)

        vector_label_start = MathTex(r" B = \begin{bmatrix} 12 \\ 5 \\ 6 \end{bmatrix}", color=YELLOW) \
            .scale(.6) \
            .to_edge(DL)

        vector_label_end = MathTex(r"X = \begin{bmatrix} 7.65 \\ -0.45 \\ -0.25 \end{bmatrix}", color=YELLOW) \
            .scale(.6) \
            .to_edge(DL)

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

class NeuralNetworkScene(Scene):

    def connect_layers(self,
                       layer: list[Circle],
                       next_layer: list[Circle],
                       start_index=0,
                       label_directions=None):

        if label_directions is None:
            label_directions = [UP]*100

        i = start_index
        arrows = VGroup()

        for layer_node in layer:
            node_grp = VGroup()
            for next_layer_node in next_layer:
                arrow = Arrow(
                            start= layer_node.get_right(),
                            end= next_layer_node.get_left(),
                            color=BLACK,
                            tip_length=0.2
                        )
                label = MathTex(f"w_{i+1}", color=BLACK) \
                            .scale(.6) \
                            .next_to(arrow.get_midpoint(), label_directions[i-start_index])

                node_grp.add(arrow, label)
                i+=1

            arrows.add(node_grp)
        return arrows

    def construct(self):
        input_layer = VGroup(
            Circle(color=RED, fill_opacity=1),
            Circle(color=GREEN, fill_opacity=1),
            Circle(color=BLUE, fill_opacity=1)
        ).arrange(DOWN)

        hidden_layer = VGroup(
            Circle(color=BLACK),
            Circle(color=BLACK),
            Circle(color=BLACK)
        ).arrange(DOWN)

        output_layer = VGroup(
            Circle(color=BLACK)
        ).arrange(DOWN)

        nn_layers = VGroup(input_layer, hidden_layer, output_layer) \
            .arrange(RIGHT, buff=2)

        self.add(nn_layers)

        input_to_hidden_arrows = self.connect_layers(
            input_layer,
            hidden_layer,
            start_index=0,
            label_directions = [UP, UP, DR] + [UP]*3 + [UP*1.75, UP, UP]
        )

        hidden_to_output_arrows = self.connect_layers(hidden_layer, output_layer, 3)



        # declare matrices
        w_hidden_latex = MathTex(r"W_{hidden} = \begin{bmatrix} w_1 & w_2 & w_3\\"
                                 r"w_4 & w_5 & w_6 \\"
                                 r" w_7 & w_8 & w_9 "
                                 r"\end{bmatrix}", color=BLACK) \
                            .scale(.75) \
                            .to_edge(UR)

        self.add(*[mobj for i,mobj in enumerate(w_hidden_latex[0]) if i not in range(10,28)])

        self.wait()

        # start cycling through weights on input-hidden
        for i,grp in enumerate(input_to_hidden_arrows):
            self.add(grp, *[mobj for j,mobj in enumerate(w_hidden_latex[0]) if j in range(6*i+10,6*i+16)])
            self.wait(3)
            self.remove(grp)

        for grp in hidden_to_output_arrows:
            self.add(grp)
            self.wait(3)
            self.remove(grp)

        self.add(input_to_hidden_arrows, hidden_to_output_arrows)
        self.wait()

if __name__ == "__main__":
    render_scenes(q='l', scene_names=['NeuralNetworkScene'])
