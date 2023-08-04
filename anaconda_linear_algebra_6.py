import numpy as np
from manim import *
from numpy import array
from threemds.utils import render_scenes, mobj_to_png

config.background_color = "WHITE"

light_axis_config = {
               "stroke_width": 4,
               "include_tip": False,
               "line_to_number_buff": SMALL_BUFF,
               "label_direction": DR,
               "font_size": 24,
               "color" : BLACK,
           }

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

class NNNode(VGroup):
    def __init__(self, mathtex_lbl="", label_scale=None, color=BLACK):
        super().__init__()
        self.circle = Circle(color=color, fill_opacity=1 if color!=BLACK else 0)
        self.label = MathTex(mathtex_lbl, color=BLACK)
        if label_scale:
            self.label.scale(label_scale)

        self.label.move_to(self.circle)
        self.add(self.circle, self.label)

class NNConnection(VGroup):

    def __init__(self, i: int, left_node: NNNode, right_node: NNNode, label_direction, *vmobjects, **kwargs):
        super().__init__()

        self.arrow = Arrow(
            start=left_node.get_right(),
            end=right_node.get_left(),
            color=BLACK,
            tip_length=0.2
        )
        self.label = MathTex(f"w_{i + 1}", color=BLACK) \
            .scale(.6) \
            .next_to(self.arrow.get_midpoint(), label_direction)

        self.add(self.arrow, self.label)

class NeuralNetworkScene(MovingCameraScene):

    def connect_layers(self,
                       layer: list[NNNode],
                       next_layer: list[NNNode],
                       start_index=0,
                       label_directions=None):

        if label_directions is None:
            label_directions = [UP]*100

        i = start_index
        layer_grp = VDict()
        layer_grp["groups"] = VGroup()
        layer_grp["arrows"] = VGroup()
        layer_grp["labels"] = VGroup()

        for right_node in next_layer:
            node_grp = VGroup()
            for left_node in layer:
                nn_connection = NNConnection(i,
                                             left_node,
                                             right_node,
                                             label_directions[i - start_index])

                layer_grp["arrows"].add(nn_connection.arrow)
                layer_grp["labels"].add(nn_connection.label)

                node_grp.add(nn_connection)
                i+=1

            layer_grp["groups"].add(node_grp)

        return layer_grp

    def construct(self):

        # INPUT LAYER
        input_layer = VGroup(
            NNNode("x_1",color=RED),
            NNNode("x_2",color=GREEN),
            NNNode("x_3",color=BLUE)
        ).arrange(DOWN)

        # HIDDEN LAYER
        hidden_layer = VGroup(
            NNNode(mathtex_lbl=r"w_1 x_1 + w_2 x_2 \\+ w_3 x_3 + b_1", label_scale=0.6),
            NNNode(mathtex_lbl=r"w_4 x_1 + w_5 x_2 \\+ w_6 x_3 + b_2", label_scale=0.6),
            NNNode(mathtex_lbl=r"w_7 x_1 + w_8 x_2 \\+ w_9 x_3 + b_3", label_scale=0.6)
        ).arrange(DOWN)

        # OUTPUT LAYER
        output_label_text = r"w_4(w_1 x_1 + w_2 x_2 + w_3 x_3 + b_1) \\ " \
                            r"+ w_5(w_4 x_1 + w_5 x_2 + w_6 x_3 + b_2) \\" \
                            r"+ w_6(w_7 x_1 + w_8 x_2 + w_9 x_3 + b_3) \\" \
                            r"+ b_4"

        output_layer = VGroup(
            NNNode(mathtex_lbl=output_label_text, label_scale=.25)
        ).arrange(DOWN)

        ## PACKAGE ENTIRE NEURAL NETWORK AND ADD
        nn_layers = VGroup(input_layer, hidden_layer, output_layer) \
            .arrange(RIGHT, buff=2) \
            .to_edge(LEFT)

        self.add(nn_layers)

        # CONNECT INPUT TO HIDDEN
        input_to_hidden = self.connect_layers(
            input_layer,
            hidden_layer,
            start_index=0,
            label_directions = [UP, UP, DR] + [UP]*3 + [UP*1.75, UP, UP]
        )

        # CONNECT HIDDEN TO OUTPUT
        hidden_to_output = self.connect_layers(hidden_layer, output_layer, 3)

        # DECLARE WEIGHT AND BIAS MATRICES
        w_hidden_latex = MathTex(r"W_{hidden} = \begin{bmatrix} w_1 & w_2 & w_3\\"
                                 r"w_4 & w_5 & w_6 \\"
                                 r" w_7 & w_8 & w_9 "
                                 r"\end{bmatrix}", color=BLACK) \
                            .scale(.75) \
                            .to_edge(UR)

        b_hidden_latex = MathTex(r"B_{hidden} = \begin{bmatrix} b_1 \\ b_2 \\ b_3 \end{bmatrix}", color=BLACK) \
                            .scale(.75) \
                            .next_to(w_hidden_latex, DOWN) \
                            .to_edge(RIGHT)

        b_output_latex = MathTex(r"B_{output} = \begin{bmatrix} b_4 \end{bmatrix}", color=BLACK) \
                            .scale(.75) \
                            .to_edge(DR)

        w_output_latex = MathTex(r"W_{output} = \begin{bmatrix} w_{10} & w_{11} & w_{12} \end{bmatrix}", color=BLACK) \
                            .scale(.75) \
                            .next_to(b_output_latex, UP) \
                            .to_edge(RIGHT)

        # add only brackets and variable name for W_hidden
        self.add(*[mobj for i,mobj in enumerate(w_hidden_latex[0]) if i not in range(10,28)])
        self.add(*[mobj for i,mobj in enumerate(b_hidden_latex[0]) if i not in range(10,16)])

        # remove the node labels containing expressions in hidden layer
        self.remove(*[node.label for node in output_layer],
                    *[node.label for node in hidden_layer]
                    )

        #self.add(b_hidden_latex, index_labels(b_hidden_latex[0], color=RED))
        self.wait()

        # start cycling through weights on input-hidden
        for i,grp in enumerate(input_to_hidden["groups"]):
            self.add(grp,
                     hidden_layer[i].label,
                     *[mobj for j,mobj in enumerate(w_hidden_latex[0]) if j in range(6*i+10,6*i+16)],
                     *[mobj for j, mobj in enumerate(b_hidden_latex[0]) if j in range(2*i+10, 2*i+12)]
                     )
            self.wait(3)
            self.remove(grp)

        # start cycling through weights on hidden-output
        self.add(*[mobj for i,mobj in enumerate(w_output_latex[0]) if i not in range(9,18)])
        for i,grp in enumerate(hidden_to_output["groups"]):
            self.add(grp,
                     *[mobj for j,mobj in enumerate(w_output_latex[0]) if j in range(3*i+9,3*i+18)],
                     output_layer[i].label,
                     b_output_latex
                     )

            self.wait(3)

        # zoom in on output node
        self.camera.frame.save_state()
        self.play(
            FadeOut(w_hidden_latex,b_hidden_latex, w_output_latex, b_output_latex),
            self.camera.frame.animate.scale(.5).move_to(output_layer[0])
        )
        self.wait(3)

        # show light and dark font thresholds
        lbl = output_layer[0].label
        new_lbl1 = MathTex(r"y", r"<", r"0.5", color=BLACK).move_to(output_layer[0])
        new_lbl2 = MathTex(r"y", r"\geq", r"0.5", color=BLACK).move_to(output_layer[0])

        dark_label = Text("DARK", color=BLACK).next_to(output_layer[0], DOWN)
        light_label = Text("LIGHT", color=WHITE).add_background_rectangle(color=BLACK).next_to(output_layer[0], DOWN)
        self.play(FadeIn(dark_label), ReplacementTransform(lbl, new_lbl1))
        self.wait()
        self.play(dark_label.animate.become(light_label),

                  Rotate(new_lbl1[1], 180*DEGREES),

                  FadeIn(Line(color=BLACK,
                              start=new_lbl1[1].get_left(),
                              end=new_lbl2[1].get_right(),
                              stroke_width=3
                              ).next_to(new_lbl2[1], DOWN, buff=0)
                         )
                  )
        self.wait()

        # zoom back out, show all arrows
        self.play(
            Restore(self.camera.frame),
            # FadeIn(w_hidden_latex,b_hidden_latex, w_output_latex, b_output_latex),
            FadeOut(hidden_to_output["labels"]),
            FadeIn(input_to_hidden["arrows"])
        )
        self.wait()

        # Show activation functions
        # ReLU
        relu_ax = Axes(x_range=[-1,3,1], y_range=[-1,3,1], axis_config=light_axis_config)
        relu_plot = relu_ax.plot(lambda x: x if x>0 else 0, color=RED)
        relu_label = Text("ReLU", color=RED).scale(5).next_to(relu_ax, DOWN)

        relu = VDict({"ax": relu_ax, "plot": relu_plot, "label" : relu_label}) \
            .scale(.10) \
            .next_to(hidden_layer, RIGHT,aligned_edge=UP, buff=0.5)

        relu_rect = DashedVMobject(
            Rectangle(height=hidden_layer.height + .25,
                      width=hidden_layer.width + .25,
                      stroke_color=RED,
                      fill_opacity=0).move_to(hidden_layer),
            num_dashes=50
        )
        # sigmoid
        sigmoid_ax = Axes(x_range=[-3, 3, 1], y_range=[-.5, 1, 1], axis_config=light_axis_config)
        sigmoid_plot = sigmoid_ax.plot(lambda x: 1 / (1 + np.exp(-x)), color=RED)
        sigmoid_label = Text("Sigmoid", color=RED).scale(5).next_to(sigmoid_ax, DOWN)

        sigmoid = VDict({"ax": sigmoid_ax, "plot": sigmoid_plot, "label": sigmoid_label}) \
            .scale(.10) \
            .next_to(output_layer, RIGHT, aligned_edge=UP, buff=0.5)

        sigmoid_rect = DashedVMobject(
            Rectangle(height=output_layer.height + .25,
                      width=output_layer.width + .25,
                      stroke_color=RED,
                      fill_opacity=0).move_to(output_layer),
            num_dashes=25
        )
        self.play(FadeIn(relu), FadeIn(relu_rect), FadeIn(sigmoid), FadeIn(sigmoid_rect))
        self.wait()

if __name__ == "__main__":
    render_scenes(q='k', scene_names=['NeuralNetworkScene'])
