import math

from manim import *
import numpy as np
import sympy as sp

from threemds.utils import render_scenes


class NNNode(VGroup):
    def __init__(self, mathtex_lbl="",
                 alt_tex_lbl="",
                 alt2_tex_lbl="",
                 label_scale=None,
                 color=WHITE):
        super().__init__()
        self.color=color
        self.circle = Circle(color=color, fill_opacity=1 if color!=WHITE else 0)

        self.alt_tex_lbl = MathTex(alt_tex_lbl)
        self.alt_tex_lbl.add_updater(lambda t: t.move_to(self.circle))

        self.alt2_tex_lbl = MathTex(alt2_tex_lbl)
        self.alt2_tex_lbl.add_updater(lambda t: t.move_to(self.circle))

        self.mathtex_lbl = MathTex(mathtex_lbl)

        if label_scale:
            self.mathtex_lbl.scale(label_scale)
            self.alt_tex_lbl.scale(label_scale)
            self.alt2_tex_lbl.scale(label_scale)

        self.mathtex_lbl.move_to(self.circle)
        self.alt_tex_lbl.move_to(self.circle)
        self.alt2_tex_lbl.move_to(self.circle)

        self.add(self.circle, self.mathtex_lbl)


class NNConnection(VGroup):

    def __init__(self, i: int, left_node: NNNode, right_node: NNNode, right_node_connect_angle: float):
        super().__init__()

        self.arrow = Arrow(
            start=left_node.get_right(),
            end= right_node.circle.point_at_angle(math.pi + right_node_connect_angle),
            tip_length=0.2,
            buff=.05
        )
        self.label = MathTex("w_{" + str(i + 1) + "}") \
            .scale(.6) \
            .next_to(self.arrow.get_midpoint(), UP*.5) \
            .rotate(self.arrow.get_angle(), about_point=self.arrow.get_midpoint())

        self.add(self.arrow, self.label)

class NeuralNetworkScene(MovingCameraScene):

    def connect_layers(self,
                       layer: list[NNNode],
                       next_layer: list[NNNode],
                       start_index=0):


        i = start_index
        layer_grp = VDict()
        layer_grp["groups"] = VGroup()
        layer_grp["arrows"] = VGroup()
        layer_grp["labels"] = VGroup()

        for right_node in next_layer:
            node_grp = VGroup()
            for left_node in layer:
                connect_angle = Line(left_node.get_center(), right_node.get_center()).get_angle()

                nn_connection = NNConnection(i,
                                             left_node,
                                             right_node,
                                             connect_angle
                                             )

                layer_grp["arrows"].add(nn_connection.arrow)
                layer_grp["labels"].add(nn_connection.label)

                node_grp.add(nn_connection)
                i+=1

            layer_grp["groups"].add(node_grp)

        return layer_grp

    def construct(self):
        skip_flag = True
        # =====================================================================================================
        self.next_section("Declare and initialize", skip_animations=skip_flag)

        # INPUT LAYER
        input_layer = VGroup(
            NNNode("x_1", alt_tex_lbl="1.0", color=RED),
            NNNode("x_2", alt_tex_lbl=".34", color=GREEN),
            NNNode("x_3", alt_tex_lbl=".2", color=BLUE)
        ).arrange(DOWN)

        # HIDDEN LAYER
        hidden_layer = VGroup(
            NNNode(mathtex_lbl=r"w_1 x_1 +\\ w_2 x_2 +\\ w_3 x_3 +\\ b_1",
                   alt_tex_lbl=r"w_1 (1.0) +\\ w_2 (.34) +\\ w_3 (.2) +\\ b_1",
                   label_scale=0.6),
            NNNode(mathtex_lbl=r"w_4 x_1 +\\ w_5 x_2 +\\ w_6 x_3 +\\ b_2",
                   alt_tex_lbl=r"w_4 (1.0) +\\ w_5 (.34) +\\ w_6 (.2) +\\ b_1",
                   label_scale=0.6),
            NNNode(mathtex_lbl=r"w_7 x_1 +\\ w_8 x_2 +\\ w_9 x_3 +\\ b_3",
                   alt_tex_lbl=r"w_7 (1.0) +\\ w_8 (.34) +\\ w_9 (.2) +\\ b_1",
                   label_scale=0.6)
        ).arrange(DOWN)

        # OUTPUT LAYER
        output_label_text = r"w_{10}(w_1 x_1 + w_2 x_2 + w_3 x_3 + b_1) \\ " \
                            r"+ w_{11}(w_4 x_1 + w_5 x_2 + w_6 x_3 + b_2) \\" \
                            r"+ w_{12}(w_7 x_1 + w_8 x_2 + w_9 x_3 + b_3) \\" \
                            r"+ b_4"

        output_layer = VGroup(
            NNNode(mathtex_lbl=output_label_text, label_scale=.25)
        ).arrange(DOWN)

        # PACKAGE ENTIRE NEURAL NETWORK AND ADD
        nn_layers = VGroup(input_layer, hidden_layer, output_layer) \
            .arrange(RIGHT, buff=2) \
            .to_edge(LEFT)

        self.add(nn_layers)

        # CONNECT INPUT TO HIDDEN
        input_to_hidden = self.connect_layers(
            input_layer,
            hidden_layer,
            start_index=0
        )

        # CONNECT HIDDEN TO OUTPUT
        hidden_to_output = self.connect_layers(hidden_layer, output_layer, 9)

        # DECLARE WEIGHT AND BIAS MATRICES
        w_hidden_latex = MathTex(r"W_{hidden} = \begin{bmatrix} w_1 & w_2 & w_3\\"
                                 r"w_4 & w_5 & w_6 \\"
                                 r" w_7 & w_8 & w_9 "
                                 r"\end{bmatrix}") \
                            .scale(.75) \
                            .to_edge(UR)

        b_hidden_latex = MathTex(r"B_{hidden} = \begin{bmatrix} b_1 \\ b_2 \\ b_3 \end{bmatrix}") \
                            .scale(.75) \
                            .next_to(w_hidden_latex, DOWN) \
                            .to_edge(RIGHT)

        b_output_latex = MathTex(r"B_{output} = \begin{bmatrix} b_4 \end{bmatrix}") \
                            .scale(.75) \
                            .to_edge(DR)

        w_output_latex = MathTex(r"W_{output} = \begin{bmatrix} w_{10} & w_{11} & w_{12} \end{bmatrix}") \
                            .scale(.75) \
                            .next_to(b_output_latex, UP) \
                            .to_edge(RIGHT)

        # add only brackets and variable name for W_hidden
        self.add(*[mobj for i,mobj in enumerate(w_hidden_latex[0]) if i not in range(10,28)])
        self.add(*[mobj for i,mobj in enumerate(b_hidden_latex[0]) if i not in range(10,16)])

        # remove the node labels containing expressions in hidden layer
        self.remove(*[node.mathtex_lbl for node in output_layer],
                    *[node.mathtex_lbl for node in hidden_layer]
                    )

        #self.add(b_hidden_latex, index_labels(b_hidden_latex[0], color=RED))
        self.wait()

        # =====================================================================================================
        self.next_section("Start cycling through weights and biases", skip_animations=skip_flag)

        # start cycling through weights on input-hidden
        for i,grp in enumerate(input_to_hidden["groups"]):
            self.play(*[FadeIn(mobj) for mobj in [grp,
                     hidden_layer[i].mathtex_lbl,
                     *[mobj for j,mobj in enumerate(w_hidden_latex[0]) if j in range(6*i+10,6*i+16)],
                     *[mobj for j, mobj in enumerate(b_hidden_latex[0]) if j in range(2*i+10, 2*i+12)]
                     ]])
            self.wait(3)
            self.play(FadeOut(grp))

        # start cycling through weights on hidden-output
        self.add(*[mobj for i,mobj in enumerate(w_output_latex[0]) if i not in range(9,18)])
        for i,grp in enumerate(hidden_to_output["groups"]):
            self.play(*[FadeIn(mobj) for mobj in [grp,
                     *[mobj for j,mobj in enumerate(w_output_latex[0]) if j in range(3*i+9,3*i+18)],
                     output_layer[i].mathtex_lbl,
                     b_output_latex
                     ]])

            self.wait(3)
        # =====================================================================================================
        self.next_section("Zoom in on output node", skip_animations=skip_flag)

        # zoom in on output node
        self.camera.frame.save_state()
        self.play(
            FadeOut(w_hidden_latex,b_hidden_latex, w_output_latex, b_output_latex),
            self.camera.frame.animate.scale(.5).move_to(output_layer[0])
        )
        self.wait(3)

        # show light and dark font thresholds
        lbl = output_layer[0].mathtex_lbl
        new_lbl1 = MathTex(r"y", r"<", r"0.5").move_to(output_layer[0])
        new_lbl2 = MathTex(r"y", r"\geq", r"0.5").move_to(output_layer[0])

        light_label = Text("LIGHT", color=WHITE) \
            .next_to(output_layer[0], DOWN)

        dark_label = Text("DARK") \
            .next_to(output_layer[0], DOWN)

        self.play(FadeIn(light_label), ReplacementTransform(lbl, new_lbl1))
        self.wait()

        self.play(light_label.animate.become(dark_label),
                  ReplacementTransform(new_lbl1[1], new_lbl2[1])
                  )
        self.wait()

        # =====================================================================================================
        # zoom back out, show all arrows and the activation functions
        self.next_section("Zoom back out, show activation functions", skip_animations=skip_flag)

        self.play(
            Restore(self.camera.frame),
            FadeOut(hidden_to_output["labels"]),
            FadeIn(input_to_hidden["arrows"])
        )
        self.wait()

        # ReLU
        relu_ax = Axes(x_range=[-1,3,1], y_range=[-1,3,1],x_length=4,y_length=4)
        relu_plot = relu_ax.plot(lambda x: x if x>0 else 0, color=YELLOW)
        relu_label = Text("ReLU", color=RED).scale(4).next_to(relu_ax, DOWN)

        relu = VDict({"ax": relu_ax, "plot": relu_plot, "label" : relu_label}) \
            .scale(.2) \
            .next_to(hidden_layer, RIGHT,aligned_edge=UP, buff=0.5)

        relu_rect = DashedVMobject(
            Rectangle(height=hidden_layer.height + .25,
                      width=hidden_layer.width + .25,
                      stroke_color=RED,
                      fill_opacity=0).move_to(hidden_layer),
            num_dashes=50
        )
        # sigmoid
        sigmoid_ax = Axes(x_range=[-3, 3, 1], y_range=[-.5, 1, 1],x_length=4,y_length=4)
        sigmoid_plot = sigmoid_ax.plot(lambda x: 1 / (1 + np.exp(-x)), color=YELLOW)
        sigmoid_label = Text("Sigmoid", color=RED).scale(5).next_to(sigmoid_ax, DOWN)

        sigmoid = VDict({"ax": sigmoid_ax, "plot": sigmoid_plot, "label": sigmoid_label}) \
            .scale(.2) \
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

        # =====================================================================================================
        self.next_section("Shift camera to the right, zoom out slightly", skip_animations=skip_flag)

        self.play(
            self.camera.frame.animate.move_to(nn_layers).scale(1.2)
        )
        self.wait()
        # =====================================================================================================
        self.next_section("Create and process salmon pink input", skip_animations=skip_flag)

        salmon_tex = VGroup(Tex("Light"), Tex("Salmon")).arrange(DOWN) \
            .next_to(nn_layers, LEFT, buff=1)

        salmon_box = Rectangle(color="#FFC5B9",
                                  fill_opacity=1.0,
                                  width=salmon_tex.width * 1.5,
                                  height=salmon_tex.height * 2
                                  ).move_to(salmon_tex)

        input_box = VGroup(salmon_box, salmon_tex)
        self.play(FadeIn(input_box))
        self.wait()

        # declare the salmon color rgb values
        salmon_rgb = VGroup(MathTex("255",color=RED),
                            MathTex("197",color=GREEN),
                            MathTex("185",color=BLUE)
                            )

        # align them to input nodes
        for mobj, node in zip(salmon_rgb, input_layer):
            mobj.move_to(node).shift(LEFT*2)

        self.play(ReplacementTransform(input_box, salmon_rgb))
        self.wait()

        # present division operation
        salmon_rgb_div = VGroup(MathTex(r"\frac{255}{255}",color=RED),
                            MathTex(r"\frac{197}{255}",color=GREEN),
                            MathTex(r"\frac{185}{255}",color=BLUE)
                            )

        for mobj1,mobj2 in zip(salmon_rgb, salmon_rgb_div):
            mobj2.move_to(mobj1)

        self.play(
            # fade in the denominators
            *[FadeIn(mobj) for mobj in
              (salmon_rgb_div[0][0][3:7], salmon_rgb_div[1][0][3:7], salmon_rgb_div[2][0][3:7])
              ],
            # move rgb values to numerators
            *[FadeTransform(mobj1,mobj2) for mobj1,mobj2 in zip(salmon_rgb,
                (salmon_rgb_div[0][0][:3], salmon_rgb_div[1][0][:3], salmon_rgb_div[2][0][:3])
            )]
        )
        self.wait()

        # show result of scaling division
        salmon_rgb_final = VGroup(MathTex(r"\frac{255}{255}", "=", r"1.0", color=RED),
                            MathTex(r"\frac{197}{255}", "=", r".77",color=GREEN),
                            MathTex(r"\frac{185}{255}", "=", r".72",color=BLUE)
                            )

        for mobj1,mobj2 in zip(salmon_rgb_div,salmon_rgb_final):
            mobj2.move_to(mobj1, aligned_edge=RIGHT)

        # scoot the fractions over and show equality side
        self.play(*[FadeTransform(mobj1, mobj2[0]) for mobj1, mobj2 in zip(salmon_rgb_div, salmon_rgb_final)],
                  *[FadeIn(mobj[1:]) for mobj in salmon_rgb_final]
                  )

        self.wait()

        # move the scaled values into the input nodes
        self.play(*[
            ReplacementTransform(rgb_tex[2], node.alt_tex_lbl) for rgb_tex, node in zip(salmon_rgb_final, input_layer)
        ],
                  *[FadeOut(node.mathtex_lbl) for node in input_layer],
                  *[FadeOut(mobj[:2]) for mobj in salmon_rgb_final]
                  )

        self.wait()

        # =====================================================================================================
        self.next_section("Propagate inputs through the hidden layer", skip_animations=False)

        # fade out all connections except for first hidden node's
        # and zoom in on the input and hidden layers
        self.play(
            *[FadeOut(c) for c in input_to_hidden["groups"][1:]],
            *[FadeOut(c) for c in hidden_to_output["groups"]]
        )
        self.wait()
        self.play(
            self.camera.frame.animate.set(height=VGroup(input_layer, hidden_layer).height * 1.2) \
                .move_to(VGroup(input_layer, hidden_layer))
        )
        # "skate" the values along the arrows

        skate_alpha = ValueTracker(.154)


        h_labels = VGroup(*[node.alt_tex_lbl.copy() \
                         .set_color(node.color) \
                         .add_background_rectangle(BLACK, opacity=.8)
                         .rotate(arrow.get_angle())
                         .add_updater(lambda mobj, arrow=arrow:
                             mobj.move_to(arrow.point_from_proportion(skate_alpha.get_value()))
                         )

                     for arrow,node in zip(input_to_hidden["arrows"], input_layer)
                     ])


        self.play(FadeIn(h_labels))
        self.wait()

        # skate the labels by updating the alpha
        # also color the x variables
        self.play(skate_alpha.animate.set_value(1),
                  *[mobj.animate.set_color(c)
                    for n in hidden_layer
                    for mobj,c in
                    zip([n.mathtex_lbl[0][2:4], n.mathtex_lbl[0][7:9], n.mathtex_lbl[0][12:14]],
                        (RED,GREEN,BLUE))
                    ],
                  run_time=3)
        self.wait()

        #  remove the updaters
        for mobj in h_labels:
            mobj.clear_updaters()

        # force update the equations in hidden layer
        for n in hidden_layer:
            n.alt_tex_lbl.move_to(n.circle)

        self.play(
            # Swap out the hidden node equations with partially substituted ones
            *[FadeTransform(
                VGroup(*[t for i,t in enumerate(hidden_layer[0].mathtex_lbl[0]) if i not in (2,3,7,8,12,13)]),
                VGroup(*[t for i,t in enumerate(hidden_layer[0].alt_tex_lbl[0]) if i not in (3,4,5,11,12,13,19,20)]),
            )],
            *[FadeOut(t) for i,t in enumerate(hidden_layer[0].mathtex_lbl[0]) if i in (2,3,7,8,12,13)],

            # "jump" the labels into the hidden nodes
            *[FadeTransform(t,n)
                for t,n in
                zip(
                    h_labels,
                    (
                        VGroup(*[t for i,t in enumerate(hidden_layer[0].alt_tex_lbl[0]) if i in (3,4,5)]).set_color(RED),
                        VGroup(*[t for i, t in enumerate(hidden_layer[0].alt_tex_lbl[0]) if i in (11,12,13)]).set_color(GREEN),
                        VGroup(*[t for i, t in enumerate(hidden_layer[0].alt_tex_lbl[0]) if i in (19,20)]).set_color(BLUE)
                    )
                )
            ]
        )
        self.wait()


        # =====================================================================================================
        # move camera to the right
        self.next_section("Zoom into ReLU", skip_animations=skip_flag)

        # zoom into ReLU
        INITIAL_FRAME_WIDTH_MULTIPLE = self.camera.cairo_line_width_multiple
        INITIAL_FRAME_WIDTH = config.frame_width

        def updater_camera(mobj):
            proportion = self.camera.frame.width / INITIAL_FRAME_WIDTH
            self.camera.cairo_line_width_multiple = INITIAL_FRAME_WIDTH_MULTIPLE * proportion

        relu.add_updater(updater_camera)

        self.play(
            self.camera.frame.animate.set(height=relu["ax"].height * 1.2).move_to(relu["ax"])
        )
        self.wait()

if __name__ == "__main__":
    render_scenes(q='l', scene_names=['NeuralNetworkScene'])
