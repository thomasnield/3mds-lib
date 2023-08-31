import math

from manim import *
import numpy as np

from threemds.utils import render_scenes

w_hidden = np.array([
    [3.55748018, 8.48639024, 1.59453643],
    [4.2898201,  8.35518251, 1.36713926],
    [3.72074292, 8.13223257, 1.48165939]
])

w_output = np.array([
    [4.27394194, 3.65634072, 2.63047526]
])

b_hidden = np.array([
    [-6.67311751],
    [-6.34084123],
    [-6.10933577]
])

b_output = np.array([
    [-5.46880991]
])

class NNNode(VGroup):
    def __init__(self, mathtex_lbl=MathTex(""),
                 alt_tex_lbl=MathTex(""),
                 alt2_tex_lbl=MathTex(""),
                 label_scale=None,
                 color=WHITE):
        super().__init__()
        self.color=color
        self.circle = Circle(color=color, fill_opacity=1 if color!=WHITE else 0)

        self.alt_tex_lbl = alt_tex_lbl
        self.alt_tex_lbl.add_updater(lambda t: t.move_to(self.circle))

        self.alt2_tex_lbl = alt2_tex_lbl
        self.alt2_tex_lbl.add_updater(lambda t: t.move_to(self.circle))

        self.mathtex_lbl = mathtex_lbl

        if label_scale:
            self.mathtex_lbl.scale(label_scale)
            self.alt_tex_lbl.scale(label_scale)
            self.alt2_tex_lbl.scale(label_scale)

        self.mathtex_lbl.move_to(self.circle)
        self.alt_tex_lbl.move_to(self.circle)
        self.alt2_tex_lbl.move_to(self.circle)

        self.add(self.circle, self.mathtex_lbl)

    def color_x(self):
        for lbl in (self.mathtex_lbl, self.alt_tex_lbl, self.alt2_tex_lbl):
            for x_tex, c in zip(lbl.x_texs, (RED,GREEN,BLUE)):
                x_tex.set_color(c)



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
        skip_flag = False
        # =====================================================================================================
        self.next_section("Declare and initialize", skip_animations=skip_flag)

        # INPUT LAYER
        input_layer = VGroup(
            NNNode(MathTex("x_1"), alt_tex_lbl =MathTex("1.0"), color=RED),
            NNNode(MathTex("x_2"), alt_tex_lbl=MathTex(".34"), color=GREEN),
            NNNode(MathTex("x_3"), alt_tex_lbl=MathTex(".2"), color=BLUE)
        ).arrange(DOWN)

        class HiddenNodeTex(MathTex):
            def __init__(self, *tex_strings, **kwargs):
                super().__init__(*tex_strings, **kwargs)

                self.w_texs = VGroup(*[self[i] for i,t in enumerate(tex_strings) if i in (0, 3, 6)])
                self.b_tex = VGroup(*[self[i] for i,t in enumerate(tex_strings) if i == 9])
                self.x_texs = VGroup(*[self[i] for i,t in enumerate(tex_strings) if i in (1,4,7)])
                self.plus_signs = VGroup(*[self[i] for i,t in enumerate(tex_strings) if i in (2,5,8)])
                self.all_but_x = VGroup(*self.w_texs, *self.b_tex, *self.plus_signs)
                self.all_but_w = VGroup(*self.x_texs, *self.b_tex, *self.plus_signs)

        # HIDDEN LAYER
        hidden_layer = VGroup(
            NNNode(mathtex_lbl=HiddenNodeTex("w_1", "x_1",     r"+\\", "w_2", "x_2",    r"+\\", "w_3", "x_3",  r"+\\",  "b_1"),
                   alt_tex_lbl=HiddenNodeTex("w_1", r"(1.0)", r"+\\",  "w_2", r"(.34)", r"+\\", "w_3", r"(.2)", r"+\\", "b_1"),
                   alt2_tex_lbl=HiddenNodeTex(f"({round(w_hidden[0,0],2)})", r"(1.0)", r"+\\",
                                              f"({round(w_hidden[0,1],2)})",  r"(.34)", r"+\\",
                                              f"({round(w_hidden[0,2],2)})", r"(.2)", r"+\\", "b_1"),
                   label_scale=0.6),

            NNNode(mathtex_lbl=HiddenNodeTex("w_4", "x_1",     r"+\\", "w_5", "x_2",    r"+\\", "w_6", "x_3",   r"+\\", "b_2"),
                   alt_tex_lbl=HiddenNodeTex(r"w_4", r"(1.0)", r"+\\", "w_5", r"(.34)", r"+\\", "w_6", r"(.2)", r"+\\", "b_2"),
                   alt2_tex_lbl=HiddenNodeTex(f"({round(w_hidden[1,0],2)})", r"(1.0)", r"+\\",
                                              f"({round(w_hidden[1,1],2)})",  r"(.34)", r"+\\",
                                              f"({round(w_hidden[1,2],2)})",r"(.2)",  r"+\\", "b_2"),
                   label_scale=0.6),

            NNNode(mathtex_lbl=HiddenNodeTex("w_7", "x_1",    r"+\\", "w_8", "x_2",   r"+\\", "w_9", "x_3",  r"+\\", "b_3"),
                   alt_tex_lbl=HiddenNodeTex("w_7", r"(1.0)", r"+\\", "w_8", r"(.34)", r"+\\", "w_9", r"(.2)", r"+\\", "b_3"),
                   alt2_tex_lbl=HiddenNodeTex(f"({round(w_hidden[2, 0], 2)})", r"(1.0)", r"+\\",
                                              f"({round(w_hidden[2, 1], 2)})", r"(.34)", r"+\\",
                                              f"({round(w_hidden[2, 2], 2)})", r"(.2)", r"+\\", "b_3"),
                   label_scale=0.6)
        ).arrange(DOWN)


        # color-code x1, x2, and x3
        for hidden_node in hidden_layer:
            hidden_node.color_x()

        # OUTPUT LAYER
        output_label_text = MathTex(r"w_{10}(w_1 x_1 + w_2 x_2 + w_3 x_3 + b_1) \\ "
                                    r"+ w_{11}(w_4 x_1 + w_5 x_2 + w_6 x_3 + b_2) \\"
                                    r"+ w_{12}(w_7 x_1 + w_8 x_2 + w_9 x_3 + b_3) \\"
                                    r"+ b_4")

        output_layer = VGroup(
            NNNode(mathtex_lbl=output_label_text, label_scale=.25)
        ).arrange(DOWN)

        # PACKAGE ENTIRE NEURAL NETWORK AND ADD
        nn_layers = VGroup(input_layer, hidden_layer, output_layer) \
            .arrange(RIGHT, buff=2)

        self.add(nn_layers)


        # CONNECT INPUT TO HIDDEN
        input_to_hidden = self.connect_layers(
            input_layer,
            hidden_layer,
            start_index=0
        )

        # CONNECT HIDDEN TO OUTPUT
        hidden_to_output = self.connect_layers(hidden_layer, output_layer, 9)

        # remove the node labels containing expressions in hidden layer
        self.remove(*[node.mathtex_lbl for node in output_layer],
                    *[node.mathtex_lbl for node in hidden_layer]
                    )

        self.wait()

        # =====================================================================================================
        self.next_section("Start cycling through weights and biases", skip_animations=skip_flag)

        # start cycling through weights on input-hidden
        for i,grp in enumerate(input_to_hidden["groups"]):
            self.play(*[FadeIn(mobj) for mobj in [grp,
                     hidden_layer[i].mathtex_lbl,
                     ]])
            self.wait(1)
            self.play(FadeOut(grp))
            self.wait(1)

        # start cycling through weights on hidden-output
        for i,grp in enumerate(hidden_to_output["groups"]):
            self.play(*[FadeIn(mobj) for mobj in [grp,
                     output_layer[i].mathtex_lbl,
                     ]])

            self.wait(3)
        # =====================================================================================================
        self.next_section("Zoom in on output node", skip_animations=skip_flag)

        # zoom in on output node
        self.camera.frame.save_state()
        self.play(
            self.camera.frame.animate.scale(.5).move_to(output_layer[0])
        )
        self.wait(3)

        # show light and dark font thresholds
        lbl = output_layer[0].mathtex_lbl
        lbl_copy = lbl.copy()
        lbl.save_state()

        new_lbl1 = MathTex(r"y", r"<", r"0.5").move_to(output_layer[0])
        new_lbl2 = MathTex(r"y", r"\geq", r"0.5").move_to(output_layer[0])

        light_label = Text("LIGHT", color=WHITE) \
            .add_background_rectangle(color="#FFC5B9") \
            .next_to(output_layer[0], DOWN)

        dark_label = Text("DARK", color=BLACK) \
            .add_background_rectangle(color="#FFC5B9") \
            .next_to(output_layer[0], DOWN)

        self.play(FadeIn(light_label), ReplacementTransform(lbl, new_lbl1))
        self.wait()

        self.play(Rotate(new_lbl1[1], 180*DEGREES), FadeTransform(light_label, dark_label))
        self.play(new_lbl1[1].animate.move_to(new_lbl2[1], aligned_edge=UP))

        self.play(*[TransformMatchingShapes(t1[i],t2[i]) for t1,t2,i in zip(new_lbl1, new_lbl2, range(0,3)) if i != 1],
                  FadeIn(new_lbl2[1])
                  )
        self.remove(new_lbl1[1])

        self.wait()
        self.play(
            FadeOut(dark_label),
            FadeOut(new_lbl1[1]),
            FadeOut(new_lbl1[2]),
            FadeOut(new_lbl2),
            FadeIn(lbl_copy)
        )
        self.add(lbl.restore())
        self.remove(lbl_copy)
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
        relu_ax = Axes(x_range=[-7,15,1], y_range=[-1,15,1],x_length=5,y_length=4, tips=False)
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
        sigmoid_ax = Axes(x_range=[-3, 3, 1], y_range=[-.5, 1, 1],x_length=5,y_length=4, tips=False)
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
        self.play(
            self.camera.frame.animate.move_to(VGroup(nn_layers, relu, sigmoid)).scale(1.05),
            FadeIn(relu),
            FadeIn(relu_rect),
            FadeIn(sigmoid),
            FadeIn(sigmoid_rect))
        self.wait()

        self.camera.frame.save_state()

        # ======================================================================================================
        self.next_section("Create and process salmon pink input", skip_animations=skip_flag)

        self.play(
            self.camera.frame.animate.set(height=VGroup(input_layer, hidden_layer).height * 1.2) \
                .move_to(VGroup(input_layer, hidden_layer))
        )
        self.wait()

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

        class FractionTex(MathTex):
            def __init__(self, *tex_strings, **kwargs):
                super().__init__(*tex_strings, **kwargs)

                self.denominator = VGroup(*[t for i,t in enumerate(self[0]) if i in (3,4,5,6)])
                self.numerator = VGroup(*[t for i,t in enumerate(self[0]) if i in (0,1,2)])

        # present division operation
        salmon_rgb_div = VGroup(FractionTex(r"\frac{255}{255}",color=RED),
                            FractionTex(r"\frac{197}{255}",color=GREEN),
                            FractionTex(r"\frac{185}{255}",color=BLUE)
                            )

        for mobj1,mobj2 in zip(salmon_rgb, salmon_rgb_div):
            mobj2.move_to(mobj1)

        self.play(
            # fade in the denominators
            *[FadeIn(t.denominator) for t in salmon_rgb_div],
            # move rgb values to numerators
            *[FadeTransform(t1, t2.numerator) for t1,t2 in zip(salmon_rgb, salmon_rgb_div)]
        )
        self.wait()

        class FractionTexEqual(MathTex):
            def __init__(self, *tex_strings, **kwargs):
                super().__init__(*tex_strings, **kwargs)
                self.denominator = VGroup(*[t for i,t in enumerate(self[0]) if i in (3,4,5,6)])
                self.numerator = VGroup(*[t for i,t in enumerate(self[0]) if i in (0,1,2)])
                self.fraction = self[0]
                self.equals_sign = self[1]
                self.right_side = self[2]
                self.equals_and_right_side = self[1:]
                self.equals_and_left_side = self[:-1]

        # show result of scaling division
        salmon_rgb_final = VGroup(FractionTexEqual(r"\frac{255}{255}", "=", r"1.0", color=RED),
                            FractionTexEqual(r"\frac{197}{255}", "=", r".77",color=GREEN),
                            FractionTexEqual(r"\frac{185}{255}", "=", r".72",color=BLUE)
                            )

        # line up the group above to the left of the input nodes
        for mobj1,mobj2 in zip(salmon_rgb_div,salmon_rgb_final):
            mobj2.move_to(mobj1, aligned_edge=RIGHT)

        # scoot the fractions over and show equality side
        self.play(*[FadeTransform(t1, t2.fraction) for t1, t2 in zip(salmon_rgb_div, salmon_rgb_final)],
                  *[FadeIn(t.equals_and_right_side) for t in salmon_rgb_final]
                  )

        self.wait()

        # move the scaled values into the input nodes
        for input_node in input_layer:
            input_node.alt_tex_lbl.move_to(input_node)

        self.play(*[
            FadeTransform(rgb_tex.right_side, node.alt_tex_lbl)
            for rgb_tex, node in zip(salmon_rgb_final, input_layer)
        ],
          *[FadeOut(node.mathtex_lbl) for node in input_layer],
          *[FadeOut(t.equals_and_left_side) for t in salmon_rgb_final],
          Restore(self.camera.frame)
        )

        self.wait()

        # =====================================================================================================
        self.next_section("Propagate inputs and weights through the hidden layer", skip_animations=skip_flag)

        # fade out all connections except for first hidden node's
        # and zoom in on the input and hidden layers
        self.play(
            *[FadeOut(c) for c in input_to_hidden["arrows"]],
            *[FadeOut(c) for c in hidden_to_output["arrows"]]
        )
        self.wait()

        # "skate" the values along the arrows
        for i in range(0,3):
            i_arrows: list[Arrow] = [g.arrow for g in input_to_hidden["groups"][i]]
            i_hidden_node: NNNode = hidden_layer[i]

            self.play(*[FadeIn(arrow) for arrow in i_arrows])
            skate_alpha = ValueTracker(.154)

            h_labels = VGroup(*[input_node.alt_tex_lbl.copy() \
                             .set_color(input_node.color) \
                             .add_background_rectangle(BLACK, opacity=.8)
                             .rotate(arrow.get_angle())
                             .add_updater(lambda mobj, arrow=arrow:
                                 mobj.move_to(arrow.point_from_proportion(skate_alpha.get_value()))
                             )

                         for arrow,input_node in zip(i_arrows, input_layer)
                         ])


            self.play(FadeIn(h_labels))
            self.wait()

            # skate the labels by updating the alpha
            # also color the x variables
            self.play(skate_alpha.animate.set_value(1),
                      run_time=3)
            self.wait()

            #  remove the updaters
            for mobj in h_labels:
                mobj.clear_updaters()

            # force update the equations in hidden layer
            i_hidden_node.alt_tex_lbl.move_to(i_hidden_node.circle)

            self.play(
                # Swap out the hidden node equations with partially substituted ones
                FadeTransformPieces(
                    i_hidden_node.mathtex_lbl.all_but_x,
                    i_hidden_node.alt_tex_lbl.all_but_x
                ),
            )
            self.wait()
            self.play(
                FadeOut(i_hidden_node.mathtex_lbl.x_texs),
            )
            self.wait()
            self.play(
                # "jump" the labels into the hidden nodes
                FadeTransformPieces(h_labels, i_hidden_node.alt_tex_lbl.x_texs)
            )
            self.wait()


            # fade in weights on edges
            w_labels =  VGroup(*[mobj.label.copy() for mobj in input_to_hidden["groups"][i]])

            self.play(FadeIn(w_labels))
            self.wait()

            # slide over to reveal weight values
            w_hidden_texs =  VGroup(*[
                MathTex(f"w_{i * 3 + j + 1}", "=", round(w_value, 2)) \
                    .scale(.6) \
                    .next_to(arrow.get_midpoint(), UP * .5) \
                    .rotate(arrow.get_angle(), about_point=arrow.get_midpoint())

                for w_value, arrow, j in zip(w_hidden[i], i_arrows, range(0, 3))
            ])

            self.play(
                TransformMatchingShapes(w_labels, VGroup(*[t[0] for t in w_hidden_texs])),
                FadeIn(VGroup(*[t[1:] for t in w_hidden_texs]))
            )
            self.wait()

            # move weight latex labels to each node
            # by transitioning from alt_tex_lbl to alt2_tex_lbl
            i_hidden_node.alt2_tex_lbl.scale(.8).move_to(i_hidden_node)

            self.play(
                FadeTransform(i_hidden_node.alt_tex_lbl.all_but_w, i_hidden_node.alt2_tex_lbl.all_but_w),
                FadeTransform(VGroup(*[t[-1] for t in w_hidden_texs]), i_hidden_node.alt2_tex_lbl.w_texs),
                FadeOut(i_hidden_node.alt_tex_lbl.w_texs),
                FadeOut(VGroup(*[t[:-1] for t in w_hidden_texs]))
            )

            # replace the bias with is value
            self.wait()
            bias_tex = MathTex(round(b_hidden[i,0],2)) \
                .scale(.6 * .8) \
                .move_to(i_hidden_node.alt2_tex_lbl.b_tex, aligned_edge=RIGHT)

            self.play(
                Circumscribe(i_hidden_node.alt2_tex_lbl.b_tex),
                FadeTransform(i_hidden_node.alt2_tex_lbl.b_tex, bias_tex)
            )
            self.wait()

            # fade out arrows and move onto next hidden node
            self.play(*[FadeOut(arrow) for arrow in i_arrows])
            self.wait()

            # solve the node
            solve_expr = (w_hidden[i] @ (np.array([255,197,1845]).T / 255) + b_hidden[i]).flatten()[0]

            node_solved_tex = MathTex(round(solve_expr, 3)).move_to(i_hidden_node)

            self.play(ReplacementTransform(VGroup(i_hidden_node.alt2_tex_lbl[:-1], bias_tex), node_solved_tex))
            self.wait()

            # jump to ReLU to trace the value
            self.camera.frame.save_state()

            # change line width behavior on camera zoom`
            INITIAL_LINE_WIDTH_MULTIPLE = self.camera.cairo_line_width_multiple
            INITIAL_FRAME_WIDTH = config.frame_width

            def line_scale_down_updater(mobj):
                proportion = self.camera.frame.width / INITIAL_FRAME_WIDTH
                self.camera.cairo_line_width_multiple = INITIAL_LINE_WIDTH_MULTIPLE * proportion

            mobj = Mobject()
            mobj.add_updater(line_scale_down_updater)
            self.add(mobj)

            # create objects for RelU graph
            x_lookup_dot = Dot(relu_ax.c2p(solve_expr, 0), color=YELLOW).scale(.2)
            y_lookup_dot = Dot(relu_ax.c2p(0, solve_expr), color=YELLOW).scale(.2)
            lookup_dot = Dot(relu_ax.c2p(solve_expr, solve_expr), color=YELLOW).scale(.2)

            x_lookup_line = DashedLine(color=YELLOW, start=x_lookup_dot, end=lookup_dot)
            y_lookup_line = DashedLine(color=YELLOW, start=lookup_dot, end=y_lookup_dot)

            # jump to Relu graph
            self.play(
                FadeOut(relu_label),
                LaggedStart(
                    node_solved_tex.animate.scale(.15).next_to(x_lookup_dot, DOWN, buff=.05),
                    self.camera.frame.animate.set(height=relu_ax.height * 1.3).move_to(relu_ax),
                    FadeIn(x_lookup_dot),
                    lag_ratio=.3
                ),
                run_time=3
            )

            relu_output_tex = node_solved_tex.copy().next_to(y_lookup_dot, LEFT, buff=.05)

            self.wait()
            self.play(Write(x_lookup_line))
            self.play(Write(lookup_dot))
            self.play(Write(y_lookup_line))
            self.play(Write(y_lookup_dot), Write(relu_output_tex))
            self.wait()

            self.play(LaggedStart(relu_output_tex.animate.scale(20/3).move_to(i_hidden_node),
                                  Restore(self.camera.frame),
                                  FadeIn(relu_label),
                                  FadeOut(VGroup(lookup_dot, x_lookup_line, x_lookup_dot, y_lookup_line, y_lookup_dot, node_solved_tex)),
                                  lag_ratio=.1))
            self.wait()
            mobj.clear_updaters()

        # restore all arrows in hidden layer
        self.play(FadeIn(input_to_hidden["arrows"]))
        self.wait()

        # ========================================================================================
        self.next_section("Propagate output node, skate incoming values", skip_animations=False)
        self.play(FadeIn(hidden_to_output["arrows"]))
        self.wait()

        hidden_solved = (w_hidden @ (np.array([255,197,1845]) / 255).T + b_hidden.T).flatten()
        skate_alpha = ValueTracker(.20)

        skating_labels = VGroup(*[
            DecimalNumber(v,num_decimal_places=3).scale(.6) \
                             .add_background_rectangle(BLACK, opacity=.8)
                             .rotate(arrow.get_angle())
                             .add_updater(lambda mobj, arrow=arrow:
                                 mobj.move_to(arrow.point_from_proportion(skate_alpha.get_value())),
                                 call_updater=True
                             )
            for v, arrow in zip(hidden_solved, hidden_to_output["arrows"])
        ])

        self.play(
            FadeIn(skating_labels)
        )
        self.wait()
        self.play(
            skate_alpha.animate.set_value(.9), run_time=3
        )
        self.wait()

        # zoom in on the output node
        self.camera.frame.save_state()

        self.play(
            self.camera.frame.animate.move_to(VGroup(skating_labels, output_layer, sigmoid_label, sigmoid)) \
                .set(width=VGroup(skating_labels, output_layer, sigmoid_label, sigmoid).width * 1.1)
        )

        # move the skated values into the node
        self.next_section("Propagate output node, trace incoming values", skip_animations=False)

if __name__ == "__main__":
    render_scenes(q='k', play=True, scene_names=['NeuralNetworkScene'])
