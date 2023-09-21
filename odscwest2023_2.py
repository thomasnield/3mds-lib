import urllib

from manim import *

from threemds.utils import render_scenes
import sympy as sp
import scipy
import numpy as np
import os

TITLE_SCALE=1.4
SUBTITLE_SCALE=1.0
BULLET_BUFF=.75
BULLET_TEX_SCALE=.8

class Die(VGroup):
    def __init__(self, num_dots, *vmobjects, **kwargs):
        super().__init__(*vmobjects, **kwargs)
        highlighted_dots = set()
        if num_dots == 1:
            highlighted_dots = (4,)
        elif num_dots == 2:
            highlighted_dots = (0,8)
        elif num_dots == 3:
            highlighted_dots = (0,4,8)
        elif num_dots == 4:
            highlighted_dots = (0,2,6,8)
        elif num_dots == 5:
            highlighted_dots = (0,2,4,6,8)
        elif num_dots == 6:
            highlighted_dots = (0,1,2,6,7,8)

        for i in range(9): self.add(Circle(.2, color=RED if i in highlighted_dots else None))
        self.arrange_in_grid(3,3).rotate(90*DEGREES)
        self.add(Rectangle(height=self.height * 1.2, width=self.width * 1.2))

class Coin(VGroup):
    def __init__(self, *vmobjects, **kwargs):
        super().__init__(*vmobjects, **kwargs)

        _coin = Circle(radius=.75, color=WHITE)
        dollar = Tex(r"\$", color=RED) \
            .scale_to_fit_height(_coin.height * .6)

        self.add(_coin, dollar)


class ProbabilityTitle(Scene):
    def construct(self):
        title = Title("Probability", color=BLUE).scale(TITLE_SCALE)

        self.play(FadeIn(title))
        self.wait()

        ax = Axes(x_range=(-3,3,1), y_range=(-.1))

        v = Vector([2, 1, 0], color=YELLOW) \
            .put_start_and_end_on(np.c2p(0,0), np.c2p(2,1))

        VGroup(np,v).scale(.7).next_to(title, DOWN, buff=1)

        self.play(Write(np))
        self.wait()
        self.play(GrowArrow(v))
        self.wait()

        self.play(Unwrite(np), Unwrite(v), FadeOut(title))
        self.wait()

class WhatIsProbability(Scene):
    def construct(self):
        title = Title("What is Probability?", color=BLUE)
        tex = VGroup(
                Tex(r"\textbf{Probability} ", "is how strongly we believe an event will happen."),
                Tex(r"Some questions that warrant a probability for an answer:"),
                Tex(r"\begin{itemize}"
                    r"\item How likely is a tornado to happen today?"
                    r"\item What are my chances in winning an election?"
                    r"\item Will my flight be late?"
                    r"\item How likely is my product to be defective?"
                    r"\end{itemize}"
                )
             ).arrange_in_grid(cols=1, cell_alignment=LEFT, buff=BULLET_BUFF) \
            .scale(.8)  \
            .next_to(title.get_corner(DL), DOWN, aligned_edge=LEFT, buff=1)

        tex[0][0].set_color(RED)

        self.play(*[Write(m) for m in (title, tex)], lag_ratio=.5)
        self.wait()

class ProbabilityNotation(Scene):
    def construct(self):

        vt = ValueTracker(0.0)

        tex = MathTex(r"P(X) = ").scale(2)
        p = DecimalNumber(0, num_decimal_places=2).add_updater(lambda m: m.set_value(vt.get_value())).scale(2)

        tex_and_p = VGroup(tex,p).arrange(RIGHT)
        rng = MathTex(r"0 \leq P(x) \leq 1", color=BLUE).scale(1.5).next_to(tex_and_p, DOWN)

        self.play(tex_and_p)
        self.wait()
        self.play(vt.animate.set_value(1.0), run_time=3)
        self.wait()
        self.play(vt.animate.set_value(0.0), run_time=3)
        self.wait()
        self.play(vt.animate.set_value(.5), run_time=1.5)
        self.wait()

        self.play(Write(rng))
        self.wait()

class ThinkingProbability(Scene):
    def construct(self):
        title = Title("Where Does Probability Come From?", color=BLUE)
        tex = VGroup(
            Tex(r"Probabilities can be thought of in two ways:"),
            Tex(r"\begin{itemize}"
                r"\item Probability based on belief"
                r"\item Probability based on data"
                r"\item Will my flight be late?"
                r"\item How likely is my product to be defective?"
                r"\end{itemize}"
                )
        ).arrange_in_grid(cols=1, cell_alignment=LEFT, buff=BULLET_BUFF) \
            .scale(.8) \
            .next_to(title.get_corner(DL), DOWN, aligned_edge=LEFT, buff=1)

        tex[0][0].set_color(RED)

        self.play(*[Write(m) for m in (title, tex)], lag_ratio=.5)
        self.wait()

class JointProbability(Scene):

    def construct(self):
        title = Tex("Joint Probability", color=BLUE).scale(1.3).to_edge(UL)
        tex1 = MathTex(r"P(A \cap B)").scale(2)
        tex2 = MathTex(r"P(A \text{ and } B)").scale(2).next_to(tex1, DOWN)

        self.play(Write(title))
        self.wait()
        self.play(Write(tex1))
        self.wait()
        self.play(ReplacementTransform(tex1.copy(), tex2))
        self.wait()

        tex3 = MathTex(r"P(A \text{ and } B)", r" = P(A) \times P(B)") \
            .scale(1.5)

        self.play(
            FadeOut(tex1),
            TransformMatchingShapes(tex2, tex3[0]),
        )
        self.play(Write(tex3[1]))
        self.wait()

        tex4 = MathTex(r"P(H \text{ and } 6)", r" = \frac{1}{2} \times \frac{1}{6}") \
            .scale(1.5)

        coin = Coin()
        die = Die(6).match_height(coin)

        coin_and_die = VGroup(coin, die) \
            .arrange(RIGHT, buff=1) \
            .to_edge(DOWN)

        self.play(TransformMatchingShapes(tex3, tex4), Write(coin_and_die))
        self.wait()

        answer = MathTex(r"= \frac{1}{12}").scale(1.5).move_to(tex4[1], aligned_edge=LEFT)

        VGroup(answer.generate_target(), tex4[0].generate_target()) \
            .move_to(ORIGIN) \
            .arrange(LEFT)

        self.play(ReplacementTransform(tex4[1], answer))
        self.play(MoveToTarget(answer), MoveToTarget(tex4[0]))
        self.wait()

class UnionProbability(Scene):
    def construct(self):
        title = Tex("Union Probability", color=BLUE).scale(1.3).to_edge(UL)
        tex1 = MathTex(r"P(A \cup B)").scale(2)
        tex2 = MathTex(r"P(A \text{ or } B)").scale(2).next_to(tex1, DOWN)

        self.play(Write(title))
        self.wait()
        self.play(Write(tex1))
        self.wait()
        self.play(ReplacementTransform(tex1.copy(), tex2))
        self.wait()

        tex3 = MathTex(r"P(A \text{ or } B)", r" = P(A) + P(B)", r"- P(A) \times P(B)") \
            .scale(1)

        add_only = tex3[0:2]
        add_only.save_state()
        add_only.move_to(ORIGIN)

        self.play(
            FadeOut(tex1), # get rid of top equation
            TransformMatchingShapes(tex2, add_only[0]),
        )
        self.play(Write(add_only[1]))
        self.wait()

        # bring in joint probability
        self.play(Restore(add_only))
        self.wait()
        self.play(Write(tex3[2]))
        self.wait()
        self.play(Circumscribe(tex3[2]))
        self.wait()

        tex4 = MathTex(r"P(H \text{ or } 6)", r" = \frac{1}{2} + \frac{1}{6} - \frac{1}{2} \times \frac{1}{6}") \
            .scale(1)

        coin = Coin()
        die = Die(6).match_height(coin)

        coin_and_die = VGroup(coin, die) \
            .arrange(RIGHT, buff=1) \
            .to_edge(DOWN)

        self.play(TransformMatchingShapes(tex3, tex4), Write(coin_and_die))
        self.wait()

        answer = MathTex(r"= \frac{7}{12}").scale(1).move_to(tex4[1], aligned_edge=LEFT)

        VGroup(answer.generate_target(), tex4[0].generate_target()) \
            .move_to(ORIGIN) \
            .arrange(LEFT)

        self.play(ReplacementTransform(tex4[1], answer))
        self.play(MoveToTarget(answer), MoveToTarget(tex4[0]))
        self.wait()

class ConditionalProbability(Scene):
    def construct(self):
        pass


if __name__ == "__main__":
    render_scenes(q='l',play=True, scene_names=["UnionProbability"])