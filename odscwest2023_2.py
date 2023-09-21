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
    def __init__(self, num_dots, dot_color=RED, *vmobjects, **kwargs):
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

        for i in range(9): self.add(Circle(.2, color=dot_color if i in highlighted_dots else None))
        self.arrange_in_grid(3,3).rotate(90*DEGREES)
        self.add(Rectangle(height=self.height * 1.2, width=self.width * 1.2))

class Coin(VGroup):
    def __init__(self, symbol="\$", tex_color=RED, *vmobjects, **kwargs):
        super().__init__(*vmobjects, **kwargs)

        _coin = Circle(radius=.75, color=WHITE)
        _symbol = Tex(symbol, color=tex_color) \
            .scale_to_fit_height(_coin.height * .6)

        self.add(_coin, _symbol)


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

class DieAndCoinSequence(VGroup):
    def __init__(self, *vmobjects, **kwargs):
        super().__init__(*vmobjects, **kwargs)
        self.combos1 = VGroup(
            *[VGroup(Coin(cv, tex_color=BLUE if cv == "T" else RED), Die(dv, dot_color=YELLOW)).arrange(DOWN, buff=.75)
              for cv in ("H",) for dv in (1, 2, 3, 4, 5, 6)]) \
            .arrange(RIGHT, buff=.75) \
            .scale_to_fit_width(6)

        self.combos2 = VGroup(
            *[VGroup(Coin(cv, tex_color=BLUE if cv == "T" else RED), Die(dv, dot_color=YELLOW)).arrange(DOWN, buff=.75)
              for cv in ("T",) for dv in (6, 5, 4, 3, 2, 1)]) \
            .arrange(RIGHT, buff=.75) \
            .scale_to_fit_width(6)

        self.all_combos = VGroup(self.combos1, self.combos2).arrange(RIGHT)
        self.add(self.all_combos)

        self.heads_outcome_brace = Brace(self.combos1, DOWN, color=RED)
        self.heads_outcome_brace_txt = MathTex(r"\frac{6}{12}", color=RED).next_to(self.heads_outcome_brace, DOWN)

        self.six_outcome_brace = Brace(VGroup(self.combos1[-1], self.combos2[0]), DOWN, color=YELLOW).shift(DOWN * .5)
        self.six_outcome_brace_txt = MathTex(r"\frac{2}{12}", color=YELLOW).next_to(self.six_outcome_brace, DOWN)

        self.heads_and_six_outcome_brace = Brace(self.combos1[-1], DOWN, color=ORANGE)
        self.heads_and_six_outcome_brace_txt = MathTex(r"\frac{1}{12}", color=ORANGE).next_to(self.heads_and_six_outcome_brace, DOWN)

        self.joint_box = DashedVMobject(Rectangle(color=PURPLE) \
                                   .stretch_to_fit_height(self.combos1[-1].height * 1.2) \
                                   .stretch_to_fit_width(self.combos1[-1].width * 1.2) \
                                   .move_to(self.combos1[-1])
                                   )

        self.add(self.heads_outcome_brace, self.heads_outcome_brace_txt, self.six_outcome_brace, self.six_outcome_brace_txt, self.joint_box)


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

class JointProbabilityExplanation(Scene):
    def construct(self):
        top_tex =  MathTex(r"P(H \text{ and } 6)", "=", r" \frac{1}{2} \times \frac{1}{6}").to_edge(UP)
        top_tex[0][2].set_color(RED)
        top_tex[0][6].set_color(YELLOW)

        seq = DieAndCoinSequence()

        self.play(Write(top_tex))
        self.wait()

        self.play(Write(seq.all_combos))
        self.wait()
        self.play(Write(seq.heads_and_six_outcome_brace))
        self.play(Write(seq.heads_and_six_outcome_brace_txt))
        self.wait()
        self.play(seq.heads_and_six_outcome_brace_txt.animate.move_to(top_tex[-1], aligned_edge=LEFT),
                  FadeOut(top_tex[-1]),
                  FadeOut(seq.heads_and_six_outcome_brace)
                  )
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

class UnionProbabilityExplanation(Scene):
    def __init__(self):
        super().__init__()

        union_tex = MathTex(r"P(H \text{ or } 6) = ", r"\frac{6}{12}", "+", r"\frac{2}{12}", r"-", r"\frac{6}{12}", r"\times", r"\frac{2}{12}").to_edge(UP)
        sequence = DieAndCoinSequence()
        self.add(union_tex, sequence)


class ConditionalProbability(Scene):
    def construct(self):
        title = Tex("Conditional Probability", color=BLUE).scale(1.3).to_edge(UL)
        tex1 = MathTex(r"P(A|B)").scale(2)
        tex2 = MathTex(r"P(A \text{ given } B)").scale(2).next_to(tex1, DOWN)

        self.play(Write(title))
        self.wait()
        self.play(Write(tex1))
        self.wait()
        self.play(ReplacementTransform(tex1.copy(), tex2))
        self.wait()

        p_rain = MathTex(r"P(", r"\text{flood}", r") = .05").scale(1.3).move_to(tex1)
        p_rain[1].set_color(RED)

        p_rain_given_flood = MathTex(r"P(", r"\text{flood}", r"\text{ given }", r"\text{rain}", r") = .80").scale(1.3).next_to(p_rain, DOWN)
        p_rain_given_flood[1].set_color(RED)
        p_rain_given_flood[3].set_color(BLUE)

        self.play(ReplacementTransform(tex1, p_rain), ReplacementTransform(tex2, p_rain_given_flood))
        self.wait()

class BayesTheorem(Scene):
    def construct(self):
        title = Tex("Bayes Theorem", color=BLUE).scale(1.3).to_edge(UL)
        self.play(Write(title))
        self.wait()

        bt_tex = MathTex(r"P(A|B) = \frac{P(B|A) \times P(A)}{P(B)}").scale(1.3)
        self.play(Write(bt_tex))
        self.wait()

class BayesTheoremTex(MathTex):
    def __init__(self, a:str, b: str,  **kwargs):

        tex_strings = []
        p_a = r"P(\text{" + a + r"})"
        p_b = r"P(\text{" + b + "})"
        p_a_b = r"P(\text{" + a + r"}|\text{" + b + r"})"
        p_b_a = r"P(\text{" + b + r"}|\text{" + a + r"})"

        tex =  p_a_b + r" = \frac{" + p_b_a  + r" \times " + p_a + "}{" + p_b + "}"
        print(tex)
        super().__init__(tex, **kwargs)

        global i
        i = 2

        def incr(j):
            global i
            if type(j) == str:
                i += len(j)
            else:
                i += j
            return i

        self.a1 = self[0][i:incr(a)] # capture A
        incr(1)
        self.b1 = self[0][i:incr(b)] # capture b
        self.a_given_b = self[0][0:i+1]
        incr(4)
        self.b2 = self[0][i:incr(b)] # capture b
        incr(1)
        self.a2 = self[0][i:incr(a)] # capture a
        self.b_given_a = self[0][i-len(a)-len(b)-3:i+1]
        incr(4)
        self.a3 = self[0][i:incr(a)] # capture a
        self.p_a = self[0][i-len(a)-2:i+1]
        incr(4)
        self.b3 = self[0][i:incr(b)] # capture b
        self.p_b = self[0][-len(b)-3:]

        VGroup(self.a1, self.a2, self.a3).set_color(RED)
        VGroup(self.b1, self.b2, self.b3).set_color(BLUE)


class VideoGameHomicidalExample1(Scene):
    def construct(self):
        self.add(Tex("Bayes Theorem", color=BLUE).scale(1.3).to_edge(UL))

        p_homicidal_gamer = MathTex(r"P(", r"\text{homicidal}", r"|", r"\text{gamer}", r")", "=", ".85").scale(1.3)
        p_homicidal_gamer[1].set_color(RED)
        p_homicidal_gamer[3].set_color(BLUE)

        self.play(Write(p_homicidal_gamer))
        self.wait()

        p_gamer_homicidal = MathTex(r"P(", r"\text{gamer}", r"|", r"\text{homicidal}", r")", "=", r"\text{ ? }").scale(1.3)
        p_gamer_homicidal[1].set_color(BLUE)
        p_gamer_homicidal[3].set_color(RED)

        VGroup(p_homicidal_gamer.generate_target(), p_gamer_homicidal).arrange(DOWN, buff=.75)

        self.play(MoveToTarget(p_homicidal_gamer), Write(p_gamer_homicidal))
        self.wait()

class VideoGameHomicidalExample2(Scene):
    def construct(self):

        self.add(Tex("Bayes Theorem", color=BLUE).scale(1.3).to_edge(UL))

        bt1 = BayesTheoremTex("A", "B")
        self.play(Write(bt1))
        self.wait()

        bt2 = BayesTheoremTex("Homicidal", "Gamer")
        self.play(ReplacementTransform(bt1, bt2))
        self.wait()

        p_solve = MathTex(r" = \frac{.85 \times .0001 }{.19}")
        p_solve[0][5:10].set_color(RED)
        p_solve[0][12:15].set_color(BLUE)

        a_given_b = bt2.a_given_b.copy()
        self.add(a_given_b)
        VGroup(a_given_b.generate_target(), p_solve).arrange(RIGHT)
        self.play(FadeOut(bt2))

        self.play(MoveToTarget(a_given_b), Write(p_solve))
        self.wait()

        p_solved = MathTex("= .0004").next_to(p_solve, DOWN, buff=.75, aligned_edge=LEFT)
        self.play(Write(p_solved))
        self.wait()
        self.play(Circumscribe(p_solved))
        self.wait()




if __name__ == "__main__":
    render_scenes(q='l',play=True, scene_names=["VideoGameHomicidalExample2"])