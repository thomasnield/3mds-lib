from manim import *
import numpy as np
import pandas as pd
import math

from threemds.utils import render_scenes, file_to_base_64, mobj_to_svg, mobj_to_png
import sympy as sp


class PX(Scene):
    def construct(self):
        tex = MathTex("0 <= P(X) <= 1",
                      r"P(X) + P(\text{not } X) = 1",
                      r"P(A) + P(B) + P(C) = 1") \
            .arrange(DOWN, buff=.75)

        mobj_to_svg(tex, 'out.svg')


class LogisticRegression(Scene):
    def construct(self):
        ax = Axes(x_range=(-1,10,1), y_range=(-.5,1.1, .5))
        plt = ax.plot(lambda x: 1.0 / (1.0 + math.exp(-(-2.823 + .62 * x))), color=WHITE)
        pts = [Dot(ax.c2p(p.x,p.y), color=RED if p.y == 0.0 else BLUE) for p in pd.read_csv("https://tinyurl.com/y2cocoo7").itertuples()]

        grp = VGroup(ax, plt, *pts)

        mobj_to_svg(grp, 'out.svg')

        self.add(grp)

class Odds(Scene):
    def construct(self):
        tex1 = MathTex(
            r"O(X) = \frac{P(X)}{1 - P(X)}",
            r"P(X) = \frac{O(X)}{1 + O(X)}"
            ).arrange(DOWN, buff=.75)

        tex2 = MathTex(
            r"P(X) = .70",
            r"O(X) &= \frac{.70}{1 - .70} \\ &= 2.\overline{333}"
        ).arrange(DOWN, buff=.75)

        #grp = VGroup(tex1, tex2).arrange(DOWN, buff=2)

        mobj_to_svg(tex2, 'out.svg')

class LogOdds(Scene):
    def construct(self):
        tex1 = MathTex(
            r"O(X) = 2.\overline{333}",
            f"log(O(X)) = {round(math.log(.7/(.3)),3)}"
            ).arrange(DOWN, buff=.75)

        #grp = VGroup(tex1, tex2).arrange(DOWN, buff=2)

        mobj_to_svg(tex1, 'out.svg')


class LogisticRegressionLogOdds(Scene):
    def construct(self):
        ax = Axes(x_range=(-1,10,1), y_range=(-.5,1.1, .5))
        plt1 = ax.plot(lambda x: 1.0 / (1.0 + math.exp(-(-2.823 + .62 * x))), color=WHITE)
        plt2 = ax.plot(lambda x: -2.823 + .62 * x, color=GRAY)
        pts = [Dot(ax.c2p(p.x,p.y), color=RED if p.y == 0.0 else BLUE) for p in pd.read_csv("https://tinyurl.com/y2cocoo7").itertuples()]

        grp = VGroup(ax, plt1, plt2, *pts)

        mobj_to_svg(grp, 'out.svg')

        self.add(grp)

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


        self.heads_and_six_outcome_brace_top = Brace(self.combos1[-1], UP, color=PURPLE)
        self.heads_and_six_outcome_brace_txt_top = MathTex(r"\frac{1}{12}", color=PURPLE) \
            .next_to(self.heads_and_six_outcome_brace_top, UP)


        self.joint_box = DashedVMobject(Rectangle(color=PURPLE) \
                                   .stretch_to_fit_height(self.combos1[-1].height * 1.2) \
                                   .stretch_to_fit_width(self.combos1[-1].width * 1.2) \
                                   .move_to(self.combos1[-1])
                                   )

        self.add(self.joint_box)


class JointProbabilityNotation(Scene):
    def construct(self):
        grp = VGroup(
            MathTex(r"P(A \cap B)"),
            MathTex(r"P(A \text{ and } B)")
        ).arrange(DOWN, buff=.75)

        mobj_to_svg(grp)

class JointProbabilityCalc(Scene):
    def construct(self):
        top_tex =  MathTex(r"P(H \text{ and } 6)", "&=", r" \frac{1}{2} \times \frac{1}{6}", r"\\ &= \frac{1}{12}" )
        top_tex[0][2].set_color(RED)
        top_tex[0][6].set_color(YELLOW)
        seq = DieAndCoinSequence()

        mobj_to_svg(top_tex)

class LogarithmicAddition(Scene):
    def construct(self):

        raw_code = """import math
p = 1

# probability of heads 10 times in a row 
for i in range(10):
    p *= .5

print(p) # 9.332636185032189e-302

# using logarithmic addition 
p = 0
for i in range(10):
    p += math.log(.5)

print(math.exp(p)) # 9.332636185154842e-302"""

        code = Code(code=raw_code, language="Python", font="Monospace", style="monokai", background="window")

        mobj_to_svg(code)

class UnionProbabilityNotation(Scene):
    def construct(self):
        tex1 = MathTex(r"P(A \cup B)", r"P(A \text{ or } B)").arrange(DOWN, buff=.75)
        mobj_to_svg(tex1)

class UnionProbabilityFormula(Scene):
    def construct(self):
        tex1 = MathTex(r"P(A \text{ or } B) = P(A) + P(B) - P(A) \times P(B)",
                       r"P(H \text{ or } 6) &= \frac{1}{2} + \frac{1}{6} - \frac{1}{2} \times \frac{1}{6} \\ \\ &= \frac{7}{12}"
                       ).arrange(DOWN, buff=.75)
        mobj_to_svg(tex1)

class UnionProbabilityExplanation(Scene):
    def construct(self):
        seq = DieAndCoinSequence()

        grp = VGroup(seq,
                     seq.heads_outcome_brace,
                     seq.heads_outcome_brace_txt,
                     seq.six_outcome_brace,
                     seq.six_outcome_brace_txt,
                     seq.joint_box)

        grp2 = VGroup(seq,
                      seq.heads_outcome_brace,
                      seq.heads_outcome_brace_txt,
                      seq.six_outcome_brace,
                      seq.six_outcome_brace_txt,
                      seq.heads_and_six_outcome_brace_top,
                      seq.heads_and_six_outcome_brace_txt_top,
                      seq.joint_box)


        mobj_to_svg(grp2,h_padding=3)

        self.add(grp)

class ConditionalProbabilityNotation(Scene):
    def construct(self):
        tex1 = MathTex(r"P(A|B)", r"P(A \text{ given } B)").arrange(DOWN, buff=.75)

        mobj_to_svg(tex1)

class ConditionalProbabilityExample(Scene):
    def construct(self):
        p_rain = MathTex(r"P(", r"\text{flood}", r") = .05")
        p_rain[1].set_color(RED)

        p_rain_given_flood = MathTex(r"P(", r"\text{flood}", r"\text{ given }", r"\text{rain}", r") = .80")
        p_rain_given_flood[1].set_color(RED)
        p_rain_given_flood[3].set_color(BLUE)

        grp = VGroup(p_rain, p_rain_given_flood).arrange(DOWN, buff=.75)
        mobj_to_svg(grp)

class BayesTheoremNotation(Scene):
    def construct(self):
        bt_tex = MathTex(r"P(A|B) = \frac{P(B|A) \times P(A)}{P(B)}")
        mobj_to_svg(bt_tex)


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

        p_homicidal_gamer = MathTex(r"P(", r"\text{homicidal}", r"|", r"\text{gamer}", r")", "=", ".85").scale(1.3)
        p_homicidal_gamer[1].set_color(RED)
        p_homicidal_gamer[3].set_color(BLUE)


        p_gamer_homicidal = MathTex(r"P(", r"\text{gamer}", r"|", r"\text{homicidal}", r")", "=", r"\text{ ? }").scale(1.3)
        p_gamer_homicidal[1].set_color(BLUE)
        p_gamer_homicidal[3].set_color(RED)

        grp = VGroup(p_homicidal_gamer, p_gamer_homicidal).arrange(DOWN, buff=.75)
        mobj_to_svg(grp)

class VideoGameHomicidalExample2(Scene):
    def construct(self):

        stats = VGroup(
            MathTex(r"P(", r"\text{homicidal}", r"|", r"\text{gamer}", r")", "=", ".85"),
            MathTex(r"P(", r"\text{Gamer}", ") = .19"),
            MathTex(r"P(", r"\text{Homicidal}", ") = .0001"),
            MathTex(r"P(", r"\text{gamer}", r"|", r"\text{homicidal}", r")", "=", r"\text{ ? }")
        ).scale(1.3).arrange(DOWN, buff=.75)

        VGroup(stats[0][1], stats[2][1], stats[3][3]).set_color(RED)
        VGroup(stats[0][3], stats[1][1], stats[3][1]).set_color(BLUE)

        mobj_to_svg(stats)


class VideoGameHomicidalExample3(Scene):
    def construct(self):


        p_solve = MathTex(r"&= \frac{.85 \times .0001 }{.19}", r"\\&= .0004" )

        mobj_to_svg(p_solve)


class VennDiagramBayes(MovingCameraScene):
    def construct(self):

        # change line width behavior on camera zoom
        INITIAL_LINE_WIDTH_MULTIPLE = self.camera.cairo_line_width_multiple
        INITIAL_FRAME_WIDTH = config.frame_width

        def line_scale_down_updater(mobj):
            proportion = self.camera.frame.width / INITIAL_FRAME_WIDTH
            self.camera.cairo_line_width_multiple = INITIAL_LINE_WIDTH_MULTIPLE * proportion

        mobj = Mobject()
        mobj.add_updater(line_scale_down_updater)
        self.add(mobj)

        whole = Circle(radius=3.5,color=YELLOW)
        whole_txt = Tex("100K Population").move_to(whole)
        self.play(*[Write(m) for m in (whole, whole_txt)])
        self.wait()

        gamers = Circle(radius=1.5, color=BLUE).move_to([0,-2,0])
        gamers_txt = Tex("19K Gamers").scale(.75).move_to(gamers)
        self.play(*[Write(m) for m in (gamers, gamers_txt)])
        self.wait()

        homicidals = Circle(radius=.01, color=RED) \
            .move_to(gamers.get_top()) \
            .shift(.005 * DOWN) \
            .rotate(45*DEGREES, about_point=gamers.get_center())

        homicidals_txt = Tex("10 Homicidals") \
            .scale_to_fit_width(homicidals.width * .6) \
            .move_to(homicidals)

        self.play(*[Write(m) for m in (homicidals, homicidals_txt)])
        self.wait()

        self.wait()
        self.camera.frame.save_state()

        self.play(
            self.camera.frame.animate.set(height=homicidals.height * 1.2) \
                .move_to(homicidals),
            run_time=3
        )
        self.wait()

        homicidals_txt.save_state()
        homicidals_play_games_txt = Tex(r"8.5 homicidals","are gamers").arrange(DOWN) \
            .scale_to_fit_width(homicidals.width * .6) \
            .move_to(homicidals) \
            .rotate(45 * DEGREES)

        homicidals_dont_play_games_txt = Tex(r"1.5 homicidals","are not gamers").arrange(DOWN) \
            .scale_to_fit_width(homicidals.width * .4) \
            .move_to(homicidals.get_top()) \
            .next_to(gamers.get_top(), UP, buff=.001) \
            .rotate(45 * DEGREES, about_point=gamers.get_center())

        self.play(Transform(homicidals_txt,
                                       VGroup(homicidals_play_games_txt,
                                            homicidals_dont_play_games_txt)
                                       )
                  )

        self.wait()
        self.play(Restore(homicidals_txt))
        self.wait()
        self.play(Restore(self.camera.frame), run_time=3)
        self.wait()
        self.play(Wiggle(gamers))
        self.wait()
        self.play(Circumscribe(homicidals,color=RED))
        self.wait()

        self.play(
            self.camera.frame.animate.set(height=homicidals.height * 1.2) \
                .move_to(homicidals),
            run_time=3
        )

        intersect = Intersection(homicidals, gamers, color=PURPLE, fill_opacity=.6)
        diff1 = Difference(homicidals, gamers, color=RED, fill_opacity=.6)
        diff2 = Difference(gamers, homicidals, color=BLUE, fill_opacity=.6)

        homicidals_play_games_prop = Tex(r".85") \
            .scale_to_fit_width(homicidals.width * .2) \
            .move_to(homicidals) \
            .rotate(45 * DEGREES)

        homicidals_dont_play_games_prop = Tex(r".15") \
            .scale_to_fit_width(homicidals.width * .2) \
            .move_to(homicidals.get_top()) \
            .next_to(gamers.get_top(), UP, buff=.001) \
            .rotate(45 * DEGREES, about_point=gamers.get_center())

        self.play(*[Write(m) for m in (diff1,diff2,intersect)])

        self.wait()

        self.play(Transform(homicidals_txt,
                           VGroup(homicidals_play_games_prop,
                                homicidals_dont_play_games_prop)
                           )
                  )
        self.wait()
        self.play(
            Restore(self.camera.frame),
            *[FadeOut(m) for m in (diff1,diff2,intersect)],
            run_time=3
        )
        self.wait()


class BayesTheoremCode(Scene):
    def construct(self):

        raw_code = """p_gamer = .19
p_homicidal = .0001
p_gamer_given_homicidal = .85

p_homicidal_given_gamer = (p_gamer_given_homicidal * p_homicidal
                           / p_gamer)

# 0.0004473684210526316
print(p_homicidal_given_gamer)"""

        code = Code(code=raw_code, language="Python", font="Monospace", style="monokai", background="window")

        mobj_to_svg(code)

class Door(VGroup):
    def __init__(self, is_open=False, *vmobjects, **kwargs):
        super().__init__(*vmobjects, **kwargs)
        self.rect = Rectangle(width=3,height=4)
        self.knob = Circle(radius=.25).move_to(self.rect, aligned_edge=RIGHT).shift(LEFT*.2)
        self.add(self.rect)

        if not is_open:
            self.add(self.knob)
class Goat(SVGMobject):
    def __init__(self, **kwargs):
        super().__init__('goat.svg')
        self.set_color(RED)
class Laptop(SVGMobject):
    def __init__(self, **kwargs):
        super().__init__('laptop.svg', stroke_color=WHITE,stroke_width=1)

class MontyHall1(Scene):
    def construct(self):
        doors = VGroup(
            *[Door() for _ in range(3)]
        ).arrange(RIGHT, buff=1)
        self.add(doors)
        mobj_to_svg(doors)


class MontyHall2(Scene):
    def construct(self):
        doors = VGroup(
            *[Door() for _ in range(3)]
        ).arrange(RIGHT, buff=1)
        doors[1].rect.set_color(YELLOW)
        self.add(doors)
        mobj_to_svg(doors)

class MontyHall3(Scene):
    def construct(self):
        doors = VGroup(
            *[Door(is_open= i==2) for i in range(3)]
        ).arrange(RIGHT, buff=1)

        doors[1].rect.set_color(YELLOW)
        goat = Goat().scale_to_fit_width(doors[2].width *.75).move_to(doors[2])

        grp = VGroup(doors, goat)
        mobj_to_svg(grp)
        self.add(grp)

class MontyHall4(Scene):
    def construct(self):
        doors = VGroup(
            *[Door(is_open= i!=1) for i in range(3)]
        ).arrange(RIGHT, buff=1)

        doors[0].rect.set_color(YELLOW)
        goat = Goat().scale_to_fit_width(doors[2].width *.75).move_to(doors[2])
        laptop = Laptop().scale_to_fit_width(doors[0].width *.75).move_to(doors[0])
        grp = VGroup(doors, goat, laptop)

        for i,t in enumerate((r"\frac{2}{3}", r"\frac{1}{3}")):
            grp.add(MathTex(t).next_to(doors[i], DOWN))

        mobj_to_svg(grp,h_padding=2)
        self.add(grp)

class MontyHallSimulation(Scene):
    def construct(self):

        raw_code = """from random import randint, choice

def random_door(): return randint(1, 3)

trial_count = 100000

stay_wins = 0
switch_wins = 0

for i in range(0, trial_count):
    prize_door = random_door()
    selected_door = random_door()
    opened_door = choice([d for d in range(1, 4) if d != selected_door and d != prize_door])
    switch_door = choice([d for d in range(1, 4) if d != selected_door and d != opened_door])

    if selected_door == prize_door:
        stay_wins += 1

    if switch_door == prize_door:
        switch_wins += 1

print("STAY WINS: {}, SWITCH WINS: {}".format(
    stay_wins, switch_wins))

print("STAY WIN RATE: {}, SWITCH WIN RATE: {}".format(
    float(stay_wins)/float(trial_count), float(switch_wins)/float(trial_count)))"""

        code = Code(code=raw_code, language="Python", font="Monospace", style="monokai", background="window")

        mobj_to_svg(code)
if __name__ == "__main__":
    render_scenes(q="l", play=True, scene_names=['MontyHallSimulation'])
