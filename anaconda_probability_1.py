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


class Die(VGroup):
    def __init__(self, num_dots, dot_color=BLACK, *vmobjects, **kwargs):
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

        for i in range(9): self.add(Circle(.2, color=dot_color if i in highlighted_dots else "WHITE", fill_opacity=1))
        self.arrange_in_grid(3,3).rotate(90*DEGREES)
        self.add(Rectangle(color=BLACK, fill_opacity=0, height=self.height * 1.2, width=self.width * 1.2))

class Coin(VGroup):
    def __init__(self, symbol="\$", tex_color=RED, *vmobjects, **kwargs):
        super().__init__(*vmobjects, **kwargs)

        _coin = Circle(radius=.75, color=BLACK)
        _symbol = Tex(symbol, color=tex_color) \
            .scale_to_fit_height(_coin.height * .6)

        self.add(_coin, _symbol)

class DieAndCoinSequence(VGroup):
    def __init__(self, *vmobjects, **kwargs):
        super().__init__(*vmobjects, **kwargs)
        self.combos1 = VGroup(
            *[VGroup(Coin(cv, tex_color=BLUE if cv == "T" else RED), Die(dv, dot_color=BLACK)).arrange(DOWN, buff=.75)
              for cv in ("H",) for dv in (1, 2, 3, 4, 5, 6)]) \
            .arrange(RIGHT, buff=.75) \
            .scale_to_fit_width(6)

        self.combos2 = VGroup(
            *[VGroup(Coin(cv, tex_color=BLUE if cv == "T" else RED), Die(dv, dot_color=BLACK)).arrange(DOWN, buff=.75)
              for cv in ("T",) for dv in (6, 5, 4, 3, 2, 1)]) \
            .arrange(RIGHT, buff=.75) \
            .scale_to_fit_width(6)

        self.all_combos = VGroup(self.combos1, self.combos2).arrange(RIGHT)
        self.add(self.all_combos)

        self.heads_outcome_brace = Brace(self.combos1, DOWN, color=RED)
        self.heads_outcome_brace_txt = MathTex(r"\frac{6}{12}", color=RED).next_to(self.heads_outcome_brace, DOWN)

        self.six_outcome_brace = Brace(VGroup(self.combos1[-1], self.combos2[0]), DOWN, color=BLUE).shift(DOWN * .5)
        self.six_outcome_brace_txt = MathTex(r"\frac{2}{12}", color=BLUE).next_to(self.six_outcome_brace, DOWN)

        self.heads_and_six_outcome_brace = Brace(self.combos1[-1], UP, color=ORANGE)
        self.heads_and_six_outcome_brace_txt = MathTex(r"\frac{1}{12}", color=ORANGE) \
            .next_to(self.heads_and_six_outcome_brace, UP)

        self.joint_box = DashedVMobject(Rectangle(color=PURPLE) \
                                   .stretch_to_fit_height(self.combos1[-1].height * 1.2) \
                                   .stretch_to_fit_width(self.combos1[-1].width * 1.2) \
                                   .move_to(self.combos1[-1])
                                   )

        self.add(self.heads_outcome_brace, self.heads_outcome_brace_txt, self.six_outcome_brace, self.six_outcome_brace_txt, self.joint_box)


class JointProbabilityExplanation(Scene):
    def construct(self):
        top_tex =  MathTex(r"P(H \cap 6)", "=", r" \frac{1}{2} \times \frac{1}{6}", r"= \frac{1}{12}", color=BLACK)
        top_tex[0][2].set_color(RED)
        top_tex[0][4].set_color(BLUE)
        top_tex[3].set_color(PURPLE)

        seq = DieAndCoinSequence()

        top_tex.next_to(seq, UP*2)
        grp = VGroup(seq,
                     top_tex,
                     seq.all_combos#,
                     #seq.heads_and_six_outcome_brace,
                     #seq.heads_and_six_outcome_brace_txt,
                     #seq.heads_and_six_outcome_brace_txt.copy().move_to(top_tex[-1], aligned_edge=LEFT)
                     ).scale(.75)
        mobj_to_svg(grp,h_padding=.75, w_padding=.1)
        self.add(grp)


class UnionProbabilityExplanation(Scene):
    def __init__(self):
        super().__init__()

        union_tex = MathTex(r"P(H \cup 6) = ", r"\frac{6}{12}", "+", r"\frac{2}{12}", r"-", r"\frac{6}{12} \times \frac{2}{12}", r"= \frac{7}{12}", color=BLACK)
        union_tex[0][2].set_color(RED)
        union_tex[0][4].set_color(BLUE)
        union_tex[1].set_color(RED)
        union_tex[3].set_color(BLUE)

        union_tex[5].set_color(PURPLE)

        sequence = DieAndCoinSequence()
        union_tex.next_to(sequence, UP*2)

        grp = VGroup(union_tex, sequence).scale(.75)
        mobj_to_svg(grp,h_padding=.75, w_padding=.1)
        self.add(union_tex, sequence)

if __name__ == "__main__":
    render_scenes(q="l", play=True, scene_names=['UnionProbabilityExplanation'])
    #file_to_base_64('/Users/thomasnield/git/3mds-lib/media/images/anaconda_linear_algebra_1/04_VectorExamplesDotsScene.png')