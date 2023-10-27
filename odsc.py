import urllib

from manim import *

from threemds.utils import render_scenes, mobj_to_svg
import sympy as sp
import scipy
import numpy as np
import os

TITLE_SCALE=1.4
SUBTITLE_SCALE=1.0
BULLET_BUFF=.75
BULLET_TEX_SCALE=.8

# ========================================================================
# SECTION 2 - CALCULUS
# ========================================================================

class FunctionContinuity(ThreeDScene):
    def construct(self):
        f = lambda x: 2*x + 1
        ax = Axes(x_range=[-2,2,1],y_range=[-1,4,1])
        self.add(ax)

        point_counts = (10,20,40,80,160,320)

        last_pts: VGroup = VGroup()
        self.add(last_pts)

        for pt_ct in point_counts:
            pts = VGroup(*[Dot(ax.c2p(x,f(x)), color=YELLOW) for x in np.linspace(-2,2,pt_ct)])
            self.play(
                last_pts.animate.become(pts)
            )
            self.wait(2)

        self.play(
            last_pts.animate.become(ax.plot(f, color=YELLOW))
        )
        self.wait()

class ThreeDPlot(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=70 * DEGREES, theta=-30 * DEGREES)

        ax = ThreeDAxes(x_range=(-1, 1, 1), y_range=(-1, 1, 1), z_range=(-10, 10, 1))


        def param_f(u, v):
            x1 = u
            x2 = v
            y = 2*x1**2 + 4*x2**3
            return ax.c2p(x1, x2, y)

        plot = Surface(
            param_f,
            resolution=(42, 42),
            v_range=[-1, +1],
            u_range=[-1, +1]
        )
        plot.set_style(fill_opacity=.5, stroke_color=WHITE, fill_color=BLUE)

        x1,x2,y = sp.symbols('x1 x2 y')
        f = 2*x1**2 + 4*x2**3

        _x1, _x2, _y = .5, .5, f.subs([(x1, .5), (x2, .5)]).evalf()

        dx1 = sp.diff(f, x1)
        dx2 = sp.diff(f, x2)

        _dx1 = dx1.subs([(x1, .5), (x2, .5)]).evalf()
        _dx2 = dx2.subs([(x1, .5), (x2, .5)]).evalf()

        f_tex = MathTex("f(x,y) = 2x^2 + 4y^3").to_edge(UL)

        self.add(ax,plot)
        self.add_fixed_in_frame_mobjects(f_tex)


class SummationAndProduct(Scene):
    def construct(self):
        x = sp.Matrix([5,2,9,6,1]).transpose()
        tex_x = MathTex(r"x_i = ", sp.latex(x))

        summation = MathTex("\sum_{i=1}^{5} x_i = 5 + 2 + 9 + 6 + 1 = 23")
        prod = MathTex(r"\prod_{i=1}^{5} x_i = & 5\times2\times9\times6\times1 = 540")

        grp = VGroup(tex_x, summation, prod).arrange(DOWN,buff=1)

        mobj_to_svg(grp, 'out.svg')


class LogarithmPlot(Scene):
    def construct(self):
        import math
        ax = Axes(x_range=(-1,5,1), y_range=(-1,2,1))
        plot = ax.plot(lambda x: math.log(x,10), x_range=(.05,5), color=BLUE)
        tex = MathTex(r"f(x) = \text{log}_{3} x").next_to(ax, DOWN)
        self.add(ax, plot, tex)
        mobj_to_svg(VGroup(ax, plot, tex),filename= 'out.svg', h_padding=1)

class LogarithmLogic(Scene):
    def construct(self):
        tex1 = MathTex(r"3^x = 9")

        tex2 = MathTex(r"\text{log}_{3}9 = 2")

        grp = VGroup(tex1, tex2).arrange(DOWN, buff=.5)

        mobj_to_svg(grp)

class LimitPlot(Scene):
    def construct(self):
        ax = Axes(x_range=(-12,12,1), y_range=(-2,2,1), x_length=6,y_length=6, tips=False)

        plot1 = ax.plot(lambda x:  1/(x-2), x_range=(3.15,12),use_smoothing=True, color=BLUE)
        plot2 = ax.plot(lambda x: 1/(x-2), x_range=(-12,2.95), use_smoothing=True, color=BLUE)
        asymptote = DashedLine(dash_length=DEFAULT_DASH_LENGTH*3, color=WHITE, start=ax.c2p(3,-12), end=ax.c2p(3,12))
        self.add(ax, plot1, plot2, asymptote)
        mobj_to_svg(VGroup(ax, plot1, plot2, asymptote), 'out.svg')

class EulersNumberPlot(Scene):
    def construct(self):
        import math
        ax = Axes(x_range=(-5,5,1), y_range=(-1,5,1), x_length=6,y_length=6, tips=False)

        plot1 = ax.plot(lambda x: math.e**x, x_range=(-5,2),use_smoothing=True, color=BLUE)
        tex = MathTex("f(x) = e^x").next_to(ax, DOWN)

        self.add(ax, plot1, tex)
        mobj_to_svg(VGroup(ax, plot1, tex), 'out.svg',h_padding=1)

class MovingSecantScene(Scene):
    def construct(self):
        x = Variable(2, num_decimal_places=2, label=r"\Delta x").scale(0.75)

        f = lambda x: 2*x**2

        x_range, y_range = (-3,3,1), (-1,9,1)
        ax = Axes(x_range,y_range)
        plot = ax.plot(f, color=BLUE)
        tangent = always_redraw(lambda:
            ax.get_secant_slope_group(x.tracker.get_value(), plot, dx=.001)[2].set_color(RED)
        )

        def x_updater(mobj):
            mobj.move_to(tangent.copy().set_length(2).rotate(90*DEGREES).get_end())

        x.add_updater(x_updater, call_updater=True)
        self.add(ax,plot,tangent, x)

        # animate
        self.play(
            x.tracker.animate.set_value(-2),
            run_time=6
        )
        self.play(
            x.tracker.animate.set_value(2),
            run_time=6
        )


class ClosingSecantLine(Scene):
    def construct(self):
        x = Variable(1, num_decimal_places=2, label=r"x").scale(0.75)

        dx = Variable(1, num_decimal_places=2, label=r"dx") \
            .scale(0.75)

        f = lambda x: 2*x ** 2

        dy_dx = Variable(2, num_decimal_places=2, label=r"\frac{\Delta y}{\Delta x}") \
            .scale(0.75) \
            .add_updater(lambda mobj: mobj.tracker.set_value((f(x.tracker.get_value() + dx.tracker.get_value()) - f(x.tracker.get_value()))/(dx.tracker.get_value())))

        x_range, y_range = (-3, 3, 1), (-1, 9, 1)
        ax = Axes(x_range, y_range)
        plot = ax.plot(f, color=BLUE)
        tangent = always_redraw(lambda:
                                ax.get_secant_slope_group(x=x.tracker.get_value(),
                                                          dx=dx.tracker.get_value(),
                                                          graph=plot,
                                                          dx_line_color=ORANGE,
                                                          dy_line_color=ORANGE,
                                                          secant_line_color=YELLOW,
                                                          secant_line_length=20)
                                )

        x_dot = Dot(color=RED).add_updater(lambda mobj: mobj.move_to(ax.c2p(x.tracker.get_value(),f(x.tracker.get_value()))),
                                            call_updater=True)

        x1_dot = Dot(color=RED).add_updater(lambda mobj: mobj.move_to(ax.c2p(x.tracker.get_value() + dx.tracker.get_value(),f(x.tracker.get_value() + dx.tracker.get_value()))),
                                            call_updater=True)

        dy_dx.add_updater(lambda mobj: mobj.move_to(tangent[2].copy() \
                                                    .set_length(2) \
                                                    .rotate(90 * DEGREES).get_end()
                                                    ),
                          call_updater=True)

        self.add(ax, plot, tangent, dy_dx, x_dot, x1_dot)

        # animate
        self.play(
            dx.tracker.animate.set_value(.001),
            run_time=6
        )
        self.wait()
        self.play(
            x.tracker.animate.set_value(-2),
            run_time=6
        )
        self.play(
            x.tracker.animate.set_value(2),
            run_time=6
        )
        self.wait()

class SimpleSecant(Scene):
    def construct(self):
        f = lambda x: 2*x**2
        x_range, y_range = (-2,2,1), (-1,4,1)
        ax = Axes(x_range,y_range, x_length=4, y_length=4)
        plot = ax.plot(f, color=BLUE)
        tangent = TangentLine(plot,alpha=.55, length=4, color=RED)

        self.add(ax,plot,tangent)
        mobj_to_svg(VGroup(ax,plot,tangent),'out.svg')


class DerivativeCode(Scene):
    def construct(self):
        raw_code = """from sympy import *
x = symbols('x')
f = 2*x**2
dx_f = diff(f,x)
print(dx_f) # 4*x
"""

        code = Code(code=raw_code, language="Python", font="Monospace", style="monokai", background="window")
        self.add(code)
        mobj_to_svg(VGroup(code), "out.svg")



class ThreeDDerivative(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=70 * DEGREES, theta=-30 * DEGREES)

        ax = ThreeDAxes(x_range=(-1, 1, 1), y_range=(-1, 1, 1), z_range=(-10, 10, 1))


        def param_f(u, v):
            x1 = u
            x2 = v
            y = 2*x1**2 + 4*x2**3
            return ax.c2p(x1, x2, y)

        plot = Surface(
            param_f,
            resolution=(42, 42),
            v_range=[-1, +1],
            u_range=[-1, +1]
        )
        plot.set_style(fill_opacity=.5, stroke_color=WHITE, fill_color=BLUE)

        x1,x2,y = sp.symbols('x1 x2 y')
        f = 2*x1**2 + 4*x2**3

        _x1, _x2, _y = .5, .5, f.subs([(x1, .5), (x2, .5)]).evalf()

        dx1 = sp.diff(f, x1)
        dx2 = sp.diff(f, x2)

        _dx1 = dx1.subs([(x1, .5), (x2, .5)]).evalf()
        _dx2 = dx2.subs([(x1, .5), (x2, .5)]).evalf()


        dot = Dot3D(point=ax.c2p(*[.5, .5, f.subs([(x1, _x1), (x2, _x1), (y,_y)]).evalf()]),
                    color=RED)

        slope1 = DashedLine(start=ax.c2p(0,0,0), end=ax.c2p(.5,0,_dx1*.5), color=YELLOW).move_to(dot)
        slope2 = DashedLine(start=ax.c2p(0,0,0), end=ax.c2p(0,.5,_dx2*.5), color=YELLOW).move_to(dot)

        self.add(ax,plot, slope1, slope2, dot)

class PartialDerivativeCode(Scene):
    def construct(self):
        raw_code = """from sympy import *
        
x, y = symbols('x y')
f = 2*x**2 + 4*y**3
dx_f = diff(f, x)
dy_f = diff(f, y)

print(dx_f) # 4*x
print(dy_f) # 12*y**2
"""

        code = Code(code=raw_code, language="Python", font="Monospace", style="monokai", background="window")
        self.add(code)
        mobj_to_svg(VGroup(code), "out.svg")

class ReimannSums(Scene):
    def construct(self):
        ax = Axes(x_range=(-3, 3, 1), y_range=(-1, 7, 1))
        f = lambda x: x ** 2
        plt = ax.plot(f, color=RED)

        rects = ax.get_riemann_rectangles(plt,
                                          x_range=(0,2.5),
                                          dx=.5,
                                          fill_opacity=.6, color=BLUE)

        self.next_section("animations", skip_animations=False)

        self.add(ax,plt)
        self.play(Write(rects))
        self.wait()

        for dx in (.25,.125,.1,.05, .025):

            new_rects = ax.get_riemann_rectangles(plt,
                                          x_range=(0,2.5),
                                          dx=dx,
                                          fill_opacity=.6, color=BLUE)

            self.play(ReplacementTransform(rects, new_rects))
            rects = new_rects

            self.wait()

        area = ax.get_area(plt, x_range=(0,2.5), color=BLUE, opacity=.6)
        self.play(
            FadeTransform(rects, area)
        )
        self.wait()
        self.next_section("final", skip_animations=False)


class IntegrationCode(Scene):
    def construct(self):
        raw_code = """from sympy import *

x = symbols('x')
f = x**2

# area from x = 0 to 2.5
area = integrate(f, (x, 0, 2.5))
print(area) # 5.20833333333333
"""

        code = Code(code=raw_code, language="Python", font="Monospace", style="monokai", background="window")
        self.add(code)
        mobj_to_svg(VGroup(code), "out.svg")


class ColdPDF(VGroup):
    def __init__(self, *vmobjects, **kwargs):
        super().__init__(*vmobjects, **kwargs)
        mean = 18
        std = 1.5
        self.mean, self.std = mean, std
        self.x_lower, self.x_upper = mean - 3 * std, mean + 3 * std

        self.ax = Axes(x_range=(self.x_lower, self.x_upper, std),
                       y_range=(0, .3, .1),
                       tips=False,
                       x_axis_config={"include_numbers" : False,
                                 "numbers_to_include" : (mean, mean+std, mean+std*2, mean-std, mean-2*std
                                                         )
                                 },
                       y_axis_config={"include_numbers": False,
                                      "numbers_to_include": (.1, .2, .3
                                                             )
                                      }
                       )
        self.plt = self.ax.plot(lambda x: scipy.stats.norm.pdf(x,mean,std), color=YELLOW)

        grp = VGroup(self.ax, self.plt)

        self.add(grp)

class ColdMedicinePDF(Scene):
    def construct(self):
        from scipy.stats import norm

        pdf = ColdPDF()
        lower_x_95 = norm.ppf(.025, pdf.mean, pdf.std)
        upper_x_95 = norm.ppf(.975, pdf.mean, pdf.std)

        center_area_95 = pdf.ax.get_area(pdf.plt,x_range=(lower_x_95, upper_x_95), color=BLUE)
        center_area_95_lbl = MathTex(.95).move_to(center_area_95)

        x = 14.5
        line = DashedLine(start=pdf.ax.c2p(x,0),
                          end=pdf.ax.c2p(x, pdf.plt.underlying_function(x) + .1),
                          color=RED
                          )

        p_value_lower = pdf.ax.get_area(pdf.plt,x_range=(-3*pdf.std+pdf.mean, x), color=RED)
        p_value_upper = pdf.ax.get_area(pdf.plt,x_range=(upper_x_95 + (lower_x_95-x), 3*pdf.std+pdf.mean), color=RED)
        p_value_lbl = MathTex("p = 0.01963",color=RED).next_to(line, UP)

        grp = VGroup(pdf)#,
                     #center_area_95_lbl,
                     #center_area_95,
                     #line,
                     #p_value_lower, p_value_upper, p_value_lbl)

        mobj_to_svg(grp, h_padding=1, w_padding=1)

        self.add(grp)


class BabyFormulaTwoTailScipy(Scene):
    def construct(self):
        raw = r"""from scipy.stats import norm

# Cold has mean recovery of 18 days
# with 1.5 standard deviations
mean = 18
std = 1.5

# Experimental drug has 14.5 days of recovery
x = 14.5

# Probability of 14.5
tail_p = norm.cdf(x, mean, std)

# Get p-value of both tails
p_value = 2*tail_p

print(p_value) # 0.019630657257290667
"""

        code = Code(code=raw, language="Python", font="Monospace", style="monokai", background="window")

        mobj_to_svg(code)
if __name__ == "__main__":
    render_scenes(q="l", scene_names=['ColdMedicinePDF'], last_frame=True)
