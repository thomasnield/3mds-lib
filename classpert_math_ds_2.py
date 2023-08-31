from manim import *
import numpy as np
from sympy import Sum

from threemds.utils import render_scenes, file_to_base_64, mobj_to_svg, mobj_to_png
import pandas as pd
import sympy as sp

#config.frame_rate = 60

CLASSPERT_ORANGE="#DF6437" #dark orange
CLASSPERT_NAVY="#1E1E27" #darky navy
text_color="#FFFFFF" #white

config.background_color = CLASSPERT_NAVY

#config.frame_width = 7
#config.frame_height = 7
#config.frame_width = 12
#config.pixel_width = 420
#config.pixel_height = 420

config.background_color=CLASSPERT_NAVY

class SimpleArea(Scene):
    def construct(self):
        ax = Axes(x_range=[-2,2,1],y_range=[-1,4,1])
        plot = ax.plot(lambda x: x**2)
        area = ax.get_area(plot, x_range=(0,1), color=CLASSPERT_ORANGE, opacity=.9)

        self.add(ax,plot,area)
        mobj_to_svg(VGroup(ax,plot,area),'out.svg')

class SimpleSecant(Scene):
    def construct(self):
        f,inv_f = lambda x: x**2, lambda y: y / 1.5
        x_range, y_range = (-.1,2,1), (-1,4,1)
        ax = Axes(x_range,y_range, x_length=4, y_length=4)
        plot = ax.plot(f)
        tangent = TangentLine(plot,alpha=.75, length=4, color=CLASSPERT_ORANGE)

        self.add(ax,plot,tangent)
        mobj_to_svg(VGroup(ax,plot,tangent),'out.svg')

class SimpleReimannSum(Scene):
    def construct(self):
        f = lambda x: 1.5*x
        inv_f = lambda y: y / 1.5
        ax = Axes(x_range=(-1,3,1), y_range=(-1,3,1), x_length=4, y_length=4)
        plot = ax.plot(f, x_range=(inv_f(-1), inv_f(3)))
        rects = ax.get_riemann_rectangles(plot,x_range=(0,1.5),dx=0.25, color=CLASSPERT_ORANGE)
        self.add(ax,plot,rects)
        mobj_to_svg(VGroup(ax,plot,rects), 'out.svg')


class CodeRender(Scene):
    def construct(self):
        raw_code = """from sympy import *

x = symbols('x')
f = 1 / (x-3)

result = limit(f, x, oo)

print(result) # 0
"""

        code = Code(code=raw_code, language="Python", font="Monospace", style="monokai", background="window")
        self.add(code)
        mobj_to_svg(VGroup(code), "out.svg")

class TexRender(Scene):
    def construct(self):

        a,b,x = sp.symbols('a b x')
        exp_f = sp.Equality(a**x, b).subs(a,2).subs(b,8)

        tex = MathTex(r"\lim_{h\to0} \frac{f(x+h) -f(x)}{h}")
        mobj_to_svg(tex, 'out.svg')

class SimplePlot(Scene):
    def construct(self):
        ax = Axes(x_range=(-10,10,1), y_range=(-.5,300,50))
        plot = ax.plot(lambda x: 3*x**2 + 1, color=CLASSPERT_ORANGE)
        self.add(ax, plot)
        mobj_to_svg(VGroup(ax, plot), 'out.svg')

class ThreeDPlot(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(70*DEGREES,-70*DEGREES)

        ax = ThreeDAxes(x_range=[-1,6,1],y_range=[-1,12,1],z_range=[-1,1.2,1])

        def param_gauss(u, v):
            x1 = u
            x2 = v
            y = 1.0 / (1 + np.exp(-(-1.70286348*x1 + 0.70820922*x2 -1.2788603)))
            return np.array([x1, x2, y])

        surface = Surface(
            param_gauss,
            resolution=(42, 42),
            v_range=[-5, +5],
            u_range=[-5, +5]
        )

        surface.set_style(fill_opacity=.5,stroke_color=ORANGE)
        surface.set_fill(CLASSPERT_ORANGE, opacity=0.2)
        grp = VGroup(ax, surface)
        self.add(grp)


class FunctionContinuity(ThreeDScene):
    def construct(self):
        f = lambda x: 2*x + 1
        ax = Axes(x_range=[-2,2,1],y_range=[-1,4,1])
        self.add(ax)

        point_counts = (5,10,20,40,80,160,320,640)

        last_pts: VGroup = VGroup()
        self.add(last_pts)

        for pt_ct in point_counts:
            pts = VGroup(*[Dot(ax.c2p(x,f(x)),color=CLASSPERT_ORANGE) for x in np.linspace(-2,2,pt_ct)])
            self.play(
                last_pts.animate.become(pts)
            )
            self.wait(2)

        self.play(
            last_pts.animate.become(ax.plot(f, color=ORANGE))
        )
        self.wait()

class Summation(Scene):
    def construct(self):
        x = sp.Matrix([5,2,9,6,1]).transpose()
        tex_x = MathTex(r"x_i = ", sp.latex(x))

        summation = MathTex(r"\prod_{i=1}^{5} x_i &= 5\times2\times9\times6\times1 \\ &= 540")
        mobj_to_svg(summation, 'out.svg')


class LogarithmPlot(Scene):
    def construct(self):
        import math
        ax = Axes(x_range=(-1,5,1), y_range=(-5,5,1))
        plot = ax.plot(lambda x: math.log(x,2), x_range=(.05,5), color=CLASSPERT_ORANGE)
        tex = MathTex(r"f(x) = \text{log}_2 x").next_to(ax, DOWN)
        self.add(ax, plot, tex)
        mobj_to_svg(VGroup(ax, plot, tex),filename= 'out.svg', h_padding=1)


class LimitPlot(Scene):
    def construct(self):
        ax = Axes(x_range=(-12,12,1), y_range=(-3,3,1), x_length=6,y_length=6, tips=False)

        plot1 = ax.plot(lambda x:  1/(x-3), x_range=(3.15,12),use_smoothing=True, color=CLASSPERT_ORANGE)
        plot2 = ax.plot(lambda x: 1/(x-3), x_range=(-12,2.95), use_smoothing=True, color=CLASSPERT_ORANGE)
        asymptote = DashedLine(dash_length=DEFAULT_DASH_LENGTH*3, color=WHITE, start=ax.c2p(3,-12), end=ax.c2p(3,12))
        self.add(ax, plot1, plot2, asymptote)
        mobj_to_svg(VGroup(ax, plot1, plot2, asymptote), 'out.svg')


class EulersNumberPlot(Scene):
    def construct(self):
        import math
        ax = Axes(x_range=(-1,6,1), y_range=(-3,3,1), x_length=6,y_length=6, tips=False)

        plot1 = ax.plot(lambda x: math.log(x), x_range=(.05,6), use_smoothing=True, color=CLASSPERT_ORANGE)

        tex = MathTex(r"f(x) = ln(x)").next_to(ax, DOWN)
        self.add(ax, plot1, tex)
        mobj_to_svg(VGroup(ax, plot1, tex), 'out.svg',h_padding=1)


class MovingSecantScene(Scene):
    def construct(self):
        x = Variable(2, num_decimal_places=2, label=r"\Delta x").scale(0.75)

        f = lambda x: x**2

        x_range, y_range = (-3,3,1), (-1,9,1)
        ax = Axes(x_range,y_range)
        plot = ax.plot(f, color=ORANGE)
        tangent = always_redraw(lambda:
            ax.get_secant_slope_group(x.tracker.get_value(), plot)[2].set_color(YELLOW)
        )

        def x_updater(mobj):
            mobj.move_to(tangent.copy().set_length(2).rotate(90*DEGREES).get_end())

        x.add_updater(x_updater, call_updater=True)
        self.add(ax,plot,tangent, x)

        # animate
        self.play(
            x.tracker.animate.set_value(-3),
            run_time=6
        )
        self.play(
            x.tracker.animate.set_value(3),
            run_time=6
        )


class ClosingSecantLine(Scene):
    def construct(self):
        x = Variable(1, num_decimal_places=2, label=r"x").scale(0.75)

        dx = Variable(1, num_decimal_places=2, label=r"dx") \
            .scale(0.75)

        f = lambda x: x ** 2

        dy_dx = Variable(2, num_decimal_places=2, label=r"\frac{\Delta y}{\Delta x}") \
            .scale(0.75) \
            .add_updater(lambda mobj: mobj.tracker.set_value((f(x.tracker.get_value() + dx.tracker.get_value()) - f(x.tracker.get_value()))/(dx.tracker.get_value())))

        x_range, y_range = (-3, 3, 1), (-1, 9, 1)
        ax = Axes(x_range, y_range)
        plot = ax.plot(f, color=ORANGE)
        tangent = always_redraw(lambda:
                                ax.get_secant_slope_group(x=x.tracker.get_value(),
                                                          dx=dx.tracker.get_value(),
                                                          graph=plot,
                                                          dx_line_color=BLUE,
                                                          dy_line_color=BLUE,
                                                          secant_line_color=YELLOW,
                                                          secant_line_length=20)
                                )

        x_dot = Dot(color=BLUE).add_updater(lambda mobj: mobj.move_to(ax.c2p(x.tracker.get_value(),f(x.tracker.get_value()))),
                                            call_updater=True)

        x1_dot = Dot(color=BLUE).add_updater(lambda mobj: mobj.move_to(ax.c2p(x.tracker.get_value() + dx.tracker.get_value(),f(x.tracker.get_value() + dx.tracker.get_value()))),
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


if __name__ == "__main__":
    render_scenes(q="l", scene_names=['TexRender'])
