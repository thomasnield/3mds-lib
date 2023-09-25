import scipy.stats
from scipy.stats import norm

from manim import *
import numpy as np
import pandas as pd
import sympy as sp
import math

from threemds.stats.plots import Histogram
from threemds.utils import render_scenes, file_to_base_64, mobj_to_svg, mobj_to_png

class MuSymbol(Scene):
    def construct(self):
        mobj_to_svg(MathTex(r"\mu", r"\bar{x}").arrange(DOWN))

class MeanFormulation(Scene):
    def construct(self):
        tex = MathTex(r"\mu = \frac{\sum\limits_{i=1}^N x_i}{N}",
                      r"\bar{x} = \frac{\sum\limits_{i=1}^n x_i}{n}"
                      ).arrange(RIGHT, buff=1)
        mobj_to_svg(tex)

class MeanCode(Scene):
    def construct(self):

        raw_code = """import numpy as np

x = np.array([1.73,1.73,1.73,1.75,1.72,1.69,1.76,
                         1.69,1.70,1.67,1.75,1.71,1.70,1.71,
                         1.68,1.70,1.74,1.72,1.76,1.69,1.76,
                         1.73,1.71,1.73,1.70,1.74,1.74,1.76,
                         1.67,1.74,1.66,1.67,1.70,1.69])

print(x.mean()) # 1.7155882352941176"""

        code = Code(code=raw_code, language="Python", font="Monospace", style="monokai", background="window")

        mobj_to_svg(code)

class VarianceSymbol(Scene):
    def construct(self):
        tex = MathTex(r"\sigma^2", "s^2").arrange(DOWN)
        mobj_to_svg(tex)

class VarianceFormulation(Scene):
    def construct(self):
        tex = MathTex(r"\sigma^2 = \frac{\sum\limits_{i=1}^N (x_i - \mu)^2}{N}",
                      r"s^2 = \frac{\sum\limits_{i=1}^n (x_i - \bar{x})^2}{n - 1}") \
            .arrange(RIGHT, buff=1)

        mobj_to_svg(tex)

class VarianceCode(Scene):
    def construct(self):

        raw_code = """import numpy as np

x = np.array([1.73,1.73,1.73,1.75,1.72,1.69,1.76,
                         1.69,1.70,1.67,1.75,1.71,1.70,1.71,
                         1.68,1.70,1.74,1.72,1.76,1.69,1.76,
                         1.73,1.71,1.73,1.70,1.74,1.74,1.76,
                         1.67,1.74,1.66,1.67,1.70,1.69])

print(x.var(ddof=1)) # 0008557040998217485"""

        code = Code(code=raw_code, language="Python", font="Monospace", style="monokai", background="window")

        mobj_to_svg(code)

class StandardDevSymbol(Scene):
    def construct(self):
        tex = MathTex(r"\sigma", "s").arrange(DOWN)
        mobj_to_svg(tex)


class StandardDevFormulation(Scene):
    def construct(self):
        tex = MathTex(r"\sigma = \sqrt{\frac{\sum\limits_{i=1}^N (x_i - \mu)^2}{N}}",
                      r"s = \sqrt{\frac{\sum\limits_{i=1}^n (x_i - \bar{x})^2}{n - 1}}") \
            .arrange(RIGHT, buff=1)

        mobj_to_svg(tex)


class StandardDeviationCode(Scene):
    def construct(self):

        raw_code = """import numpy as np

x = np.array([1.73,1.73,1.73,1.75,1.72,1.69,1.76,
                         1.69,1.70,1.67,1.75,1.71,1.70,1.71,
                         1.68,1.70,1.74,1.72,1.76,1.69,1.76,
                         1.73,1.71,1.73,1.70,1.74,1.74,1.76,
                         1.67,1.74,1.66,1.67,1.70,1.69])

print(x.std(ddof=1)) # 0.029252420409630185"""

        code = Code(code=raw_code, language="Python", font="Monospace", style="monokai", background="window")

        mobj_to_svg(code)

class MedianConcept(Scene):
    def construct(self):
        seq = VGroup(*[MathTex(m) for m in (3,5,8,11,15)]).arrange(DOWN)
        bracket = Brace(seq[2], LEFT).set_color(YELLOW)
        txt = Tex("Median", color=YELLOW).next_to(bracket, LEFT)
        grp = VGroup(seq, bracket, txt)
        mobj_to_svg(grp, w_padding=3)


class MedianCode(Scene):
    def construct(self):
        raw_code = r"""import numpy as np

x = np.array([1.73,1.73,1.73,1.75,1.72,1.69,1.76,
                         1.69,1.70,1.67,1.75,1.71,1.70,1.71,
                         1.68,1.70,1.74,1.72,1.76,1.69,1.76,
                         1.73,1.71,1.73,1.70,1.74,1.74,1.76,
                         1.67,1.74,1.66,1.67,1.70,1.69])

print(x.mean()) # 1.7155882352941176
print(np.median(x)) # 1.7149999999999999
"""

        code = Code(code=raw_code, language="Python", font="Monospace", style="monokai", background="window")

        mobj_to_svg(code)

"""
x = np.array([
    14.2, 12.8, 13.8, 15.7, 13.8, 14.9, 12.9, 15.9,
    13.5, 14.5, 15.4, 14.2, 13.1, 15.8, 13.7, 14.6,
    14.8, 13.2, 13.2, 15.5, 13.6, 13.2, 11.5, 13.3,
    12.4, 15.8, 12.8, 12.9, 16.8, 14.0, 12.7, 14.3,
    12.2, 12.6, 12.9, 13.7, 15.0, 14.1, 13.3, 14.0,
    15.5, 14.1, 15.6, 13.6, 13.2
])
"""
x = np.array([14.0,14.2,13.5,14.7,16.6,14.7,14.0,15.0,13.6,13.4,13.9,14.6,16.3,13.6,
              13.9,14.5,12.8,14.3,14.0,13.9,11.7,13.0,13.7,14.7,14.2,15.2,12.9,14.1,
              15.0,14.4,13.4,13.5,12.8,13.5,13.9,15.0,15.5,11.3,13.7,14.5,14.0,14.3,
              12.7,15.4,14.0,12.3,12.8,14.2,13.2,16.5,12.1,14.5,13.4,13.5,14.5,13.8,
              13.9,15.3,14.9,14.1,12.1,13.5,14.8,15.6,12.1,14.5,13.0,13.9,12.2,14.1,
              13.2,13.7,13.8,13.4,14.1,15.0,13.3,13.4,14.1,14.5,15.1,12.8,14.2,14.2,
              14.9,12.8,13.0,14.0,12.0,13.2,15.9,14.6,13.4,12.9,13.5,14.2,14.9,13.1,
              15.7,14.0,13.8,13.3,13.8,14.3,13.8,13.9,13.2,14.8,13.7,13.7,14.5,14.8,
              14.1,14.7,12.7,14.9,13.6,14.1,14.1,14.2,15.0,14.3,13.7,13.5,14.4,15.3,
              16.2,15.0,15.0,15.6,13.9,15.2,13.7,13.5,14.4,14.7,14.4,14.9,13.6,12.5,
              14.4,15.0,12.9,14.4,13.9,15.2,13.9,13.7,12.0,13.9,13.7,15.7,13.7,13.0,13.1])

class HistogramPlot(Scene):
    def construct(self):
        hist = Histogram(x, 8,
                         show_points=True,
                         show_normal_dist=True
                         )
        self.add(hist)
        mobj_to_svg(hist,w_padding=4, h_padding=10)

class CoffeeNormalPDF(VGroup):
    def __init__(self, *vmobjects, **kwargs):
        super().__init__(*vmobjects, **kwargs)
        mean = round(x.mean(),1)
        std = round(x.std(ddof=1))
        self.mean, self.std = mean, std
        self.x_lower, self.x_upper = mean - 3 * std, mean + 3 * std

        self.ax = Axes(x_range=(self.x_lower, self.x_upper, std),
                       y_range=(0, .5, .25),
                       tips=False,
                       x_axis_config={"include_numbers" : False,
                                 "numbers_to_include" : (mean, mean+std, mean+std*2, mean-std, mean-2*std
                                                         )
                                 },
                       y_axis_config={"include_numbers": False,
                                      "numbers_to_include": (.25, .5
                                                             )
                                      }
                       )
        self.plt = self.ax.plot(lambda x: scipy.stats.norm.pdf(x,mean,std), color=YELLOW)

        self.dots = VGroup(*[Dot(self.ax.c2p(_x,0), color=BLUE) for _x in x])
        grp = VGroup(self.ax, self.plt)

        self.add(grp)

class NormalDistributionPlot(Scene):

    def construct(self):
        norm_dist = CoffeeNormalPDF()
        self.add(norm_dist)
        mobj_to_svg(norm_dist)

class CoffeeParameters(Scene):
    def construct(self):
        tex = MathTex(r"\bar{x} = 14.005", r"s = .955") \
            .arrange_in_grid(rows=2,cols=1,col_alignments='r', buff=.75) \

        mobj_to_svg(tex)

class CoffeeData(Scene):
    def construct(self):
        raw = r"""import numpy as np

x = np.array([14.0,14.2,13.5,14.7,16.6,14.7,14.0,15.0,13.6,13.4,13.9,14.6,16.3,13.6,
              13.9,14.5,12.8,14.3,14.0,13.9,11.7,13.0,13.7,14.7,14.2,15.2,12.9,14.1,
              15.0,14.4,13.4,13.5,12.8,13.5,13.9,15.0,15.5,11.3,13.7,14.5,14.0,14.3,
              12.7,15.4,14.0,12.3,12.8,14.2,13.2,16.5,12.1,14.5,13.4,13.5,14.5,13.8,
              13.9,15.3,14.9,14.1,12.1,13.5,14.8,15.6,12.1,14.5,13.0,13.9,12.2,14.1,
              13.2,13.7,13.8,13.4,14.1,15.0,13.3,13.4,14.1,14.5,15.1,12.8,14.2,14.2,
              14.9,12.8,13.0,14.0,12.0,13.2,15.9,14.6,13.4,12.9,13.5,14.2,14.9,13.1,
              15.7,14.0,13.8,13.3,13.8,14.3,13.8,13.9,13.2,14.8,13.7,13.7,14.5,14.8,
              14.1,14.7,12.7,14.9,13.6,14.1,14.1,14.2,15.0,14.3,13.7,13.5,14.4,15.3,
              16.2,15.0,15.0,15.6,13.9,15.2,13.7,13.5,14.4,14.7,14.4,14.9,13.6,12.5,
              14.4,15.0,12.9,14.4,13.9,15.2,13.9,13.7,12.0,13.9,13.7,15.7,13.7,13.0,13.1])


print(x.mean()) # 14.005806451612901
print(x.std(ddof=1)) # 0.9551799116367922"""

        code = Code(code=raw, language="Python", font="Monospace", style="monokai", background="window")

        mobj_to_svg(code)

class CoffeeMatplotLib(Scene):
    def construct(self):
        raw = r"""import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

mean = 14.005
std = 0.955

# Create a grid of x-values
x = np.linspace(-3.5*std+mean, 3.5*std+mean, 1000)

# Calculate the normal probability density function
y = norm.pdf(x, mean, std)

# Plot the normal distribution
plt.plot(x, y, color='blue', linewidth=2)

# Set the title and axis labels
plt.title('Normal Distribution')
plt.xlabel('Grams of Coffee')
plt.ylabel('Barista Recommends Likelihood')

# Show the plot
plt.show()"""

        code = Code(code=raw, language="Python", font="Monospace", style="monokai", background="window")

        mobj_to_svg(code)

class TraceLikelihood(Scene):
    def construct(self):
        norm_dist = CoffeeNormalPDF()

        x = 15.5
        vert_line = DashedLine(
            start=norm_dist.ax.c2p(x, 0),
            end=norm_dist.ax.c2p(x,norm_dist.plt.underlying_function(x)),
            color=RED
        )
        horz_line = DashedLine(
            start=norm_dist.ax.c2p(x,norm_dist.plt.underlying_function(x)),
            end=norm_dist.ax.c2p(norm_dist.x_lower, norm_dist.plt.underlying_function(x)),
            color=RED
        )
        x_lbl = MathTex(x, color=RED).next_to(vert_line, DOWN, buff=.25)

        likelihood_lbl = MathTex(r"\times", color=RED).next_to(horz_line, LEFT, buff=.25)
        grp = VGroup(norm_dist, vert_line, horz_line, x_lbl, likelihood_lbl)

        self.add(grp)
        mobj_to_svg(grp)

class TraceArea(Scene):
    def construct(self):
        norm_dist = CoffeeNormalPDF()
        mean, std = norm_dist.mean, norm_dist.std
        x_lower, x_upper = 15, 16

        area = norm_dist.ax.get_area(norm_dist.plt, x_range=(x_lower,x_upper), color=RED)
        area_lbl = MathTex(round(norm.cdf(x=x_upper, loc=mean, scale=std) -
                                 norm.cdf(x=x_lower, loc=mean, scale=std), 2)
                           ).move_to(area).shift(DOWN)

        grp = VGroup(norm_dist,area, area_lbl)

        self.add(grp)
        mobj_to_svg(grp)

class CoffeeSciPy(Scene):
    def construct(self):
        raw = r"""import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

mean = 14.005
std = 0.955

# Create a grid of x-values
x = np.linspace(-3.5*std+mean, 3.5*std+mean, 1000)

# Calculate the normal probability density function
y = norm.pdf(x, mean, std)

# Create a mask for the area to be shaded
shade = np.logical_and(x >= 15, x <= 16)

# Plot the normal distribution and shade the area of interest
plt.plot(x, y, color='blue', linewidth=2)
plt.fill_between(x, y, where=shade, color='red', alpha=0.5)

# Set the title and axis labels
plt.title('Normal Distribution')
plt.xlabel('Grams of Coffee')
plt.ylabel('Barista Recommends Likelihood')

# Show the plot
plt.show()"""

        code = Code(code=raw, language="Python", font="Monospace", style="monokai", background="window")

        mobj_to_svg(code)

class CoffeeMatplotLibArea(Scene):
    def construct(self):
        raw = r"""import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

mean = 14.005
std = 0.955

# Create a grid of x-values
x = np.linspace(-3.5*std+mean, 3.5*std+mean, 1000)

# Calculate the normal probability density function
y = norm.pdf(x, mean, std)

# Create a mask for the area to be shaded
shade = np.logical_and(x >= 15, x <= 16)

# Plot the normal distribution and shade the area of interest
plt.plot(x, y, color='blue', linewidth=2)
plt.fill_between(x, y, where=shade, color='red', alpha=0.5)

# Set the title and axis labels
plt.title('Normal Distribution')
plt.xlabel('Grams of Coffee')
plt.ylabel('Barista Recommends Likelihood')

# Show the plot
plt.show()"""

        code = Code(code=raw, language="Python", font="Monospace", style="monokai", background="window")

        mobj_to_svg(code)
class CoffeeNormalCDF(VGroup):
    def __init__(self, *vmobjects, **kwargs):
        super().__init__(*vmobjects, **kwargs)
        mean = round(x.mean(),1)
        std = round(x.std(ddof=1))
        self.mean, self.std = mean, std
        self.x_lower, self.x_upper = mean - 3 * std, mean + 3 * std

        self.ax = Axes(x_range=(self.x_lower, self.x_upper, std),
                       y_range=(0, 1, .25),
                       tips=False,
                       x_axis_config={"include_numbers": False,
                                      "numbers_to_include": (mean,
                                                             mean + std,
                                                             mean + std * 2,
                                                             mean - std,
                                                             mean - 2 * std
                                                             )
                                      }
                       ,
                       y_axis_config={"include_numbers" : False,
                                 "numbers_to_include" : (.25, .5, .75, 1.0 )
                                 }
                       )

        self.plt = self.ax.plot(lambda x: norm.cdf(x,mean,std), color=RED)

        grp = VGroup(self.ax, self.plt)
        self.add(grp)

class PDFandCDF(Scene):
    def construct(self):
        pdf = CoffeeNormalPDF()
        cdf = CoffeeNormalCDF()
        grp = VGroup(pdf, cdf).arrange(UP, buff=.5).scale_to_fit_height(7).to_edge(RIGHT)

        pdf_label = Tex("Probability Density Function (PDF)").scale(.7).next_to(pdf, LEFT)
        cdf_label = Tex("Cumulative Density Function (CDF)").scale(.7).next_to(cdf, LEFT)
        grp.add(pdf_label, cdf_label)
        grp.move_to(ORIGIN)

        self.add(grp)
        mobj_to_svg(grp)

class PDFandCDFAnimated(Scene):
    def construct(self):
        pdf = CoffeeNormalPDF()
        cdf = CoffeeNormalCDF()

        mean, std = pdf.mean, pdf.std

        grp = VGroup(pdf, cdf).arrange(UP, buff=.5).scale_to_fit_height(7).to_edge(RIGHT)

        pdf_label = Tex("Probability Density Function (PDF)").scale(.7).next_to(pdf, LEFT)
        cdf_label = Tex("Cumulative Density Function (CDF)").scale(.7).next_to(cdf, LEFT)
        grp.add(pdf_label, cdf_label)
        grp.move_to(ORIGIN)

        self.add(grp)
        vt = ValueTracker(-3*std+mean)
        projecting_line = always_redraw(lambda: DashedLine(
                start=pdf.ax.c2p(vt.get_value(), 0),
                end=cdf.ax.c2p(vt.get_value(), cdf.plt.underlying_function(vt.get_value())),
                color=RED
            )
        )
        horz_projecting_line = always_redraw(lambda: DashedLine(
                start=cdf.ax.c2p(-3*std+mean, cdf.plt.underlying_function(vt.get_value())),
                end=cdf.ax.c2p(vt.get_value(), cdf.plt.underlying_function(vt.get_value())),
                color=RED
            )
        )
        projecting_area = always_redraw(lambda: pdf.ax.get_area(pdf.plt, x_range=(-3*std+mean, vt.get_value()), color=RED))
        self.add(projecting_line, projecting_area, horz_projecting_line)

        self.play(vt.animate.set_value(3*std+mean),run_time=3)
        self.play(vt.animate.set_value(-3*std+mean),run_time=3)
        self.play(vt.animate.set_value(3*std+mean),run_time=3)
        self.play(vt.animate.set_value(-3*std+mean),run_time=3)
        self.play(vt.animate.set_value(mean),run_time=1.5)
        self.wait()


class PDFandCDFPieces(Scene):
    def construct(self):
        pdf = CoffeeNormalPDF()
        cdf = CoffeeNormalCDF()

        mean, std = pdf.mean, pdf.std

        grp = VGroup(pdf, cdf).arrange(UP, buff=.5).scale_to_fit_height(7).to_edge(RIGHT)

        pdf_label = Tex("PDF").scale(.7).next_to(pdf, LEFT)
        cdf_label = Tex("CDF").scale(.7).next_to(cdf, LEFT)
        grp.add(pdf_label, cdf_label)
        grp.to_edge(LEFT)

        self.add(grp)
        vt = ValueTracker(-3*std+mean)
        projecting_line = always_redraw(lambda: DashedLine(
                start=pdf.ax.c2p(vt.get_value(), 0),
                end=cdf.ax.c2p(vt.get_value(), cdf.plt.underlying_function(vt.get_value())),
                color=RED
            )
        )
        horz_projecting_line = always_redraw(lambda: DashedLine(
                start=cdf.ax.c2p(-3*std+mean, cdf.plt.underlying_function(vt.get_value())),
                end=cdf.ax.c2p(vt.get_value(), cdf.plt.underlying_function(vt.get_value())),
                color=RED
            )
        )
        projecting_area = always_redraw(lambda: pdf.ax.get_area(pdf.plt, x_range=(-3*std+mean, vt.get_value()), color=RED))
        self.add(projecting_line, projecting_area, horz_projecting_line)

        self.play(vt.animate.set_value(16))
        area_16 = pdf.ax.get_area(pdf.plt, x_range=(-3*std+mean, 16), color=RED) \
            .move_to(projecting_area)

        area_16.generate_target().scale(.5).to_edge(UR * 2)

        self.wait()
        self.play(MoveToTarget(area_16))
        self.wait()

        subtract_sign = Rectangle(height=.125, width=1, fill_opacity=.8) \
            .next_to(area_16, DOWN, buff=.75)

        area_15 = pdf.ax.get_area(pdf.plt, x_range=(-3*std+mean, 15), color=RED) \
            .move_to(projecting_area)

        area_15.generate_target() \
            .scale(.5) \
            .next_to(subtract_sign, DOWN, buff=.75) \
            .align_to(area_16, LEFT)

        self.play(vt.animate.set_value(15))
        self.wait()

        self.play(
            FadeIn(subtract_sign),
            MoveToTarget(area_15)
        )
        self.wait()

        equals_sign = VGroup(subtract_sign.copy(), subtract_sign.copy()) \
            .arrange(DOWN) \
            .next_to(VGroup(area_15, area_16), DOWN, buff=.75)


        self.play(
            FadeIn(equals_sign)
        )
        self.wait()

if __name__ == "__main__":
    render_scenes(q="l", play=True, scene_names=['PDFandCDFPieces'])
