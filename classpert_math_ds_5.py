import scipy.stats
from scipy.stats import norm

from manim import *
import numpy as np
import pandas as pd
import sympy as sp
import scipy
from sklearn.linear_model import LinearRegression, LogisticRegression
import math

from threemds.utils import render_scenes, file_to_base_64, mobj_to_svg, mobj_to_png

df = pd.read_csv(r"https://raw.githubusercontent.com/thomasnield/machine-learning-demo-data/master/regression/linear_normal.csv")

class LinearFunction(Scene):
    def construct(self):
        ax = Axes(x_range=(0,5,1), y_range=(0,5,1),
                  x_length=5,
                  y_length=5,
                  axis_config={"include_tip": False, "include_numbers" :True})
        plt = ax.plot(lambda x: .75*x + 1, color=YELLOW)
        grp = VGroup(ax, plt)

        tex = MathTex(r"y = mx + b")
        mobj_to_svg(grp, w_padding=2, h_padding=2)
        mobj_to_svg(tex)
        self.add(grp)

class LinearRegressionModel(VGroup):
    def __init__(self, df: pd.DataFrame,
                 x_length=5,
                 y_length=5,
                 *vmobjects,
                 **kwargs):
        super().__init__(*vmobjects, **kwargs)

        # extract columns
        X = df.values[:, :-1]
        Y = df.values[:, -1]

        # Axes and dots
        ax = Axes(x_range=(0,100,10),
                  y_range=(-20,200,20),
                  x_length=x_length,
                  y_length=y_length,
                  axis_config={"include_tip": False}
                  )

        dots = VGroup(*[Dot(ax.c2p(d.x, d.y), color=BLUE) for d in df.itertuples()])

        # LR model and line
        lr = LinearRegression().fit(X,Y)
        m, b = lr.coef_.flatten()[0], lr.intercept_.flatten()[0]
        print(m, b)

        line = Line(start=ax.c2p(0, b),
                    end=ax.c2p(100, m * 100 + b), color=YELLOW)


        # residuals
        residuals = VGroup(*[
            Line(start=ax.c2p(x, y), end=ax.c2p(x, m * x + b), color=RED)
            for d,x,y in zip(dots, X.flatten(), Y.flatten())
            ]
        )

        # squares
        squares = VGroup(*[
            Rectangle(color=RED, fill_opacity=.6) \
                         .stretch_to_fit_height(r.get_length()) \
                         .stretch_to_fit_width(r.get_length()) \
                         .next_to(r, LEFT, buff=0)
            for d,x,y,r in zip(dots, X.flatten(), Y.flatten(), residuals)
            ]
        )

        # package up and add everything
        grp = VGroup(ax, dots, line)
        self.add(grp)

        self.X, self.Y, self.ax, self.dots, self.lr, self.m, self.b, self.line, self.residuals, self.squares = \
            X, Y, ax, dots, lr, m, b, line, residuals, squares


class LinearRegressionSimple(Scene):
    def construct(self):
        lr_model = LinearRegressionModel(df)
        mobj_to_svg(lr_model)
        self.add(lr_model)

class LinearRegressionResiduals(Scene):
    def construct(self):
        lr_model = LinearRegressionModel(df.sample(n=10, random_state=15))
        residuals = lr_model.residuals
        grp = VGroup(lr_model, residuals)
        mobj_to_svg(grp)
        self.add(grp)

class ResidualsCode(Scene):
    def construct(self):
        raw = """import numpy as np

x = np.array([84,37,58,52,47,78,93,15,12,60])
y = np.array([155.8,102.0,164.8,120.9,86.8,93.0,201.6,25.2,14.7,118.6])

m, b = 1.8561908186503653, 8.848172120340422

y_predict = m*x + b
residuals = y_predict - y

for r in residuals:
    print(r)"""

        code = Code(code=raw, language="Python", font="Monospace", style="monokai", background="window")

        output=Code(code="""OUTPUT:
8.96820088697109
-24.47276758959606
-48.29276039793841
-15.529905309840586
9.289140596907586
60.631055975068904
-20.12608174517561
11.491034400095902
16.422461944144807
1.6196212393623455""", language="text",  font="Monospace", background="window")

        mobj_to_svg(output)

class LinearRegressionSquares(Scene):
    def construct(self):
        lr_model = LinearRegressionModel(df.sample(n=10, random_state=15))
        squares = lr_model.squares
        grp = VGroup(lr_model, squares)
        mobj_to_svg(grp)
        self.add(grp)

class SquaresCode(Scene):
    def construct(self):
        raw = """import numpy as np

x = np.array([84,37,58,52,47,78,93,15,12,60])
y = np.array([155.8,102.0,164.8,120.9,86.8,93.0,201.6,25.2,14.7,118.6])

m, b = 1.8561908186503653, 8.848172120340422

y_predict = m*x + b
squares = (y_predict - y)**2

sum_of_squares = sum(squares)
mean_of_squares = sum(squares) / len(squares)

print(sum_of_squares) # 7824.550195373367
print(mean_of_squares) # 782.4550195373367"""

        code = Code(code=raw, language="Python", font="Monospace", style="monokai", background="window")

        mobj_to_svg(code)

class StackOnesOnX(Scene):
    def construct(self):

        x = np.array([84, 37, 58, 52, 47, 78, 93, 15, 12, 60])
        y = np.array([155.8, 102.0, 164.8, 120.9, 86.8, 93.0, 201.6, 25.2, 14.7, 118.6])
        x_1 = np.vstack([x, np.ones(len(x))]).T

        tbl = Table([[str(_x)] for _x in x],
                    col_labels=(Tex("x")),
                    include_outer_lines=True
                    )
        tbl1 = Table([[str(int(_x_1[0])), str(int(_x_1[1]))] for _x_1 in x_1],
                     col_labels=(Tex("x"),Tex("x1")),
                     include_outer_lines=True,
                     )

        arrow = Arrow(start=LEFT, end=RIGHT, color=YELLOW)

        grp = VGroup(tbl, arrow, tbl1).arrange(RIGHT)
        mobj_to_svg(grp)


class StackOnesOnXCode(Scene):
    def construct(self):
        raw = """import numpy as np

x = np.array([84,37,58,52,47,78,93,15,12,60])
y = np.array([155.8,102.0,164.8,120.9,86.8,93.0,201.6,25.2,14.7,118.6])

x_1 = np.vstack([x, np.ones(len(x))]).T

print(x_1)
"""

        code = Code(code=raw, language="Python", font="Monospace", style="monokai", background="window")

        output=Code(code="""OUTPUT:
[[84.  1.]
 [37.  1.]
 [58.  1.]
 [52.  1.]
 [47.  1.]
 [78.  1.]
 [93.  1.]
 [15.  1.]
 [12.  1.]
 [60.  1.]]""", language="text",  font="Monospace", background="window")

        mobj_to_svg(output)

class InverseMatrixTechnique(Scene):
    def construct(self):
        tex = MathTex(r"b = (X^T \cdot X)^{-1} \cdot X^T \cdot Y")
        mobj_to_svg(tex)


class TransposeMatrix(Scene):
    def construct(self):
        m = sp.Matrix([[1,2],[3,4]])
        grp = VGroup(
            MathTex("A = ", sp.latex(m)),
            MathTex("A^{-1} = ", sp.latex(m.transpose()))
        )
        grp.arrange(RIGHT, buff=1)
        mobj_to_svg(grp)

class InverseMatrixTechniqueCode(Scene):
    def construct(self):
        raw = """import numpy as np
from numpy.linalg import inv

x = np.array([84,37,58,52,47,78,93,15,12,60])
y = np.array([155.8,102.0,164.8,120.9,86.8,93.0,201.6,25.2,14.7,118.6])

x_1 = np.vstack([x, np.ones(len(x))]).T

coeffs = inv(x_1.T @ x_1) @ (x_1.T @ y)

print(coeffs) # [1.85619082 8.84817212]"""

        code = Code(code=raw, language="Python", font="Monospace", style="monokai", background="window")

        mobj_to_svg(code)

class QRFormula(Scene):
    def construct(self):
        qr_tex = MathTex(r"X = Q \cdot R")
        qr_solve = MathTex(r"b = R^{-1} \cdot Q^{T} \cdot Y")

        mobj_to_svg(qr_solve)

class QRTechniqueCode(Scene):
    def construct(self):
        raw = """import numpy as np
from numpy.linalg import qr, inv

x = np.array([84,37,58,52,47,78,93,15,12,60])
y = np.array([155.8,102.0,164.8,120.9,86.8,93.0,201.6,25.2,14.7,118.6])

x_1 = np.vstack([x, np.ones(len(x))]).T

Q, R = qr(x_1)
coeffs = inv(R) @ Q.transpose() @ y

print(coeffs) # [1.85619082 8.84817212]"""

        code = Code(code=raw, language="Python", font="Monospace", style="monokai", background="window")

        mobj_to_svg(code)

class LinearRegressionLoss(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=70 * DEGREES, theta=-70 * DEGREES, zoom=.5)

        ax = ThreeDAxes(x_range=(-10, 10, 1), y_range=(-10, 10, 1), z_range=(-1_000_000, 5_000_000, 1_000_000))

        _x = np.array([84,37,58,52,47,78,93,15,12,60])
        _y = np.array([155.8,102.0,164.8,120.9,86.8,93.0,201.6,25.2,14.7,118.6])

        m, b, i, n = sp.symbols('m b i n')
        x, y = sp.symbols('x y', cls=sp.Function)

        _sum_of_squares = sp.Sum((m * x(i) + b - y(i)) ** 2, (i, 0, n)) \
            .subs(n, len(_x) - 1).doit() \
            .replace(x, lambda i: _x[i]) \
            .replace(y, lambda i: _y[i])

        sum_of_squares = sp.lambdify([m, b], _sum_of_squares)

        def param_loss(u, v):
            m = u
            b = v
            s = sum_of_squares(m, b)
            return ax.c2p(m, b, s)

        loss_plane = Surface(
            param_loss,
            resolution=(42, 42),
            u_range=[-10, +10],
            v_range=[-10, +10],
            fill_opacity=.5,
            stroke_color=BLUE,
            fill_color=BLUE
        )

        self.add(ax, loss_plane)

class SymPyDerivatives(Scene):
    def construct(self):
        code = Code(code="""from sympy import *

m, b, i, n = symbols('m b i n')
x, y = symbols('x y', cls=Function)

sum_of_squares = Sum((m*x(i) + b - y(i)) ** 2, (i, 0, n))

d_m = diff(sum_of_squares, m)
d_b = diff(sum_of_squares, b)
print(d_m)
print(d_b)

# OUTPUTS
# Sum(2*(b + m*x(i) - y(i))*x(i), (i, 0, n))
# Sum(2*b + 2*m*x(i) - 2*y(i), (i, 0, n))""", language="Python", font="Monospace", style="monokai", background="window")

        mobj_to_svg(code)

class DerivativesMandB(Scene):
    def construct(self):

        tex = MathTex(r"\frac{\partial}{\partial m} &= \sum_{i=0}^{n} 2(b + mx_i - y_i)x_i \\",
                    r"\frac{\partial}{\partial b} &= \sum_{i=0}^{n} 2 b + 2 m x_i - 2 y_i")

        mobj_to_svg(tex)

class GradientDescentCode(Scene):
    def construct(self):
        code = Code(code="""import numpy as np

x = np.array([84,37,58,52,47,78,93,15,12,60])
y = np.array([155.8,102.0,164.8,120.9,86.8,93.0,201.6,25.2,14.7,118.6])

# Building the model
m, b, L, iterations = 0.0, 0.0, .00001, 500_000

# Perform Gradient Descent
for i in range(iterations):
    # slope with respect to m
    d_m = np.sum(2 * (b + m*x - y) * x)
    # slope with respect to b
    d_b = np.sum(2*b + 2*m*x - 2*y)
    # update m and ba
    m -= L * d_m
    b -= L * d_b
    print(f"y = {m}x + {b}")

# last iteration prints
# y = 1.8561908195210568x + 8.848172062832369""", language="Python", font="Monospace", style="monokai", background="window")

        mobj_to_svg(code)

class LinearRegressionSklearn(Scene):
    def construct(self):

        code = Code(code="""import numpy as np
from sklearn.linear_model import LinearRegression

x = np.array([84,37,58,52,47,78,93,15,12,60])
y = np.array([155.8,102.0,164.8,120.9,86.8,93.0,201.6,25.2,14.7,118.6])

lr = LinearRegression().fit(x.reshape(-1, 1),y)

m,b = lr.coef_[0], lr.intercept_

# y = 1.8561908186503653x + 8.848172120340422
print(f"y = {m}x + {b}")""", language="Python", font="Monospace", style="monokai", background="window")

        mobj_to_svg(code)

class BinaryOutcomeTable(Scene):
    def construct(self):
        # data
        x = np.array([7.8, 7.9, 4.0, 7.8, 6.4, 7.2, 6.6, 7.6, 4.9, 5.8, 4.8, 4.0, 4.4, 3.0, 2.1, 3.8, 2.5, 5.6, 3.7, 3.3, 3.4, 4.3, 3.4, 0.7])
        y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        tbl = Table([[str(_x), str(_y)] for _x,_y in zip(x,y)],
                    col_labels=(Tex("x"), Tex("y")),
                    include_outer_lines=True
                    )

        mobj_to_svg(tbl)

class SimpleLogisticRegression(VGroup):
    def __init__(self, *vmobjects, **kwargs):
        super().__init__()

        # data
        x = np.array([7.8, 7.9, 4.0, 7.8, 6.4, 7.2, 6.6, 7.6, 4.9, 5.8, 4.8, 4.0, 4.4, 3.0, 2.1, 3.8, 2.5, 5.6, 3.7, 3.3, 3.4, 4.3, 3.4, 0.7])
        y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        # axes, dots, and top line
        ax = Axes(x_range=(x.min()-1, x.max()+1, 1),
                  y_range=(0,1.25, .25),
                  x_axis_config={
                      "include_numbers" : True,
                  },
                  y_axis_config= {
                      "include_numbers" : True
                  }#,
                  #x_length=7,
                  #y_length=5,
            )

        lbls = VGroup(
            Tex("Hours of Rain") \
                .scale(.75) \
                .next_to(ax, DOWN, buff=.5),

            Tex("P(Flood)") \
                .scale(.75) \
                .rotate(90 * DEGREES) \
                .next_to(ax, LEFT, buff=.5)
        )

        dots = VGroup(*[
            Dot(ax.c2p(_x,_y), color=BLUE if _y == 1 else RED, radius=.1) for _x,_y in zip(x,y)
        ])

        top_line = DashedLine(start=ax.c2p(0,1), end=ax.c2p(x.max()+1,1), color=WHITE)

        # logistic regression model and plot
        lr_model = LogisticRegression(penalty=None)
        lr_model.fit(x.reshape(-1, 1), y)

        plot = ax.plot(lambda _x: 1.0 / (1.0 + math.exp(-(-7.73556604 + 1.68361151*_x))),
                       color=YELLOW)

        grp = VGroup(ax, top_line, plot, dots, lbls).scale(.9)

        self.add(grp)

        # likelihood lines
        likelihood_lines = VGroup(*[
            DashedLine(start=ax.c2p(_x, _y),
                       end=ax.c2p(_x, plot.underlying_function(_x)),
                       color=BLUE if _y == 1 else RED)
                for d, _x, _y in zip(dots, x, y)
        ])

        likelihood_lines_normalized = VGroup(*[
            DashedLine(start=ax.c2p(_x, 1),
                       end=ax.c2p(_x, plot.underlying_function(_x)),
                       color=BLUE if _y == 1 else RED)
                for d, _x, _y in zip(dots, x, y)
        ])

        dots_normalized = VGroup(*[
            d.copy().move_to(l.get_top()) for d,l in zip(dots, likelihood_lines_normalized)
        ])

        self.x, self.y, self.ax, self.dots, self.dots_normalized, \
            self.top_line, self.plot, self.lr_model, self.likelihood_lines, \
            self.likelihood_lines_normalized, self.axis_lbls = \
            x, y, ax, dots, dots_normalized, top_line, plot, lr_model, likelihood_lines, likelihood_lines_normalized, lbls


class BasicLogisticRegression(Scene):
    def construct(self):
        lr = SimpleLogisticRegression()


        lookup_line_vert = DashedLine(start= lr.ax.c2p(5,0),
                                      end = lr.ax.c2p(5, lr.plot.underlying_function(5)),
                                      color=YELLOW)

        lookup_line_horz = DashedLine(start = lr.ax.c2p(5, lr.plot.underlying_function(5)),
                                      end= lr.ax.c2p(0, lr.plot.underlying_function(5)),
                                      color=YELLOW)

        y_lookup_lbl = MathTex(round(lr.plot.underlying_function(5), 2), color=YELLOW) \
            .scale(.6) \
            .next_to(lookup_line_horz, LEFT, buff=.1)

        lr = VGroup(lr.ax, lr.top_line,
                    lr.plot,
                    lr.dots,
                    lookup_line_vert,
                    lookup_line_horz,
                    lr.axis_lbls,
                    y_lookup_lbl)

        mobj_to_svg(lr,h_padding=2, w_padding=2)
        self.add(lr)

class LogisticRegressionFormula(Scene):
    def construct(self):
        tex = MathTex(r"L(x) = \frac{1}{1 + e^{-(mx + b)}")
        mobj_to_svg(tex)

class LogisticRegressionSklearn(Scene):
    def construct(self):

        code = Code(code="""import numpy as np
from sklearn.linear_model import LogisticRegression

x = np.array([7.8,7.9,4.0,7.8,6.4,7.2,6.6,7.6,4.9,5.8,4.8,4.0,
              4.4,3.0,2.1,3.8,2.5,5.6,3.7,3.3,3.4,4.3,3.4,0.7])

y = np.array([1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0])

lr = LogisticRegression().fit(x.reshape(-1, 1),y)
m,b = lr.coef_[0,0], lr.intercept_[0]

# L(x) = 1 / (1 + e^(-(1.2436430952492856x + -5.798891442057214)
print(f"L(x) = 1 / (1 + e^(-({m}x + {b})")""", language="Python", font="Monospace", style="monokai", background="window")

        mobj_to_svg(code)

class ReshapeCode(Scene):
    def construct(self):

        code = Code(code="""import numpy as np

x = np.array([1, 2, 3])

print(x) 
# [1 2 3]

print(x.reshape(-1,1))
# [[1]
#  [2]
#  [3]]""", language="Python", font="Monospace", style="monokai", background="window")

        mobj_to_svg(code)

if __name__ == "__main__":
    render_scenes(q="l", play=False, scene_names=['ReshapeCode'])
