import scipy.stats
from scipy.stats import norm

from manim import *
import numpy as np
import pandas as pd
import sympy as sp
import scipy
from sklearn.linear_model import LinearRegression
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

if __name__ == "__main__":
    render_scenes(q="l", play=False, scene_names=['InverseMatrixTechnique'])
