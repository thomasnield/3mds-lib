import pandas as pd
from math import sqrt

from manim import *
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def create_model() -> tuple:
    data = list(pd.read_csv("https://bit.ly/2KF29Bd").itertuples())
    m = ValueTracker(1.93939)
    b = ValueTracker(4.73333)

    ax = Axes(
        x_range=[0, 10],
        y_range=[0, 25, 5],
        axis_config={"include_tip": False},
    )
    # plot points
    points = [Dot(point=ax.c2p(p.x, p.y), radius=.15, color=BLUE) for p in data]

    # plot function
    line = Line(start=ax.c2p(0, b.get_value()), end=ax.c2p(10, m.get_value() * 10 + b.get_value())).set_color(YELLOW)

    # make line follow m and b value
    line.add_updater(
        lambda l: l.become(
            Line(start=ax.c2p(0, b.get_value()), end=ax.c2p(10, m.get_value() * 10 + b.get_value()))).set_color(YELLOW)
    )

    return data, m, b, ax, points, line


class EquationScene(Scene):

    def construct(self):
        # three versions of linear function
        eq1 = MathTex("y", "=m", r"x", "+b")
        eq1a = MathTex("f(x)", "=m", r"x", "+b")

        eq2 = MathTex(r"y", r"= \beta_1", "x", r"+ \beta_0")

        eq3 = MathTex(r"y = \beta_2 x_2 + \beta_1 x_1 + \beta_0")
        eq4 = MathTex(r"y = \beta_3 x_3 + \beta_2 x_2 + \beta_1 x_1 + \beta_0")
        eq5 = MathTex(r"y = \beta_4 x_4 + \beta_3 x_3 + \beta_2 x_2 + \beta_1 x_1 + \beta_0")
        eq6 = MathTex(r"y = \beta_5 x_5 + \beta_4 x_4 + \beta_3 x_3 + \beta_2 x_2 + \beta_1 x_1 + \beta_0")

        framebox1 = SurroundingRectangle(eq1[0])
        framebox1a = SurroundingRectangle(eq1a[0])

        framebox2 = SurroundingRectangle(eq1a[2])

        # populate equation
        self.play(
            Write(eq1)
        )
        self.wait(2)

        text1 = Text("Output", font_size=14, color=YELLOW).next_to(framebox1, DOWN)
        text2 = Text("Input", font_size=14, color=YELLOW).next_to(framebox2, DOWN)

        self.play(
            Write(framebox1),
            Write(text1)
        )
        self.wait(2)
        self.play(
            ReplacementTransform(eq1, eq1a),
            ReplacementTransform(framebox1, framebox1a)
        )
        self.wait(2)

        self.play(
            ReplacementTransform(framebox1a, framebox2),
            ReplacementTransform(text1, text2)
        )
        self.wait(2)

        self.play(
            FadeOut(text2, framebox2)
        )
        self.wait(2)

        self.play(
            ReplacementTransform(eq1a, eq2)
        )

        self.wait(2)

        self.play(
            ReplacementTransform(eq2, eq3)
        )
        self.wait(1)
        self.play(
            ReplacementTransform(eq3, eq4)
        )
        self.wait(1)
        self.play(
            ReplacementTransform(eq4, eq5)
        )
        self.wait(1)
        self.play(
            ReplacementTransform(eq5, eq6)
        )
        self.wait(2)

class FirstScene(Scene):

    def construct(self):
        data, m, b, ax, points, line = create_model()

        # add elements to VGroup
        graph = VGroup(ax, *points)

        # three versions of linear function
        eq1 = MathTex("f(x) = ", r"m ", r"x + ", "b").move_to((RIGHT + DOWN))
        eq1[1].set_color(RED)
        eq1[3].set_color(RED)

        eq2 = MathTex(r"f(x) = ", r"\beta_1", r"x + ", r"\beta_0").move_to((RIGHT + DOWN))
        eq2[1].set_color(RED)
        eq2[3].set_color(RED)

        eq3 = MathTex("f(x) = ", f'{m.get_value()}', r"x + ", f'{b.get_value()}').move_to((RIGHT + DOWN))
        eq3[1].set_color(RED)
        eq3[3].set_color(RED)

        # populate charting area
        self.play(
            DrawBorderThenFill(graph),
            run_time=2.0
        )

        # draw line
        self.play(
            Create(line),
            run_time=2.0
        )

        # transform the math equation to three variants
        # equation 1 create
        self.play(
            Create(eq1)
        )

        self.wait()

        # animate the coefficients m and b
        def blink(item, value, increment):
            self.play(ScaleInPlace(item, 4 / 3), value.animate.increment_value(increment))

            for i in range(0, 1):
                self.play(ScaleInPlace(item, 3 / 4), value.animate.increment_value(-2 * increment))
                self.play(ScaleInPlace(item, 4 / 3), value.animate.increment_value(2 * increment))

            self.play(ScaleInPlace(item, 3 / 4), value.animate.increment_value(-increment))
            self.wait()

        blink(eq1[1], m, .50)
        blink(eq1[3], b, 2.0)

        self.wait()

        # transform to beta coefficients
        self.play(ReplacementTransform(eq1, eq2))

        self.wait()

        # transform with coefficent values
        self.play(
            ReplacementTransform(
                eq2,
                eq3
            )
        )

        self.wait()

        # remove equation
        self.play(
            FadeOut(eq3, shift=DOWN),
        )


class NormalDistTrace(Scene):

    def construct(self):
        data, m, b, ax, points, line = create_model()

        self.add(ax, line, *points)

        mean = 0
        std_dev = 1
        graph = FunctionGraph(
            lambda x: (1 / (std_dev * np.sqrt(2 * PI))) * np.exp(-.5 * ((x - mean) / std_dev) ** 2),
            x_range=[-3, 3],
            color=RED
        ).stretch_to_fit_height(3) \
            .scale(.5)

        note_text = Text("A normal distribution follows the line!", font_size=26) \
            .shift(RIGHT * 2 + DOWN * 2)

        self.play(
            LaggedStart(
                FadeIn(note_text),
                MoveAlongPath(graph, line)
            ),
            rate_func=linear,
            run_time=3
        )

class ThreeDScatter(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=70 * DEGREES, theta=-30 * DEGREES)

        axes = ThreeDAxes()
        self.add(axes)

        """ # used to generate data 
        x1 = np.random.random(15) * 2.0
        x2 = np.random.random(15) * 2.0

        y = (.3 * x1 + .6 * x2 - 1)  + np.random.normal(0,.5,15)

        print(np.array([x1,x2,y]).transpose())
        pd.DataFrame(np.array([x1,x2,y]).transpose()).to_csv("foo.csv",index=False)
        """

        data = list(pd.read_csv("https://bit.ly/35ebET5").itertuples())

        points = []
        for i in range(0, 45):
            points += Sphere(center=(data[i].x1, data[i].x2, data[i].y), radius=.05)


        def param_gauss(u, v):
            x1 = u
            x2 = v
            y = 0.3192737*x1 + 0.6086176*x2 -1.0212873
            return np.array([x1, x2, y])

        linear_plane = Surface(
            param_gauss,
            resolution=(42, 42),
            v_range=[-5, +5],
            u_range=[-5, +5]
        )

        linear_plane.set_style(fill_opacity=.5,stroke_color=BLUE)
        linear_plane.set_fill(BLUE, opacity=0.2)

        self.play(*[Write(p) for p in points], run_time=.1)

        self.play(Write(linear_plane), run_time=2)


def create_residual_model(scene,data,m,b,ax,points,line) -> tuple:
  residuals: list[Line] = []
  for d in data:
    residual = Line(start=ax.c2p(d.x, d.y), end=ax.c2p(d.x, m.get_value() * d.x + b.get_value())).set_color(RED)
    scene.play(Create(residual), run_time=.3)
    residual.add_updater(lambda r,d=d: r.become(Line(start=ax.c2p(d.x, d.y), end=ax.c2p(d.x, m.get_value()*d.x+b.get_value())).set_color(RED)))
    residuals += residual

  # flex residuals changing the coefficients m and b
  def flex_residuals():
    m_delta=1.1
    scene.play(m.animate.increment_value(m_delta))
    for i in (-1,1,-1,1):
        scene.play(m.animate.increment_value(i*m_delta))
        scene.play(m.animate.increment_value(i*m_delta))
    scene.play(m.animate.increment_value(-m_delta))

    scene.wait()

  return residuals, flex_residuals


class SecondScene(Scene):

    def construct(self):
        # add base graph
        data,m,b,ax,points,line = create_model()
        self.add(ax,line,*points)

        residuals, flex_residuals = create_residual_model(self,data,m,b,ax,points,line)

        flex_residuals()


class ThirdScene(Scene):

    def construct(self):
        # add base graph
        data, m, b, ax, points, line = create_model()
        self.add(ax, line, *points)

        residuals, flex_residuals = create_residual_model(self, data, m, b, ax, points, line)

        squares: list[Square] = []
        for i, d in enumerate(data):
            square = Square(color=RED,
                            fill_opacity=.6,
                            side_length=residuals[i].get_length()
                            ).next_to(residuals[i], LEFT, 0)

            square.add_updater(lambda s=square, r=residuals[i]: s.become(
                Square(color=RED,
                       fill_opacity=.6,
                       side_length=r.get_length()
                       ).next_to(r, LEFT, 0)
            ))

            squares += square
            self.play(Create(square), run_time=.1)

        flex_residuals()
        length = 0.0

        for s in squares:
            length = sqrt(length ** 2 + s.side_length ** 2)
            total_square = Square(color=RED, fill_opacity=1, side_length=length).move_to(3 * LEFT + 2.5 * UP)
            self.play(
                ReplacementTransform(s, total_square),
                run_time=.3
            )

        self.play(DrawBorderThenFill(Text("SSE").scale(.8).move_to(total_square)))
        self.wait()


from sympy import symbols, Function, Sum, lambdify, diff

class LinearRegressionLoss3D(ThreeDScene):
    def construct(self):

        self.set_camera_orientation(phi=70 * DEGREES, theta=-30 * DEGREES)

        loss_axes = ThreeDAxes(x_range=(-10, 10, 1), y_range=(-10, 10, 1), z_range=(-1, 100_000, 10_000))

        points = list(pd.read_csv("https://bit.ly/2KF29Bd").itertuples())
        m, b, i, n = symbols('m b i n')
        x, y = symbols('x y', cls=Function)

        _sum_of_squares = Sum((m * x(i) + b - y(i)) ** 2, (i, 0, n)) \
            .subs(n, len(points) - 1).doit() \
            .replace(x, lambda i: points[i].x) \
            .replace(y, lambda i: points[i].y)

        sum_of_squares = lambdify([m, b], _sum_of_squares)

        def param_loss(u, v):
            m = u
            b = v
            y = sum_of_squares(m, b)
            return loss_axes.c2p(m, b, y)

        loss_plane = Surface(
            param_loss,
            resolution=(42, 42),
            v_range=[-10, +10],
            u_range=[-10, +10]
        )
        loss_plane.set_style(fill_opacity=.5, stroke_color=BLUE)
        loss_plane.set_fill(BLUE)

        # self.begin_ambient_camera_rotation(rate=2.0*PI/10.0)

        d_m = lambdify([m, b], diff(_sum_of_squares, m))
        d_b = lambdify([m, b], diff(_sum_of_squares, b))

        # Building the model
        m = -5.0
        b = 5.0

        # The learning Rate
        L = .0025

        # The number of iterations
        iterations = 1000

        path = VMobject(stroke_color=RED)

        dot = Sphere(radius=.05, stroke_color=YELLOW, fill_opacity=.2).move_to(param_loss(m, b))
        path.set_points_as_corners([dot.get_center(), dot.get_center()])

        def update_path(path):
            previous_path = path.copy()
            previous_path.add_points_as_corners([dot.get_center()])
            path.become(previous_path)

        path.add_updater(update_path)

        path_points = []

        # Perform Gradient Descent
        for i in range(iterations):
            # update m and b
            m -= d_m(m, b) * L
            b -= d_b(m, b) * L
            print(m, b)
            if round(b, 4) == 4.7333 and round(m, 4) == 1.9394:
                break

            path_points.append((m, b))

        m_b_label = always_redraw(lambda: MathTex("m &= ", round(m, 4),
                                                  "\\\\b &= ", str(round(b, 4)),
                                                  font_size=40).move_to(LEFT * 5 + DOWN * 2))

        right_graph = VGroup(loss_plane, path, dot, loss_axes).move_to(RIGHT * 3.6 + UP).scale(.4)

        self.add(right_graph)
        self.add_fixed_in_frame_mobjects(m_b_label)

        # line graph
        linear_axes = Axes(x_range=(0, 10, -1), y_range=(0, 26, 2))
        points = [Dot(point=linear_axes.c2p(p.x, p.y), radius=.25, color=BLUE) for p in points]

        line = Line(start=linear_axes.c2p(b, 0), end=linear_axes.c2p(10, m * 10 + b))

        left_graph = VGroup(linear_axes, line, *points).move_to(LEFT * 3.3 + UP).scale(.4)

        self.add_fixed_in_frame_mobjects(left_graph)

        for i, p in enumerate(path_points):
            m = p[0]
            b = p[1]

            self.play(
                dot.animate.move_to(param_loss(p[0], p[1])),
                Transform(line, Line(start=linear_axes.c2p(10, m * 10 + b), end=linear_axes.c2p(0, b))),
                run_time=max(.1, 1.0 - (i * .01))
            )

        self.wait(4)

    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split

def create_train_test_model() -> tuple:
    # data = list(pd.read_csv("https://bit.ly/3TUCgh2").itertuples())

    df = pd.read_csv('https://bit.ly/3TUCgh2', delimiter=",")

    X = df.values[:, :-1]
    Y = df.values[:, -1]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1 / 3, random_state=7)

    model = LinearRegression()
    fit = model.fit(X_train, Y_train)
    result = model.score(X_test, Y_test)

    m = ValueTracker(fit.coef_.flatten()[0])
    b = ValueTracker(fit.intercept_.flatten()[0])

    ax = Axes(
        x_range=[0, 100, 20],
        y_range=[-40, 200, 40],
        axis_config={"include_tip": False},
    )
    # plot points
    train_points = [Dot(point=ax.c2p(p.x, p.y), radius=.15, color=BLUE) for p in
                    pd.DataFrame(data={'x': X_train.flatten(), 'y': Y_train.flatten()}).itertuples()]
    test_points = [Dot(point=ax.c2p(p.x, p.y), radius=.15, color=BLUE) for p in
                   pd.DataFrame(data={'x': X_test.flatten(), 'y': Y_test.flatten()}).itertuples()]

    # plot function
    line = Line(start=ax.c2p(0, b.get_value()), end=ax.c2p(200, m.get_value() * 200 + b.get_value())).set_color(
        YELLOW)

    # make line follow m and b value
    line.add_updater(
        lambda l: l.become(
            Line(start=ax.c2p(0, b.get_value()), end=ax.c2p(200, m.get_value() * 200 + b.get_value()))).set_color(
            YELLOW)
    )

    return m, b, ax, train_points, test_points, line


class TrainTestScene(Scene):

    def construct(self):
        m, b, ax, train_points, test_points, line = create_train_test_model()

        train_group = VGroup(*train_points)
        test_group = VGroup(*test_points)

        # populate charting area
        self.add(ax, train_group, test_group)

        # Move train/test split to box visual
        training_rect = Rectangle(color=BLUE, height=1.5, width=1.5, fill_opacity=.8).shift(LEFT * 4 + UP * 2.5)
        training_text = Text("TRAIN").scale(.6).move_to(training_rect)
        training_rect_grp = VGroup(training_rect, training_text)

        test_rect = Rectangle(color=RED, height=1.5 / 2, width=1.5, fill_opacity=.8).next_to(training_rect, DOWN)
        test_text = Text("TEST").scale(.6).move_to(test_rect)
        test_rect_grp = VGroup(test_rect, test_text)

        self.wait()
        self.play(*[p.animate.set_fill(RED) for p in test_points])
        self.wait()

        # copy training and testing points for reverse transformation
        train_points_copy = [p.copy() for p in train_points]
        test_points_copy = [p.copy() for p in test_points]

        self.play(ReplacementTransform(test_group, test_rect_grp))
        self.wait()
        self.play(ReplacementTransform(train_group, training_rect_grp))

        b1 = Brace(test_rect, direction=RIGHT)
        b1_label = MathTex(r"\frac{1}{3}").scale(.7).next_to(b1, RIGHT)
        self.play(Create(b1), Create(b1_label))

        b2 = Brace(training_rect, direction=RIGHT)
        b2_label = MathTex(r"\frac{2}{3}").scale(.7).next_to(b2, RIGHT)
        self.play(Create(b2), Create(b2_label))

        self.wait()
        self.play(FadeOut(b1), FadeOut(b2), FadeOut(b1_label), FadeOut(b2_label))
        self.wait()
        train_points_copy_grp = VGroup(*train_points_copy)
        test_points_copy_grp = VGroup(*test_points_copy)

        self.play(ReplacementTransform(training_rect_grp, train_points_copy_grp))
        self.wait()
        self.play(Create(line))
        self.wait()
        self.play(FadeOut(train_points_copy_grp))
        self.play(ReplacementTransform(test_rect_grp, test_points_copy_grp))
        self.wait()



class MetricScene(Scene):

  def construct(self):

      title = Text("Performance Metrics", color=BLUE).shift(UP * 3 + LEFT * 3)
      self.play(Create(title))
      self.wait()

      term0 = MathTex(r"R^2", color=YELLOW)
      term0desc = MathTex(r"\text{ - Scores the variation predictability of variables between 0 and 1.").next_to(term0, RIGHT)
      term0grp = VGroup(term0, term0desc) \
        .scale(.60) \
        .next_to(title, DOWN, buff=1) \
        .align_to(title, LEFT)

      self.play(Create(term0grp))
      self.wait()

      term1 = MathTex(r"\text{Standard Error of the Estimate}", color=YELLOW)
      term1desc = MathTex(r"\text{ - Measures the overall error of the linear regression.").next_to(term1, RIGHT)
      term1grp = VGroup(term1, term1desc) \
        .scale(.60) \
        .next_to(term0, DOWN, buff=1) \
        .align_to(title, LEFT)

      self.play(Create(term1grp))
      self.wait()

      term2 = MathTex(r"\text{Prediction Interval}", color=YELLOW)
      term2desc = MathTex(r"\text{ - Makes a prediction as a statistical range rather than a single value.}").next_to(term2, RIGHT)
      term2grp = VGroup(term2, term2desc) \
        .scale(.60) \
        .next_to(term1, DOWN, buff=1) \
        .align_to(title, LEFT)

      self.play(Create(term2grp))
      self.wait()

      term3 = MathTex(r"\text{Statistical Significance}", color=YELLOW)
      term3desc = MathTex(r"\text{ - Uses a p-value to determine probability linear relationship is coincidental.}").next_to(term3, RIGHT)
      term3grp = VGroup(term3, term3desc) \
        .scale(.60) \
        .next_to(term2, DOWN, buff=1) \
        .align_to(title, LEFT)

      self.play(Create(term3grp))
      self.wait()