from manim import *

from threemds.utils import render_scenes, mobj_to_svg
import urllib.request

class LogoScene(Scene):

    def construct(self):
        circle = Circle(1.0, color=BLUE)
        rectangle = Rectangle(height=1.0, width=2.0, color=BLUE).move_to(circle, UP)

        handle = RoundedRectangle(corner_radius=.5, height=1.5, width=2.0, color=BLUE) \
            .move_to(circle).shift(LEFT * .8)

        handle_inner = Difference(handle, handle.copy().scale(.8), color=BLUE, fill_opacity=0.0)

        cup = VGroup(circle, rectangle, handle_inner)

        self.play(
            Create(cup)
        )

        self.play(
            cup.animate.set_fill(BLUE, opacity=1)
        )

        # create sin wave steam

        def get_sine_wave(dx=0):
            return FunctionGraph(
                lambda x: np.sin((x + dx)),
                x_range=[-3, 3]
            )

        sine_function = get_sine_wave()
        d_theta = ValueTracker(0)

        def update_wave(func):
            func.become(
                get_sine_wave(dx=d_theta.get_value())
            )
            return func

        sine_function.add_updater(update_wave)

        self.play(Create(sine_function))
        self.play(d_theta.animate.increment_value(4 * PI), run_time=2)

        # create steam functions
        steam_functions = []
        steam_waves = VGroup()
        for i in range(3):
            steam_function = get_sine_wave() \
                .rotate(PI / 2.0) \
                .scale(.2) \
                .next_to(cup, UP) \
                .shift([i * .5, 0, 0])

            steam_waves.add(steam_function)

            d_theta = ValueTracker(0)

            def update_wave(func, d=d_theta, i=i):
                func.become(
                    get_sine_wave(dx=d.get_value()) \
                        .rotate(PI / 2.0) \
                        .scale(.2) \
                        .next_to(cup, UP) \
                        .shift([i * .5, 0, 0])
                )
                return func

            steam_function.add_updater(update_wave)

            steam_functions += d_theta

            self.play(Create(steam_function), run_time=.3)

        text = Text("3-Minute Data Science").scale(.8).shift(DOWN * 1.5)
        self.play(Write(text), run_time=.5)
        mobj_to_svg(VGroup(cup, steam_waves, sine_function), 'logo.svg', h_padding=1)
        self.play(*(d.animate.increment_value(4 * PI) for d in steam_functions), run_time=8, rate_func=linear)
        self.wait()


class TitleScene(Scene):

    def construct(self):
        title = Text("Neural Networks")
        subtitle = Text("in 3 Minutes",color=BLUE).scale(.75).next_to(title, DOWN)

        self.play(FadeIn(title), FadeIn(subtitle), run_time=2)
        self.wait()
        self.play(FadeOut(title), FadeOut(subtitle), run_time=2)


class ClosingCard(Scene):
  def construct(self):
    urllib.request.urlretrieve(r"https://images-na.ssl-images-amazon.com/images/I/51yHtuQ9wAL._SX379_BO1,204,203,200_.jpg", "image1.jpg")
    urllib.request.urlretrieve(r"https://images-na.ssl-images-amazon.com/images/I/41khDop3M4L._SX379_BO1,204,203,200_.jpg", "image2.jpg")

    title = Text("Thank You! Please Support").set_color(BLUE).to_edge(UL)

    books = Group(
      ImageMobject(r"image1.jpg"),
      ImageMobject(r"image2.jpg")
    ).scale(.9).arrange(RIGHT,buff=.8).next_to(title,DOWN,buff=.8,aligned_edge=LEFT)

    source_code = Tex("Source code in description", color=BLUE) \
        .next_to(books, DOWN, aligned_edge=LEFT,buff=1)

    email = Text(r"thomas@nieldconsultinggroup.com") \
        .scale(.5) \
        .next_to(source_code, DOWN, aligned_edge=LEFT)

    self.play(*[FadeIn(mobj) for mobj in (title, books, source_code, email)])

    self.wait(2)

# execute all scene renders
if __name__ == "__main__":
    #render_scenes(q="l", play=True, scene_names=["ConstantsExamples"])
    render_scenes(q="k", scene_names=["ClosingCard"])
