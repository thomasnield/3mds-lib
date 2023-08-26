from manim import *

from threemds.utils import render_scenes


class CircleTraceTest(Scene):
    def construct(self):
        circle = Circle(radius=2)
        self.add(circle)

        vt = ValueTracker(0.0)
        point: Dot = always_redraw(lambda: Dot(point=circle.point_from_proportion(vt.get_value())))

        tex = always_redraw(lambda:
                            DecimalNumber(vt.get_value(), num_decimal_places=1) \
                                .move_to(
                                    circle.point_from_proportion(vt.get_value())*1.3
                                )
                            )

        self.add(vt,tex,point)
        self.wait()
        self.play(vt.animate.set_value(1), run_time=3)
        self.wait()
        self.play(vt.animate.set_value(0), run_time=3)
        self.wait()


if __name__ == "__main__":
    render_scenes(q='h', scene_names=['CircleTraceTest'])


