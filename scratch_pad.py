import pandas as pd
from manim import *
from threemds.utils import *

# configure the grid size
def resize_grid(m_width, m_height):
    config.frame_width = m_width
    config.frame_height = m_height
    config.pixel_width = int(config.frame_width * config.pixel_width / config.frame_width)
    config.pixel_height = int(config.pixel_width * config.frame_height / config.frame_width)


class Overfitting(Scene):
    def construct(self):
        df = pd.read_csv('https://bit.ly/3goOAnt', delimiter=",")

        plane = NumberPlane(
            x_range = (0, 11),
            y_range = (0, 26),
            x_length = 7,
            y_length = 7
            #axis_config={"include_numbers": True},
        )
        plane.center()
        line_graph = plane.plot_line_graph(
            x_values = df.values[:,0],
            y_values = df.values[:,1],
            line_color=BLUE,
            vertex_dot_style=dict(stroke_width=3,  fill_color=PURPLE),
            stroke_width = 4,
        )
        self.add(plane, line_graph)

        def f(x): return 1.93939*x + 4.73333
        regr_line = Line(start=plane.c2p(0,f(0)), end=plane.c2p(11, f(11)), color=RED)
        mobj_to_png(VGroup(plane, line_graph, regr_line), "overfitting.png")

if __name__ == "__main__":
    render_scenes(q="k", frames_only=False, scene_names="Overfitting")