from manim import *
from threemds.utils import render_scenes


class TextSlide(VGroup):
    def __init__(self,
                 title,
                 bullet_buffer=.6,
                 title_buffer=.7,
                 title_scale=1.4,
                 bullet_scale=.8,
                 corner_mobj=None,
                 bullets=None):
        super().__init__()

        # format title
        if bullets is None:
            bullets = []

        self.title = title \
            .scale(title_scale) \
            .to_edge(UL)

        self.add(title)

        # format bullets
        self.bullets = VGroup(*bullets) \
            .scale(bullet_scale) \
            .arrange(DOWN,buff= bullet_buffer) \
            .next_to(title, DOWN, buff=title_buffer)

        for a in self.bullets:
            a.to_edge(LEFT)

        self.add(self.bullets)

        # format corner mobj
        if corner_mobj:
            self.corner = corner_mobj
            corner_mobj.next_to(title, DOWN).to_edge(RIGHT)
            self.add(corner_mobj)


# Examples
class BulletSlideScene(Scene):
    def construct(self):
        self.add(TextSlide(
                title=Tex("My Title", color=BLUE),
                corner_mobj=Circle(color=RED),
                bullets = [
                    Tex("Here is a point "),
                    Tex("Here is another point"),
                    Tex("This is also a good point"),
                    Tex("Have you considered this point too?"),
                    Tex("Here is a multiline point that","can cover multiple lines.") \
                        .arrange(DOWN, aligned_edge=LEFT)
                ]
            ).to_edge(LEFT)
        )

# execute all scene renders
if __name__ == "__main__":
    render_scenes(q="l", play=True, scene_names=["BulletSlideScene"])
    #render_scenes(q="k")
