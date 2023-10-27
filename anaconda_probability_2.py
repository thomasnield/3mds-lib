from manim import *
from threemds.utils import render_scenes, mobj_to_svg, mobj_to_png, file_to_base_64
from numpy import array

config.background_color = "WHITE"

light_background_line_style = {
                "stroke_color": LIGHT_GRAY,
                "stroke_width": 2,
                "stroke_opacity": 1,
            }
light_axis_config = {
               "stroke_width": 4,
               "include_ticks": False,
               "include_tip": False,
               "line_to_number_buff": SMALL_BUFF,
               "label_direction": DR,
               "font_size": 24,
               "color" : BLACK,
           }

class VennDiagramBayes(MovingCameraScene):
    def construct(self):

        # change line width behavior on camera zoom
        INITIAL_LINE_WIDTH_MULTIPLE = self.camera.cairo_line_width_multiple
        INITIAL_FRAME_WIDTH = config.frame_width

        def line_scale_down_updater(mobj):
            proportion = self.camera.frame.width / INITIAL_FRAME_WIDTH
            self.camera.cairo_line_width_multiple = INITIAL_LINE_WIDTH_MULTIPLE * proportion

        mobj = Mobject()
        mobj.add_updater(line_scale_down_updater)
        self.add(mobj)

        whole = Circle(radius=3.5,color=BLACK)
        whole_txt = Tex("100K Population", color=BLACK).move_to(whole)
        self.add(whole, whole_txt)

        gamers = Circle(radius=1.5, color=BLUE).move_to([0,-2,0])
        gamers_txt = Tex("19K Gamers", color=BLACK).scale(.75).move_to(gamers)
        self.add(gamers, gamers_txt)
        self.wait()

        homicidals = Circle(radius=.01, color=RED) \
            .move_to(gamers.get_top()) \
            .shift(.005 * DOWN) \
            .rotate(45*DEGREES, about_point=gamers.get_center())

        homicidals_txt = Tex("10 Homicidals", color=BLACK) \
            .scale_to_fit_width(homicidals.width * .6) \
            .move_to(homicidals)

        self.add(homicidals, homicidals_txt)
        self.wait()

        self.wait()
        self.camera.frame.save_state()

        self.play(
            self.camera.frame.animate.set(height=homicidals.height * 1.2) \
                .move_to(homicidals),
            run_time=3
        )
        self.wait()

        homicidals_txt.save_state()
        homicidals_play_games_txt = Tex(r"8.5 homicidals","are gamers", color=BLACK).arrange(DOWN) \
            .scale_to_fit_width(homicidals.width * .6) \
            .move_to(homicidals) \
            .rotate(45 * DEGREES)

        homicidals_dont_play_games_txt = Tex(r"1.5 homicidals","are not gamers", color=BLACK).arrange(DOWN) \
            .scale_to_fit_width(homicidals.width * .4) \
            .move_to(homicidals.get_top()) \
            .next_to(gamers.get_top(), UP, buff=.001) \
            .rotate(45 * DEGREES, about_point=gamers.get_center())

        self.play(Transform(homicidals_txt,
                                       VGroup(homicidals_play_games_txt,
                                            homicidals_dont_play_games_txt)
                                       )
                  )

        self.wait()
        self.play(Restore(homicidals_txt))
        self.wait()
        self.play(Restore(self.camera.frame), run_time=3)
        self.wait()
        self.play(Wiggle(gamers))
        self.wait()
        self.play(Circumscribe(homicidals,color=RED))
        self.wait()

        self.play(
            self.camera.frame.animate.set(height=homicidals.height * 1.2) \
                .move_to(homicidals),
            run_time=3
        )

        intersect = Intersection(homicidals, gamers, color=PURPLE, fill_opacity=.6)
        diff1 = Difference(homicidals, gamers, color=RED, fill_opacity=.6)
        diff2 = Difference(gamers, homicidals, color=BLUE, fill_opacity=.6)

        homicidals_play_games_prop = Tex(r".85", color=BLACK) \
            .scale_to_fit_width(homicidals.width * .2) \
            .move_to(homicidals) \
            .rotate(45 * DEGREES)


        homicidals_dont_play_games_prop = Tex(r".15", color=BLACK) \
            .scale_to_fit_width(homicidals.width * .2) \
            .move_to(homicidals.get_top()) \
            .next_to(gamers.get_top(), UP, buff=.001) \
            .rotate(45 * DEGREES, about_point=gamers.get_center())

        self.bring_to_front(homicidals_txt, homicidals_dont_play_games_prop, homicidals_play_games_prop)
        self.remove(homicidals_dont_play_games_prop, homicidals_play_games_prop)

        self.play(*[Write(m) for m in (diff1,diff2,intersect)])

        self.wait()

        self.play(Transform(homicidals_txt,
                           VGroup(homicidals_play_games_prop,
                                homicidals_dont_play_games_prop)
                           )
                  )
        self.wait()
        self.play(
            Restore(self.camera.frame),
            *[FadeOut(m) for m in (diff1,diff2,intersect)],
            run_time=3
        )
        self.wait()


if __name__ == "__main__":
    render_scenes(q="k", play=True, scene_names=['VennDiagramBayes'])
    #file_to_base_64('/Users/thomasnield/git/3mds-lib/media/images/anaconda_linear_algebra_1/04_VectorExamplesDotsScene.png')