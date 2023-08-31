from manim import *
import os

class MovingCameraLineWidthTest(MovingCameraScene):

    def construct(self):
        line = Line(start=[-1,1,0], end=[1,-1,0])

        self.add(line)

        INITIAL_LINE_WIDTH_MULTIPLE = self.camera.cairo_line_width_multiple
        INITIAL_FRAME_WIDTH = config.frame_width

        def line_scale_down_updater(mobj):
            proportion = self.camera.frame.width / INITIAL_FRAME_WIDTH
            self.camera.cairo_line_width_multiple = INITIAL_LINE_WIDTH_MULTIPLE * proportion

        mobj = Mobject()
        mobj.add_updater(line_scale_down_updater)
        self.add(mobj)

        self.camera.frame.save_state()
        self.play(
            self.camera.frame.animate.move_to(line.get_center()).scale(.1)
        )
        self.wait()
        self.play(Restore(self.camera.frame))
        self.wait()


# Execute rendering
if __name__ == "__main__":
    os.system( r"manim -ql -v WARNING -p --disable_caching -o MovingCameraLineWidthTest.mp4 scratch_pad.py MovingCameraLineWidthTest")