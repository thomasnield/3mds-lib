from manim import *
from numpy import array
from threemds.utils import render_scenes, file_to_base_64, mobj_to_svg

config.background_color = "WHITE"
config.frame_rate = 60
#config.frame_width = 7
#config.frame_height = 7
#config.frame_width = 12
#config.pixel_width = 420
#config.pixel_height = 420


light_background_line_style = {
                "stroke_color": LIGHT_GRAY,
                "stroke_width": 2,
                "stroke_opacity": 1,
                "axis_config" : {
                   "stroke_width": 4,
                   "include_tip": False,
                   "line_to_number_buff": SMALL_BUFF,
                   "label_direction": DR,
                   "font_size": 24,
                   "color" : BLACK
               }
            }
light_axis_config = {
               "stroke_width": 4,
               "include_tip": False,
               "line_to_number_buff": SMALL_BUFF,
               "label_direction": DR,
               "font_size": 24,
               "color" : BLACK,
           }

class DeterminantScene(LinearTransformationScene):
    """
    config.frame_width = 7
    config.frame_height = 7
    config.pixel_width = 420
    config.pixel_height = 420
    """
    def __init__(self):
        LinearTransformationScene.__init__(
            self,
            show_coordinates=True,
            leave_ghost_vectors=False,
            show_basis_vectors=True,
            foreground_plane_kwargs = light_background_line_style | { "x_range" : [-5.5,5.5,1],
                                                                      "y_range" : [-5.5, 5.5, 1]},
            background_plane_kwargs = { "x_range" : [-5.5,5.5,1], "y_range" : [-5.5, 5.5, 1]}
        )

    def construct(self):
        #self.add_vector([1,1], animate=False, color=BLACK)
        sq = Square(side_length=1, fill_opacity=.4, fill_color=ORANGE, stroke_opacity=0) \
            .move_to(self.plane.c2p(0,0), aligned_edge=DL)

        self.add_transformable_mobject(sq)
        self.apply_matrix(array([[3,0],[0,2]]))

class DeterminantSheerScene(LinearTransformationScene):
    """
    config.frame_width = 7
    config.frame_height = 7
    config.pixel_width = 420
    config.pixel_height = 420
    """
    def __init__(self):
        LinearTransformationScene.__init__(
            self,
            show_coordinates=True,
            leave_ghost_vectors=False,
            show_basis_vectors=True,
            foreground_plane_kwargs = light_background_line_style | { "x_range" : [-5.5,5.5,1],
                                                                      "y_range" : [-5.5, 5.5, 1]},
            background_plane_kwargs = { "x_range" : [-5.5,5.5,1], "y_range" : [-5.5, 5.5, 1]}
        )

    def construct(self):
        #self.add_vector([1,1], animate=False, color=BLACK)
        sq = Square(side_length=1, fill_opacity=.4, fill_color=ORANGE, stroke_opacity=0) \
            .move_to(self.plane.c2p(0,0), aligned_edge=DL)

        self.add_transformable_mobject(sq)
        self.apply_matrix(array([[1,1],[0,2]]))


class DeterminantInversionScene(LinearTransformationScene):
    """
    config.frame_width = 7
    config.frame_height = 7
    config.pixel_width = 420
    config.pixel_height = 420
    """
    def __init__(self):
        LinearTransformationScene.__init__(
            self,
            show_coordinates=True,
            leave_ghost_vectors=False,
            show_basis_vectors=True,
            foreground_plane_kwargs = light_background_line_style | { "x_range" : [-5.5,5.5,1],
                                                                      "y_range" : [-5.5, 5.5, 1]},
            background_plane_kwargs = { "x_range" : [-5.5,5.5,1], "y_range" : [-5.5, 5.5, 1]}
        )

    def construct(self):
        #self.add_vector([1,1], animate=False, color=BLACK)
        sq = Square(side_length=1, fill_opacity=.4, fill_color=ORANGE, stroke_opacity=0) \
            .move_to(self.plane.c2p(0,0), aligned_edge=DL)

        self.add_transformable_mobject(sq)
        self.apply_matrix(array([[0,2],[1,1]]))

class ThreeDDeterminantScene(ThreeDScene):
    def construct(self):

        ax = ThreeDAxes(axis_config={"color" : BLACK})

        i_hat_start = Line(start=ax.c2p(0, 0, 0), end=ax.c2p(1, 0, 0), color=GREEN)
        j_hat_start = Line(start=ax.c2p(0, 0, 0), end=ax.c2p(0, 1, 0), color=RED)
        k_hat_start = Line(start=ax.c2p(0, 0, 0), end=ax.c2p(0, 0, 1), color=PURPLE)

        A = array([
            [1.5,-1,2.5],
            [0,2,1],
            [0,0,1.5]
        ])

        i_hat_end = Line(start=ax.c2p(0, 0, 0), end=ax.c2p(*A[:, 0]), color=GREEN)
        j_hat_end = Line(start=ax.c2p(0, 0, 0), end=ax.c2p(*A[:, 1]), color=RED)
        k_hat_end = Line(start=ax.c2p(0, 0, 0), end=ax.c2p(*A[:, 2]), color=PURPLE)
        basis_start = VGroup(i_hat_start,j_hat_start,k_hat_start)

        cube = Cube(side_length=1.0, fill_color=YELLOW) \
            .move_to(basis_start)

        self.set_camera_orientation(phi=75 * DEGREES, theta=-30 * DEGREES)
        self.add(ax, cube, basis_start)
        
        self.wait()
        self.play(
            cube.animate.become(cube.copy().apply_matrix(A)),
            i_hat_start.animate.become(i_hat_end),
            j_hat_start.animate.become(j_hat_end),
            k_hat_start.animate.become(k_hat_end),
            run_time=3
        )
        self.wait()



class ZeroDeterminantScene(LinearTransformationScene):
    """
    config.frame_width = 7
    config.frame_height = 7
    config.pixel_width = 420
    config.pixel_height = 420
    """
    def __init__(self):
        LinearTransformationScene.__init__(
            self,
            show_coordinates=True,
            leave_ghost_vectors=False,
            show_basis_vectors=True,
            foreground_plane_kwargs = light_background_line_style | { "x_range" : [-5.5,5.5,1],
                                                                      "y_range" : [-5.5, 5.5, 1]},
            background_plane_kwargs = { "x_range" : [-5.5,5.5,1], "y_range" : [-5.5, 5.5, 1]}
        )

    def construct(self):
        sq = Square(side_length=1, fill_opacity=.4, fill_color=ORANGE, stroke_opacity=0) \
            .move_to(self.plane.c2p(0,0), aligned_edge=DL)

        self.add_transformable_mobject(sq)
        self.apply_matrix(array([[-2,2],[1,-1]]))


class ThreeDZeroDeterminantScene(ThreeDScene):
    def construct(self):
        ax = ThreeDAxes(axis_config={"color": BLACK})

        i_hat = array([1,0,0])
        j_hat = array([0,1,0])
        k_hat = array([0,0,1])

        i_hat_start = Line(start=ax.c2p(0, 0, 0), end=ax.c2p(*i_hat), color=GREEN)
        j_hat_start = Line(start=ax.c2p(0, 0, 0), end=ax.c2p(*j_hat), color=RED)
        k_hat_start = Line(start=ax.c2p(0, 0, 0), end=ax.c2p(*k_hat), color=PURPLE)
        basis_start = VGroup(i_hat_start, j_hat_start, k_hat_start)

        A = np.array([[1, 0, 0], [-1, 2, 0], [0, 0, 0]])

        i_hat_end = i_hat_start.copy().apply_matrix(A)
        j_hat_end = j_hat_start.copy().apply_matrix(A)
        k_hat_end = k_hat_start.copy().apply_matrix(A)

        cube = Cube(side_length=1, fill_color=YELLOW) \
            .move_to(basis_start)

        self.set_camera_orientation(phi=75 * DEGREES, theta=-30 * DEGREES)
        self.add(ax, cube, basis_start)

        self.wait()
        self.play(
            cube.animate.become(cube.copy().apply_matrix(A)),
            i_hat_start.animate.become(i_hat_end),
            j_hat_start.animate.become(j_hat_end),
            k_hat_start.animate.become(k_hat_end),
            run_time=3
        )
        self.wait()


if __name__ == "__main__":
    render_scenes(q='l', scene_names=['ThreeDZeroDeterminantScene'])