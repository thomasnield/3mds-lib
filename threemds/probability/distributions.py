from manim import *
import numpy as np
import scipy
import sympy

NO_LABELS = 0
X_LABELS = 1
Z_SCORE_LABELS = 2

class NormalPDF(VGroup):

    def __init__(self,
                 mean: float = 0,
                 std: float = 0,
                 x_range_sigma = 4,
                 show_x_axis_labels = NO_LABELS,
                 show_area_plot = False
                 ):
        super().__init__()

        # area trackers
        self.area_lower_range = ValueTracker(0)
        self.area_upper_range = ValueTracker(0)

        # PDF function
        self.f_pdf = lambda x: scipy.stats.norm.pdf(x, mean, std)
        self.z_score = lambda x: (x-mean) / std

        # declare axes with default sizing
        self.axes = Axes(
            x_range=[mean-x_range_sigma*std, mean+x_range_sigma*std, std],
            y_range=[0, self.f_pdf(mean)+.1, (self.f_pdf(mean)+.1) / 4],
            x_axis_config={"include_numbers": False,
                         "numbers_to_exclude" : [mean-4*std]
            },
            y_axis_config={"include_numbers": True,
                           "decimal_number_config": {
                               "num_decimal_places": 2
                           }
            }
        )
        # create both sets of labels
        self.x_labels = VGroup(*[MathTex(round(mean+i*std,2)) \
                                   .scale(.7) \
                                   .next_to(self.axes.c2p(mean+i*std, 0), DOWN)
                               for i in range(-x_range_sigma,x_range_sigma+1)
                               ])

        self.x_sigma_labels = VGroup(*[MathTex(round(self.z_score(mean+i*std)), r"\sigma") \
                                    .scale(.7) \
                                    .next_to(self.axes.c2p(mean+i*std, 0), DOWN)
                               for i in range(-x_range_sigma,x_range_sigma+1)
                               ])

        if show_x_axis_labels == X_LABELS:
            self.add(self.x_labels)
        elif show_x_axis_labels == Z_SCORE_LABELS:
            self.add(self.x_sigma_labels)

        # PDF plot
        self.pdf_plot = self.axes.plot(self.f_pdf, color=BLUE)


        # area plot
        self.area_plot = always_redraw(lambda: self.axes.get_area(self.pdf_plot,
                                                                  (self.area_lower_range.get_value(),
                                                            self.area_upper_range.get_value()),
                                                                  color=[BLUE, RED])
                                       )

        self.get_area = lambda: round(
            scipy.stats.norm.cdf(self.area_upper_range.get_value(), mean, std) -
            scipy.stats.norm.cdf(self.area_lower_range.get_value(), mean, std),
            4)

        self.add(self.axes, self.pdf_plot)

        if show_area_plot:
            self.add(self.area_plot)

        # mean and std line
        self.mean_line = always_redraw(lambda: DashedLine(start=self.axes.c2p(mean,0),
                                                     end=self.axes.c2p(mean, self.f_pdf(mean)),
                                                     color=YELLOW
                                                )
                                       )
        self.std_line = always_redraw(lambda: DashedLine(start=self.axes.c2p(mean,0),
                                                     end=self.axes.c2p(mean, self.f_pdf(mean)),
                                                     color=BLUE
                                                )
                                      )
        # area range label
        self.area_range_label = always_redraw(
            lambda: MathTex("P(", round(self.area_lower_range.get_value()), r"\leq X \leq",
                            round(self.area_upper_range.get_value()), ") = ",
                            format(self.get_area(), ".3f")
                            ).next_to(self.axes.c2p(mean, self.f_pdf(mean)), UP)
        )

        # sigma area range label
        self.sigma_area_range_label = always_redraw(
            lambda: MathTex("P(", round(self.z_score(self.area_lower_range.get_value())), r"\sigma \leq Z \leq",
                            round(self.z_score(self.area_upper_range.get_value())), r"\sigma) = ",
                            format(self.get_area(), ".3f")
                            ).next_to(self.axes.c2p(mean, self.f_pdf(mean)), UP)
        )


class NormalCDF(VGroup):
    def __init__(self,
                 mean=0,
                 std=1,
                 x_range_sigma=3,
                 show_x_axis_labels=NO_LABELS,
                 piece_mode=False
                 ):
        super().__init__()

        # PDF and CDF functions
        self.f_pdf = lambda x: scipy.stats.norm.pdf(x, mean, std)
        self.f_cdf = lambda x: scipy.stats.norm.cdf(x, mean, std)
        self.z_score = lambda x: (x-mean) / std

        # axes should accomodate both PDF and CDF if needed
        self.axes = Axes(
            x_range=[mean-x_range_sigma*std, mean+x_range_sigma*std, std],
            y_range=[0,1.1,.25],
            y_axis_config={"include_numbers": True,
                           "decimal_number_config": {
                               "num_decimal_places": 2
                           }
            }
        )

        # PDF and CDF plot
        self.pdf_plot = self.axes.plot(self.f_pdf, color=BLUE)
        self.cdf_plot = self.axes.plot(self.f_cdf, color=RED)

        # X labels
        # create both sets of labels
        self.x_labels = VGroup(*[MathTex(round(mean+i*std,2)) \
                                   .scale(.7) \
                                   .next_to(self.axes.c2p(mean+i*std, 0), DOWN)
                               for i in range(-x_range_sigma,x_range_sigma+1)
                               ])

        self.x_sigma_labels = VGroup(*[MathTex(round(self.z_score(mean+i*std)), r"\sigma") \
                                    .scale(.7) \
                                    .next_to(self.axes.c2p(mean+i*std, 0), DOWN)
                               for i in range(-x_range_sigma,x_range_sigma+1)
                               ])

        if show_x_axis_labels == X_LABELS:
            self.add(self.x_labels)
        elif show_x_axis_labels == Z_SCORE_LABELS:
            self.add(self.x_sigma_labels)

        # add components if not piece mode
        if not piece_mode:
            self.add(self.axes, self.cdf_plot)
            if show_x_axis_labels == X_LABELS:
                self.add(self.x_labels)
            elif show_x_axis_labels == Z_SCORE_LABELS:
                self.add(self.x_sigma_labels)

        # project PDF area to CDF
        self.x_tracker = ValueTracker(self.axes.x_range[0])

        # always redraw PDF area
        self.pdf_area = always_redraw(lambda: self.axes.get_area(
            self.pdf_plot,
            (self.axes.x_range[0],
            self.x_tracker.get_value()),
            color=[BLUE, RED]
            )
        )

        # always update the connecting line
        def create_pdf_to_cdf_line():
            x_raw = self.x_tracker.get_value()
            cdf_y_raw = self.f_cdf(x_raw)
            x = self.axes.c2p(x_raw, 0)[0]
            pdf_y = self.axes.c2p(x_raw, self.f_pdf(x_raw))[1]
            cdf_y = self.axes.c2p(x_raw, self.f_cdf(x_raw))[1]
            line = DashedLine(
                start=[x,pdf_y,0],
                end=[x,cdf_y,0],
                color=RED
            )
            dot = Dot(color=RED).move_to([x,cdf_y,0])
            label = MathTex(round(cdf_y_raw,2), color=RED).scale(.8).next_to(dot, UL*.5)
            return VGroup(line, dot, label)

        self.pdf_to_cdf_line = always_redraw(create_pdf_to_cdf_line)

class NormalPPF(VGroup):
    def __init__(self,
                 mean=0,
                 std=1,
                 x_range_sigma=3):
        super().__init__()

        # PDF and CDF functions
        self.f_pdf = lambda x: scipy.stats.norm.pdf(x, mean, std)
        self.f_cdf = lambda x: scipy.stats.norm.cdf(x, mean, std)
        self.z_score = lambda x: (x-mean) / std
        self.z_to_x = lambda z: z*std + mean

        self.f_ppf = lambda q: self.z_to_x(scipy.stats.norm.ppf(q))

        # axes should accomodate both PDF and CDF if needed
        self.axes = Axes(
            x_range=[.0002,.9998,.05],
            y_range=[mean-x_range_sigma*std, mean+x_range_sigma*std, std]
        )

        # PDF and CDF plot
        self.ppf_plot = self.axes.plot(self.f_ppf, color=ORANGE, use_smoothing=True)
        self.add(self.axes, self.ppf_plot)


