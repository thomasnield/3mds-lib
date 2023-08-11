"""
Script
* Plot numberline
* Plot histogram with different sized bins, and fit curve
* Show area of curve is not useful, but when animated to scaled down to 1.0 it creates a probability distribution.
* Emphasize properties including symmetry and apread
* Show probability of single point is 0, and how areas of ranges creates probabilities
* Show reimann sums to calculate area briefly
* Zoom out and show CDF being drawn by a projecting libe from PDf
* Show subtraction operations to calculate middle ranges
* Zoom out and show CDF transforming a copy of its plot into an inverse CDF on another tile
* Demonstrate inverse CDF, and trace an inverse CDF lookup back to the PDF


"""

import math

import scipy.stats
from scipy.stats import norm
from threemds.stats.plots import DottedNumberLine, Histogram
from threemds.stats.formulas import *
from threemds.probability.distributions import  *
from threemds.utils import render_scenes

data = np.array([65.27153711, 61.69996242, 60.98565375, 65.30031155, 63.51806848, 68.19351011
                    , 66.95478689, 64.55759847, 63.39196506, 67.54289154, 63.19717054, 67.49928145
                    , 63.19766386, 72.39460819, 65.06618895, 61.47292356, 63.65363793, 67.40224834
                    , 69.5453564, 60.68221867, 61.8700392, 61.83843595, 68.61144043, 65.56311344
                    , 64.75033447, 58.11010178, 58.05563835, 64.47898656, 61.58413871, 62.73034603
                    , 71.43664684, 63.16241713, 64.67237221, 57.72834468, 69.02743165, 61.51629892
                    , 63.11713525, 66.89447031, 64.641517, 64.27097855, 68.33858714, 61.08831661
                    , 70.21508249, 61.28410416, 66.28841856, 66.36769216, 66.85034888, 60.67420648
                    , 60.26603058, 65.16350301, 65.50142234, 63.43139858, 65.77102549, 62.59483884
                    , 64.99651396, 63.70706524, 63.53666649, 63.56936279, 67.49544015, 65.61582933
                    , 70.70856991, 64.50069849, 58.86870671, 66.22804048, 65.12576775, 58.25042313
                    , 60.85683308, 63.962537, 63.76654245, 62.21488775, 66.89233257, 66.27436643
                    , 66.06183922, 62.58999392, 62.01327474, 69.776555, 65.86363553, 66.37112032
                    , 65.0016078, 68.24377827, 60.02304379, 64.70747144, 62.58229384, 69.80994781
                    , 67.92226975, 62.4487201, 63.18153599, 65.53453952, 65.39880782, 59.37181606
                    , 67.58819312, 65.053418, 62.32011733, 65.51449093, 61.70972692, 66.08806211
                    , 63.49776397, 68.8884009, 63.55453324, 66.02846214])

create_number_line = lambda: DottedNumberLine(data)

class NumberLineScene(Scene):
    def construct(self):
        self.play(LaggedStart(Write(create_number_line())))
        self.wait()

class HistogramScene(Scene):
    def construct(self):

        # Transition numberline to histogram
        nl = create_number_line()
        hist = Histogram(data, 300, show_points=False, show_normal_dist=False)

        bin_text_fn = lambda bin_ct, hist: MathTex("k = ", bin_ct).next_to(hist, DOWN)
        bin_text = bin_text_fn(300,hist)

        self.add(nl)
        self.wait()
        self.play(FadeOut(nl.nl))
        self.wait()
        self.play(ReplacementTransform(nl.dots,hist.dots))
        self.wait()
        self.play(Create(hist, lag_ratio=.1), Create(bin_text, lag_ratio=.1))
        self.wait()

        # Animate bin changes
        bin_sizes = [200, 100, 80, 70, 50, 30, 20,14,13,12,11]
        include_norm_dist_inds = [b < 30 for b in bin_sizes]

        for i,s,b in zip(range(0,len(bin_sizes)), bin_sizes, include_norm_dist_inds):
            new_hist = Histogram(data,
                                 bin_count=s,
                                 show_points=False,
                                 show_normal_dist=False)

            new_bin_text = bin_text_fn(s, new_hist)

            replace_hist = lambda new_hist=new_hist, hist=hist: self.play(ReplacementTransform(hist, new_hist),
                                             ReplacementTransform(hist.dots, new_hist.dots))

            replace_hist_and_plot = lambda new_hist=new_hist, hist=hist: self.play(
                                                      ReplacementTransform(hist, new_hist),
                                                      ReplacementTransform(hist.dots, new_hist.dots),
                                                      ReplacementTransform(hist.normal_dist_plot, new_hist.normal_dist_plot)
                                                     )

            # update bin count label
            self.remove(bin_text)
            self.add(new_bin_text)
            bin_text = new_bin_text

            # handle transition to showing the normal distribution
            if i > 0 and (not include_norm_dist_inds[i-1]) and b:
                replace_hist()
                self.wait()
                self.play(Write(new_hist.normal_dist_plot))
            elif b:
                replace_hist_and_plot()
            else:
                replace_hist()

            hist = new_hist
            self.wait()

        ## brace bar width
        braced_bar = hist.bars[2]
        brace = Brace(braced_bar, DOWN)
        brace_text = brace.get_text(round(hist.bin_width,2))
        self.play(LaggedStart(Write(brace), Write(brace_text)), FadeOut(bin_text))
        self.wait()

class ConstantsExamples(ZoomedScene):

  def construct(self):
        normpdf = NormalPDF(mean=data.mean(),
                          std=data.std(),
                          show_x_axis_labels=X_LABELS,
                          show_area_plot=False
                          )

        formula = NormalDistributionTex().scale(1).to_edge(UR)

        self.add(normpdf)

        self.camera.frame.save_state()
        self.camera.frame.scale(0.45).move_to(formula)
        self.play(Write(formula))
        self.wait()


        # highlight mu and sigma
        self.play(formula.mu.animate.set_color(YELLOW),
                  Circumscribe(formula.mu, color=YELLOW)
        )
        self.wait()
        self.play(formula.sigmas.animate.set_color(BLUE),
                  *[Circumscribe(s, color=BLUE) for s in formula.sigmas]
                  )
        self.wait()

        # highlight e and pi
        pi_equals = MathTex(r"\pi = 3.14159...", color=RED).next_to(formula, DOWN)
        e_equals = MathTex(r"e = 2.71828...", color=RED).next_to(formula, DOWN)
        self.play(
            Circumscribe(formula.pi),
            FadeIn(pi_equals),
            formula.pi.animate.set_color((RED))
        )
        self.wait()
        self.play(FadeOut(pi_equals))
        self.wait()
        self.play(
            Circumscribe(formula.eulers_number, color=RED),
            FadeIn(e_equals),
            formula.eulers_number.animate.set_color(RED)
        )
        self.play(FadeOut(e_equals, color=RED))
        self.wait()

        # zoom out, restore colored elements to WHITE
        self.play(
            Restore(self.camera.frame),
            *[e.animate.set_color((WHITE)) for e in [formula.mu, formula.eulers_number, formula.sigmas, formula.pi]]
        )
        self.wait()
        self.play(FadeOut(formula))
        self.wait()

class PDFScene(Scene):
    def construct(self):

        # intialize objects and helper functions
        normpdf = NormalPDF(mean=data.mean(),
                            std=data.std(),
                            show_x_axis_labels=X_LABELS,
                            show_area_plot=False
                            )
        axes = normpdf.axes
        x_labels = normpdf.x_labels
        mu_tracker = ValueTracker(data.mean())
        sigma_tracker = ValueTracker(data.std())

        def get_mu(): return mu_tracker.get_value()
        def get_sigma(): return sigma_tracker.get_value()
        def get_pdf(x): return norm.pdf(x, get_mu(), get_sigma())
        def get_cdf(x): return norm.cdf(x, get_mu(), get_sigma())
        def get_ppf(p): return norm.ppf(p, get_mu(), get_sigma())
        def z_to_x(z): return get_mu() + z*get_sigma()
        def get_mu_output(): return norm.pdf(get_mu(), get_sigma(),get_sigma())
        def get_sigma_output(): return norm.pdf(z_to_x(1), get_mu(), get_sigma())

        def get_area_x(a,b,color=BLUE): return \
            normpdf.axes.get_area(graph=normpdf.pdf_plot,
                                  x_range=(a, b),
                                  color=color)

        def get_area_z(a_sigma,b_sigma,color=BLUE): return get_area_x(z_to_x(a_sigma), z_to_x(b_sigma), color)

        def get_vert_line_x(x: float, color=YELLOW): return DashedLine(
            start= axes.c2p(x, 0),
            end= axes.c2p(x, get_pdf(x)),
            color=color
        )

        def get_vert_line_z(z: float, color=YELLOW): return get_vert_line_x(z_to_x(z), color)

        def get_horz_line_x(x: float, color=YELLOW): return DashedLine(
            start= axes.c2p(x, get_pdf(x)),
            end= axes.c2p(z_to_x(-4), get_pdf(x)),
            color=color
        )

        def get_horz_line_z(z: float, color=YELLOW): return get_horz_line_x(z_to_x(z), color)

        # Create PDF
        self.add(normpdf)

        # show PDF label
        pdf_label = Tex("Probability Density Function (PDF)").to_edge(UP)
        self.play(FadeIn(pdf_label))
        self.wait()
        self.play(FadeOut(pdf_label))
        self.wait()

        # highlight mu and sigma with lines and labels
        mu_trace = always_redraw(lambda: get_vert_line_z(0))

        mu_label = always_redraw(lambda: MathTex(r"\mu = ", round(get_mu(), 2), color=YELLOW) \
                                 .scale(.75) \
                                 .next_to(mu_trace, UP)
                                 )

        sigma_trace = always_redraw(lambda: get_vert_line_z(1, color=BLUE))
        sigma_label = always_redraw(lambda: MathTex(r"\sigma = ", round(sigma_tracker.get_value(), 2), color=BLUE) \
                                    .scale(.75) \
                                    .next_to(sigma_trace, UR)
                                    )

        # animate mu and sigma tracing
        self.play(LaggedStart(Write(mu_trace)))
        self.wait()
        self.play(Write(mu_label))
        self.wait()
        self.play(Write(sigma_trace))
        self.wait()
        self.play(Write(sigma_label))

        # Move mean and standard deviation around
        moving_plot = always_redraw(lambda: normpdf.axes.plot(
            lambda x: norm.pdf(x, mu_tracker.get_value(), sigma_tracker.get_value()),
            color=BLUE
        )
                                    )
        self.remove(normpdf.pdf_plot)
        self.add(moving_plot)
        self.wait()
        self.play(
            sigma_tracker.animate.set_value(data.std() * 1.25)
        )
        self.wait()
        self.play(
            sigma_tracker.animate.set_value(data.std() * 1.5)
        )
        self.wait()
        self.play(
            sigma_tracker.animate.set_value(data.std())
        )
        self.wait()
        self.play(
            mu_tracker.animate.set_value(data.mean() + 4)
        )
        self.wait()
        self.play(
            mu_tracker.animate.set_value(data.mean())
        )
        self.wait()
        self.play(
            mu_tracker.animate.set_value(data.mean() - 4)
        )
        self.wait()
        self.play(
            mu_tracker.animate.set_value(data.mean())
        )
        self.wait()

        # animate sigma length
        sigma_length = Line(start=normpdf.axes.c2p(mu_tracker.get_value(), 0),
                            end=normpdf.axes.c2p(mu_tracker.get_value() + sigma_tracker.get_value(), 0))

        sigma_brace = Brace(sigma_length, UP)
        self.remove(sigma_label)
        sigma_label = sigma_label.copy()
        self.add(sigma_label)
        sigma_label.save_state()
        self.play(sigma_label.animate.scale(.9).next_to(sigma_brace, UP), Write(sigma_brace))
        self.wait()
        self.play(FadeOut(sigma_brace), Restore(sigma_label))
        self.wait()
        self.add(normpdf.pdf_plot)
        self.remove(moving_plot)
        self.play(*[FadeOut(mobj) for mobj in [mu_trace, mu_label, sigma_trace, sigma_label]])
        self.wait()

        # Show area under the entire curve is 1.0
        full_area_plot = get_area_z(-4,4,color=BLUE)
        full_area_plot_label = MathTex("A = 1.0").move_to(full_area_plot)

        self.play(Write(full_area_plot))
        self.wait()
        self.play(Create(full_area_plot_label))
        self.wait()
        self.play(FadeOut(full_area_plot_label), FadeOut(full_area_plot))
        self.wait()

        # Negate looking up a likelihood value
        likelihood_vert_line = get_vert_line_z(1.5)
        self.play(FadeOut(x_labels))
        self.wait()

        likelihood_horz_line = get_horz_line_z(1.5)

        x_value_lookup_label = MathTex(round(z_to_x(1.5),2), color=YELLOW) \
            .next_to(likelihood_vert_line, DOWN) \
            .scale(.75)

        self.play(Write(x_value_lookup_label), Write(likelihood_vert_line))
        self.wait()

        likelihood_lookup_label = MathTex(r"\times", color=RED) \
            .next_to(likelihood_horz_line, LEFT)

        self.play(Write(likelihood_horz_line))
        self.play(Write(likelihood_lookup_label))

        self.wait()
        self.play(
            likelihood_horz_line.animate.set_color(RED),
            x_value_lookup_label.animate.set_color(RED)
        )
        self.play(
            FadeOut(likelihood_horz_line),
            FadeOut(likelihood_lookup_label),
            FadeOut(x_value_lookup_label)
        )
        self.wait()

        # Show a narrow area range
        narrow_area = get_area_x(69,70,color=BLUE)
        self.play(ReplacementTransform(likelihood_vert_line, narrow_area))
        self.wait()

        # add range labels for range 69-70
        a_label = MathTex(69, color=YELLOW).scale(.75).next_to(axes.c2p(69,0), DOWN)
        b_label = MathTex(70, color=YELLOW).scale(.75).next_to(axes.c2p(70,0), DOWN)

        self.play(Write(a_label), Write(b_label))
        self.wait()

        # show area label for range 69-70
        callout_line = Line(start=narrow_area.get_center(),
                            end=narrow_area.get_center() + UP + RIGHT,
                            color=YELLOW)

        area_label_a_b = MathTex("A = ", round(get_cdf(70) - get_cdf(69), 2), color=YELLOW) \
            .next_to(callout_line.end, RIGHT)

        self.play(Write(callout_line))
        self.play(Write(area_label_a_b))

        # widen range to 68-71
        self.play(
            Transform(narrow_area, get_area_x(68,71,color=BLUE)),

            Transform(area_label_a_b, MathTex("A = ", round(get_cdf(71) - get_cdf(68), 2), color=YELLOW) \
                      .next_to(callout_line.end, RIGHT)),

            Transform(a_label, MathTex(68, color=YELLOW).scale(.75).next_to(axes.c2p(68,0), DOWN)),

            Transform(b_label, MathTex(71, color=YELLOW).scale(.75).next_to(axes.c2p(71,0), DOWN))
        )
        self.wait()

        # widen range to 64-77
        self.play(
            Transform(narrow_area, get_area_x(64,77,color=BLUE)),

            Transform(area_label_a_b, MathTex("A = ", round(get_cdf(77) - get_cdf(64), 2), color=YELLOW) \
                .next_to(callout_line.end, RIGHT)),

            Transform(a_label, MathTex(64, color=YELLOW)\
                      .scale(.75)\
                      .next_to(axes.c2p(64,0), DOWN)),

            Transform(b_label, MathTex(77, color=YELLOW)\
                      .scale(.75)\
                      .next_to(axes.c2p(77,0), DOWN))
        )
        self.wait()
        self.play(*[FadeOut(mobj) for mobj in [narrow_area, area_label_a_b, a_label, b_label, callout_line]])
        self.wait()

class TiledScene(MovingCameraScene):

    def construct(self):
        skip_animations=False

        # Declare PDF model
        class PDFPlot(VGroup):

            def __init__(self, mean, std):
                super().__init__()
                f = lambda x: norm.pdf(x, mean,std)

                self.mean = mean
                self.std = std
                self.lower_x = mean-std*3
                self.upper_x = mean+std*3

                axes = Axes(x_range=[self.lower_x, self.upper_x, std],
                            y_range=[0, f(mean) + .1, (f(mean) + .1) / 4],
                            x_axis_config={"include_numbers": False,
                                           "numbers_to_exclude": [mean - 4 * std]
                                           },
                            y_axis_config={"include_numbers": True,
                                           "decimal_number_config": {
                                               "num_decimal_places": 2
                                           }
                                           }
                            )

                plot = axes.plot(f, color=BLUE)
                self.add(axes, plot)

                self.f=f
                self.axes=axes
                self.plot=plot

            def x2p(self, x):
                return self.axes.c2p(x, self.f(x))

            def area_range(self, x_start, x_end, color=BLUE):
                return self.axes.get_area(self.plot, color=color, x_range=(x_start, x_end))

        # Declare CDF model
        class CDFPlot(VGroup):
            def __init__(self, mean, std):
                super().__init__()
                f = lambda x: norm.cdf(x, mean,std)

                axes = Axes(x_range=[mean - 3 * std, mean + 3 * std, std],
                            y_range=[0, 1.1, .25],
                            y_axis_config={"include_numbers": True,
                                           "decimal_number_config": {
                                               "num_decimal_places": 2
                                            }
                                           }
                )

                plot = axes.plot(f, color=RED)

                self.add(axes, plot)
                self.f=f
                self.axes=axes
                self.plot=plot

            def x2p(self, x):
                return self.axes.c2p(x, self.f(x))

            def plot_to_x(self, x):
                return self.axes.plot(self.f, color=RED, x_range=[mean - 3 * std, x])

            def vertical_line(self, x):
                return DashedLine(
                    start=self.axes.c2p(x, self.f(x)),
                    end=self.axes.c2p(x, 0),
                    color=RED
                )

            def horizontal_line(self, x):
                return DashedLine(
                    start=self.axes.c2p(-3*std+mean, self.f(x)),
                    end=self.axes.c2p(x, self.f(x)),
                    color=RED
                )

        # Declare PPF model
        class PPFPlot(VGroup):
            def __init__(self, mean, std):
                super().__init__()
                f = lambda x: norm.ppf(x, mean,std)

                axes = Axes(x_range=[.001,.999,.05],
                            y_range=[mean-3*std, mean+3*std, std],
                            x_length=4,
                            x_axis_config= {
                                "numbers_to_include" : [0,.25,.5,.75,1]
                            }
                )

                plot = axes.plot(f, color=ORANGE, use_smoothing=True)

                self.add(axes, plot)
                self.f=f
                self.axes=axes
                self.plot=plot

            def p2p(self, p):
                return self.axes.c2p(p, self.f(p))

            def vertical_line(self, p):
                return DashedLine(
                    start=self.axes.c2p(p,-3*std+mean),
                    end=self.axes.c2p(p, self.f(p)),
                    color=RED
                )

            def horizontal_line(self, p):
                return DashedLine(
                    start=self.axes.c2p(p, self.f(p)),
                    end=self.axes.c2p(0, self.f(p)),
                    color=RED
                )

        mean,std = data.mean(), data.std()

        # create PDF and CDF
        self.next_section("Create PDF and CDF, and zoom out", skip_animations=skip_animations)
        pdf_model = PDFPlot(mean, std)
        cdf_model = CDFPlot(mean, std)

        # stack the PPF to the right for later
        ppf_model = PPFPlot(mean, std).to_edge(RIGHT)

        # stack the PDF and CDF to the left
        left_panel = VGroup(cdf_model, pdf_model) \
            .arrange(DOWN) \
            .scale_to_fit_height(7)

        # add PDF to the scene
        self.add(pdf_model)

        # start camera zoomed on PDF
        self.camera.frame.save_state()
        self.camera.frame.scale(0.6).move_to(pdf_model)
        self.wait()

        # zoom out the camera to reveal the CDF model axes
        self.play(
            Restore(self.camera.frame),
            FadeIn(cdf_model.axes)
        )
        self.wait()

        self.next_section("Draw area between PDF and CDF, label both", skip_animations=skip_animations)

        # Declare the range for x values on PDF
        x_upper_tracker = ValueTracker(mean-std*3)

        # Declare the area for the PDF which will update based on the trackers above
        area_color=BLUE
        area: Mobject = always_redraw(lambda: pdf_model.area_range(-3*std+mean, x_upper_tracker.get_value(), color=area_color))

        # Draw the connecting dashed line between the PDF and CDF projecting the area
        connecting_line: DashedLine = always_redraw(lambda: DashedLine(
            start=pdf_model.x2p(x_upper_tracker.get_value()),
            end=cdf_model.x2p(x_upper_tracker.get_value()),
            color=RED
            ))

        # Project the area to the CDF as x_upper increases, also show the area as a decimal
        cdf_partial_plot = always_redraw(lambda: cdf_model.plot_to_x(x_upper_tracker.get_value()))
        area_label = always_redraw(lambda: DecimalNumber(cdf_model.f(x_upper_tracker.get_value()), num_decimal_places=2) \
                                        .scale(.8) \
                                        .next_to(cdf_model.x2p(x_upper_tracker.get_value()), RIGHT)
                                   )

        # Populate the area plot, connecting line, partial CDF plot, area label
        self.play(*[FadeIn(mobj) for mobj in (area, connecting_line, cdf_partial_plot, area_label)])

        # Run the animation for 5 seconds by filling the whole PDF/CDF
        self.play(
            x_upper_tracker.animate.set_value(mean+3*std),
            run_time=5
        )
        self.wait()
        self.play(
            *[FadeOut(mobj) for mobj in (area, connecting_line, cdf_partial_plot, cdf_model.plot, area_label)]
        )

        # bump the PDF and CDF to the left edge
        self.play(
            left_panel.animate.to_edge(LEFT)
        )
        self.wait()

        # label both the PDF and CDF
        cdf_label = Tex("Cumulative Density Function (CDF)") \
            .scale(.75) \
            .next_to(cdf_model, RIGHT)

        pdf_label = Tex("Probability Density Function (PDF)") \
            .scale(.75) \
            .next_to(pdf_model, RIGHT)

        self.play(FadeIn(cdf_label), FadeIn(pdf_label))
        self.wait()
        self.play(FadeOut(cdf_label), FadeOut(pdf_label))
        self.wait()

        self.next_section("Show an area lookup from PDF -> CDF at x=70", skip_animations=skip_animations)

        # show an area lookup from PDF -> CDF at x=70

        # label 70 on the x-axis
        label_x_eq_70 = MathTex("70").scale(.8) \
            .next_to(pdf_model.axes.c2p(70, 0), DOWN)

        # reset the x_upper
        x_upper_tracker.set_value(mean-3*std)

        # fade in the label, move the PDF/CDF to x = 70
        self.play(FadeIn(label_x_eq_70))
        self.add(area, connecting_line)
        self.play(x_upper_tracker.animate.set_value(70))
        self.wait()

        # label the area in the CDF, then move it into the area of the PDF
        label_cdf_70 = DecimalNumber(cdf_model.f(70), num_decimal_places=3) \
            .next_to(connecting_line.get_top(), UL)

        area_center=pdf_model.axes.c2p(mean,pdf_model.f(mean)*.5)
        area_question = Tex("?").move_to(area_center)
        self.play(FadeIn(area_question))
        self.wait()

        # draw the line to look up the area on CDF
        cdf_horz_line: DashedLine = always_redraw(lambda: DashedLine(
            start=cdf_model.x2p(x_upper_tracker.get_value()),
            end=cdf_model.axes.c2p(-3*std+mean, cdf_model.f(x_upper_tracker.get_value())),
            color=RED
            ))
        self.play(Write(cdf_horz_line))
        self.wait()

        # show the area lookup label for the CDF, then move it to the area
        self.play(FadeIn(label_cdf_70))
        self.wait()
        self.remove(area_question)
        self.play(
            label_cdf_70.animate.move_to(area_center)
        )

        self.next_section("Show an area middle range lookup from PDF -> CDF", skip_animations=skip_animations)

        # get ready to break up the area into two pieces, from 65 to 70 and left tail to 65
        self.wait()
        area_65_70 = always_redraw(lambda: pdf_model.area_range(x_upper_tracker.get_value(),70, BLUE))
        self.add(area_65_70)

        # move the area up to 70 to the right side of the screen
        area_70_panel = VGroup(area, label_cdf_70).copy()
        self.play(area_70_panel.animate.scale(.8).next_to(left_panel).to_edge(UR))
        self.wait()
        self.play(FadeOut(label_cdf_70))
        self.wait()

        # label 65 on the x-axis
        label_x_eq_65 = MathTex("65").scale(.8) \
            .next_to(pdf_model.axes.c2p(65, 0), DOWN)

        # change area color to red, then look up area for x<=65
        area_color=RED
        self.play(x_upper_tracker.animate.set_value(65),
                  FadeIn(label_x_eq_65)
                  )

        # draw the horizontal connecting line on the CDF, also draw question mark
        area_question = Tex("?").next_to(area_65_70.get_left(), LEFT).shift(LEFT*.20)
        self.wait()
        self.play(Write(area_question), Write(cdf_horz_line))
        self.wait()

        # label the area for x<=65 in the CDF, then move it down
        label_cdf_65 = DecimalNumber(cdf_model.f(65), num_decimal_places=3) \
            .next_to(connecting_line.get_top(), UL)
        self.wait()
        self.play(FadeIn(label_cdf_65))
        self.wait()
        self.remove(area_question)
        self.play(FadeOut(cdf_horz_line),
                  label_cdf_65.animate.next_to(area_65_70.get_left(), LEFT))
        self.wait()

        # move x<=65 area to the right panel, show minus sign and position under it
        area_65_panel = VGroup(area.copy(), label_cdf_65)
        minus_sign = MathTex("-").scale(4).next_to(area_70_panel, DOWN,buff=.5)

        self.remove(area)
        self.play(FadeIn(minus_sign),
                  area_65_panel.animate.scale(.8).next_to(
                        minus_sign, DOWN, aligned_edge=area_70_panel.get_left(), buff=.5
                  )
        )
        self.wait()

        # create equals sign and put it under the other two plots on right side
        equals_sign = MathTex("=").scale(5) \
            .match_width(minus_sign) \
            .next_to(VGroup(area_65_panel, area_70_panel), DOWN,buff=.5)

        # label the area x between 65 and 70
        label_65_70 = DecimalNumber(cdf_model.f(70) - cdf_model.f(65), num_decimal_places=3) \
            .move_to(area_65_70) \
            .shift(DOWN * .5)

        # bring in the label, remove the connecting line
        self.play(
            FadeIn(label_65_70),
            FadeOut(connecting_line)
        )
        self.wait()

        # move the area 65 through 70 to right side
        area_65_70_panel = VGroup(label_65_70, area_65_70).copy()
        self.play(
            FadeIn(equals_sign),
            area_65_70_panel.animate.scale(.8).next_to(
                VGroup(area_65_panel, area_70_panel, equals_sign),
                direction=DOWN,
                buff=.5
            ).to_edge(RIGHT)
        )
        self.wait()

        # Fade out all areas
        self.play(
            FadeOut(VGroup(area_65_70_panel,
                           equals_sign,
                           minus_sign,
                           area_65_panel,
                           area_70_panel,
                           area_65_70,
                           label_65_70,
                           label_x_eq_65,
                           label_x_eq_70))
        )
        self.wait()

        self.next_section("Invert CDF to become PPF", skip_animations=skip_animations)
        # bring in PPF on right, starting with the axes
        cdf_axes_copy = cdf_model.axes.copy()

        self.play(
            Rotate(cdf_axes_copy, 180 * DEGREES, axis=X_AXIS)
        )
        self.play(
            Rotate(cdf_axes_copy, 90 * DEGREES)
        )
        self.play(
            FadeTransform(cdf_axes_copy, ppf_model.axes)
        )
        self.wait()

        # next bring in the CDF plot and make it the PPF plot
        cdf_copy = cdf_model.plot.copy()

        self.play(
            Rotate(cdf_copy, 180 * DEGREES, axis=X_AXIS)
        )
        self.play(
            Rotate(cdf_copy, 90 * DEGREES)
        )
        self.play(
            cdf_copy.animate.become(ppf_model.plot),
        )
        self.wait()

        self.next_section("Show a lookup for PPF", skip_animations=False)

        # label the PPF
        ppf_label = Tex("Probability Point Function (PPF)") \
            .rotate(90*DEGREES) \
            .scale(.75) \
            .next_to(ppf_model, LEFT)

        self.wait()
        self.play(FadeIn(ppf_label))
        self.wait()
        self.play(FadeOut(ppf_label))
        self.wait()

        # Get .75 of area for PPF
        area_color=BLUE
        x_upper_tracker.set_value(-3*std+mean)
        x_area_75 = ppf_model.f(.75)

        # draw the .75 area
        area_75: VMobject = always_redraw(lambda: pdf_model.area_range(-3*std+mean,x_upper_tracker.get_value(), color=BLUE))
        self.wait()
        self.add(area_75)
        self.play(x_upper_tracker.animate.set_value(x_area_75))
        self.wait()

        # label the .75 area
        area_75_label = MathTex(".75") \
            .move_to(pdf_model.axes.c2p(mean, pdf_model.f(mean)*.5))

        self.play(FadeIn(area_75_label))
        self.wait()

        # move question mark to the x-axis
        area_question.next_to(pdf_model.axes.c2p(x_area_75, 0), DOWN,buff=.15)
        self.play(Write(area_question))
        self.wait()

        # draw the PPF vertical line, and CDF horizontal line equivalent
        ppf_vert_line = ppf_model.vertical_line(.75)
        cdf_horz_line = cdf_model.horizontal_line(x_area_75)
        self.play(Write(ppf_vert_line), Write(cdf_horz_line))
        self.wait()

        # draw the PPF horizontal line, reveal x-value
        ppf_horz_line = ppf_model.horizontal_line(.75)
        cdf_vert_line = cdf_model.vertical_line(x_area_75)
        self.play(Write(ppf_horz_line), Write(cdf_vert_line))
        self.wait()

        # reveal the x-value
        x_label_cdf = DecimalNumber(x_area_75, num_decimal_places=2) \
            .scale(.8) \
            .next_to(cdf_model.axes.c2p(x_area_75, 0), DOWN,buff=.15)

        x_label_ppf = x_label_cdf.copy() \
            .next_to(ppf_model.axes.c2p(0, x_area_75), LEFT, buff=.15)

        self.play(Write(x_label_cdf), Write(x_label_ppf))
        self.wait()

        # copy the x-value over the question mark
        self.play(
            FadeOut(area_question),
            x_label_cdf.animate.move_to(area_question)
        )
        self.wait()


# execute all scene renders
if __name__ == "__main__":
    render_scenes(q="k", scene_names=["TiledScene"])
    # render_scenes(q="k")
