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

class PDFScene(Scene):
    def construct(self):

        # standard deviation initial
        normpdf = NormalPDF(mean=data.mean(),std=data.std(), show_x_axis_labels=X_LABELS)

        mu_tracker = ValueTracker(data.mean())
        sigma_tracker = ValueTracker(data.std())

        mu_output = lambda: norm.pdf(mu_tracker.get_value(),
                                mu_tracker.get_value(),
                                sigma_tracker.get_value()
                             )

        sigma_output = lambda: norm.pdf(mu_tracker.get_value() + sigma_tracker.get_value(),
                                mu_tracker.get_value(),
                                sigma_tracker.get_value()
                             )

        # highlight mu and sigma with lines and labels
        mu_trace = always_redraw(lambda: DashedLine(start=normpdf.axes.c2p(mu_tracker.get_value(),0),
                              end=normpdf.axes.c2p(mu_tracker.get_value(), mu_output()),
                              color=YELLOW
                              )
        )

        mu_label = always_redraw(lambda: MathTex(r"\mu = ", round(mu_tracker.get_value(), 2)) \
            .scale(.75) \
            .set_color(YELLOW) \
            .next_to(mu_trace, UP)
        )

        sigma_trace = always_redraw(lambda: DashedLine(start=normpdf.axes.c2p(mu_tracker.get_value() + sigma_tracker.get_value(),0),
                              end=normpdf.axes.c2p(mu_tracker.get_value() + sigma_tracker.get_value(),sigma_output()),
                              color=BLUE
                              )
        )
        sigma_label = always_redraw(lambda: MathTex(r"\sigma = ", round(sigma_tracker.get_value(), 2)) \
            .scale(.75) \
            .set_color(BLUE) \
            .next_to(sigma_trace, UR)
        )

        # animate mu and sigma tracing
        self.play(LaggedStart(Create(normpdf, lag_ratio=.25)))
        self.wait()
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
        sigma_length = Line(start=normpdf.axes.c2p(mu_tracker.get_value(),0),
                            end=normpdf.axes.c2p(mu_tracker.get_value() + sigma_tracker.get_value(),0))

        sigma_brace = Brace(sigma_length, UP)
        self.remove(sigma_label)
        sigma_label = sigma_label.copy()
        self.add(sigma_label)
        sigma_label.save_state()
        self.play(sigma_label.animate.scale(.9).next_to(sigma_brace, UP), Write(sigma_brace))
        self.wait()
        self.play(FadeOut(sigma_brace), Restore(sigma_label))
        self.wait()


class PDFAreaScene(Scene):
    def construct(self):
        mean = data.mean()
        std = data.std()
        normpdf = NormalPDF(mean=mean,std=std, show_x_axis_labels=X_LABELS)
        self.add(normpdf)

        def get_sigma_line(sigma):
            grp = VDict()

            grp["line"] = DashedLine(start=normpdf.axes.c2p(mean + std*sigma, 0),
                               end=normpdf.axes.c2p(mean+std*sigma,
                                                    norm.pdf(mean+std*sigma,mean,std)
                                                    ),
                               color=BLUE
                               )

            grp["label"] = MathTex(sigma, r"\sigma").scale(.6).next_to(grp["line"], UP + (LEFT if sigma < 0 else RIGHT) *.5)

            return grp

        sigma_lines = [get_sigma_line(i) for i in range(-3,4,1)]
        self.play(*[Write(sl["line"]) for sl in sigma_lines])
        self.wait()
        self.play(LaggedStart(*[Write(sl["label"]) for sl in sigma_lines]))
        self.wait()


class FormulaScene(Scene):
    def construct(self):
        f = NormalDistributionTex()

        # write formula
        self.play(Write(f))
        self.wait()

        # highlight mu
        self.play(
            f.mu.animate.set_color(RED),
            Circumscribe(f.mu)
        )
        self.wait()

        # highlight sigma
        self.play(
            f.sigmas.animate.set_color(BLUE),
            *[Circumscribe(s) for s in f.sigmas]
        )

        self.wait()

        # highlight Euler's number and Pi
        euler_constant = VGroup(MathTex("e = "), DecimalNumber(math.e.real, num_decimal_places=4, show_ellipsis=True)) \
            .arrange(RIGHT) \
            .to_edge(DOWN)

        pi_constant = VGroup(MathTex(r"\pi = "), DecimalNumber(math.pi.real, num_decimal_places=4, show_ellipsis=True)) \
            .arrange(RIGHT) \
            .next_to(euler_constant, UP)

        for v in euler_constant:
            v.set_color(YELLOW)
        for v in pi_constant:
            v.set_color(YELLOW)

        self.play(
            f.eulers_number.animate.set_color(YELLOW),
            Circumscribe(f.eulers_number),
            FadeIn(euler_constant)
        )
        self.wait()
        self.play(
            f.pi.animate.set_color(YELLOW),
            Circumscribe(f.pi),
            FadeIn(pi_constant)
        )
        self.wait(1)
        self.play(FadeOut(euler_constant), FadeOut(pi_constant))
        self.wait()


import math

class ConstantsExamples(ZoomedScene):

  def construct(self):
        func1 = lambda x: 1.0 / (math.sqrt(2*math.pi)) * math.exp(-.5*x**2)
        ax1 = Axes(
            axis_config={"include_numbers": True},
            x_range=[-3,3,1],
            y_range=[-.1,.6,.25]
        )
        plot1 = ax1.plot(func1,color=BLUE)

        formula = MathTex(r"f(x) = \frac{1}{\sigma \sqrt{2\pi}}e^{\frac{1}{2}(\frac{x-\mu}{\sigma})^2}").scale(1).to_edge(UR)

        # self.add(index_labels(formula[0]))
        self.add(ax1, plot1)

        alpha = ValueTracker(.1)
        dot = always_redraw(lambda: Dot().move_to(plot1.point_from_proportion(alpha.get_value())))
        line = always_redraw(lambda: DashedLine(start=dot.get_center(),
                                                end=[dot.get_center()[0],ax1.c2p(0,0)[1],0]
                                                ).set_color(YELLOW)
                                                )

        self.add(dot,line)

        self.camera.frame.save_state()
        self.camera.frame.scale(0.45).move_to(formula)
        self.play(Write(formula))
        self.wait()


        # highlight constants and variable
        self.play(Circumscribe(formula[0][11],time_width=1,color=RED),
                  Circumscribe(formula[0][12],time_width=1,color=RED),
                  *[f.animate.set_color(RED) for i,f in enumerate(formula[0]) if i in {11,12}]
                  )
        self.wait()
        self.play(Circumscribe(formula[0][7],time_width=1,color=BLUE),
                  Circumscribe(formula[0][19],time_width=1,color=BLUE),
                  Circumscribe(formula[0][21],time_width=1,color=BLUE),
                  *[f.animate.set_color(BLUE) for i,f in enumerate(formula[0]) if i in {7,19,21}]
                  )
        self.wait()
        self.play(Circumscribe(formula[0][17],time_width=1,color=YELLOW),
                  Circumscribe(formula[0][2],time_width=1,color=YELLOW),
                *[f.animate.set_color(YELLOW) for i,f in enumerate(formula[0]) if i in {2,17}]
                )

        self.wait()
        key = MathTex(r"\mu = 0", r"\sigma=1").arrange(DOWN).next_to(formula,DOWN).to_edge(RIGHT).scale(.8)
        key[0][0].set_color(BLUE)
        key[1][0].set_color(BLUE)

        # zoom out
        self.play(Restore(self.camera.frame), FadeIn(key))
        self.play(alpha.animate.set_value(.9), run_time=5)
        self.play(alpha.animate.set_value(.1), run_time=5)

        self.wait()


class TiledScene(Scene):
    def construct(self):
        pdf_dist = NormalPDF(mean=data.mean(), std=data.std())
        cdf_dist = NormalCDF(mean=data.mean(), std=data.std())

        left = VGroup(pdf_dist, cdf_dist) \
            .arrange(UP) \
            .scale_to_fit_height(7)

        # create PPF off CDF through rotation
        ppf_dist = cdf_dist.copy().rotate(180 * DEGREES, axis=X_AXIS) \
            .rotate(90 * DEGREES)
        
        ppf_dist.cdf_plot.set_color(ORANGE)

        right = VGroup(ppf_dist)

        # add and then remove PPF
        self.add(VGroup(left, right).arrange(RIGHT))
        right.remove(ppf_dist)

        # transition PPF
        self.wait()
        cdf_copy = cdf_dist.copy()
        cdf_copy.generate_target(ppf_dist)
        self.play(
            Rotate(cdf_copy, 180 * DEGREES, axis=X_AXIS)
        )
        self.play(
            Rotate(cdf_copy, 90 * DEGREES)
        )
        self.play(
            cdf_copy.animate.become(ppf_dist),
        )

# execute all scene renders
if __name__ == "__main__":
    render_scenes(q="l", play=True, scene_names=["ConstantsExamples"])
    #render_scenes(q="k")
