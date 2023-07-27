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

        def get_area_x(a,b,color=YELLOW): return \
            normpdf.axes.get_area(graph=normpdf.pdf_plot,
                                  x_range=(a, b),
                                  color=color)

        def get_area_z(a_sigma,b_sigma,color=YELLOW): return get_area_x(z_to_x(a_sigma), z_to_x(b_sigma), color)

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
        pdf_label = Text("Probability Density Function (PDF)").to_edge(UP)
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
        full_area_plot = get_area_z(-4,4,color=YELLOW)
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
        narrow_area = get_area_x(69,70,color=YELLOW)
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
            Transform(narrow_area, get_area_x(68,71,color=YELLOW)),

            Transform(area_label_a_b, MathTex("A = ", round(get_cdf(71) - get_cdf(68), 2), color=YELLOW) \
                      .next_to(callout_line.end, RIGHT)),

            Transform(a_label, MathTex(68, color=YELLOW).scale(.75).next_to(axes.c2p(68,0), DOWN)),

            Transform(b_label, MathTex(71, color=YELLOW).scale(.75).next_to(axes.c2p(71,0), DOWN))
        )
        self.wait()

        # widen range to 64-77
        self.play(
            Transform(narrow_area, get_area_x(64,77,color=YELLOW)),

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

class TiledScene(Scene):
    def construct(self):
        pdf_dist = NormalPDF(mean=data.mean(), std=data.std())
        cdf_dist = NormalCDF(mean=data.mean(), std=data.std())

        pdf_dist_start = pdf_dist.copy()

        left = VGroup(pdf_dist, cdf_dist) \
            .arrange(UP) \
            .scale_to_fit_height(7)

        # create PPF off CDF through rotation
        ppf_dist = cdf_dist.copy().rotate(180 * DEGREES, axis=X_AXIS) \
            .rotate(90 * DEGREES)

        ppf_dist.cdf_plot.set_color(ORANGE)

        right = VGroup(ppf_dist)
        tiles = VGroup(left, right).arrange(RIGHT)

        # initialize PDF
        self.add(pdf_dist_start)
        self.wait()
        self.play(
            ReplacementTransform(pdf_dist_start, pdf_dist)
        )
        self.wait()
        self.play(
            FadeIn(tiles)
        )
        # add and then remove PPF
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



from manim import config

# execute all scene renders
if __name__ == "__main__":
    render_scenes(q="l", play=True, scene_names=["TiledScene"])
    # render_scenes(q="k")
