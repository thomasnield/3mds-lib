import math

from manim import *
from threemds.stats.plots import DottedNumberLine, Histogram
from threemds.stats.formulas import NormalDistributionTex
from threemds.probability.distributions import  NormalPDF, X_LABELS, NormalCDF

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
        self.play(Write(create_number_line()))
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
        self.play(Create(hist), Create(bin_text))
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

class PDFScene(Scene):
    def construct(self):

        normpdf1 = NormalPDF(mean=data.mean(),std=data.std(), show_x_axis_labels=X_LABELS)
        normpdf2 = NormalPDF(mean=data.mean(),std=data.std()*2, show_x_axis_labels=X_LABELS)
        normpdf3 = NormalPDF(mean=data.mean(),std=data.std()*3, show_x_axis_labels=X_LABELS)

        mu_tracker = ValueTracker(data.mean())
        sigma_tracker = ValueTracker(data.std())

        mu_sigma_label = always_redraw(lambda: MathTex(r"\mu &= ",
                                                       round(mu_tracker.get_value(),2),
                                                       r"\\ \sigma &= ",
                                                       round(sigma_tracker.get_value(),2)
                                                       ).to_edge(UR, buff=.5)
                                       )
        self.play(Create(normpdf1), Write(mu_sigma_label))
        self.wait()
        normpdf1.save_state()
        self.play(Transform(normpdf1.pdf_plot, normpdf2.pdf_plot),
                  Transform(normpdf1.x_labels, normpdf2.x_labels),
                  sigma_tracker.animate.set_value(data.std() * 2))
        self.wait()
        self.play(Transform(normpdf1.pdf_plot, normpdf3.pdf_plot),
                  Transform(normpdf1.x_labels, normpdf3.x_labels),
                  sigma_tracker.animate.set_value(data.std() * 2))
        self.wait()
        self.play(Restore(normpdf1), sigma_tracker.animate.set_value(data.std()))
        self.wait()

        # move mean around
        self.play(
            Transform(normpdf1.pdf_plot,
                      normpdf1.copy().shift(RIGHT * normpdf1.axes.c2p(data.mean() + data.std(),0)[0]).pdf_plot
                      ),
            mu_tracker.animate.set_value(data.mean() + data.std())
        )
        self.wait()
        self.play(Restore(normpdf1), mu_tracker.animate.set_value(data.mean()))
        self.wait()

class PDFAreaScene(Scene):
    def construct(self):

        mean = data.mean()
        std = data.std()

        # transition histogram to PDF
        hist = Histogram(data,
                  bin_count=11,
                  show_points=True,
                  show_normal_dist=True)

        norm_pdf = NormalPDF(mean=mean,
                             std=std,
                             show_area_plot=True)
        self.add(hist)
        self.wait()
        self.play(
            FadeOut(hist.bars),
            FadeOut(hist.dots),
        )
        self.wait()
        self.play(
            ReplacementTransform(hist.normal_dist_plot, norm_pdf.pdf_plot),
            FadeTransform(hist.axes, VGroup(norm_pdf.axes, norm_pdf.x_labels)),
        )
        self.play(
            Write(norm_pdf.area_range_label)
        )
        self.wait()

        # Start highlighting areas for 1 through 4 standard deviations
        self.play(
            Create(norm_pdf.area_plot)
        )

        self.wait()
        norm_pdf.area_lower_range.set_value(mean)
        norm_pdf.area_upper_range.set_value(mean)

        for sigma in range(1,4):
            self.play(
                norm_pdf.area_lower_range.animate.set_value(mean - std * sigma),
                norm_pdf.area_upper_range.animate.set_value(mean + std * sigma)
            )
            self.wait()


        # switch to sigma mode
        self.play(
            ReplacementTransform(norm_pdf.x_labels, norm_pdf.x_sigma_labels),
            ReplacementTransform(norm_pdf.area_range_label, norm_pdf.sigma_area_range_label)
        )
        self.wait()

        for sigma in range(1,4):
            self.play(
                norm_pdf.area_lower_range.animate.set_value(mean - std * sigma),
                norm_pdf.area_upper_range.animate.set_value(mean + std * sigma)
            )
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

class CDFScene(Scene):
    def construct(self):
        pdf_dist = NormalPDF(mean=data.mean(), std=data.std())
        cdf_dist = NormalCDF(mean=data.mean(), std=data.std())

        self.add(pdf_dist.axes, pdf_dist.pdf_plot, pdf_dist.x_labels)
        self.wait()
        self.play(
            ReplacementTransform(pdf_dist.pdf_plot, cdf_dist.pdf_plot),
            ReplacementTransform(pdf_dist.axes, cdf_dist.axes),
            ReplacementTransform(pdf_dist.x_labels, cdf_dist.x_labels)
        )
        self.wait()
        self.play(
            Create(cdf_dist.cdf_plot),
            Create(cdf_dist.pdf_to_cdf_line)
        )
        self.wait()

        # label PDF and CDF
        pdf_label = Text("PDF", color=BLUE).next_to(
            cdf_dist.axes.c2p(data.mean(), cdf_dist.f_pdf(data.mean())), UR
        )
        cdf_label = Text("CDF", color=RED).next_to(
            cdf_dist.axes.c2p(data.mean(), cdf_dist.f_cdf(data.mean())), UL
        )

        self.play(
            Write(pdf_label), Write(cdf_label)
        )
        self.wait()
        self.play(
            FadeOut(pdf_label), FadeOut(cdf_label)
        )

        self.wait()
        self.add(cdf_dist, cdf_dist.pdf_plot, cdf_dist.pdf_area, cdf_dist.pdf_to_cdf_line)
        self.wait()
        self.play(cdf_dist.x_tracker.animate.set_value(cdf_dist.axes.x_range[1]), run_time=7)
        self.wait()
        self.play(cdf_dist.x_tracker.animate.set_value(cdf_dist.axes.x_range[0]), run_time=7)
        self.wait()

class InverseCDF(Scene):
    def construct(self):
        pass