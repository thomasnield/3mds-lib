from manim import *

class NormalDistributionTex(MathTex):
    def __init__(self, fill_color=WHITE):
        super().__init__(r"f(x) = \frac{1}{\sigma \sqrt{2\pi}}e^{\frac{1}{2}(\frac{x-\mu}{\sigma})^2}")
        self.fill_color = fill_color

    def with_index_labels(self):
        return VGroup(self, index_labels(self))

    @property
    def mu(self):
        return self[0][19]

    @property
    def sigmas(self):
        return VGroup(*[t for i,t in enumerate(self[0]) if i in (7,21)])

    @property
    def x_variables(self):
        return VGroup(*[t for i,t in enumerate(self[0]) if i in (2,17)])

    @property
    def left_hand(self):
        return self[0][0:4]

    @property
    def right_hand(self):
        return self[0][5:]

    @property
    def eq_sign(self):
        return self[0][4:5]

    @property
    def denominator(self):
        return self[0][7:12]

    @property
    def numerator(self):
        return self[0][5:6]

    @property
    def fraction_nut(self):
        return self[0][6]

    @property
    def eulers_number(self):
        return self[0][12]

    @property
    def euler_with_exponent(self):
        return self[0][12:]

    @property
    def euler_exponent(self):
        return self[0][13:]

    @property
    def pi(self):
        return self[0][11]