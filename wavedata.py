
# description:
# wavedata - processing on waveform-like data
# tools to make it easier to process xy data using numpy
# to store, analyze, transform, and display xy data
# such as time series or waveform data
# perform common scientific operations on xy data

from waveclass import Wave
from wave2d import Wave2D
from vector import Array,V,V3,Vs,Polar,Piecewise
from util import *
# Point = Vector = V
# Path = Points = Vectors = Vs

if __name__ == '__main__':
    from wavetests import  *
    plotit = 1
    wavetest(plot=plotit)
    wave2Dtest(plot=plotit)
    plottest(plot=plotit)
    complextest()
    histtest(plot=plotit)
    arraytest()
    vtest()
    vstest(1,plot=plotit)
    arctest(plot=plotit)
    circletest(plot=plotit)

    # TODO: wave2d[:,0], wave2d.flatten(), wave2d.max() should return wave not wave2d
    # TODO: fix np.ones_like(wave2d) returns wave2d without .xs,.ys
    # TODO: research pandas multidim, pandas plotting
    # TODO: bevel,round joint instead of miter, round corners
