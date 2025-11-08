# coding: utf8 # only needed for python 2

import numpy as np
import matplotlib
import pickle
from waveclass import Wave
from wave2d import Wave2D
from vector import Array,V,V3,Vs,Polar,Piecewise
from util import wrange,timeit,Vec,trapezoidintegrate,discretepdfsample
from util import coscurvefit,curvefit,quadfit,polyfit,stepclimb,list2str
from util import storecallargs,maplistargs
from numpy import sqrt,pi,sin,cos,exp,nan,inf
from waveclass import interpolate1d
from wavedata import gaussianproduct
# from rich.traceback import install
# install()
# import pandas as pd
# from wavepd import Wavepd as Wave

def wavetest(plot=0):
    #a = np.array([1,2,3]); a.x = 5 # error
    #class C(np.ndarray): pass; c = C(a); c.x = 5 # no error
    # print(set(dir(Wave(1)))-set(dir(np.ndarray)))
    # print( dir(Wave(1)), dir(np.ndarray) )

    # ps = pd.Series(data=[1,2,3],index=[0,1,2])
    # ps+7
    # print(ps)
    # print(ps+7)
    Wave([1,2,3]) # print(Wave([1,2,3]))

    if 0:
        class Testwave(pd.Series):
            def __init__(self, data=None, index=None, name=None, dtype=None, copy=False, fastpath=False, indexname=None):
                if isinstance(data, (int, float)) and index is None:
                    data = np.zeros(data)
                if data is None and index is not None:
                    data = index # make it easy to create a wave that's just an x wave, ie. wx = Wave(index=[0,1,2],indexname='wx')
                    name,indexname = None,name
                if indexname:
                    index = pd.Series(data=index,name=indexname)
                if isinstance(data,Testwave) and index is None:
                    super(Testwave, self).__init__(data=data, index=data.index, dtype=type(data[0]), name=name, copy=copy, fastpath=fastpath)
                super(Testwave, self).__init__(data=data, index=index, dtype=float, name=name, copy=copy, fastpath=fastpath)
            def __str__(self):
                return str(np.array2string(np.array(self), threshold=19)).replace('\n','') + ' x:' + str(np.array2string(np.array(self.x), threshold=19)).replace('\n','')
                # def format(a): return '['+' '.join([('%.2f'%n).rstrip('0').rstrip('.') for n in a])+']'
                # return format(self) + ' x:' + format(self.x)
            @property
            def _constructor(self):
                return Testwave
            @property
            def p(self):
                return Testwave(np.arange(len(self)),index=self.index)
            @property
            def y(self):
                return self.values
            @property
            def x(self):
                return Testwave(data=np.array(self.index),index=np.array(self.index),name=self.index.name)
            # def array(self): # causes failure in pandas 1.0.3
            #     return np.asarray(self)

        # print()
        w = Testwave([1,2,3],[-1,0,1])
        w+5
        # print('w',w)
        # print('w+5',w+5) # fails in pandas 1.0.3
        # print('w.index',w.index)
        # print()
        # print('***')
        # print('*** passed new tests ***')
        # print('***')
        # print()

    a = 1+np.arange(3); w = Wave(a)
    w+7
    # print('w[0]',type(w[0]),w[0])
    # print('w.x[0]',type(w.x[0]),w.x[0])
    # print('list(w)',type(list(w)),list(w))
    # print('list(w.x)',type(list(w.x)),list(w.x))
    # print()
    # print('w+7',w+7)
    # print()

    # print('w.p',w.p)
    # print('list(9+w.p)',type(list(9+w.p)),list(9+w.p))
    # print('w',type(w),w)
    # print('w.x',type(w.x),w.x)
    # print('w.p',type(w.p),w.p)
    w.x = list(9+w.p) # w.setdx(9,dx=1)
    assert list(w)==[1,2,3]
    assert list(w.x)==[9,10,11]
    assert list(w.p)==[0.,1.,2.]

    v = np.exp(w)
    assert list(w.x)==list(v.x) # print(list(w.x),list(v.x))

    # w = Wave(a); 
    # w.setix(0,20) # print('w',w); print('w.x',w.x)
    # assert 0==w.x[0] # print('w.x[0]',w.x.iloc[0],w.x[0])
    # assert 10==w.x[1] # print('w.x[1]',w.x.iloc[1],w.x[1])
    # # assert [w.atx(x) for x in [-5,5,15,25]]==[1.0,1.5,2.5,3.0] # print(w,w.x,'atx 0,5,15,20:',w.atx(0),w.atx(5),w.atx(15),w.atx(20))
    # assert [w.atx(x) for x in [0,5,15,20]]==[1.0,1.5,2.5,3.0], [w.atx(x) for x in [0,5,15,20]]# 

    # w = Wave(a); w.setix(xleft=0,xright=20)
    # assert [w.atx(x) for x in [0,5,15,20]]==[1.0,1.5,2.5,3.0] # print(w,w.x,'atx -5,5,15,25:',w.atx(-5),w.atx(5),w.atx(15),w.atx(25))

    # w = Wave(a); w.setix(20,0)
    # assert [w.atx(x) for x in [0,5,15,20]]==[x for x in reversed([1.0,1.5,2.5,3.0])] # print('w',w);print('w.x',w.x);print('atx -5,5,15,25:',w.atx(-5),w.atx(5),w.atx(15),w.atx(25),reversed([1.0,1.5,2.5,3.0]))

    # w = Wave(a); w.setpx(100,dx=100) # print(w.index.name,w.idxmin(),w.values.argmin())
    # assert (w.xmin(),w.xmax())==(100,300) # print(w,w.x,'xmin,xmax:',w.xmin(),w.xmax())

    # w = Wave(a); w.setx(x0=10,xn=13)
    # assert [10,11,12]==list(w.x)

    # b,c = np.arange(5),np.arange(5); c[1:4] = 10+b[1:4]; print('b,c',b,c)
    # b = np.arange(11)-5; c = Wave(b,name='c').setix(-5,5)
    b = np.arange(11)-5; c = Wave(b,wrange(-5,5,1),name='c')
    assert type(c.xwave)==type(np.negative(c.xwave)) # print(type(c.xwave),type(np.negative(c.xwave)))
    assert list(c)==[-5,-4,-3,-2,-1,0,1,2,3,4,5] and list(c.xwave)==[-5.,-4.,-3.,-2.,-1.,0.,1.,2.,3.,4.,5.] # print('c',c); print('c.xwave',c.xwave)
    assert list(c.xwave)==[-5.,-4.,-3.,-2.,-1.,0.,1.,2.,3.,4.,5.] # print('c',c); print('c.xwave',c.xwave)
    assert list( (1*(c.xwave>0)-1*(c.xwave<0)).x )==list(c.xwave)
    c = 1*(c.xwave>0) - 1*(c.xwave<0) # print('cc',cc); print('c.xwave',c.xwave)

    a = Wave([1,6,7]) # print( Wave( data=np.array(a), index=np.array(a) ) )

    d = Wave(c,name='d') # print('c',c); print('c.xwave',c.xwave); print('d',d); print('d.x',d.x)
    assert list(d)==[-1,-1,-1,-1,-1,0,1,1,1,1,1] and list(d.x)==[-5.,-4.,-3.,-2.,-1.,0.,1.,2.,3.,4.,5.] # 
    # assert isinstance(d,Wave) and isinstance(d,np.ndarray)
    # assert isinstance(d,Wave) and (isinstance(d,pd.Series) or isinstance(d,np.ndarray))

    if 0:
        e = Wave(9)# 
        print('d',d)
        print('e',e)
        assert list(e)==[0]*9 # print(d+e) # not an error, result has combined index
        def eq(u,v): return (u!=u and v!=v) or u==v
        assert all([ eq(a,b) for a,b in zip( list(d+e), [np.NaN]*5+[0,1,1,1,1,1]+[np.NaN]*3) ])
    
    # print(Wave(a)+[0,1]) # correctly raises broadcast error
    assert [2,7,8]==list(Wave(a)+[1,1,1]) # no error

    # r = 10*Wave(np.random.rand(21)).setix(-5,5); r += (r.x-4)**2; r[5:10] = r.p[5:10]
    r = 10*Wave(np.random.rand(21),np.linspace(-5,5,21)); r += (r.x-4)**2; r[5:10] = r.p[5:10]
    assert 8==r.atx(-1)
    rcopy = r[5:10] # rcopy.printwave('rcopy')
    assert list(rcopy)==[5,6,7,8,9] and list(rcopy.x)==[-2.5,-2,-1.5,-1,-0.5] and list(rcopy.p)==[0,1,2,3,4]
    rcopy += 10-2*rcopy.p # rcopy.printwave('rcopy'); r[5:10].printwave('r[5:10]')
    assert list(rcopy)==[15,14,13,12,11] and list(rcopy.x)==[-2.5,-2,-1.5,-1,-0.5] and list(rcopy.p)==[0,1,2,3,4]
    
    if 0:
        print((r[5:10]),[5,6,7,8,9] , list(r[5:10].x)==[-2.5,-2,-1.5,-1,-0.5] , list(r[5:10].p)==[0,1,2,3,4])
        assert list(r[5:10])==[5,6,7,8,9] and list(r[5:10].x)==[-2.5,-2,-1.5,-1,-0.5] and list(r[5:10].p)==[0,1,2,3,4]
        r.smooth(11,savgol=False)

    rrr = Wave(21).p.smooth(11,savgol=False)
    assert 5==rrr[5]
    #rrrr = Wave(21).p.smooth(11,savgol=True)
    #assert 1==round(rrrr[1]*1e14)/1e14


    # import matplotlib.pyplot as plt
    # plt.rcParams['axes.facecolor']=plt.rcParams['savefig.facecolor']=plt.rcParams['figure.facecolor']='white'
    # plt.rc('font',family='Arial')
    # plt.plot(r.x.y,r.y)
    # plt.plot(r.x.y,r.y,'darkred',rr.x.y,rr.y,'darkblue',rcopy.x.y,rcopy.y,'darkgreen',marker='o')
    # plt.show()

    w = Wave(np.arange(4)*2); w.tmp=0; del w.tmp; # w.p = np.arange(len(w)) # del w.x # correctly raises error
    w = Wave(np.arange(4)*2); w.x=np.arange(-2,6,2); #w.printwave('w')
    u = Wave(np.arange(4),w.xwave); #u.printwave('u')
    uu = Wave(w.xwave,w.xwave); #uu.printwave('uu')
    uuu = Wave(w.xwave); #uuu.printwave('uuu')
    # w [0 2 4 6] [-2  0  2  4] [0 1 2 3]
    # u [0 1 2 3] [-2  0  2  4] [0 1 2 3]
    # uu [-2  0  2  4] [-2  0  2  4] [0 1 2 3]
    # uuu [-2  0  2  4] [-2  0  2  4] [0 1 2 3]
    assert list(w.xwave)==list(u.xwave)==list(uu.xwave)==list(uuu.xwave)

    z = Wave(1+np.arange(3)); z.x *= 2; #z.printwave('z')
    assert list(z.x)==[0,2,4]
    zz = Wave(1+np.arange(3)); zz.x = 3*zz.p; #zz.printwave('zz')
    assert list(zz.x)==[0,3,6]
    # v = Wave(w)
    # v.printwave('v')
    # v.x.printwave('v.x') # need this to return its own xwave
    # np.exp(v.p).printwave()
    # np.exp(v.x).printwave()

    u = Wave([10,12,13],name='y (nm)')
    u.x = (1,2,3)
    # u.index.name = 'x (mm)'
    if plot:
        wx = np.linspace(-2.,5.,17)
        ws = [Wave(np.abs(wx)**a,wx) for a in np.linspace(1.0,1.5,6)]
        # ws[0].plot(ws,groups=2,m=1)
        u.plot(*ws,m=True,xlim=(-1,None),ylim=(None,15),groupsize=2)

    g = Wave([4,5,6],[0,2,3],name='g')#,indexname='gx')
    assert 4==g[0] and 5==g[1] and 6==g[-1]
    assert 0==g.x[0] and 2==g.x[1] and 3==g.x[-1]
    assert 4.5==g(1) # print(g(1))
    assert 1==g.xaty(4.5) # print(g.xaty(4.5)) #g.plot()
    assert 'g'==g.name==(g[0:1]).name
    # assert 'gx'==g.index.name==g.x.name==Wave(g).index.name==(2*g).index.name
    #print('2g',(2*g).name,Wave(g).name) # erase name after operations?
    # print( str(g));print( str(g/3));print( str(g[:2]));print( str(g[1:]));print( str(g[1:2]))
    # assert str(g)=='[4. 5. 6.] x:[0 2 3]'
    # assert str(g/3)=='[1.33333333 1.66666667 2.        ] x:[0 2 3]'
    # assert str(g[:2])=='[4. 5.] x:[0 2]'
    # assert str(g[1:])=='[5. 6.] x:[2 3]'
    # assert str(g[1:2])=='[5.] x:[2]'

    f = Wave(np.random.random(5),np.random.random(5))
    h = Wave(f) # duplicate a wave
    assert all(fi==hi for fi,hi in zip(f,h)) and all(fi==hi for fi,hi in zip(f.x,h.x))
    h = f.copy() # duplicate a wave
    assert all(fi==hi for fi,hi in zip(f,h)) and all(fi==hi for fi,hi in zip(f.x,h.x))

    assert [0,0,0]==list(Wave(3))
    assert [0,1,2]==list(Wave(3).x)==list(Wave(3).p)

    try:
        from scipy.signal import savgol_filter
        savgol_filter(np.arange(21), window_length=11, polyorder=2)
    except FutureWarning:
        print('no savgol_filter')

    k = Wave([2,2,3,3,3],[10,11,12,13,14])
    assert np.allclose(k.zeropad(scale=1).y,[0,2,2,3,3,3,0,0])
    assert np.allclose(k.zeropad(scale=2).y,[0,0,0,0,0,2,2,3,3,3,0,0,0,0,0,0])
    assert np.allclose(k.zeropad(scale=1,poweroftwo=False).y,k.y)
    assert np.allclose(k.zeropad(scale=2,poweroftwo=False).y,[0,0,2,2,3,3,3,0,0,0])
    # print(Wave([0,0,1,1,0,0]).zeropad(1,1))
    assert np.allclose(Wave([0,0,1,1,0,0]).zeropad(1,1).y,[0,0,0,1,1,0,0,0])
    # print(Wave([0,0,1,1,0,0]).zeropad(1,1).fft(0).x)
    assert np.allclose(Wave([0,0,1,1,0,0]).zeropad(1,1).fft(0).x,[-0.5, -0.375, -0.25, -0.125, 0.0, 0.125, 0.25, 0.375])
    # k.fft().plot()
    # Wave([1,0,1,0,1,0,1,0]).fft().plot()

    m = Wave([0,0,1,1],[0,0.4,0.6,1])
    mmx = Wave(np.linspace(0,1,11),np.linspace(0,1,11))
    mm = m(mmx)
    # print(mm,type(mm))
    assert np.allclose(mm.x,[0.,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.])
    assert isinstance(mm,Wave)

    assert np.allclose(mm(0.3,0.6).x,[0.3,0.4,0.5,0.6])
    assert np.allclose(mm(0.19,0.49).x,[0.2,0.3,0.4])
    assert np.allclose(mm(0.21,0.51).x,[0.3,0.4,0.5])

    # with open('__pycache__/test.pickle', 'wb') as outfile:
    # # assert np.allclose(mm(0.2,0.5).x,[0.2,0.3,0.4,0.5])
    #     pickle.dump(m, outfile)
    # with open('__pycache__/test.pickle', 'rb') as infile:
    #     mp = pickle.load(infile)
    # assert np.array_equal(m,mp) # print('m:',m) print('mp:',mp)
    p = pickle.dumps(m)
    mp = pickle.loads(p)
    assert np.array_equal(m,mp) # print('m:',m) print('mp:',mp)

    kk = Wave([2,4,5],[100,101,103]) # print(kk)
    assert np.allclose(kk.upsample().x,[100.0,100.1,100.2,100.3,100.4,100.5,100.6,100.7,100.8,100.9,101.0,101.2,101.4,101.6,101.8,102.0,102.2,102.4,102.6,102.8,103.0]) # print(list(kk.upsample().x))
    assert np.allclose(kk.upsample(2).y,Wave([2,3,4,4.5,5],[100,100.5,101,102,103]).y) # print(kk.upsample(2))

    # fwhmtest
    x,xx = np.linspace(-1,1,3),np.linspace(-1,1,1001)
    y,yy = 1-abs(x),1-xx**2
    Δx,Δxx = Wave(y,x).fwhm(),Wave(yy,xx).fwhm()#;print('Δx,Δxx');print(Δx,Δxx)
    assert np.allclose([Δx,Δxx], [1,sqrt(2)]), [Δx,Δxx]
    xxx = np.linspace(-10,10,1001)
    yyy = Wave(10*np.sinc(xxx),xxx) # print(yyy.fwhm()) # yyy.plot()
    assert 1.2066926064705057==yyy.fwhm()

    w = Wave([1,2,3],[0,10,20])
    assert w[::-1]==Wave([3,2,1],[20,10,0])

    print('wave tests passed')

    # f.plot([g,h])
    # f.igor(waves=[g,h])
    # make sure index.name is always transfered
    # g[3.5]
    # g[-1.5]? (at end of list)
    # g[-0.]?
    # g.xy (return n x 2 array of xy points)
    # x2p,p2x,area,insertpoints,appending waves,integrate
    # deletepoints reset w.p, w.x=w.p
    # w[5:10] = 10+w.p[5:10]
    # wave1(4,5) = wave2(x+1)
    # dimension labels for plotting?
def wave2Dtest(plot=0):
    w = Wave2D(xs=[0,1,2],ys=[10,20])
    assert list(w.xs)==[0,1,2]
    assert list(w.ys)==[10,20]
    assert isinstance(w.xx,Wave2D)
    assert isinstance(w.xx+1,Wave2D)
    assert isinstance(2*w.yy,Wave2D)
    assert isinstance(w.xx+w.yy,Wave2D)
    # print('w',w,'w.xx',w.xx,type(w.xx),'w.xs',w.xs)
    # print('w.xx+w.yy',w.xx+w.yy)
    w = 1+w.xx+w.yy/10
    if plot: w.plot(legendtext='The Title')
    # print(w.atxy(0,10),w.atxy(0.5,10),w.atxy(0,15),w.atxy(0.5,15))
    # print(w.atxy(0,10,method='linear'),w.atxy(0.5,10,method='linear'),w.atxy(0,15,method='linear'),w.atxy(0.5,15,method='linear'))
    # print(w.atxy(0,10,method='nearest'),w.atxy(0.5,10,method='nearest'),w.atxy(0,15,method='nearest'),w.atxy(0.5,15,method='nearest'))
    assert np.allclose( [2,2.5,2.5,3], [w.atxy(0,10),w.atxy(0.5,10),w.atxy(0,15),w.atxy(0.5,15)] )
    assert np.allclose( [2,2,2,2], [w.atxy(0,10,method='nearest'),w.atxy(0.5,10,method='nearest'),w.atxy(0,15,method='nearest'),w.atxy(0.5,15,method='nearest')] )
    assert np.allclose( [2,2.5,2.5,3], [w(0,10),w(0.5,10),w(0,15),w(0.5,15)] )
    assert np.allclose( [2,2.5,2.5,3], [w(0,10,'linear'),w(0.5,10,'linear'),w(0,15,'linear'),w(0.5,15,'linear')] )
    # print(w(1,10),w(1,15),w(1,20),w(1,21),w(1,22))
    # print(w(1,21)) # cubic can handle out of bounds but linear can't
    # print(w(1,21,'linear')) # raises ValueError
    assert 4==w(1,21)
    # print(w,w.pmax(),w.xymax(),w.pmin(),w.xymin())
    assert [w.pmax(),w.xymax(),w.pmin(),w.xymin()]==[(2, 1),(2.0, 20.0),(0, 0),(0.0, 10.0)]
    # print(w.min(),w.max())
    assert w.min()==2 and w.max()==5
    # print(w,w.atxy(0,10),w.atxy(0.5,10),w.atxy(0,15),w.atxy(0.5,15))
    # print(w(0,10),w.atxy(0,10),w(0.5,10),w.atxy(0.5,10))
    w2 = Wave2D(xs=[0,1,2],ys=[10,20])
    w2 = w2.xx+w2.yy
    # print([w2.atxy(0,10),w2.atxy(0.5,10),w2.atxy(0,15),w2.atxy(0.5,15)])
    assert np.allclose([w2.atxy(0,10),w2.atxy(0.5,10),w2.atxy(0,15),w2.atxy(0.5,15)],[10.0, 10.5, 15.0, 15.5])
    # print('w2.xx',w2.xx);print('w2.yy',w2.yy)
    # print('w2[0,:]',np.array(w2[0,:]));print('w2[:,0]',np.array(w2[:,0]))
    if plot: w2.plot()
    # print(w2[1].wave2textarray(),'[11. 21.] x:[10. 20.]')
    assert w2[1].wave2textarray()=='[11. 21.] x:[10. 20.]'
    assert w2[1,:].wave2textarray()=='[11. 21.] x:[10. 20.]'
    assert str(w2[1:2])=='[[11. 21.]] xs[1.] ys[10. 20.]'
    assert str(w2[1:2,:])=='[[11. 21.]] xs[1.] ys[10. 20.]'
    assert np.allclose([list(w2.atx(0.5)),list(w2.atx(0.5,method='nearest')),list(w2.atx(0.5,method='linear')),list(w2.atx(0.5,method='cubic'))], [[10,20],[10,20],[10.5,20.5],[10.5,20.5]] ) # print(w2[0,:],w2[1,:],w2.atx(0.5),w2.atx(0.5,method='nearest'),w2.atx(0.5,method='linear'),w2.atx(0.5,method='cubic'))
    assert np.allclose([list(w2.aty(15)),list(w2.aty(15,method='nearest')),list(w2.aty(15,method='linear')),list(w2.aty(15,method='cubic'))], [[10, 11, 12],[10, 11, 12],[15, 16, 17],[15, 16, 17]] )
    # uxx,uyy = Wave2D(xs=np.linspace(-4,4,9),ys=np.linspace(-4,4,9),returngrid=1)
    uxx,uyy = Wave2D(xs=np.linspace(-4,4,9),ys=np.linspace(-4,4,9)).grid()
    u = np.exp(-uxx**2-uyy**2)
    # u.plot(colormesh=1)
    assert np.allclose(u.ys,[-4,-3,-2,-1,0,1,2,3,4])
    assert np.isclose(u.min(),0) and u.max()==1
    # print(u.pmax(),u.xymax(),u.pmin(),u.xymin())
    assert [u.pmax(),u.xymax(),u.pmin(),u.xymin()]==[(4, 4),(0.0, 0.0),(0, 0),(-4.0, -4.0)]
    # xx,yy = Wave2D(xs=np.linspace(-10,10,21),ys=np.linspace(-10,10,21),returngrid=1)
    xx,yy = Wave2D(xs=np.linspace(-10,10,21),ys=np.linspace(-10,10,21)).grid()
    zz = (1/(1+xx**2+yy**2)).reduced(size=3)
    assert np.allclose( zz.flatten(), [0.00497512,0.00990099,0.00497512,0.00990099,1.,0.00990099,0.00497512,0.00990099,0.00497512] )
    assert str(zz[:1,:2])==str(zz[:1][:,:2])=='[[0.00497512 0.00990099]] xs[-10.] ys[-10.   0.]'
    if 0:
        assert isinstance(zz[0],Wave)
        assert isinstance(zz[:,0],Wave)
    assert isinstance(zz[0:1,0:1],Wave2D)
    assert isinstance(zz[0,0],np.float64)
    # with open('__pycache__/test.pickle', 'wb') as outfile: # https://stackoverflow.com/questions/26598109/preserve-custom-attributes-when-pickling-subclass-of-numpy-array
    #     pickle.dump(zz, outfile)
    # with open('__pycache__/test.pickle', 'rb') as infile:
    #     zp = pickle.load(infile)
    p = pickle.dumps(zz)
    zp = pickle.loads(p)
    assert np.array_equal(zz,zp) # print('zz:',zz) print('zp:',zp)
    # print(type(zz.flatten()),type(zz[0]))
    aa = Wave2D([[1,1,1,1],[1,np.nan,0,1],[1,1,1,1]])
    bb = Wave2D([[1,1,1,1],[1,1,0,1],[1,1,1,1]])
    assert np.all(aa.replacenanwithmedian()==bb)
    print('Wave2D tests passed')
def plottest(verbose=0,plot=1):
    d1,d2,dd1,dd2 = Wave([1,2,3]),Wave([2,3,2]),Wave([3,2,1]),Wave([2,1,2])
    # for n in range(14):
    #     Wave.plots(d1+0.1,d2+0.2,dd1+0.3,dd2+0.4,d1+5.1,d2+5.2,dd1+5.3,dd2+5.4,m=1,linewidth=4,seed=n,legendtext=str(n))
    # Wave.plots(Wave([1,2,3]),Wave([2,3,2]),Wave([3,2,1]),colors=['#801515','#804515','#0D4D4D','#116611'])
    # Wave.plots(d1,d2,dd1,dd2,colors=['#D46A6A','#D49A6A','#407F7F','#55AA55'])
    # Wave.plots(d1,d2,dd1,dd2,m=1,colors=['#801515','#804515','#0D4D4D','#116611'],xlabel='output',ylabel='input')
    # Wave.plots(d1,d2,dd1,dd2,d1-0.5,d2-0.5,dd1-0.5,dd2-0.5,m=1,colors=['#003f5c','#2f4b7c','#665191','#a05195','#d45087','#f95d6a','#ff7c43','#ffa600'],xlabel='out',ylabel='in')
    wx = np.linspace(0,2*np.pi,81)
    # Wave(np.cos(wx),wx).plot(xphase=True)
    xs,ys = np.linspace(-2,4,7),np.linspace(-1,2,7)
    zs = xs[:,None]+ys[None,:]
    if verbose: print(zs,zs.shape)
    # Wave2D( zs,xs,ys ).plot()
    # import matplotlib.pyplot as plt
    # lines = [ plt.Line2D((3,0), (0,1), lw=2.5, marker='.', markersize=15, markerfacecolor='r', markeredgecolor='k', alpha=0.5), plt.Line2D((2,0), (0,0.5), color='k')]
    lines = [ {'xdata':(3,0), 'ydata':(0,1), 'lw':2.5, 'marker':'.', 'markersize':15, 'markerfacecolor':'r', 'markeredgecolor':'k', 'alpha':0.5} , {'xdata':(2,0), 'ydata':(0,0.5), 'color':'k'}]
    texts = [ {'x':-1,'y':-1,'s':'x','fontsize':36} ]
    if plot:
        Wave2D( zs,xs,ys ).plot(legendtext='γΓαβχδεφγηιϕκλµνοπθρστυϖωξψζ',corner='lower left',lines=lines,texts=texts)
        wx = np.linspace(0,3,101)
        Wave.plots(*[Wave(f(wx**2),wx) for f in [sin,cos]],m='_|',l='.—')
def fttest():
    u  = Wave([0,1,1,1,1],[0,0,0.5,0.5,1])
    uu = Wave([  1,1,1,1],[  0,0.5,0.5,1])
    f = 0.1
    # print(fourierintegral( u, u.x,f)==fourierintegral(uu,uu.x,f))
    for f in np.linspace(1.,17.,17):
        assert np.allclose([fourierintegral( u, u.x,f),fourierintegral(uu,uu.x,f)],[0,0]), f' f{f} ft:{fourierintegral( u, u.x,f)}'
    def fintegralstep(f): # returns integral(exp(i2pfx),0,0.5)
        k = 2*pi*f
        if 0==k: return 0.5
        return (sin(k/2) + 1j*2*sin(k/4)**2)/k # Sin[k/2]/k + I (2 Sin[k/4]^2)/k
    u  = Wave([0,1,1,0,0],[0,0,0.5,0.5,1])
    uu = Wave([  1,1,0,0],[  0,0.5,0.5,1])
    # f = 1; print(u.ft(f),uu.ft(f),fintegralstep(f))
    # f = 0.5; print(u.ft(f),uu.ft(f),fintegralstep(f))
    # f = 0.; print(u.ft(f),uu.ft(f),fintegralstep(f))
    for f in np.linspace(0.,2.,17):
        assert np.allclose( u.ft(f),fintegralstep(f)), f' f{f} ftu: { u.ft(f)} ft:{fintegralstep(f)}'
        assert np.allclose(uu.ft(f),fintegralstep(f)), f' f{f} ftuu:{uu.ft(f)} ft:{fintegralstep(f)}'
    def sinint0(k,x0,x1,u0,u1):
        # integrate (0 + (1-0)/(1-0)*(x-0)) * sin(k x) from 0 to 1
        return (k*u0*cos(k*x0)-k*u1*cos(k*x1)-((u0-u1)*(np.sin(k*x0)-np.sin(k*x1)))/(x0-x1))/k**2
        # du,dx = u1-u0,x1-x0
        # return du/dx/k**2*(cos(k*x1)-cos(k*x0))+(u1*np.sin(k*x1)-u0*np.sin(k*x0))/k
    def sinint(k,x0,x1,u0,u1):
        return -(u1-u0)/(x1-x0)/k**2 * (np.sin(k*x0)-np.sin(k*x1)) + 1/k * (u0*cos(k*x0)-u1*cos(k*x1))
        # return (u1-u0)/(x1-x0)/k**2 * (cos(k*x1)-cos(k*x0)) + 1/k * (u1*np.sin(k*x1)-u0*np.sin(k*x0))
    # k = 2*pi*0.25; x0,x1 = 0,1; u0,u1 = 0,1
    # print(sinint0(k,x0,x1,u0,u1),sinint(k,x0,x1,u0,u1))
    def fintegralxfrom0to1(f): # returns integral(x*exp(i2πfx),0,1)
        k = 2*pi*f
        if 0==k: return 0.5
        return (cos(k)-1+k*np.sin(k))/k**2 + 1j*(-k*cos(k)+np.sin(k))/k**2
    wx = np.linspace(0,1,2); w = Wave(wx,wx)
    wwx = np.linspace(0,1,10); ww = Wave(wwx,wwx)
    wwwx = np.linspace(1.3,2.3,23); www = Wave(wwwx-1.3,wwwx)
    f = 0.43
    # print(f,w.ft(f),ww.ft(f),fintegralxfrom0to1(f))
    # print(f,abs(w.ft(f)),abs(ww.ft(f)),abs(fintegralxfrom0to1(f)))
    for f in np.linspace(0.0,2.00,17):
        assert np.allclose(w.ft(f),fintegralxfrom0to1(f)), f'f:{f} ft:{w.ft(f)} ft0:{fintegralxfrom0to1(f)}'
        assert np.allclose(ww.ft(f),fintegralxfrom0to1(f)), f'f:{f} ft:{ww.ft(f)} ft0:{fintegralxfrom0to1(f)}'
        assert np.allclose(abs(www.ft(f)),abs(fintegralxfrom0to1(f))), f'f:{f} ft:{abs(www.ft(f))} ft0:{abs(fintegralxfrom0to1(f))}'
def complextest():
    w = Wave([1j,2j],[0,1])
    # print(w)
    # assert str(w)=='[0.+1.j 0.+2.j] x:[0 1]'
    # print(w.wave2textarray())
    assert w.wave2textarray()=='[0.+1.j 0.+2.j] x:[0 1]'
    ww = Wave([0,2j],[0,1])
    assert isinstance(ww[0],complex)
    www = w[:]
    # print(www.wave2textarray())
    # print('[0.+1.j 0.+2.j] x:[0 1]')
    assert www.wave2textarray()=='[0.+1.j 0.+2.j] x:[0 1]'
    xs = np.linspace(-10,10,6)
    u = Wave2D(xs=xs,ys=xs) + 1j
    # assert isinstance(u[0][0],np.complex)
    assert isinstance(u[0][0],complex)
    assert u[0][0]==u[0,0]==1j
def histtest(plot=0):
    wx = Wave(np.linspace(-10,10,1001))
    # print(wx.gaussian(100,10)[:3],wx.random()[:3],wx.choice([-1,2,99])[:3])
    wx.gaussian(100,10)[:3]
    wx.random()[:3]
    wx.choice([-1,2,99])[:3]
    wy = Wave(wx).gaussian(y0=5,dy=0.1)
    if plot: wy.histogram().plot()
def arraytest(verbose=0):
    def typetest(a):
        fs = [
            (Array, lambda a: np.sin(1-a**2)),
            (Array, lambda a: np.cumsum(a,axis=1)),
            (Array, lambda a: a[:1]),
            (np.float64, lambda a: np.max(a)),
            (np.float64, lambda a: a.min()),
            (np.ndarray, lambda a: np.sum(a,axis=1)),
            (np.ndarray, lambda a: a.max(axis=1)),
            (np.ndarray, lambda a: np.min(a,axis=1)),
            ]
        for t,f in fs:
            if verbose: print(f(a),type(f(a)))
            assert type(f(a))==t or type(f(a))==np.complex128
    a = Array([[0,0],[1,1],[2,2]])
    typetest(a)
    typetest(Array([[1]]))
    typetest(Array([[1+1j,1-1j]]))
    assert (Array([5,5,5])*Array(6)*Array([[1],[1]])).shape==(2,3)
    assert 6==Array(6)
    assert ()==Array(6).shape
    if verbose: print('repr:',[a],'str:',a)
def vtest(verbose=0):
    a = V(1,2)*2+V(0,1)
    assert np.allclose( a, np.array([2,5]) )
    a[0:2] = [0,1]
    assert np.allclose( a, np.array([0,1]) )
    a.x,a.y = 3,4
    assert np.allclose( a, np.array([3,4]) )
    assert np.all(V(1,4)==V([1,4])) 
    assert np.all(V(1,4)==V([[[1,4]]])) 
    assert np.all(V(np.array([2,7]))==V([2,7])) 
    v = V(3,4)
    assert v.mag()==5 
    assert np.all((v+(3,2))/6==[1,1])
    assert np.allclose(V(np.sqrt(2),np.sqrt(2)).rotate(np.pi*9/4),[0,2])
    if verbose: print(v,[v])
def vstest(verbose=0,plot=0):
    vs = Vs([[1,2],[2,3],[3,5]])
    assert np.all( vs.copy()==[[1,2],[2,3],[3,5]] )
    assert np.all( vs-[[1,2],[2,3],[3,5]]==[[0,0],[0,0],[0,0]] )
    assert np.all( vs-[1,2]==[[0,0],[1,1],[2,3]] )
    assert np.all( vs-[[1,2]]==[[0,0],[1,1],[2,3]] )
    assert np.all( vs[:2]==[[1,2],[2,3]])
    assert vs.wave()==Wave([2.,3.,5.],[1.,2.,3.])
    # print( str(vs.list()) )
    # print( '[V:[1. 2.], V:[2. 3.], V:[3. 5.]]' )
    # assert str(vs.list())=='[V:[1. 2.], V:[2. 3.], V:[3. 5.]]'
    assert str(vs.list())=='[[1.0, 2.0], [2.0, 3.0], [3.0, 5.0]]'
    assert all([type(v)==V for v in vs])
    assert type(vs[0])==V
    assert type(vs[:1])==Vs
    assert all(vs.x==[1,2,3]) and type(vs.x)==np.ndarray
    assert all(vs.y==[2,3,5]) and type(vs.y)==np.ndarray
    vvs = Vs([3,3],size=4); vvs.x,vvs.y = [0,0,1,1],[0,1,1,0]; assert np.all(vvs==[(0,0),(0,1),(1,1),(1,0)])
    assert np.all(Vs(x=[0,2],y=[3,7])==[[0,3],[2,7]])
    assert np.all(Vs([3,3],size=3).concat(Vs(size=3)).concat([[1,1]])==Vs([[3,3]]*3+[[0,0]]*3+[[1,1]])) 
    assert np.all(Vs([3,3],size=2).concat(Vs(size=2),[[1,1]])==Vs([[3,3]]*2+[[0,0]]*2+[[1,1]])) 
    assert np.all((Vs(y=[2,2,2])==[[0,2],[1,2],[2,2]]))
    assert np.all(vvs.segmentlengths()==np.array([[0],[1],[1],[1]])) and 3==vvs.length()
    assert np.allclose((2*vvs).thickline(1),[[0.5,0],[0.5,1.5],[1.5,1.5],[1.5,0],[2.5,0],[2.5,2.5],[-0.5,2.5],[-0.5,0],[0.5,0]])
    if verbose: print(vs,[vs])
    vr = vs.thickribbons(widths=[15,1,3,1,2,1,3,1,2])
    vs.thickline(3).signedarea()
    assert Vs(Vs([[0,0],[0,1]],width=8)).width==8
    def concatenationtest():
        c = [[0,0]] ++ Vs([[0,1],[0,2]]) ++ Vs([[0,3],[0,4]])
        if verbose: print(c,type(c))
        assert np.all(c==[[0,0],[0,1],[0,2],[0,3],[0,4]]) and isinstance(c,Vs)
        # assert Vs([[0,0]]++Vs([[0,0],[0,1]],width=8)).width==8
    concatenationtest()
    def eqtest():
        assert [[1,2],[2,3],[3,5]]==vs and ((1,2),(2,3),(3,5))==vs and np.array([[1,2],[2,3],[3,5]])==vs
        assert not [[1,2],[2,3],[3,55]]==vs and not ((1,2),(2,3),(3,55))==vs and not np.array([[1,2],[2,3],[3,55]])==vs
        assert vs==[[1,2],[2,3],[3,5]] and vs==((1,2),(2,3),(3,5)) and vs==np.array([[1,2],[2,3],[3,5]])
        assert not vs==[[1,2],[2,3],[3,55]] and not vs==((1,2),(2,3),(3,55)) and not vs==np.array([[1,2],[2,3],[3,55]])
        assert not [[1,],[3,]]==vs and not ((1,),(3,))==vs and not np.array([[1,],[3,]])==vs
    eqtest()
    if plot:
        vs.thickline(3).plot(x='x (µm)',y='y (µm)',grid=1)
        vvs.rotate(-pi/4,p0=[10,-10]).thickline(0.5,cw=1,widthcorrection=1).plot()
        vs.thickline(0.5,cw=1,normal=(1,0),widthcorrection=1).plot()
        Vs.plots(*vr,m=1) # Wave.plots(*[us.wave() for us in vr],m=1,aspect=1)
    vs = Vs([[0,-1000],[0,-200]])
    assert np.all(vs[0.5]==[0,-600])
    assert np.all(vs.upsample(num=5)==[[0,-1000],[0,-840],[0,-680],[0,-520],[0,-360],[0,-200]])
    def bsplinetest(plot):
        vs = [Vs([[0,0],[0,1],[2-x,3],[3,3]]) for x in [0,1,2,3,4]] + [Vs([[0,0],[0,1+x],[2,3],[3,3]]) for x in [0,1,2,3,4]]
        bs = [v.bspline(degree=3) for v in vs]
        if plot: Vs.plots(*bs,*vs,l=len(bs)//2*'0'+len(bs)//2*'1'+'"'*len(vs),m='"'*len(bs)+'o'*len(vs),groupsize=len(bs)//2)
        c = Vs([[50,25],[59,12],[50,10],[57,2],[40,4],[40,14]])
        b = c.bspline(n=100,degree=2,closed=1) # print(len(b),b[:2],b[-2:])
        if plot: Vs.plots(b,c++c[:1],m='"o')
    bsplinetest(plot)
def arctest(plot=0):
    Arc = Vs.arc
    ws = [Arc(5,0,pi*3/2,1,V(-10,-10)),Arc(5,pi/2,pi,1,V(0,-10)),Arc(5,0,pi/2,1,V(10,-10))] # ccw arcs, φ0<φ1
    wws = [Arc(5,0,-pi*3/2,2,V(-10,10)),Arc(5,-pi/2,-pi,2,V(0,10)),Arc(5,0,-pi/2,2,V(10,10))] # cw arcs, φ1<φ0
    p = Arc(5,0,2*pi) # full circle
    if plot: Vs.plots(*(ws+wws+[p]),m=1,ms=2)
    p += [20,0]; p *= 2; p += 5
    p2 = p.thickline(2)
    p4 = p.thickline(4)
    if plot: Vs.plots(p2,p4,m=1)
    assert type(ws[0])==Vs
def circletest(plot=0):
    ws = [Vs.circle(5,V(-10,-10)),Vs.circle(5,V(0,-10)),Vs.circle(5,V(10,-10))]
    if plot: Vs.plots(*(ws),m=1,ms=2)
def backendtest(backend=None): # https://stackoverflow.com/a/49348512/12322780
    if backend: matplotlib.use(backend)
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    x, y = np.meshgrid(np.linspace(-1,1,31),np.linspace(-1,1,31))
    fig, ax = plt.subplots(figsize=(2.8,2.8))
    fig.subplots_adjust(.02,.02,.98,.9)
    # ax.set_title('scatter')
    sc = ax.scatter(x,y,s=2)
    ax.axis('off')
    ax.set_xlim(-1.2,1.2)
    ax.set_ylim(-1.2,1.2)
    ax.set_aspect('equal')
    def generator():
        X=np.copy(x); Y=np.copy(y)
        for i in range(100):
            X += 0.0015*np.cos(x+3*y)
            Y += 0.0015*np.sin(6*x-4*y)
            yield (X,Y)
    def animate(i): 
        x,y = i
        sc.set_offsets(np.c_[x.flatten(),y.flatten()])
    ani = FuncAnimation(fig, animate, frames=generator, interval=50, repeat=False)
    ani.save('skewedgrid_scatter.gif', writer='imagemagick')
    plt.show()
def linelabeltest(showplot):
    wx = np.linspace(0,10,101)
    ws = [Wave(sin(wx),wx,'sin x\n'),Wave(cos(wx),wx,'cos x\n')]
    wws = [Wave(9*sin(sqrt(wx)),wx,'sin √x'),Wave(9*cos(sqrt(wx)),wx,'cos √x')]
    import plot
    if showplot: plot.plot(waves=ws,x='x',y='y',rightlabel='z',rightwaves=wws,li=3)
def markertest(showplot):
    x = np.linspace(11,15,5)
    ws = [Wave(n*x,x,n) for n in range(20,31)]
    import plot
    if showplot: plt = plot.plot(waves=ws,x='X',y='Y',m='d+*xt"D"♣♦♥♠',l='01234',c='01234',mf='0011',li='234'+9*'1',ms=9,loglog=1,groupsize=4,fewerticks=1,fewerticklabels=1,show=showplot) # ♦♥ won't fill
    # plot.plot(waves=ws,m='d"o"D"♣♦♥♠',l='0123401234',c='0123456789',markerfacecolor='none',ms=9,groupsize=2,loglog=1)
def bartest(showplot):
    from plot import plot
    ws = [Wave([i//2,1,2,3-i//2],np.array([0,2,4,6])+i/6,f'#{i}')+i for i in range(6)]
    if showplot: plot(waves=ws[::-1],bar=1)
def plotordertest(showplot):
    ws = [Wave([i//2,1,2,3-i//2],name=f'{i}')+i for i in range(6)]
    from plot import plot
    if showplot: plot(waves=ws,m='o"s',groupsize=3)
    ws = [Wave([i//2,1,2,3-i//2])+i for i in range(6)]
    # plot(waves=ws,m='o"s',groupsize=3,legendtext='Title')
    if showplot: plot(waves=ws,m='o"s',groupsize=3,swap=1,lw=4,legendtext='Title')
def plot2Dtest(showplot):
    xs,ys = np.linspace(-10,10,21),np.linspace(-10,10,21)
    ww = Wave2D(xs=xs,ys=ys)
    ww = sin(abs(ww.xx)+abs(ww.yy))
    from plot import plot
    if showplot:
        plot(colormesh=ww,waves=[Wave(abs(ys),xs,'up'),Wave(-abs(ys),xs,'dn')],c=['k','r'])
        plot(image=ww,waves=[Wave(abs(ys),xs,'up'),Wave(-abs(ys),xs,'dn')],c='k',lw=5)
        plot(contourf=ww,contour=ww,levels=2)
def errorbartest(showplot):
    from plot import plot
    ws = [Wave([i//2,1,2,3-i//2],np.array([0,2,4,6]),f'#{i}')+i for i in range(6)]
    if showplot: plot(waves=ws,errorbars=[[(n+2)/5 for n,_ in enumerate(w)] for w in ws],m=1)
def rotationtest():
    v = V(1,2,3,4)
    assert np.allclose([-1.,-2.,-3.,-4.],v.rotation(pi,0,1).rotation(pi,1,2).rotation(pi,2,3).rotation(pi,0,2).rotation(pi,1,3).rotation(pi,0,3))
    vv = V(1,2,3,4,5)
    assert (list(vv.rotation(+sqrt(77),0,4))==
            list(vv.rotation(-sqrt(77),4,0)))
def test3d():
    assert np.allclose(V3(np.sqrt(3)/2,0.5,0), V3(1,0,0).rotate(np.pi/6, V3(0,0,1), V3(0,0,-1)))
    assert np.allclose(V3(0,0,1), V3(1,0,0).rotate(np.pi*4/3, V3(2,2,2), V3(1,1,1)))
    assert np.allclose(V3(0,1,10), V3(1,0,10).rotate(np.pi*2/3, V3(2,2,12), V3(1,1,11)))
    assert np.allclose(V3(-1,1,1), V3(1,1,1).rotate(np.pi/2, V3(0,0,1), V3(0,0,0)))
    assert np.allclose(V3(1,19,-1), V3(1,1,1).rotate(np.pi, V3(1,10,0), V3(-1,10,0)))
    assert np.allclose(V3(2.74911638,4.77180932,1.91629719), V3(3,5,0).rotate(1.2, V3(4,4,1), V3(0,0,0)))
    assert np.allclose(V3(0,0,0.5).sphericalinvert(2), V3(0,0,8))
    assert np.allclose(V3(0,0,1.5).sphericalinvert(1,(0,0,1)), V3(0,0,3))
    assert np.allclose( ([0,0,0], [0,0,1]), V3.lineintersection(V3(0,2,0),V3(0,1,0),V3(1,0,1),V3(2,0,1)) )
    assert np.allclose( ([0.5,0.5,0], [0.5,0.5,1]), V3.lineintersection(V3(0,0,0),V3(1,1,0),V3(1,0,1),V3(0,1,1)) )
def voronoitest(showplot):
    p = Vs(Vs.polygon(6,23).sep()) ++ Vs(Vs.polygon(3.5,17).sep()) ++ Vs(Vs.polygon(3,27).sep())+V(0,0.5) ++ -Vs(Vs.polygon(2,15).sep())+V(0,-0.5) ++ Vs(Vs.polygon(1,7).sep())
    ts,cs = Vs.voronoi(p.unsep())
    xlim = ylim = (-1,1)
    ps = Vs.random(99,xlim,ylim)
    cells,centers = Vs.voronoi(ps)
    if showplot:
        Vs.plots(ts,cs,m=' o',l='0 ')
        cells.plot(xlim=xlim,ylim=ylim)
    a,b = Vs.polygon(9,2,end=0),Vs.polygon(8,2,end=0)
    u,v = a.concat(b,sep=1),Vs.concats(a,b,sep=1)
    assert str(u)==str(v)=='[[-0.000e+00  9.000e+00] [-1.102e-15 -9.000e+00] [       nan        nan] [-0.000e+00  8.000e+00] [-9.797e-16 -8.000e+00]]'
    assert V(0,-1).reflection(V(1,0),V(0,1))==V(2,1)
    c = Vs.polygon(1,5)
    # Vs.plots(c,c.reflection(V(1,0),V(0,1)))
    if showplot:
        Vs.plots(c,c.reflection(V(0.5,0),V(0.5,1)))
def shapelytest(showplot):
    a = Vs([(0,5),(5,0),(0,-5),(-5,0),(0,5)],name='a')
    b, c, d = a+V(0,2), a+V(0,10), a+V(0,11)
    assert str(a.shapelypolygon())=='POLYGON ((0 5, 5 0, 0 -5, -5 0, 0 5))' # auto-closed
    assert a.intersects(b) and a.intersects(c) and not a.intersects(d)
    # assert str((a.intersection(b)))=='[[ 0.  5.] [ 4.  1.] [ 0. -3.] [-4.  1.] [ 0.  5.]]'
    assert str((a.intersection(b)))=='[[ 4.  1.] [ 0. -3.] [-4.  1.] [ 0.  5.] [ 4.  1.]]'
    buff = a.buffer(2)
    if showplot: Vs.plots(a,b.rename('b'),a.intersection(b).rename('a∩b'),buff)
def concattest():
    v0,v1 = V(0,1),V(2,3)
    vs0 = vs1 = Vs([v0,v1])
    vv = Vs.concats(vs0,vs1)
    assert np.allclose(np.array(vv),np.array([[0,1],[2,3],[0,1],[2,3]]))
    vvv = Vs.concats(vs0,vs1,sep=True)
    assert np.allclose(np.array(vvv),np.array([[0,1],[2,3],[nan,nan],[0,1],[2,3]]),equal_nan=True)
def filltest(showplot):
    x,y = [0,0,1,1],[0,1,1,0]
    w = Wave(y,x)
    Wave.plots([1.5*w+n for n in range(0,20,2)],fill=1,lw=3,l='0123',seed=30,show=showplot)
    x = np.linspace(0,1,101)
    w0 = Wave(0.25*np.sin(2*np.pi*x)**2,x,'sin')
    w1 = Wave(0.5*(1+np.cos(2*np.pi*x)**2),x,'cos')
    Wave.plots(w1,w0,fillbetween=1,seed=30,show=showplot)
def chaintest():
    vvs = Vs(x=[0,1,2,np.nan,0,1,2,np.nan,0,1,2,np.nan],y=[0,1,2,np.nan,0,1,2,np.nan,0,1,2,np.nan])
    vs = Vs(x=[0,1,2,np.nan,0,1,2,np.nan,0,1,2],y=[0,1,2,np.nan,0,1,2,np.nan,0,1,2])
    a,aa = vs.chain2list(),vvs.chain2list() # print(aa) print(a)
def radialinterpolatetest(showplot):
    s = Polar.square() # s.plot(show=1)
    assert np.isclose(s.radialinterpolate(θ=np.pi/4),1/np.sqrt(2)) # print('pi/4')
    assert np.isclose(s.radialinterpolate(θ=np.pi/2),1) # print('pi/2')
    ss = Polar.triangle()
    s.radialsum(ss).plot(show=showplot)
    ss.radialsum(s).plot(show=showplot)
    s.radialaverage(ss,0.1).plot(show=showplot)
    s.radialaverage(ss,0.9).plot(show=showplot)
def randomshapetest(showplot):
    # print([endlessshuffle(t,[0,1,2])[0] for t in range(40)])
    # print([9]+[endlessshuffle(t,[0,1,2])[1] for t in range(40)])
    # for t in [0,0.25,0.5,0.75,1,1.5,2]:
    # for t in [0,1,2]:
    for t in [0,1,2,3,4,5,6]:
        Polar.random(t).plot(seed=0,show=showplot)
        # Polar.random(t/60).plot(seed=0,show=showplot)
def animate(showplot):
    def f(i):
        x = np.linspace(0, 4, 1000)
        y = np.sin(2 * np.pi * (x - 0.01 * i))
        return x,y
    import plot
    if showplot:
        plot.animate(f,100,delay=10)
def animateshapetest(showplot):
    ts = np.linspace(0,10,200)
    ts = list(ts[:]) + list(ts[::-1])
    def f(i):
        s = Polar.random(ts[i],ease=5).polar2wave()
        # s = Polar.random(ts[i],ease=0).polar2wave()
        return s.x,s.y
    import plot
    if showplot:
        plot.animate(f,len(ts),delay=50,lw=2,xlim=(-2,2),ylim=(-2,2),aspect=1,show=1)
    # plot.animate(f,len(ts),delay=50,lw=2,xlim=(-2,2),ylim=(-2,2),aspect=1,show=showplot,save='animate')
def polartest(showplot):
    p = Polar.polygon()
    star = Polar.star().upsample()
    t = Polar.triangle()
    s = Polar.square()
    c = Polar.circle()
    sc = s.radialsum(c)
    def tri(x): return 1-2*abs(x-0.5)
    h = star*tri(star.x)
    sc = s.radialsum(c)
    assert type(sc)==Polar
    Polar.plots(c,p,t,star,0.2*sc,0.5*h,show=showplot)
def flowertest(showplot):
    f0 = Polar.flower()
    f1 = 0.9*Polar.flower(m=9)
    f2 = 0.5*Polar.flower(m=2,saw=1)
    Polar.plots(f0,f1,f2,show=showplot)
    f2.polar2vs().gridarray(3,3).plot(show=showplot)
    f1.polar2vs().radialarray(25,1).plot(show=showplot)
def plotpropertytest(showplot):
    w = Wave([2,0,1])
    ww = Wave([1,0,2])
    w.m = 'D'
    ww.m = 's'
    # print(hasattr(w,'m'),w.m)
    assert hasattr(w,'m')
    assert w.m=='D'
    Wave.plots(w,ww.setplot(m='x',l='1'),m='o',show=showplot)
def trapezoidintegratetest(showplot):
    def test(ys,xs):
        f = trapezoidintegrate(ys=ys,xs=xs)
        # print([f(x) for x in xs])
        w = Wave([f(x) for x in xs],xs).setplot(m='o',l=' ')
        xx = np.linspace(xs[0]-1,xs[-1]+1,101)
        ww = Wave([f(x) for x in xx],xx)
        ff = trapezoidintegrate(ys=ys,xs=xs,invfunc=1)
        aa = np.linspace(ww[0],ww[-1],101)
        www = Wave(aa,[ff(a) for a in aa]).setplot(l=1)
        Wave.plots(w,ww,www,show=showplot)
    test(ys=[0,2,0],xs=[0,1,2])
    test(ys=[0,2,3,0],xs=[0,1,3,6])
def pdfsampletest(verbose=False):
    a0 = [discretepdfsample(xs=[0,1,2,3],ys=[0.1,0.2,0.3,0.4]) for _ in range(5)]
    xs = np.linspace(-pi,pi,17)
    ys = sin(xs)**2
    a1 = [discretepdfsample(xs,ys) for _ in range(5)]
    if verbose:
        print(a0)
        print(a1)
def transposetest(showplot):
    def plotit(w):
        w.plot(legendtext='original',show=showplot)
        w.transpose().plot(legendtext='transpose',show=showplot)
        w.transpose(mirrorx=1).plot(legendtext='mirrorx\n+90° rotation\nccw',show=showplot)
        w.transpose(mirrory=1).plot(legendtext='mirrory\n-90° rotation\ncw',show=showplot)
    xs,ys = np.linspace(-2,4,61),np.linspace(-1,2,31)
    plotit(Wave2D(xs[:,None]+ys[None,:],xs,ys).addcircle(0,0,2,10))
    xs,ys = np.linspace(-10,10,201),np.linspace(-10,10,201)
    ww = Wave2D(xs=xs,ys=ys)
    plotit(0.1*sin(abs(2*ww.xx)+abs(ww.yy)).addrectangle(2,5,10,5,1.9))
def interpolate1dtest(verbose=False):
    xs = [0,2,4]
    ys = [2,1,10]
    assert np.isnan(interpolate1d(-2,xs,ys,extrapolate=None))
    assert 2==interpolate1d(-2,xs,ys,extrapolate='const')
    assert 3==interpolate1d(-2,xs,ys,extrapolate='lin')
    assert np.isnan(interpolate1d(6,xs,ys,extrapolate=None))
    assert  10==interpolate1d(6,xs,ys,extrapolate='const')
    assert  19==interpolate1d(6,xs,ys,extrapolate='lin')
    assert np.isclose(100,interpolate1d(6,xs,ys,extrapolate='log'))
    for x in [3,[1,3],np.array([1,3]),-1,[-1,5]]:
        if verbose: print(x)
        y = [interpolate1d(x,xs,ys,extrapolate=e) for e in [None,'const','lin','log']]
        if verbose: print('   ',list(y))
def piecewisetest(showplot):
    # showplot = 1
    a,b = Piecewise([0,2,0],[0,2,4]),Piecewise([0,2,2,0],[0,1,3,5])
    Piecewise.plots(a,b,a+b,m='o',show=showplot); # print(a+b)
    a,b = Piecewise([0,2,0],[0,2,4]),Piecewise([0,0,2,2,0,0],[0,1,1,3,3,5])
    Piecewise.plots(a,b,a+b,m='o',show=showplot); # print(a+b)
    a,b = Piecewise([0,2,0],[0,2,4]),Piecewise([0,0,nan,2,2,nan,0,0],[0,1,1,1,3,3,3,5])
    Piecewise.plots(a,b,a+b,m='o',show=showplot); # print(a+b)
    a = Piecewise([1,2,1],[0,2,4])
    assert list(a+1)==list(1+a)==[2,3,2], f"{list(a+1)} {list(1+a)}"
    assert list(a+1)==list(1+a)==[2,3,2], f"{list(a+1)} {list(1+a)}"
    assert list(a-2)==list(-2+a)==[-1,0,-1], f"{list(a-2)} {list(-2+a)}"
    assert list(a*2)==list(2*a)==[2,4,2]
    assert list((2*a)/2)==list((a/2)*2)==[1,2,1]
def cosfittest(showplot):
    w = Wave([0,1,1,2,1,1,0,1,1,2],[10,20,30,40,50,60,70,80,90,100])
    fix = (True,True,False,False)
    w0 = w.cosfit(coef=0,upsample=20,fix=fix,guess=(0.5,1.5,50,1))
    w1 = w.cosfit(coef=0,upsample=20,fix=fix,guess=(0.5,1.5,20,1))
    Wave.plots(w,w0,w1,m='o  ',l=' 00',show=showplot)
    def cosfitfunc2(x,a,y0,period):
        return y0+a*np.cos(2*np.pi*x/period)
    assert np.allclose(coscurvefit(xs=np.array([1,2,2,1]),ys=np.array([1,2,3,4]),guess=(10,1,12,3),fix=(0,1,0,1)),(1.5000000059128809, 1, 1201.7929168495443, 3.0))
    assert np.allclose(coscurvefit(xs=np.array([1,2,3,4]),ys=np.array([1,2,2,1]),guess=(10,1,12,3),fix='0101'),(0.8706569411379551, 1, 7.624585705922912, 3.0))
    assert np.allclose(coscurvefit(xs=np.array([1,2,3,4]),ys=np.array([1,2,2,1]),guess=(0.5,1,1.5,0),fix=(0,0,0,1)),(0.8944271909999149, 1.7236067977499785, 1.6666666666666665, 0.0))
    assert np.allclose(curvefit(np.array([1,2,3,4]),np.array([1,2,2,1]),cosfitfunc2,guess=(0.5,1,1.5),fix=(0,0,0)),(0.8944271909999149, 1.7236067977499785, 1.6666666666666665))
    # w0 = Wave([1,2,2,1,1,1,2,2,1]).setplot(m='o',l='')
    # w = w0.cosfit(coef=0,upsample=100)
    # Wave.plots(w0,w,show=showplot)
def correlatetest(showplot):
    xs = np.linspace(-10,10,21)
    u = Wave((5-abs(xs)).clip(0,4),xs)
    v = 3*sqrt(u/2)
    u0,v0,u1 = u[:-2],v[4:-4],u[4:-4]
    w0,w1 = u0.correlate(v0,normalize=False),u0.correlate(u1,normalize=False)
    n0,n1,n2 = u0.correlate(v0),u0.correlate(u1),u0.correlate(-u1)
    assert w0(0)==w0.max() and w1(0)==w1.max()
    if showplot:
        Wave.plots(v0.rename('v'),u0.rename('u'),l='03',m=1,lw=1,ms=3,scale=(1,0.5),seed=2)
        Wave.plots(w0.rename('cor(v,u)'),w1.rename('cor(u,u)'),m=1,lw=1,ms=3,scale=(1,1),seed=2)
        Wave.plots(n0.rename('cor(v,u)'),n1.rename('cor(u,u)'),n2.rename('cor(u,-u)'),m=1,lw=1,ms=3,grid=1,fewerticks=1,scale=(1,1),seed=2)
def multicrosscorrelationtest(showplot):
    ps = np.arange(101.)
    def f(x): return 1+np.clip(1-np.abs(x/3),0,1)
    ws = [Wave(f(ps-50-x0),ps) for x0 in range(-3,3+1,2)]
    if showplot: Wave.plots(*ws)
    ws = [w.y for w in ws]
    a0,dxs = multicrosscorrelation(ws,[0]); #print('\n','a0',a0,'dxs',dxs)
    a,dxs = multicrosscorrelation(ws,Δi=0.5); #print('\n','a',a,'dxs',dxs)
    if showplot: Wave(a,dxs).plot()
    # a,dxs = multicrosscorrelation(ws,[Wave(a,dxs).maxloc()],Δi=0.5,plot=1)
def ffttest(showplot,L=10,d=200,num=512):
    def ftpolarplot():
        def ftpolar(x0):
            xs = np.linspace(x0,x0+d,101)
            aa = Wave((xs<L),xs).ft(np.linspace(-2,2,2001))
            return Wave(aa.imag,aa.real)
        Wave.plots(ftpolar(0),ftpolar(-5),ftpolar(-10),lw=1,x='Re',y='Im')
    if showplot:
        ftpolarplot()
    def plotffttest(x0,showplot):
        xs = np.linspace(x0,x0+d,num)
        ys = (0<xs)&(xs<L)
        dx = xs[1]-xs[0]
        df = 1/dx
        f0 = np.linspace(-2,2,num)
        a0 = Wave(ys,xs).ft(f0)
        w0 = Wave(a0,f0) # w0.plot(m=1,lw=1,ms=1)
        f1 = np.fft.fftshift(np.fft.fftfreq(ys.size)) * df
        a1 = np.fft.fftshift(np.fft.fft(ys)) * dx
        a1 *= exp(-1j*2*np.pi*x0*f1)
        w1 = Wave(a1,f1) # w1.plot(m=1,lw=1,ms=1)
        w2 = Wave(ys,xs).fft()
        Wave.plots(w0.real(),w0.imag(),w1.real(),w1.imag(),w2.real(),w2.imag(),
            c='001122',l='0303  ',m=1,lw=1,ms=1,grid=1,seed=5,xlim=(-0.2,1),legendtext=f'x0={x0}',show=showplot)
    # plotffttest(0,showplot)
    plotffttest(-7,showplot)
    xs = np.linspace(0,0+d,num)
    w = Wave((xs<L),xs)
    @timeit
    def slow(n):
        for _ in range(n): w.ft(np.linspace(-2,2,num))
    @timeit
    def fast(n): # 5800x faster
        for _ in range(n): w.fft()
    # slow(10**1)
    # fast(10**4)
    @timeit
    def memtest(k):
        xs = np.linspace(0,d,2**k)
        w = Wave((xs<L),xs)
        w.fft()
    # memtest(20) # 2**20 = 0.35s
    # memtest(26) # 2**26 = 25s
    # memtest(28) # 2**28 = 2.7e8 = 94s
    # memtest(30) # 2**30 = 1.1e9 = 742s
def loadfiletest():
    def converttest():
        input_string = "\napple, 123, grape,\nbanana, kiwi, 4.56E+02\n \t , 78g, 000.789\n"
        lines = input_string.rstrip().split('\n')
        def convert(line):
            def string2symbol(s):
                v = s.strip()
                if not v:
                    return ' '
                try:
                    float(v)
                    return '0'
                except ValueError:
                    return 'x'
            return ','.join([string2symbol(s) for s in line.split(',')])
        output_string = '\n'.join([convert(line) for line in lines])
        print('***'+'\n'+input_string+'\n'+'***'+'\n'+output_string+'\n'+'***')
    # converttest()
    files = [
        'waveplate_scan_results_HV.txt',
        'NOV20-GAH2025-05-1.1-1.dat',
        ]
    for i,file in enumerate(files):
        print(fileformat(file,crop=1))
        names,arrays = loadfile(file,quiet=0,debug=0)
def hysteresistests():
    file = 'MAY22F_BCT2115-B11_1.4_1.8_V_DAY1.csv'
    trig,p,f,b = loadfile(file,quiet=1,waves=1)
    trig,p,f,b = trig[:-1],p[:-1],f[:-1],b[:-1]
    p.hysteresiscrossings(fraction=0.8,initial=True,debug=1)
    p.hysteresiscrossings(fraction=0.2,initial=True,debug=1)
    p.hysteresispeaks(fraction=0.9,ends=True,debug=1)
    p.recurringslopes('rise',fraction=0.9,ends=True,debug=1)
    p.recurringslopes('fall',fraction=0.9,ends=True,debug=1)
    (f/p).recurringslopes('rise',fraction=0.75,ends=True,debug=1)
    (f/p).recurringslopes('fall',fraction=0.75,ends=True,debug=1)
    import pandas
    w0 = Wave([2,4,3],[0,1,2],'x:x:x:x:x')
    w0.c = 'k'
    w0.setplot(l='6',m='o',ms=6,lw=0.5,li=6,mf=0)
    w0.plot(show=showplot)
    w0.to_pickle('tmp.dat') # print(w0,w0.c,w0.m)
    w = pandas.read_pickle('tmp.dat') # print(w,w.c,w.m) # print(' '.join(w._metadata))
    d = w.getplot() # print(d['c'],d['l'],d['li'],d['m'],d) # for a in 'name c l lw m ms f i'.split(): print(a,getattr(w,a))
def markeralphatest(showplot):
    assert 255==int(256-1e-9)
    def matplotalphamarkers():
        import matplotlib.pyplot as plt
        plt.rcParams['keymap.quit'] = ['ctrl+w','cmd+w','q','escape']
        plt.plot([0,0,2],[2,0,0],label='q',marker='o',color='g',markersize=20,mew=5,linewidth=5,clip_on=False,zorder=3,markerfacecolor='white')
        for x in (0.9,1.0,1.1): plt.plot([x    ],[1.5],marker='D',markersize=50,mew=5,color=(0,0,0,1),markerfacecolor=(0,1,0,0.5),mec=(0,0,0,1))
        for x in (0.9,1.0,1.1): plt.plot([x    ],[1.0],marker='D',markersize=50,mew=5,color='b',markerfacecolor=(0,0,1,0.5))
        for x in (0.9,1.0,1.1): plt.plot([x    ],[0.5],marker='D',markersize=50,mew=5,color='k',markerfacecolor=(1,0,0,0.5))
        for x in (0.9,1.0,1.1): plt.plot([x-0.7],[1.5],marker='D',markersize=50,mew=5,color='#FF0000',markerfacecolor='#00ff0077',mec='#000000FF')
        for x in (0.9,1.0,1.1): plt.plot([x-0.7],[1.0],marker='D',markersize=50,mew=5,color='#0000ff',markerfacecolor='#0000ff77')
        for x in (0.9,1.0,1.1): plt.plot([x-0.7],[0.5],marker='D',markersize=50,mew=5,color='k',markerfacecolor='#ff000077')
        plt.xlim(-0.05,2.05); plt.ylim(-0.05,2.05)
        plt.show()
def wavepickletest(showplot):
    # matplotalphamarkers()
    w = Wave([0,0,3],[2,0,0],m='o',c=0,ms=20,lw=5)
    v0 = Wave([0.5,2.5],[0.7,0.7],'0.1',m='D',ms=120,lw=5,mf=0.1)
    v1 = Wave([0.5,2.5],[1.0,1.0],'0.5',m='D',ms=120,lw=5,mf=0.5)
    v2 = Wave([0.5,2.5],[1.3,1.3],'0.9',m='D',ms=120,lw=5,mf=0.9)
    # Wave.plots(w,v0,v1,v2,show=showplot)
    wa = Wave([0,0,3],[2,0,0],m='o',c=0,ms=20,lw=5)
    va = Wave([1.5]*9,wrange(0.6,1.4,0.1),'0.1',m='D',ms=60,lw=0.2,mf=0.6,l=' ')
    Wave.plots(wa,va,show=showplot)
def joblibWave2Dtest():
    import joblib
    memory = joblib.Memory('j:/backup', verbose=1) # use as @memory.cache
    ww = Wave2D(xs=[0,1,2],ys=[10,20]).xx**2
    ii = Wave2D(xs=[0,1,2],ys=[10,20])
    @memory.cache
    def f(ww,ii,xoffset=0,method='linear'):
        def xys():
            for i,j in np.ndindex(ii.shape):
                yield ii.xx[i,j],ii.yy[i,j]
        conv = sum([ ii(x,y) * ww(x+xoffset,y,method=method) for x,y in xys() ])
        return ww.xx.sum()+ww.yy.sum()+ww.sum() + ii.xx.sum()+ii.yy.sum()+ii.sum() + conv
    # print(f(ww,ii))
    @memory.cache
    def overlap(ii,efield,xoffset=0,method='linear'):
        def xys():
            for i,j in np.ndindex(ii.shape):
                yield ii.xx[i,j],ii.yy[i,j]
        return sum([ ii(x,y) * efield(x+xoffset,y,method=method) for x,y in xys() ])
    # print(overlap(ii,ww,xoffset=0))
def standarderrorofthemeantest(N=10**4):
    np.random.seed(0)
    w = Wave(np.random.normal(size=N))
    avg = w.mean()
    err0 = w.sdev(ddof=1)/np.sqrt(len(w)); #print(f"{avg: .4f}±{err0:.4f}")
    err1 = w.standarderrorofthemean(); #print(f"{avg: .4f}±{err1:.4f}")
    def errmean(chunks,k=100,gethist=False):
        ws = [Wave(np.random.normal(size=N)) for _ in range(k)]
        es = Wave([w.standarderrorofthemean(chunks) for w in ws])
        # if plot: es.histogram(bins=wrange(-0.5*es.max()/20,es.max()+1.5*es.max()/20,es.max()/20)).plot()
        if gethist: return es.histogram(dx=es.max()/20)
        return sum(es)/k
    # for i in range(2,10): errN = errmean(i); print(f"{0: .4f}±{errN:.4f} {i}chunks")
    assert f"#{avg: .4f}±{err0:.4f}#"=="#-0.0184±0.0099#"
    assert f"#{avg: .4f}±{err1:.4f}#"=="#-0.0184±0.0099#"
    # Wave.plots(*[errmean(i,k=1000,gethist=1).setplot(c=i) for i in (2,2,2,3,3,3,4,4,4)],m='o')
def stepclimbtest(verbose=False):
    xs = ys = zs = np.linspace(-5,5,11)
    if verbose:
        print('xs',xs); print('ys',ys); print('zs',zs)
    def f(x,y,z,x0=-1.4,y0=0.6,z0=2.4):
        return exp(-(x-x0)**2-(y-y0)**2-(z-z0)**2)
    stepclimb(f,xs,ys,zs,verbose=verbose) # [4, 6, 7] (-1.0, 1.0, 2.0) 0.618783391806141
def quadmaxtest(verbose=False):
    xs,ys = [0,1,2,3],[2,1,2,5]
    a,b,c = quadfit(xs,ys)
    if verbose: print('a,b,c',a,b,c)
    a,b,c = quadfit(xs,ys,guess=(1,1,1))
    if verbose: print('a,b,c',a,b,c)
    a,b,c = quadfit(xs,ys,guess=(1.1,1,1),fix='100')
    if verbose: print('a,b,c',a,b,c)
    a,b,c = polyfit(xs,ys,2,guess=(1.1,1,1),fix='100')
    if verbose: print('a,b,c',a,b,c)
    z,a,b,c = polyfit(xs,ys,3,guess=(1,1.1,1,1),fix='0100')
    if verbose: print('z,a,b,c',z,a,b,c)
    # z,a,b,c = polyfit(xs,ys,3,guess=(1,1.1,1,1),fix='0000')
    # if verbose: print('z,a,b,c',z,a,b,c)
    if verbose:
        print('quadminloc',Wave(ys,xs).quadminloc(),'quadmin',Wave(ys,xs).quadmin())
        print('quadminloc',Wave([2,1.1,2.1,5],xs).quadminloc(),'quadmin',Wave([2,1.1,2.1,5],xs).quadmin())
        print('quadmaxloc',Wave([2,1.1,2.1,5],xs).quadmaxloc(edgemax=1),'quadmax',Wave([2,1.1,2.1,5],xs).quadmax(edgemax=1))
        print('quadmaxloc',Wave([2,1.1,2.1,-5],xs).quadmaxloc(edgemax=1),'quadmax',Wave([2,1.1,2.1,-5],xs).quadmax(edgemax=1))
        print('quadmaxloc',Wave([2,1.1,-2.1,-5],xs).quadmaxloc(edgemax=1),'quadmax',Wave([2,1.1,-2.1,-5],xs).quadmax(edgemax=1))
    assert np.allclose([1,-2,2],quadfit(xs,ys))
    assert np.allclose([0.10869565,2.0135869],[Wave([2,1.1,-2.1,-5],xs).quadmaxloc(edgemax=1),Wave([2,1.1,-2.1,-5],xs).quadmax(edgemax=1)])
    assert np.allclose([1,1],[Wave(ys,xs).quadminloc(),Wave(ys,xs).quadmin()])
def dottest():
    dot1 = Vec([1, 2, 3])
    dot2 = Vec([2, 3, 2])
    result_pow = dot1.pow(dot2)
    assert (result_pow == Vec([1**2, 2**3, 3**2])).all()
    dot3 = Vec([10, 20, 30])
    dot4 = Vec([5, 5, 5])
    result_add = dot3.add(dot4)
    assert (result_add == Vec([10+5, 20+5, 30+5])).all()
    dot_str = Vec(["a", "b", "c"])
    result_upper = dot_str.str.upper()
    assert (result_upper == Vec(["A", "B", "C"])).all()
    dot5 = Vec([1, 2, 3])
    result_multiply_add = (dot5 * 2) + 3
    assert (result_multiply_add == Vec([5, 7, 9])).all()
    assert (Vec([-1, 2, -3]).abs()==[1,2,3]).all()
    assert (Vec(['a', 'b', 'C']).upper()==['A','B','C']).all()
    assert (Vec([1, 2, 3]).pow(Vec([2, 3, 2]))==[1,8,9]).all()
    # currently fails:
    dot1 = Vec([1, 2, 3],dtype=int)
    dot2 = Vec([10, 20, 30],dtype=int)
    dot3 = Vec([100, 200, 300],dtype=int)
    dot4 = Vec([1000, 2000, 3000],dtype=int)
    # print(list(dot1.add(dot2,dot3)))
    assert list(dot1.add(dot2,dot3))==[11,22,33]
    # print(dot1.sum(dot2,dot3))
    # def f(x, y, z):
    #     return x + y + z
    # result_f = dot1.f(dot2, dot3)
    # expected_result = Vec([1+10+100, 2+20+200, 3+30+300])
    # assert (result_f == expected_result).all()
def wave2dintegratetest():
    f = Wave2D([[1,1],[1,1],[1,1],[1,1]],xs=[0,1],ys=[1,2,3,4])
    g = Wave2D([[1,1,1,1],[1,1,1,1],[1,1,1,1]],xs=[1,2,3,4],ys=[10,20,30])
    assert np.allclose(f @ g, [[4,4,4],[4,4,4]]), f @ g
    assert np.allclose(f.integrate(g), [[4,4,4],[4,4,4]]), f.integrate(g)
def gaussianproducttest():
    ('θ,σ,ρ',list2str(gaussianproduct(0,10,1, 0,10,1,degrees=1)))
    ('θ,σ,ρ',list2str(gaussianproduct(0,10,1, +90,10,1,degrees=1)))
    ('θ,σ,ρ',list2str(gaussianproduct(+45,10,1, 0,1000,1000,degrees=1)))
    ('θ,σ,ρ',list2str(gaussianproduct(0,10,1, 0,inf,inf,degrees=1)))
    ('θ,σ,ρ',list2str(gaussianproduct(+45,10,1, -45,10,1,degrees=1)))
    ('θ,σ,ρ',list2str(gaussianproduct(-5,10,1, +5,10,1,degrees=1)))
    ('θ,σ,ρ',list2str(gaussianproduct(-5,inf,1, +5,inf,1,degrees=1)))
def lightpipestest(λ=1000,z=1,xsize=1000,ω=20,N=255,plot=False):
    import LightPipes as lp
    nm,µm,mm = lp.nm,lp.um,lp.mm
    F0 = lp.Begin(xsize*µm,λ*nm,N=N)
    F0 = lp.GaussBeam(F0,w0=ω*µm,x_shift=0,y_shift=0)
    I0 = lp.Intensity(F0)
    u0 = Wave(I0[N//2]).rename('u0')
    F1 = lp.Fresnel(z*mm,F0)
    I1 = lp.Intensity(F1)
    u1 = Wave(I1[N//2]).rename('u1')
    xs = wrange(-xsize/2,+xsize/2,xsize/(N-1),format=None) # print('xs',xs)
    yy,xx = np.meshgrid(xs,xs)
    zz = exp(-(xx**2+yy**2)/ω**2)
    ww = Wave2D(zz,xs=xs,ys=xs) # ww.plot()
    w0 = Wave( ww.xslicemax().y ).rename('w0')
    www = ww.propagatelightpipes(λ,z) # www.plot()
    w1 = Wave( www.xslicemax().y ).rename('w1')
    if plot:
        Wave.plots(u0,u1,w0.magsqr(),w1.magsqr(),m=0,c='0101',l='2233',scale=(3,1))
def tophattest(x0=-2,x1=2,dx=0.25,plot=False): # show that tophat(xs,x0,x1,dx) + tophat(xs,x1,x2,dx) = tophat(xs,x0,x2,dx)
    from wavedata import tophat
    xs = wrange(-5,5,0.5)
    v0 = Wave( tophat(xs,-2.0,2.0,0.5), xs )
    v1 = Wave( tophat(xs,-2.7,2.7,0.5), xs )
    if plot:
        Wave.plot(v0,v0.mirrorx(),1+v1,1+v1.mirrorx(),c='0011',l='32',m='o',grid=1,seed=19)
    assert np.allclose(v0.y,v0.mirrorx().y)
    assert np.allclose(v1.y,v1.mirrorx().y)
    xs = wrange(-3,3,dx)
    ys = wrange(-2,2,dx)
    nn = Wave2D(xs=xs,ys=ys)
    aa = nn.rectangle(-1,1,-1,1)
    assert np.allclose(4,aa.sum()*aa.dx**2) # area = 4 exactly
    bb = nn.rectangle(0,sqrt(2),0,sqrt(2))
    assert np.allclose(2,bb.sum()*bb.dx**2) # area = 2 exactly
    xs = wrange(x0,x1,dx)
    w = Wave(tophat(xs,-1,1,dx),xs)
    u0 = Wave(tophat(xs,-1,-1/pi,dx),xs)
    u1 = Wave(tophat(xs,-1/pi,1/np.e,dx),xs)
    u2 = Wave(tophat(xs,1/np.e,1,dx),xs)
    assert np.allclose(w.y,u0.y+u1.y+u2.y)
    if plot:
        Wave.plots(w,u0,u1,u2,u0+u1+u2,l='01113',m='oDDD ')
    u0 = Wave(tophat(xs,-1,-0.1,dx),xs) # check when x0==x1 or when x1-x0<dx/2
    u1 = Wave(tophat(xs,-0.1,0,dx),xs)
    u2 = Wave(tophat(xs,0,1,dx),xs)
    if plot:
        Wave.plots(w,u0,u1,u2,u0+u1+u2,l='01113',m='oDDD ')
    assert np.allclose(w.y,u0.y+u1.y+u2.y)
def rectangletest(step=0.2,plot=True):
    xs = wrange(-3,3,step)
    ys = wrange(-2,2,step)
    nn = Wave2D(xs=xs,ys=ys).rectangle(-2,2,-1,1)
    if plot:
        nn.plot()
    return nn
def fromtilestest(plot=True):
    a = [[0,0,0],
         [0,1,0],
         [2,2,2]]
    d = {0:1.0, 1:2.0, 2:1.5}
    xs = [-2,-1,1,2]
    ys = [-2,-1,0,1]
    step = 0.1
    nn = Wave2D.fromtiles(a,d,xs,ys,step)
    if plot:
        nn.plot()
    return nn
def slabstest(plot=True):
    xs = wrange(-3,3,0.2)
    ys = wrange(-2,2,0.2)
    nn = Wave2D(xs=xs,ys=ys)
    n0 = nn.yslabs(ys=[-1.5,0.5],ns=[1,3,1.5]) # n0.plot()
    n1 = nn.yslabs(ys=[-0.5,1.5],ns=[1,3,1.5]) # n1.plot()
    n2 = nn.xslabs(xs=[-1,1],ns=[n0,n1,n0])
    if plot:
        n2.plot()
    return n2
def fourwaystomakearidge(w=2,h=1,n0=1.0,n1=1.5,n2=2.0,step=0.1,plot=True):
    a = [[0,0,0],
         [0,2,0],
         [1,1,1]]
    d = {0:n0,1:n1,2:n2}
    xs = [-w/2-1,-w/2,w/2,w/2+1]
    ys = [-h-1,-h,0,1]
    na = Wave2D.fromtiles(a,d,xs,ys,step)
    if plot:
        na.plot()
    xs,ys = wrange(-w/2-1,+w/2+1,step),wrange(-h-1,1,step)
    nn = Wave2D(xs=xs,ys=ys)
    nend = nn.yslabs(ys=[-h],ns=[n1,n0])
    nmid = nn.yslabs(ys=[-h,0],ns=[n1,n2,n0])
    nb = nn.xslabs(xs=[-w/2,w/2],ns=[nend,nmid,nend])
    if plot:
        nb.plot()
    ntop = n0
    ncen = nn.xslabs(xs=[-w/2,w/2],ns=[n0,n2,n0])
    nbot = n1
    nc = nn.yslabs(ys=[-h,0],ns=[nbot,ncen,ntop])
    if plot:
        nc.plot()
    nd = n0 + (n1-n0)*nn.yslab(-h-2,-h) + (n2-n0)*nn.rectangle(-w/2,w/2,-h,0)
    if plot:
        nd.plot()
def storecallargstest(verbose=0):
    class MyClass:
        @storecallargs
        def f(self, a, b, **kwargs):
            if verbose: print(f"   a: {a}, b: {b}, kwargs: {kwargs}")
        @storecallargs
        def g(self, *args, **kwargs):
            if verbose: print(f"   args: {args}, kwargs: {kwargs}")
    # Example usage
    obj = MyClass()
    obj.f(1, 2, c=3, d=4)
    assert obj.callargs=={'a': 1, 'b': 2, 'c': 3, 'd': 4}
    if verbose: print('obj.callargs',obj.callargs)  # Output should be {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    obj.g(1, 2, c=3, d=4)
    assert obj.callargs=={'c': 3, 'd': 4}
    if verbose: print('obj.callargs',obj.callargs)
def maplistargstest():
    @maplistargs
    def f(a, b, c):
        return a * b * c
    assert f([1, 2, 3], 2, [4, 5, 6])==[f(1, 2, 4), f(2, 2, 5), f(3, 2, 6)]
def wwavetest():
    from wavedata import WWave as Wave
def radiusedtest(plot=True,i=None):
    for j in range(30):
        if i is None:
            radiusedtest(plot=0,i=j)
    from wavedata import wrange
    a = Vs([(1,1)]+[(cos(u),sin(u)) for u in wrange(0,pi/2,pi/40,format=None)]+[(1,1)]) # a.plot(m='o',grid=1)
    i = i if i is not None else len(a)//2
    aa = a.shiftclosedcurve(i%len(a[:-1]))
    b = a.radiused(0.1) + V(0.1,0.1) # b.plot(m='o',grid=1)
    c = b.radiused(0.05,debug=0) + V(0.1,0.1)
    if plot:
        Wave.plot(a.wave(),aa[1:-1].wave(),b.wave(),c.wave(),m='o',ms=2,grid=1,aspect=1,seed=1)
def mergetest(plot=False):
    aa = Wave([1,2,3,4],[0,1,2,3])
    bb = Wave([5,6,7,8],[2,3,4,5])
    assert np.allclose( aa.mergex(bb).x, bb.mergex(aa).x )
    a = Wave([0,0,1,1,0,0],[0,1,1,2,2,4])
    b = Wave([0,0,1,1,0,0],[0,2,2,3,3,4])
    c = a.mergex(b)+b.mergex(a)
    d = a.addlayer(b)
    assert np.allclose( c.x, [0,1,1,2,2,3,3,4] )
    assert np.allclose( d.x, [0,1,1,2,2,3,3,4] )
    assert np.allclose( c.y, d.y )
    if plot: Wave.plot(a+0.0,a.mergex(b)+0.0,b+2.0,b.mergex(a)+2.0,c+4.0,m='o',l='03')
def dashtest():
    from wavedata import Wavex,Wx,WWave
    # u = Wave(data=[-5,5],index=[-5,5],name='u').plot()
    # uu = Wave(index=[-5,5],name='uu').plot()
    # uuu = Wave(data=[-5,5],name='uuu').plot()

    u1 = Wavex([0,1],'Wavex')
    u2 = Wx([0,1],'Wx')
    Wave.plot(u1,u2)

    # w0 = 0*Wavex.dashed(-2,2,dash=(0.5,0.5),phase=0.0) 
    # w1 = 1*Wavex.dashed(-2,2,dash=(0.5,0.5),phase=0.5)
    # w2 = 2*Wavex.dashed(-2,2,dash=(0.5,0.5),phase=1.0)
    # Wave.plots(w0,w1,w2)
def mesh2dtest():
    from wavedata import Mesh2D
    # import pickle
    # w0 = Wave2D([[0,1,2],[2,2,2]],xs=[-10,0,10],ys=[-5,5])
    # p = pickle.dumps(w0)
    # w = pickle.loads(p)
    # assert np.array_equal(w,w0)
    m0 = Mesh2D([[0,1,2],[2,2,2]],xs=[-10,-5,5,10],ys=[-5,0,5])
    p = pickle.dumps(m0)
    m = pickle.loads(p)
    assert np.array_equal(m,m0)
    assert np.array_equal(Mesh2D('012 222',xs=[-10,-5,5,10],ys=[-5,0,5]),m0)
    assert np.array_equal(Mesh2D('012-222',xs=[-10,-5,5,10],ys=[-5,0,5]),m0)
    assert np.array_equal(Mesh2D('012222',xs=[-10,-5,5,10],ys=[-5,0,5]),m0)
    assert np.array_equal(Mesh2D('012'
                                 '222',xs=[-10,-5,5,10],ys=[-5,0,5]),m0)
    # m0.plot()
    # m0.plot(outlinelinewidth=2)
    a = 1+m0
    assert isinstance(a,Mesh2D)
    assert hasattr(a,'xs') and hasattr(a,'ys')
if __name__ == '__main__':
    # # backendtest()
    # # backendtest('TkCairo')
    # # backendtest('Qt5Cairo')
    showplot = 0
    print('showplot',bool(showplot))
    if 1:
        wavetest(plot=showplot)
        wave2Dtest(plot=showplot)
        plottest(plot=showplot)
        complextest()
        histtest(plot=showplot)
        arraytest()
        vtest()
        # vstest(0,plot=showplot)
        arctest(plot=showplot)
        circletest(plot=showplot)
        linelabeltest(showplot)
        markertest(showplot)
        plotordertest(showplot)
        plot2Dtest(showplot)
        bartest(showplot)
        errorbartest(showplot)
        rotationtest()
        test3d()
        voronoitest(showplot)
        shapelytest(showplot)
        concattest()
        filltest(showplot)
        chaintest()
        radialinterpolatetest(showplot)
        randomshapetest(showplot)
        polartest(showplot)
        flowertest(showplot)
        plotpropertytest(showplot)
        trapezoidintegratetest(showplot)
        pdfsampletest()
        transposetest(showplot)
        interpolate1dtest()
        piecewisetest(showplot)
        cosfittest(showplot)
        correlatetest(showplot)
        ffttest(showplot)
        markeralphatest(showplot)
        wavepickletest(showplot)
        joblibWave2Dtest()
        standarderrorofthemeantest()
        stepclimbtest()
        quadmaxtest()
        dottest()
        wave2dintegratetest()
        gaussianproducttest()
        lightpipestest(plot=showplot)
        tophattest(plot=showplot)
        rectangletest(plot=showplot)
        fromtilestest(plot=showplot)
        slabstest(plot=showplot)
        fourwaystomakearidge(plot=showplot)
        storecallargstest()
        maplistargstest()
        tophattest(plot=0)
        stepclimbtest()
        radiusedtest(showplot)
    # dashtest()
    mergetest(plot=0)
    mesh2dtest()
    slabstest(plot=0)

    print('all tests passed')

    # fttest()
    # multicrosscorrelationtest(showplot)
    # loadfiletest()
    # hysteresistests()
    # animate(showplot)
    # animateshapetest(showplot)
