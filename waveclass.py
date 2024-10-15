
import pandas as pd
import numpy as np
from numpy import sqrt,pi,nan,inf,sign,exp,log,sin,cos,floor,ceil
from util import sincsqr,sincsqrcurvefit,coscurvefit,cosfitfunc,fourierintegral,quadfit,lerp

class Wave(pd.Series):
    """
    Data structure similar to Igor Pro's wave (https://www.wavemetrics.net/doc/igorman/II-05%20Waves.pdf)
    containing both x values and y values like the data in an x-y plot. Acts as a numpy array of y values,
    but also has an array of x values of the same length to accompany it that are used in operations where
    appropriate e.g. wave.area(). x values are equal to np.arange(len(yvalues)) if unspecified.
    """
    def __init__(self, data=None, index=None, name=None, dtype=None, copy=False, fastpath=False, **kwargs):
        if isinstance(data, (int, float, complex)) and index is None:
            data = np.zeros(data)
        if data is None and index is not None:
            data = index # create a wave that's just an x wave, ie. wx = Wave(index=[0,1,2])
        if isinstance(data,Wave) and index is None:
            super(Wave, self).__init__(data=data, index=data.x, dtype=type(data[0]), name=name, copy=copy, fastpath=fastpath)
        # super(Wave, self).__init__(data=data, index=index, dtype=float, name=name, copy=copy, fastpath=fastpath)
        # print(data,type(data),data[0] if hasattr(data,'__len__') else 0)
        try:
            assert any([isinstance(d,complex) for d in data])
            super(Wave, self).__init__(data=data, index=index, dtype=complex, name=name, copy=copy, fastpath=fastpath) # support complex dtype
        except:
            super(Wave, self).__init__(data=data, index=index, dtype=float, name=name, copy=copy, fastpath=fastpath)
        # if hasattr(data,'name') and name is None: self.name = None # erase name after operation?
        for k,v in kwargs.items():
            setattr(self,k,v)
    def __getitem__(self, p): # we want Wave to act like a list, not a dict like in Series
        try:
            if isinstance(p,slice):                             # seems to have fixed complex wave w being cast to float in w[:] operation
                return Wave( pd.Series(self).iloc[p] ).copy()   # but still a mystery why type(w.iloc[:][0])==<class 'numpy.float64'>
            return self.iloc[p]
        except pd.core.indexing.IndexingError:  # matplotlib counts on IndexError being raised, whereas pandas raises IndexingError
            raise IndexError
    def __setitem__(self, p, val): # https://stackoverflow.com/questions/31282764/when-subclassing-a-numpy-ndarray-how-can-i-modify-getitem-properly
        if not isinstance(p,slice) and int(p)==p: p=int(p)
        self.iloc[p] = val # sets val at iloc
    def __call__(self,x,x1=None,extrapolate=None):
        if x1 is not None:
            # p0,p1 = nudge(self.p(x)),nudge(self.p(x1))
            p0,p1 = self.x2p(x),self.x2p(x1)
            p0,p1 = 0 if np.isnan(p0) else p0, len(self)-1 if np.isnan(p1) else p1
            return self[int(np.ceil(p0)):int(np.floor(p1))+1]
        if isinstance(x,Wave):
            return Wave(self.atx(x,extrapolate=extrapolate),x.y)
        return self.atx(x,extrapolate=extrapolate)
    def __str__(self):
        return self.wavestring()
        return (str(np.array2string(np.array(self), threshold=19)).replace('\n','') + ' x:' + 
                str(np.array2string(np.array(self.x), threshold=19)).replace('\n','') + self.name*bool(self.name) )
        # def format(a): return '['+' '.join([('%.2f'%n).rstrip('0').rstrip('.') for n in a])+']'
        # return format(self) + ' x:' + format(self.x)
    def __repr__(self):
        return str(self)
    # https://stackoverflow.com/questions/57237906/
    # https://pandas.pydata.org/pandas-docs/stable/development/extending.html#define-original-properties
    _metadata = 'name c l lw li m ms mf mew'.split()
    def getplot(self):
        return {k:getattr(self,k) for k in self._metadata if not k=='name'} # _metadata = 'name c l lw li m ms mf mew'.split()
    def newplot(self,override=True,**kwargs):
        return self.copy().setplot(override=override,**kwargs)
    def setplot(self,s='',override=True,**kwargs):
        if s:
            c,l,m,mf,ms,lw,mew = (list(s)+[None]*7)[0:7]
            kwargs.update(c=c,l=l,m=m,mf=mf,ms=ms,lw=lw,mew=mew)
        assert 'f' not in kwargs, 'use mf not f for marker fill value'
        for k,v in kwargs.items():
            if override or k not in self:
                setattr(self,k,v)
        return self
    def sp(self,s='',override=True,**kwargs):
        return self.setplot(s,override,**kwargs)
    @property
    def _constructor(self):
        # return Wave
        return type(self)
    @property
    def p(self):
        return Wave(np.arange(len(self)),index=self.index)
    @property
    def y(self):
        return self.values
    @property
    def x(self):
        return self.index
    @property
    def xwave(self):
        return Wave(data=np.array(self.index),index=np.array(self.index))
    @property
    def ywave(self):
        return Wave(data=np.array(self.y),name=self.name)
    def dx(self,exact=True):
        deltas = np.diff(self.index)
        if exact:
            assert np.allclose(deltas, deltas[0]), 'inconsistent delta x'
            return self.index[1]-self.index[0]
        return np.mean(deltas)
    @property
    def np(self):
        return np.asarray(self)
    # def array(self): # causes failure in pandas 1.0.3
    #     return np.asarray(self)
    def ylist(self):
        return list(self.y)
    def xlist(self):
        return list(self.x)
    def yarray(self):
        return np.asarray(self.y)
    def xarray(self):
        return np.asarray(self.x)
    def copy(self,**kwargs):
        # if kwargs: print('kwargs in Wave.copy()',kwargs)
        return Wave(self.yarray().copy(),self.xarray().copy(),self.name)
    def __eq__(self,w):
        return all([a==b for a,b in zip(self.y,w.y)]) and all([a==b for a,b in zip(self.x,w.x)])
    def vs(self):
        # return Vs([(x,y) for x,y in zip(self.x,self.y)])
        return Vs(x=self.x,y=self.y)
    def wave2vs(self):
        return self.vs()
    def rename(self,name):
        self.name = name
        return self
    def appendname(self,name):
        self.name = str(self.name) + str(name)
        return self
    def swapxy(self):
        return Wave(data=self.x, index=self.y, name=self.name)
    @x.setter
    def x(self,x): # this is what is runs when you do w.x = [1,2,3]
        self.index = list(x)
    def x2p(self,x,nearest=False):
        if nearest:
            return int(np.round(np.clip( self.p(x,extrapolate='lin'), 0, len(self)-1 )))
        def isint(z):
            return np.isclose(z,np.round(z),rtol=1e-40)
        def nudge(z):
            return int(np.round(z)) if isint(z) else z
        return nudge(self.p(x))
    def xaty(self,y,reverse=False,i1d=False):
        def lerp(x,x0,x1,y0,y1):
            return y0 + (y1-y0)*(x-x0)/(x1-x0)
        # print('TODO: update wave.xaty to using interpolate1d')
        if reverse:
            return Wave(self.y[::-1],self.x[::-1]).xaty(y,i1d=i1d)
        if i1d:
            Wave(self.x,self.y).atx(y,checkvalidity=True)
        for i,(y0,y1) in enumerate(zip(self[:-1],self[1:])):
            if (y0<=y<=y1) or (y1<=y<=y0):
                return lerp(y,y0,y1,self.x[i],self.x[i+1])
        return np.NaN
    # def addlayer(self,*ws,extrapolate=None,rtol=1e-5,atol=1e-8):
    #     if not ws: return self
    #     a,b = self,ws[0]
    #     assert a.xwave.monotonicincreasing() and b.xwave.monotonicincreasing()
    #     x,y,i,j = [],[],0,0
    #     while i<len(a) and j<len(b):
    #         if np.isclose(a.x[i],b.x[j],rtol=rtol,atol=atol):
    #             x,y,i,j = x+[a.x[i]], y+[a[i]+b[j]], i+1, j+1
    #         elif a.x[i]<b.x[j]:
    #             x,y,i = x+[a.x[i]], y+[a[i]+b(a.x[i],extrapolate=extrapolate)], i+1
    #         elif b.x[j]<a.x[i]:
    #             x,y,j = x+[b.x[j]], y+[a(b.x[j],extrapolate=extrapolate)+b[j]], j+1
    #         else:
    #             assert 0
    #     return Wave(y,x).addlayer(*ws[1:],extrapolate=extrapolate,rtol=rtol,atol=atol)
    # def mergex(self,w,extrapolate=None,rtol=1e-5,atol=1e-8):
    #     a,b = self,w
    #     assert a.xwave.monotonicincreasing() and b.xwave.monotonicincreasing()
    #     x,y,i,j = [],[],0,0
    #     while i<len(a) and j<len(b):
    #         if np.isclose(a.x[i],b.x[j],rtol=rtol,atol=atol):
    #             x,y,i,j = x+[a.x[i]], y+[a[i]], i+1, j+1
    #         elif a.x[i]<b.x[j]:
    #             x,y,i = x+[a.x[i]], y+[a[i]], i+1
    #         elif b.x[j]<a.x[i]:
    #             x,y,j = x+[b.x[j]], y+[a(b.x[j],extrapolate=extrapolate)], j+1
    #         else:
    #             assert 0
    #     return Wave(y,x,self.name)
    def mergex(self,*ws,extrapolate=None,rtol=1e-5,atol=1e-8):
        if not ws: return self
        a,b = self,ws[0]
        assert a.xwave.monotonicincreasing() and b.xwave.monotonicincreasing()
        x,y,i,j = [],[],0,0
        while i<len(a) and j<len(b):
            if np.isclose(a.x[i],b.x[j],rtol=rtol,atol=atol):
                x,y,i,j = x+[a.x[i]], y+[a[i]], i+1, j+1
            elif a.x[i]<b.x[j]:
                x,y,i = x+[a.x[i]], y+[a[i]], i+1
            elif b.x[j]<a.x[i]:
                x,y,j = x+[b.x[j]], y+[a(b.x[j],extrapolate=extrapolate)], j+1
            else:
                assert 0
        return Wave(y,x,self.name).mergex(*ws[1:],extrapolate=extrapolate,rtol=rtol,atol=atol)
    def addlayer(self,*ws,extrapolate=None,rtol=1e-5,atol=1e-8):
        if not ws: return self
        a = self.mergex(ws[0],extrapolate=extrapolate,rtol=rtol,atol=atol)
        b = ws[0].mergex(self,extrapolate=extrapolate,rtol=rtol,atol=atol)
        assert np.allclose(a.xwave,b.xwave)
        return (Wave(a.y,b.x,self.name)+b).addlayer(*ws[1:],extrapolate=extrapolate,rtol=rtol,atol=atol)
    def minimumlayer(self,*ws,extrapolate=None,rtol=1e-5,atol=1e-8):
        if not ws: return self
        a = self.mergex(ws[0],extrapolate=extrapolate,rtol=rtol,atol=atol)
        b = ws[0].mergex(self,extrapolate=extrapolate,rtol=rtol,atol=atol)
        assert np.allclose(a.xwave,b.xwave)
        return np.minimum(Wave(a.y,b.x,self.name),b).minimumlayer(*ws[1:],extrapolate=extrapolate,rtol=rtol,atol=atol).rename(self.name)
    def monotonicincreasing(self):
        return not np.any(-1==np.sign(self.y[1:] - self.y[:-1]))
    def monotonicx(self):
        x = self.xwave
        return x.monotonicincreasing() or x.reverse().monotonicincreasing()   
    def removenonmonotonicx(self):
        inc = self.x[0]<=self.x[-1]
        ps = self.xys if inc else self.xys()[::-1]
        result = [ps[0]]
        for p in ps[1:]:
            if p[0]>result[-1][0]:
                result.append(p)
        return Wave.fromxypairs(result if inc else result[::-1],self.name)
    def monotonicxcheck(self):
        xs,ys = self.x,self.y
        if not xs[0]<xs[-1]:
            xx = [x for x in xs if not np.isnan(x)]
            if not xx[0]<xx[-1]:
                xs,ys = xs[::-1],ys[::-1]
        def allvalid(xs):
            return all([0<xi for xi in xs[1:]-xs[:-1]])
        f = allvalid(xs)
        if not f:
            failed = [(x0,x1) for x0,x1 in zip(xs[1:],xs[:-1]) if x0>=x1]
            print(len(failed),'out of',len(xs),'nonmonotonic')
            Wave(xs).plot()
        return f
    def atx(self,x,kind='linear',extrapolate=None,checkvalidity=False,monotonicx=True): #,left=None,right=None): # returns y(x) at given x, by default limited to x[0] <= x <= x[-1] unless otherwise specified by left,right
        if monotonicx:
            y = interpolate1d(x,self.x,self.y,kind=kind,extrapolate=extrapolate,checkvalidity=checkvalidity)
            # return Wave(y,x) if hasattr(x,'__len__') else y
            aa = (list,tuple,np.ndarray,Wave,pd.core.series.Series) # 1D iterables
            if isinstance(x,aa):
                # print(type(x),x.shape,x.size,[int(isinstance(x,a)) for a in aa])
                if isinstance(x,np.ndarray) and 1==x.ndim: # but not Wave2D
                    return Wave(y,x)
            return y
        if 1==len(self):
            if extrapolate is not None:
                return self.y[0]+x*0
            return np.where(x==self.x[0], self.y[0]+x*0, np.NaN+x*0)
        @np.vectorize # ok if x is list or np.array or Wave
        def f(x):
            for i,(x0,x1) in enumerate(zip(self.x[:-1],self.x[1:])):
                if (x0<=x and x<=x1) or (x1<=x and x<=x0):
                    if 'log'==extrapolate:
                        return np.exp(lerp(x,x0,x1,np.log(self[i]),np.log(self[i+1])))
                    return lerp(x,x0,x1,self[i],self[i+1])
            if x<=self.x[0]<=self.x[1] or x>=self.x[0]>self.x[1]:
                x0,x1,y0,y1 = self.x[0],self.x[1],self[0],self[1]
            elif x<=self.x[-1]<self.x[-2] or x>=self.x[-1]>=self.x[-2]:
                x0,x1,y0,y1 = self.x[-1],self.x[-2],self[-1],self[-2]
            else:
                x0,x1,y0,y1 = 4*[np.NaN]
            if extrapolate in ['const','constant']:
                return y0
            if extrapolate in ['lin','linear']:
                return lerp(x,x0,x1,y0,y1)
            if extrapolate in ['log','logarithmic']:
                return np.exp(lerp(x,x0,x1,np.log(y0),np.log(y1)))
            if extrapolate is not None:
                return extrapolate + 0*y0 # complex waves don't work without +0*y0 ?!
            return np.NaN
        if isinstance(x,(list,tuple,np.ndarray,Wave,pd.core.series.Series)):
            return f(x)
        try:
            return np.ndarray.item(f(x))
        except ValueError as e:
            print(x,f'add {type(x)} to list of interables')
            raise e
    def yatx(self,x,kind='linear',extrapolate=None,checkvalidity=False,monotonicx=True):
        return self.atx(x,kind=kind,extrapolate=extrapolate,checkvalidity=checkvalidity,monotonicx=monotonicx)
    def interpolate(self,dx=None,kind='linear'):
        from scipy.interpolate import interp1d
        x,y = self.x,self.y
        def dxmin(x):
            return (x[1:]-x[:-1]).min()
        dx = dx if dx is not None else 10**round(np.log10(dxmin(x))-1.)
        x0,x1 = (x.min()//dx)*dx,(x.max()//dx+1)*dx
        x,y = np.array([x[0]-dx*10]+list(x)+[x[-1]+dx*10]),np.array([y[0]]+list(y)+[y[-1]])
        f = interp1d(x,y,kind=kind)
        # f = interp1d(x, y, kind='cubic')
        xx = np.linspace(x0,x1,round((x1-x0)/dx)+1) # print(x0,x1,round((x1-x0)/dx)+1,dx)
        yy = f(xx)
        return Wave(yy,xx)
    def xys(self):
        return [(x,y) for x,y in zip(self.x,self.y)]
    def subrange(self,x0,x1):
        return self[self.x2p(x0,nearest=True):self.x2p(x1,nearest=True)]
    def area(self):
        return self.y.sum()*self.dx()
    def mean(self):
        return self.y.mean()
    def nanmean(self):
        return np.nanmean(self.y)
    def sdev(self,ddof=0):
        return np.std(self.y,ddof=ddof)
    def nansdev(self,ddof=0):
        return np.nanstd(self.y,ddof=ddof)
    def standarderrorofthemean(self,N=None): # if N = number of consecutive chunks is specified, returns sdev of the means of each chunk
        if N is None:
            return np.nanstd(self.y,ddof=1)/np.sqrt(sum(~np.isnan(self.y)))
        u = self.removenans()
        w = u.downsample(int(len(u)//N))
        return np.nanstd(w.y,ddof=1)/np.sqrt(len(w))
    def localsdev(self):
        # x - y = µx - µy ± √(σx²+σy²) = σ√2
        deltas = np.diff(self.y)
        return np.std(deltas)/np.sqrt(2)
    def monotony(self):
        return self.mean()/self.sdev()
    def len(self):
        return len(self)
    def pmax(self):
        return self.values.argmax()
    def pmin(self):
        return self.values.argmin()
    def xmax(self):
        # return self.idxmax()
        return self.x.max()
    def xmin(self):
        # return self.idxmin()
        return self.x.min()
    def maxloc(self,aswave=False):
        return self[self.pmax():self.pmax()+1] if aswave else self.xmax()
    def minloc(self,aswave=False):
        return self[self.pmin():self.pmin()+1] if aswave else self.xmin()
    def quadmaxlocplot(self,edgemax=False):
        w = self.quadmaxloc(aswave=1,edgemax=edgemax).setplot(m='o')
        return Wave.plots(self,w,c='00')
    def quadmaxloc(self,aswave=False,edgemax=False):
        p = self.pmax()
        assert edgemax or    0<p<len(self)-1
        p0,p1 = (p-1,p+2) if 0<p<len(self)-1 else (0,3) if 0==p else (-4,-1)
        a,b,c = quadfit(self.x[p0:p1],self.y[p0:p1])
        assert a<0
        x0 = -0.5*b/a # x-coord of max or min for y=ax²+bx+c
        return Wave([a*x0**2+b*x0+c],[x0]) if aswave else x0
    def quadmax(self,edgemax=False):
        p = self.pmax()
        assert edgemax or 0<p<len(self)-1
        # a,b,c = quadfit(self.x[p-1:p+2],self.y[p-1:p+2])
        p0,p1 = (p-1,p+2) if 0<p<len(self)-1 else (0,3) if 0==p else (-4,-1)
        a,b,c = quadfit(self.x[p0:p1],self.y[p0:p1])
        assert a<0
        x = -0.5*b/a # x-coord of max or min for y=ax²+bx+c
        return a*x**2 + b*x + c
    def quadmaxfit(self,edgemax=False):
        p = self.pmax()
        assert edgemax or 0<p<len(self)-1
        p0,p1 = (p-1,p+2) if 0<p<len(self)-1 else (0,3) if 0==p else (-4,-1)
        a,b,c = quadfit(self.x[p0:p1],self.y[p0:p1])
        assert a<0
        xs = np.linspace(self.x[p0],self.x[p1],101)
        return Wave( a*xs**2 + b*xs + c, xs )
    def quadminloc(self,aswave=False,edgemin=False):
        p = self.pmin()
        assert edgemin or 0<p<len(self)-1
        p0,p1 = (p-1,p+2) if 0<p<len(self)-1 else (0,3) if 0==p else (-4,-1)
        a,b,c = quadfit(self.x[p0:p1],self.y[p0:p1])
        assert a>0
        x0 = -0.5*b/a # x-coord of max or min for y=ax²+bx+c
        return Wave([a*x0**2+b*x0+c],[x0]) if aswave else x0
    def quadmin(self,edgemin=False):
        p = self.pmin()
        assert edgemin or 0<p<len(self)-1
        p0,p1 = (p-1,p+2) if 0<p<len(self)-1 else (0,3) if 0==p else (-4,-1)
        a,b,c = quadfit(self.x[p0:p1],self.y[p0:p1])
        assert a>0
        x = -0.5*b/a # x-coord of max or min for y=ax²+bx+c
        return a*x**2 + b*x + c
    def log(self):
        return np.log(self)
    def exp(self):
        return np.exp(self)
    def normalize(self,c=1,zeromin=False):
        if abs(self.min())>self.max():
            return (-self).normalize(c,zeromin)
        return c*(self-self.min())/(self.max()-self.min()) if zeromin else c*self/self.max()
    def hasxnans(self):
        return any(list(np.isnan(self.x)))
    def hasynans(self):
        return any(list(np.isnan(self.y)))
    def sort(self):
        assert not self.hasxnans(), "can't sort if self.x has nans"
        wx,wy = zip(*sorted(zip(self.x,self.y)))
        return Wave(wy,wx,self.name)
    def sorty(self):
        assert not self.hasynans(), "can't sort by y if self.y has nans"
        wy,wx = zip(*sorted(zip(self.y,self.x)))
        return Wave(wy,wx,self.name)
    def concat(self,w):
        return Wave(list(self.y)+list(w.y),list(self.x)+list(w.x),name=self.name)
    def reverse(self):
        return Wave(self.y[::-1],self.x[::-1],name=self.name)
    def removenans(self):
        index = np.isfinite(self.y) & np.isfinite(self.x)
        return Wave(self.y[index],self.x[index],name=self.name)
    def removeinf(self):
        return self.removenans()
    def hasdata(self):
        return any(list(np.isfinite(self.y)))
    def inf2nan(self):
        yy = self.y
        yy[yy==-inf] = np.nan
        yy[yy==+inf] = np.nan
        return Wave(yy,self.x,name=self.name)
    def removevalue(self,value):
        return Wave(self.y[value!=self.y],self.x[value!=self.y],name=self.name)
    def removezeros(self):
        return self.removevalue(0)
    def scalex(self,a):
        # self.x *= a
        # return self
        return type(self)(self.y,self.x*a,self.name)
    def shiftx(self,x):
        # self.x += x
        # return self
        return type(self)(self.y,self.x+x,self.name)
    def mirrorx(self):
        return type(self)(self.y[::-1],-self.x[::-1],self.name)
    def offsetx(self,x0,extrapolate=None):
        ys = [self.atx(x-x0,extrapolate=extrapolate) for x in self.x]
        return Wave(ys,self.x,self.name)
    def offsetp(self,p,fill=None):
        fill = fill if fill is not None else np.nan
        ys = list(self.y[p:]) + [fill]*p if 0<=p else [fill]*abs(p) + list(self.y[:p])
        return Wave(ys,self.x,self.name)
    def clipfilter(self,ymin=-inf,ymax=inf):
        ind = (ymin<self.y) & (self.y<ymax) # (self.y>ymin)*(self.y<ymax) #np.logical_and( (self.y>ymin), (self.y>ymax) )
        return Wave(self.y[ind],self.x[ind],self.name)
    def morph(self,old,new): # old = list of old values, new = list of new values
        # e.g. old=[0,1,4,inf],new=[0,2,4,4] results in 0→0,0.5→1,1→2, and all y>=4 will be clipped to 4
        w = Wave(new,old)
        return Wave([w(y) for y in self.y],self.x)
    def smooth(self,boxsize=5,medfilt=False,pctfilt=None,savgol=True,mode=None):
        def nanbounds(w):
            ww = w.copy()
            ww[0],ww[-1] = nan,nan
            return ww
        if 1==boxsize:
            return self
        elif medfilt:
            from scipy.signal import medfilt
            w = medfilt(self, kernel_size=boxsize)
        elif pctfilt is not None: # pctfilt==0 equivalent to minfilt, pctfilt==100 equivalent to maxfilt
            from scipy.ndimage import percentile_filter
            w = percentile_filter(self,percentile=pctfilt,size=boxsize)
        elif savgol:
            from scipy.signal import savgol_filter
            w,mode = (nanbounds(self),None) if 'nan'==mode else (self,mode)
            mode = mode if mode is not None else 'interp' # interp, mirror, nearest, wrap, constant(need cval)
            w = savgol_filter(w, window_length=boxsize, polyorder=2, mode=mode)
        else:
            w,mode = (nanbounds(self),None) if 'nan'==mode else (self,mode)
            mode = mode if mode is not None else 'same' # assumes zeros beyond boundary
            w = np.convolve(w, np.ones(boxsize)/boxsize, mode=mode)
        return Wave(w,self.x,self.name)
    def convolution(self,w,pad=2,abs=False):
        return self.zeropad(scale=pad).convolve(w,abs=abs)
    def convolve(self,filter,abs=False):
        assert np.isclose(self.dx(),filter.dx()), f"self.dx={self.dx()} filter.dx={filter.dx()}"
        # assert self.dx()==filter.dx(), f"self.dx={self.dx()} filter.dx={filter.dx()}"
        w = Wave( np.convolve(self, filter, mode='same'), self.x )
        return np.abs(w) if abs else w
    # def correlation(self,w,dp):
    #     print(len(w),len(w)//2-dp//2,len(w)//2+(dp-dp//2))
    #     p0,p1 = len(w)//2-dp//2,len(w)//2+(dp-dp//2)
    #     # u = self.correlate(w[p0:p1])
    #     # ux = self.xwave.correlate(w.xwave[p0:p1])
    #     u  = np.correlate(self, w[p0:p1], mode='same')
    #     ux = np.correlate(self.x, w.x[p0:p1], mode='same')
    #     p0 = Wave(ux).maxloc()
    #     # ux.plot(); exit()
    #     # Wave(ux).plot()
    #     # Wave(u).plot()
    #     assert len(u)==len(ux)
    #     Wave(Wave(u,self.dx()*(np.arange(len(u))-p0))).plot()
    #     return u
    #     return Wave(u,self.dx()*(np.arange(len(u))-p0))
    def correlate(self,v,normalize=True,samex=True):
        assert type(v) is Wave
        assert np.isclose(self.dx(),v.dx()), f"self.dx={self.dx()} v.dx={v.dx()}, Waves must have same x spacing"
        def indexoffset(): # relative offset of the x values
            return (v.x[0]-self.x[0])/v.dx()
        # if windowed:
        #     assert all(v.x==self.x)
        #     return
        n = len(self)-len(v)+1
        # print(f"n:{n} x0:{self.x[0]-v.x[0]} x0:{self.x[-1]-v.x[-1]}")
        xs = np.linspace(self.x[0]-v.x[0],self.x[-1]-v.x[-1],n)
        w = Wave( np.correlate(self,v,mode='valid'), xs )
        return w / np.sqrt(sum(self**2)) / np.sqrt(sum(v**2)) if normalize else w
    def zeropad(self,scale=50,poweroftwo=True):
        def next_power_of_2(x): # https://stackoverflow.com/a/14267669
            return 1 if x == 0 else int(2**np.ceil(np.log2(x)))
        n = next_power_of_2(scale*len(self)) if poweroftwo else scale*len(self) # total length of new wave
        n0 = (n-len(self.index))//2 # length of front padding
        # n1 = n-len(self.index)-n0 # length of end padding
        wx = (np.arange(0,n)-n0)*self.dx() + self.index[0]
        w = np.zeros_like(wx,dtype=self.dtype)
        w[n0:n0+len(self)] = self
        return Wave(w,wx)
    def oldfft(self,pad=True):
        assert len(self)<1e5, 'Wave too long for fft (can use too much ram in 64 bit python and lock up the machine)'
        w = self.zeropad() if pad else self
        f = np.fft.fftshift(np.fft.fft(w.y))
        fx = np.fft.fftshift(np.fft.fftfreq(len(w), d=w.dx()))
        # return Wave(f*f.conjugate(),fx)
        return Wave(f.real**2 + f.imag**2, fx)
        # return Wave(abs(f), fx)
    def fft(self,plot=0): # returns fft where x axis is frequency, i.e. df=1/dx
        # assert len(self)<1e5, 'Wave too long for fft (can use too much ram in 64 bit python and lock up the machine)'
        yy,x0,dx,df = self.y,self.x[0],self.dx(),1/self.dx()
        ff = np.fft.fftshift(np.fft.fftfreq(yy.size)) * df
        aa = np.fft.fftshift(np.fft.fft(yy)) * dx
        aa *= exp(-1j*2*np.pi*x0*ff)
        w = Wave(aa,ff)
        if plot: w.plot(m=1,lw=1,ms=1)
        return w
    def ft(self,f,*args,**kwargs): # 5800x slower than fft()
        return fourierintegral(self,self.x,f,*args,**kwargs) # exact fourier integral of piecewise linear function
    def ftwave(self,fs,*args,**kwargs):
        return Wave(self.ft(fs),fs)
    def sqr(self):
        return self**2
    def abs(self):
        return Wave(np.abs(self.y),self.x,self.name)
    def sqrt(self):
        return Wave(np.sqrt(self.y),self.x)
    def magsqr(self):
        return Wave(self.y.real**2 + self.y.imag**2,self.x,self.name)
    def angle(self):
        return Wave(np.angle(self.y),self.x,self.name)
    def phase(self):
        return self.angle()
    def real(self):
        return Wave(np.real(self.y),self.x,self.name)
    def imag(self):
        return Wave(np.imag(self.y),self.x,self.name)
    def conjugate(self):
        return Wave(np.conjugate(self.y),self.x,self.name)
    def diff(self):
        dw =      self.y[1:] -     self.y[:-1]
        dwx = 0.5*self.x[1:] + 0.5*self.x[:-1] # Wave(dw,dwx).plot()
        return Wave(dw,dwx)
    def differentiate(self,n=1,dx=None,extrapolate='lin'): # keeps same number of points
        if 0==n: return self
        dx = dx if dx is not None else self.dx()
        dwdx = [ (self(x+dx/2,extrapolate=extrapolate) - 
                  self(x-dx/2,extrapolate=extrapolate))/dx for x in self.x ]
        return Wave(dwdx,self.x).differentiate(n-1,dx,extrapolate=extrapolate)
    def cumsum(self):
        return Wave(self.y.cumsum(),self.x,self.name)
    def integrate(self):
        # exact integral of piecewise linear function
        return fourierintegral(self.y,self.x,freq=0).real
    def groupindex(self,extrapolate='lin'): # ng = n - λ*Δn/Δλ # assumes self = Wave(index values, wavelength values)
        return self - self.x*self.differentiate(extrapolate=extrapolate)
    def zerocrossings(self, getIndices=False):
    # copied from https://github.com/sczesla/PyAstronomy/blob/master/src/pyaC/mtools/zerocross.py
        """
        Find the zero crossing events in a discrete data set.
        Linear interpolation is used to determine the actual
        locations of the zero crossing between two data points
        showing a change in sign. Data point which are zero
        are counted in as zero crossings if a sign change occurs
        across them. Note that the first and last data point will
        not be considered whether or not they are zero. 
        If getIndices is True, also the indicies of the points preceding
        the zero crossing event will be returned. Default is False.
        """
        x,y = self.x,self.y
        assert not np.any((x[1:] - x[0:-1]) <= 0.0), 'The x-values must be sorted in ascending order'
        indi = np.where(y[1:]*y[0:-1] < 0.0)[0]
        dx = x[indi+1] - x[indi]
        dy = y[indi+1] - y[indi]
        zc = -y[indi] * (dx/dy) + x[indi]
        zi = np.where(y == 0.0)[0]
        zi = zi[np.where((zi > 0) & (zi < x.size-1))]
        zi = zi[np.where(y[zi-1]*y[zi+1] < 0.0)]
        zzindi = np.concatenate((indi, zi)) 
        zz = np.concatenate((zc, x[zi]))
        sind = np.argsort(zz)
        zz, zzindi = zz[sind], zzindi[sind]
        return zz if not getIndices else (zz, zzindi)
    def find_steepest_zero_crossings(self): # modified from ai chat
        ys,xs = self.y,self.x
        zero_crossings = []
        for i in range(1,len(ys)):
            if ys[i] == 0:
                zero_crossings.append(i-1e-9) # (-1e-9 in case of zero on last i)
            elif ys[i-1] * ys[i] < 0:
                m = (ys[i] - ys[i-1]) / (xs[i] - xs[i-1]) # Interpolate to find the exact zero crossing
                y0 = ys[i] - m * xs[i]
                zero_crossings.append(-y0 / m)
        def slope(x):
            i = int(x)
            return abs(ys[i+1]-ys[i]) / (xs[i+1]-xs[i])
        zero_crossings.sort(key=lambda x:slope(x), reverse=True) # sort by slope
        slopes = [slope(x) for x in zero_crossings]
        return np.array(zero_crossings),np.array(slopes)
    def recurringpeaks(w,threshold=None,debug=False,includeends=False,getranges=False):
        # find the location of each peak and trough in a noisy periodic signal
        # assumes that peak-to-peak noise is much less than half the peak-to-peak amplitude
        threshold = threshold if threshold is not None else 0.5*(w.max()+w.min())
        crossings = [i for i in range(len(w)-1) if w[i]<threshold<=w[i+1] or w[i+1]<threshold<=w[i]]
        def slope(i):
            return int(np.sign(w[i+1]-w[i]))
        def intervaltype(i,j): # return -1,0,+1 depending on whether the interval between two crossings is trough, noise, or peak
            intmin,intmax = w[i:j+1].min(),w[i:j+1].max()
            assert -slope(i)==slope(j), f'{i} {j} {slope(i)} {slope(j)}'
            if +1==slope(i):    # possible peak
                return +1 if 0.5*(w.max()-threshold)<intmax-threshold else 0
            elif -1==slope(i):  # possible trough
                return -1 if 0.5*(threshold-w.min())<threshold-intmin else 0
            else:
                assert 0
        intervals = [(i,j,slope(i)) for i,j in zip(crossings[:-1],crossings[1:]) if intervaltype(i,j)]
        assert all([ -slope(i)==slope(j) for i,j,s in intervals ])
        if includeends:
            (i0,_,s0),(_,j1,s1) = intervals[0],intervals[-1]
            intervals = [(0,i0+1,-s0)] + intervals + [(j1,len(w)-1,-s1)]
        peaks = [ (i+(s*w[i:j+1]).pmax(),s) for i,j,s in intervals ]
        assert all(i<=k<=j for (i,j,_),(k,_) in zip(intervals,peaks))
        if debug:
            # print('intervals',intervals); print('peaks',peaks)
            u0 = (threshold+0*w).setplot(c='k',l='3',m=' ')
            wc = Wave([w[i] for i in crossings],[w.x[i] for i in crossings])
            wfs = [Wave([w[i],w[j]],[w.x[i],w.x[j]]).setplot(m='',l='0',c='2' if +1==s else '4',f=1) for i,j,s in intervals]
            wmax = Wave([w[i] for i,s in peaks if +1==s],[w.x[i] for i,s in peaks if +1==s]).setplot(m='D',l='',c='6',f=0)
            wmin = Wave([w[i] for i,s in peaks if -1==s],[w.x[i] for i,s in peaks if -1==s]).setplot(m='D',l='',c='7',f=0)
            Wave.plots(w,wc,*wfs,wmax,wmin,u0,m=' D',l='0 ',c='10',lw=1,ms=3,seed=2) # 
        return peaks # [(index,±1) for each peak and whether it is a peak or trough]
    def hysteresiscrossings(w,fraction=None,hi2lo=None,lo2hi=None,initial=None,debug=False):
        # return same length array with value 0 if it has crossed below hi2lo and 1 if it has crossed above lo2hi
        # adapted from https://stackoverflow.com/questions/23289976/
        assert fraction is not None or (hi2lo is not None and lo2hi is not None)
        if initial is None: # try both and return the one with the most transitions
            h0 = w.hysteresiscrossings(fraction=fraction,hi2lo=hi2lo,lo2hi=lo2hi,initial=0,debug=debug)
            h1 = w.hysteresiscrossings(fraction=fraction,hi2lo=hi2lo,lo2hi=lo2hi,initial=1,debug=debug)
            def taggedmaskcount(mask):
                tagged,tag = [int(bool(mask[0]))],1
                for i,j in zip(mask[:-1],mask[1:]):
                    tagged += [tag if j else 0]
                    tag = tag+1 if (1,0)==(bool(i),bool(j)) else tag
                return tag
            return h0 if taggedmaskcount(h0)>taggedmaskcount(h1) else h1
        hi2lo,lo2hi = (fraction*w.min()+(1-fraction)*w.max(),(1-fraction)*w.min()+fraction*w.max()) if fraction is not None else (hi2lo,lo2hi)
        y,y0,y1,rev = (np.array(w)[::-1],lo2hi,hi2lo,True) if hi2lo>lo2hi else (np.array(w),hi2lo,lo2hi,False) # If thresholds are reversed, y must be reversed as well
        ishigh = (y>=y1)
        ishighorlow = (y<=y0) | ishigh
        ii = np.nonzero(ishighorlow)[0]  # indices of all that are above or below
        if not ii.size:  # prevent index error if ii is empty
            yy = np.zeros_like(y, dtype=bool) | initial
        else:
            counts = np.cumsum(ishighorlow)  # from 0 to len(y)
            yy = np.where(counts, ishigh[ii[counts-1]], initial)
        yy = yy[::-1] if rev else yy
        if debug:
            w0,w1 = (w*0+y0).rename('lower').setplot(c='k',l='3'),(w*0+y1).rename('upper').setplot(c='k',l='2')
            Wave.plots(w1,w0,w,Wave(yy*(y1-y0)+y0,w.x),seed=3,lw=1)
        return yy
    def hysteresispeaks(w,fraction=None,y0=None,y1=None,ends=False,debug=False):
        # find index of each peak or trough subject to the criterion that the signal crosses the hyseresis threshold both coming and going
        assert fraction is not None or (y0 is not None and y1 is not None)
        y0,y1 = (fraction*w.min()+(1-fraction)*w.max(),(1-fraction)*w.min()+fraction*w.max()) if fraction is not None else (y0,y1)
        f = w.hysteresiscrossings(hi2lo=y0,lo2hi=y1,initial=None,debug=debug)
        b = w.hysteresiscrossings(hi2lo=y1,lo2hi=y0,initial=None,debug=debug)
        # u = (f & b) if +1==peaksign else (~f & ~b) if -1==peaksign else ((f & b)|(~f & ~b))
        u = 1*(f & b)-1*(~f & ~b)
        u[0],u[-1] = (0,0) if ends else (u[0],u[-1])
        if debug:
            Wave.plots(Wave(f),Wave(b),seed=1,l='13')
            # Wave(u,w.x).plot()
        return u
    def periodicpeaks(w,fraction,ends=0,sign=None,debug=False):
        def taggedmask(mask):
            # [0,1,1,0,0,1,1,1,0,1,0] given mask return same array with each block tagged with block number: 
            # [0,1,1,0,0,2,2,2,0,3,0]
            tagged,tag = [int(bool(mask[0]))],1
            for i,j in zip(mask[:-1],mask[1:]):
                tagged += [tag if j else 0]
                tag = tag+1 if (1,0)==(bool(i),bool(j)) else tag # print(i,j,tagged,tag)
            return np.array(tagged)
        def splitmask(mask):
            #  [0,1,1,0,0,1,1,1,0,1,0] given input mask return list of masks, one for each block:
            # [[0,1,1,0,0,0,0,0,0,0,0],
            #  [0,0,0,0,0,1,1,1,0,0,0],
            #  [0,0,0,0,0,0,0,0,0,1,0]]
            u = taggedmask(mask)
            return np.array([(u==k).astype(int) for k in range(1,u.max()+1)])
        u = w.hysteresispeaks(fraction,ends=False,debug=debug)
        t = taggedmask(w.hysteresispeaks(fraction,ends=False))
        def localpeakindex(k):
            wlocal = np.where(t==k,w,nan)
            imax = np.nanargmax(wlocal)
            if +1==u[imax]:
                return imax
            imin = np.nanargmin(wlocal)
            assert -1==u[imin]
            return imin
        peakindices = [localpeakindex(k) for k in range(1,t.max()+1)] # ordered list of indices of both max and min peaks
        # print(peakindices)
        if 1==ends:
            peakindices = [0] + peakindices        if 0==u[ 0] else peakindices
            peakindices = peakindices + [len(u)-1] if 0==u[-1] else peakindices
        if -1==ends:
            peakindices = peakindices[1:]  if 0!=u[ 0] else peakindices
            peakindices = peakindices[:-1] if 0!=u[-1] else peakindices
        peakindices = peakindices if sign is None else [i for i in peakindices if sign==u[i]] # select only max peaks (sign=+1) or min peaks (sign=-1)
        # print(peakindices)
        if debug:
            # Wave.plots(Wave(u,w.x),Wave(t,w.x))
            Wave.plots(w.copy().setplot(c='#00000022',lw=10),Wave(w.y[peakindices],w.x[peakindices],c='k',m='o'))
        return peakindices

    def recurringslopes(w,tilt,fraction,debug=False,crop='max',getindices=False):
        # doesn't currently deal with ends correctly
        assert tilt in 'rise fall'.split()
        assert crop in 'val max mid min'.split()
        if crop not in 'val max'.split(): raise NotImplementedError
        u = w.hysteresispeaks(fraction,ends=False)
        ii = [  i for i in range(len(u)-1) if tuple(u[i:i+2])==(0,+1 if 'fall'==tilt else -1)] # section starts (tuple=(-1,0) for rise,min crop, avg of both for mid) # print('ii',ii)
        jj = [i+1 for i in range(len(u)-1) if tuple(u[i:i+2])==(-1 if 'fall'==tilt else +1,0)] # section ends   (tuple=(0,+1) for rise,min) # print('jj',jj)
        # if ends, add 0 to ii if jj[0]<ii[0] and start section is long enough (u[0]==0)
        # if ends, add -1 to jj if jj[-1]<ii[-1] and end section is long enough (u[-1]==0)
        ends,strict = True,False
        if ends and jj and (not ii or jj[0]<ii[0]) and (u[0] in (0,+1 if 'fall'==tilt else -1)):
            ii = [0] + ii
        else:
            jj = jj[1:] if jj else jj
        if ends and ii and (not jj or jj[len(jj)-1]<ii[len(jj)-1]) and (u[-1] in (0,+1 if 'fall'==tilt else -1)):
            jj = jj + [len(u)-1]
        else:
            ii = ii[:len(jj)] if ii else ii
        assert len(ii)==len(jj) and all(i<j for i,j in zip(ii,jj)), ['ii',*ii,'jj',*jj]
        vs = [w[i:j] for i,j in zip(ii,jj)]
        def valcrop(v):
            return v[v.pmax():v.pmin()+1] if 'fall'==tilt else v[v.pmin():v.pmax()+1]
        if debug:
            Wave.plots(w.copy().setplot(c='#00000022',lw=10),*[v.setplot(c='#aa000044',lw=4) for v in vs],*[valcrop(v).setplot(c='k',lw=0.5) for v in vs],lw=1,ms=1)
        if not getindices:
            return [valcrop(v) for v in vs] if 'val'==crop else vs
        def valindices(i,j):
            return (i+w[i:j].pmax(),i+w[i:j].pmin()+1) if 'fall'==tilt else (i+w[i:j].pmin(),i+w[i:j].pmax()+1)
        indices = [(i,j) for i,j in zip(ii,jj)]
        return [valindices(i,j) for i,j in indices] if 'val'==crop else indices
    def resample(self,n=2,dx=None):
        from util import wrange
        x0,x1 = self.x[0],self.x[-1]
        xs = wrange(x0,x1,dx) if dx is not None else np.linspace(x0,x1,n*len(w))
        return Wave([self(x) for x in xs],xs,self.name)
    def downsample(self,n=10):
        if n<2:
            return self
        ys = [np.mean(np.array(self[i*n:i*n+n])) for i in range(len(self)//n)]
        xs = [np.mean(np.array(self.index[i*n:i*n+n])) for i in range(len(self)//n)]
        return Wave(ys,xs,name=self.name)
    def upsample(self,num=10):
        xs,ys = list(self.x),list(self.y)
        xxs,yys = [],[]
        for x0,x1,y0,y1 in zip(xs[:-1],xs[1:],ys[:-1],ys[1:]):
            xxs += list(np.linspace(x0,x1,num=1 if x0==x1 else num,endpoint=False))
            yys += list(np.linspace(y0,y1,num=1 if x0==x1 else num,endpoint=False))
        xxs += [xs[-1]]
        yys += [ys[-1]]
        return Wave(yys,xxs)
        # return type(self)(yys,xxs)
    def oldpeakwidth(self,level=None,invert=False,debug=0,p0=None):
        p0,y0 = [self.pmin(),self.min()] if invert else [self.pmax(),self.max()]
        w0,w1 = self[:p0+1],self[p0:]
        x0,x1 = w0.xaty(level,reverse=True,i1d=False), w1.xaty(level,i1d=False)
        if debug:
            print('peakwidth x1-x0',x1-x0)
            return Wave((level,level),(x0,x1),m='o')
        return x1-x0
    def peakwidth(self,ylevel,p0=None,liberal=False,invert=False,debug=False,getwave=False):
        # for multipeaked curve, returns narrowest possible valid width, as opposed to widest possible when liberal=True
        def lerp(x,x0,x1,y0,y1):
            return y0 + (y1-y0)*(x-x0)/(x1-x0)
        ys,xs = self.y,self.x
        if invert: raise NotImplementedError
        p0 = p0 if p0 is not None else np.argmax(ys)
        assert ylevel<ys[p0]
        def halfwidthx(ys,xs,y0):
            assert len(ys)==len(xs) and ys[-1]<ys[0]
            for i in range(len(ys)-1):
                if ys[i+1] < y0:
                    return lerp(y0,ys[i],ys[i+1],xs[i],xs[i+1])
            return xs[-1]
        # x0,x1 = halfwidthx(ys[:p0+1][::-1],xs[:p0+1][::-1],ylevel), halfwidthx(ys[p0:],xs[p0:],ylevel)
        assert all(ys[:p0+1][::-1]==ys[p0::-1]) and all(xs[:p0+1][::-1]==xs[p0::-1])
        x0,x1 = ([halfwidthx(-ys[p0:][::-1],xs[p0:][::-1],-ylevel), halfwidthx(-ys[:p0+1],xs[:p0+1],-ylevel)] if liberal else 
                 [halfwidthx(ys[:p0+1][::-1],xs[:p0+1][::-1],ylevel), halfwidthx(ys[p0:],xs[p0:],ylevel)])
        w = Wave((ylevel,ylevel),(x0,x1),m='o',c='k',l='3')
        if debug:
            print('x0,x1',x0,x1,'peakwidth x1-x0',x1-x0)
            Wave.plots(self,w,xlim=(2*x0-x1,2*x1-x0))
        return w if getwave else np.abs(x1-x0)
    def fwhm(self,relativelevel=0.5,liberal=True,debug=0):
        return self.peakwidth(relativelevel*self.max(),liberal=liberal,invert=0,debug=debug)
    def gaussian(self,y0=0,dy=1):
        return Wave(np.random.normal(y0,dy,size=self.x.shape),self.x)
    def random(self):
        return Wave(np.random.random(size=self.x.shape),self.x)
    def choice(self,choices):
        return Wave(np.random.choice(choices,size=self.x.shape),self.x)
    def histogram(self,bins='auto',dx=None): # bins = 101, bins = [0,5,10,15,20]
        assert np.nanmax(self)<1e9 and -1e9<np.nanmin(self), f"min:{np.nanmin(self)} max:{np.nanmax(self)}"
        if dx is not None:
            n0,n1 = int(floor(self.min()/dx)), int(ceil(self.max()/dx))
            bins = dx*np.linspace(n0,n1,n1-n0+1)
        hist,binedges = np.histogram(self.y[~np.isnan(self.y)],bins=bins)
        histx = (binedges[:-1]+binedges[1:])/2.
        return Wave(hist,histx)
    def getpeaks(self,mins=False):
        a = np.array(self)
        m = ( (a[:-2]>a[1:-1])*(a[2:]>a[1:-1]) if mins else 
              (a[:-2]<a[1:-1])*(a[2:]<a[1:-1]) )
        # return Wave([0]+list(m)+[0],self.x) # 1 at peaks, 0 elsewhere
        return np.array([x for x,y in zip(self.x[1:-1],m) if 1==y])
    def countpeaks(self,mins=False):
        return np.sum(self.getpeaks())
        # a = np.array(self)
        # return np.sum( (a[:-2]<a[1:-1]) * (a[2:]<a[1:-1]) )
    def curvefit(self,fitfunc,x0=None,x1=None,coef=2):
        from scipy.optimize import curve_fit
        cc = curve_fit(fitfunc, self.x, self.y)[0]
        wx = np.linspace(x0,x1,101) if x0 is not None else self.x
        w = Wave(fitfunc(wx,*cc),wx)
        return w if 0==coef else (cc if 1==coef else (w,cc))
    def polyfit(self,n,guess=None,fix=None):
        return polyfit(self.x,self.y,n,guess=None,fix=None)
    def linefit(self,x0=None,x1=None,b=None,coef=2):
        def default(x,x0):
            return x0 if x is None else x
        x0,x1 = default(x0,self.x.min()),default(x1,self.x.max())
        from scipy.optimize import curve_fit
        if b is None: # line through (x,y) = (0,b)
            def fitfunc(x,aa,bb):
                return aa*x+bb
        else:
            def fitfunc(x,aa):
                return aa*x+b
        cc = curve_fit(fitfunc, self.x, self.y)[0]
        if 4==coef:
            return Wave([fitfunc(x,*cc) for x in self.x],self.x)
        w = Wave([fitfunc(x,*cc) for x in [x0,x1]],[x0,x1])
        # return w,cc
        return w if 0==coef else (cc if 1==coef else (w,cc))
    def quadraticfit(self,x0=None,x1=None,a=None,b=None,c=None,coef=2): # extremum at (x0,y0) = (-0.5*b/a, c-0.25*b**2/a)
        from scipy.optimize import curve_fit
        if b is None and c is None:
            if a is None:
                def fitfunc(x,aa,bb,cc):
                    return aa*x**2 + bb*x + cc
            else:
                def fitfunc(x,bb,cc):
                    return a*x**2 + bb*x + cc
        else: # y = ax²
            if b is None:
                def fitfunc(x,aa,bb):
                    return aa*x**2 + bb*x
            else:
                def fitfunc(x,aa):
                    return aa*x**2 + b*x + c
        cc = curve_fit(fitfunc, self.x, self.y)[0]
        # x0,x1 = default(x0,self.x.min()),default(x1,self.x.max())
        wx = np.linspace(x0,x1,101) if x0 is not None else self.x
        w = Wave(fitfunc(wx,*cc),wx)
        return w if 0==coef else (cc if 1==coef else (w,cc))
        # # fit notes
        # fit = np.polyfit(self.x,self.y,1) # highest power first!
        # series = np.polynomial.polynomial.Polynomial.fit(self.x,self.y,1) # print(series.convert().coef)
        # # cubic through zero
        # def fitfunc(x,a,b,c):
        #     return a*x**3 + b*x**2 + c*x  # d=0 is implied
        # params = curve_fit(fitfunc, x, y)
        # [a,b,c] = params[0]
        # x_fit = np.linspace(x[0], x[-1], 100)
        # y_fit = fitfunc(x_fit,a,b,c)
    def gaussfit(self,coef=False,upsample=False,sdev=None):
        height,floor,fwhm,x0 = gausscurvefit(self.x,self.y,sdev=sdev)
        # return p if coef else gauss(self.x,*p)
        if coef:
            return height,floor,fwhm,x0
        if upsample:
            xs = self.upsample().x
            return Wave( gauss(xs,height,floor,fwhm,x0), xs )
        return Wave( gauss(self.x,height,floor,fwhm,x0), self.x )
    def sincsqrfit(self,coef=0,upsample=False):
        p = sincsqrcurvefit(self.x,self.y)
        if 1==coef:
            return p
        xs = self.upsample().x if upsample else self.x
        w = Wave( sincsqr(xs,*p), xs )
        return w if 0==coef else p if 1==coef else (w,p)
    def cosfit(self,coef=0,upsample=False,guess=(None,None,None,None),fix=(0,0,0,0)):
        p = coscurvefit(self.x,self.y,guess=guess,fix=fix)
        xs = self.upsample(10 if upsample==1 else upsample).x if upsample else self.x
        w = Wave( cosfitfunc(xs,*p), xs )
        return w if 0==coef else p if 1==coef else (w,p)
    def sinfit(self,offset=0,upsample=0,coef=False,A=False):
        c = [0, self.mean(), self.quadraticfit(coef=0)][offset]
        a = self - c
        f0 = abs(a.fft().maxloc())
        print('f0',f0)
        z = a.ft(f0)
        # print(z.real,z.imag)
        φ = 2*pi*f0*a.xwave
        φ = φ.upsample() if upsample else φ
        z0 = sqrt(z.real**2 + z.imag**2) * 2/(a.x[-1]-a.x[0])
        if coef:
            return z0
        if A:
            return (c-z0)/(c+z0)
        return c + (z.real*cos(φ) + z.imag*sin(φ)) * 2/(a.x[-1]-a.x[0])
    def wavestring(self,round=False,name=True):
        sy = ','.join([str(rint(y)) if round else f'{y:g}' for y in self.y])
        sx = ','.join([str(rint(x)) if round else f'{x:g}' for x in self.x])
        return f"Wave([{sy}],[{sx}]" + (f",'{self.name}')" if self.name else ")")
    def pprint(self,format='g',xformat=None,yformat=None):
        sx = xformat if xformat is not None else format
        sy = yformat if yformat is not None else format
        print('xs=['+','.join([f"{x:{sx}}" for x in self.x])+']')
        print('ys=['+','.join([f"{y:{sy}}" for y in self.y])+']')
    def wave2textarray(self):
        return (str(np.array2string(np.array(self), threshold=19)).replace('\n','') + ' x:' + 
                str(np.array2string(np.array(self.x), threshold=19)).replace('\n','') + (self.name if self.name else '') )
        # def format(a): return '['+' '.join([('%.2f'%n).rstrip('0').rstrip('.') for n in a])+']'
        # return format(self) + ' x:' + format(self.x)

    def printwave(self,s='',type=0):
        print(s+' '+str(list(self))+' x'+str(list(self.x))+' p'+str(list(self.p)))
    def lines(self,color='k'):
        for x0,x1,y0,y1 in zip(self.x[:-1],self.x[1:],self.y[:-1],self.y[1:]):
            yield {'xdata':(x0,x1),'ydata':(y0,y1),'color':color}
    def ballandstick(self,y0=-inf): # for plotting wave as a bar chart
        ys = [yi for y in self.y for yi in (y0,y,nan)] # print('ys',ys)
        xs = [xi for x in self.x for xi in (x,x,nan)]
        return Wave(ys,xs,self.name).setplot(m='o',l='0')
    def plot(self,*a,waves=[],getpng=False,**kwargs):
        assert isinstance(waves, list)
        assert all([isinstance(w, Wave) for w in a])
        waves = [self]+list(a)+list(waves)
        import plot
        png = plot.plot(waves=waves,**kwargs)
        # assert not isinstance(m,Wave)
        return png if getpng else self
    def export(self,*waves,folder='',save='out',separator='\t',header=True):
        import itertools
        waves = [self]+list(waves)
        ynames = [w.name if (hasattr(w,'name') and w.name is not None) else f'y{n+1 if n else ""}' for n,w in enumerate(waves)]
        xnames = [w.x.name if (hasattr(w.x,'name') and w.x.name is not None) else f'x{n+1 if n else ""}' for n,w in enumerate(waves)]
        xynames = [name for names in zip(xnames,ynames) for name in names]
        xywaves = [v for vs in zip([w.x for w in waves],waves) for v in vs]
        def samexwaves():
            def samecolumns(i,j):
                return len(xywaves[i])==len(xywaves[j]) and np.allclose(list(xywaves[i]),list(xywaves[j]), equal_nan=True)
            return all([samecolumns(i,i+2) for i in range(0,len(xywaves)-2,2)])
        if samexwaves():
            xynames = [s for s in xynames[0:1]+xynames[1::2]]
            xywaves = [s for s in xywaves[0:1]+xywaves[1::2]]
        with open(folder+save+'.dat','w',encoding='utf-8',errors='replace') as file:
            if header: file.write(separator.join(xynames)+'\n') # print(separator.join(xynames))
            for xs in itertools.zip_longest(*xywaves,fillvalue=''):
                file.write(separator.join([f'{x}' for x in xs])+'\n' )
    def igorsave(self,waves=[],folder='',save='wave'):
        waves = [self]+list(waves)
        ynames = [w.name if (hasattr(w,'name') and w.name is not None) else 'w'+str(n)+'y' for n,w in enumerate(waves)]
        xnames = [w.x.name if (hasattr(w.x,'name') and w.x.name is not None) else 'w'+str(n)+'x' for n,w in enumerate(waves)]
        xynames = [name for names in zip(xnames,ynames) for name in names]
        ynames,xnames = ['w'+str(n)+'y' for n,w in enumerate(waves)],['w'+str(n)+'x' for n,w in enumerate(waves)]
        xyids = [name for names in zip(['w'+str(n)+'x' for n,w in enumerate(waves)],['w'+str(n)+'y' for n,w in enumerate(waves)]) for name in names]
        xywaves = [v for vs in zip([w.x for w in waves],waves) for v in vs]
        with open(folder+save+'.dat','w',encoding='utf-8',errors='replace') as file:
            file.write('\t'.join(xyids)+'\n') # print('\t'.join(xynames))
            for xs in itertools.zip_longest(*xywaves,fillvalue=None):
                file.write('\t'.join(['%s'%x for x in xs])+'\n' )
        with open(folder+'name.dat','w',encoding='utf-8',errors='replace') as file:
            file.write('\n'.join(['name']+xynames)+'\n')
    def igor(self,waves=[],title='Graph',xlabel='',ylabel='',legendtext='',groupsize=0,corner=None,aspect='',folder='c:/temp/igorplot/',run=False):
        from subprocess import Popen
        self.igorsave(waves=waves,folder=folder)
        def opencorner():
            slopeup = [1 for w in waves if w[0]<=w[-1]]
            return 'LT' if len(waves)<2*sum(slopeup) else 'RT'
        corner = corner if corner is not None else opencorner()
        with open(folder+'ax.dat','w',encoding='utf-8',errors='replace') as file:
            file.write(('\n'.join(['ax',title,xlabel,ylabel,legendtext,str(groupsize),corner,aspect])+'\n'))
        if run:
            Popen(r'"C:\Program Files\WaveMetrics\Igor Pro 8 Folder\IgorBinaries_x64\Igor64.exe" "C:\work\work.pxp"') # Popen(['C:\Program Files\WaveMetrics\Igor Pro 8 Folder\IgorBinaries_x64\Igor64.exe','/I','/X','print 99'])
    def clickcoords(self):
        coords = []
        import matplotlib.pyplot as plt
        plt.plot(self.x,self.y)
        def onclick(event):
            nonlocal coords
            coords.append((event.xdata, event.ydata))
            print(f" ({event.xdata:g},{event.ydata:g})")
        cid = plt.gcf().canvas.mpl_connect('button_press_event', onclick)
        plt.show()
        return coords
    def remove(self,i):
        xs,ys = self.xlist(),self.ylist()
        return Wave(ys[:i]+ys[i+1:],xs[:i]+xs[i+1:],self.name)
    def findclosestpoint(self,p):
        dx,dy = self.xmax()-self.xmin(),self.max()-self.min()
        d = Wave([(x-p[0])**2/dx**2+(y-p[1])**2/dy**2 for x,y in self.xys()])
        return d.minloc()
    def removeclosestpoint(self,p):
        dx,dy = self.xmax()-self.xmin(),self.max()-self.min()
        d = Wave([(x-p[0])**2/dx**2+(y-p[1])**2/dy**2 for x,y in self.xys()])
        return self.remove(d.minloc())
    def removeclosestpoints(self,ps):
        return self if 0==len(ps) else self.removeclosestpoint(ps[0]).removeclosestpoints(ps[1:])
    @staticmethod
    def zeros(xs,*args,**kwargs):
        return 0*Wave(xs,xs,*args,**kwargs)
    @staticmethod
    def ones(xs,*args,**kwargs):
        return 1+0*Wave(xs,xs,*args,**kwargs)
    @staticmethod
    def fromxsandys(xs,ys,*args,**kwargs):
        return Wave(ys,xs,*args,**kwargs)
    @staticmethod
    def fromxypairs(xys,name=None):
        xs,ys = zip(*xys)
        return Wave(ys,xs,name)
    @staticmethod
    def fromxys(xys,name=None):
        return Wave.fromxypairs(xys,name)
    @staticmethod
    def wavesum(*ws,extrapolate='lin'):
        ws = ws[0] if isinstance(ws[0],(list,tuple)) else ws
        w0 = ws[0].copy()
        for w in ws[1:]:
            w0 += w(w0.x,extrapolate=extrapolate)
        return w0
    @staticmethod
    def exports(*ws,**kwargs):
        wws = [Wave(w) for w in ws]
        return wws[0].export(waves=wws[1:],**kwargs)
    @staticmethod
    def igors(*ws,**kwargs):
        wws = [Wave(w) for w in ws]
        return wws[0].igor(waves=wws[1:],**kwargs)
    @staticmethod
    def plots(*ws,waves=[],**kwargs):
        assert isinstance(waves, list)
        ws = ws[0] if isinstance(ws[0],(list,tuple)) else ws
        # assert isinstance(ws[0],Wave), f'trying to plot {type(ws[0])}'
        if not isinstance(ws[0],Wave):
            print(f'warning, trying to plot {type(ws[0])}')
        import plot
        return plot.plot(waves=list(ws)+waves,**kwargs)
    @staticmethod
    def combines(*ws,**kwargs):
        xys = sorted([(x,y) for w in ws for x,y in w.xys()])
        xs,ys = zip(*xys)
        return Wave(ys,xs)
    @staticmethod
    def nudge(z):
        def isint(z):
            return np.isclose(z,round(z),rtol=1e-50)
        return round(z) if isint(z) else z
    # same name allowed? no, but... # https://stackoverflow.com/q/2589690, https://stackoverflow.com/questions/28237955/, https://stackoverflow.com/questions/395735/how-to-check-whether-a-variable-is-a-class-or-not
    # @classmethod
    # def plot(cls,*ws,**kwargs):
    #     return cls.plots(*ws,**kwargs,loglog=1)
class Wavex(Wave):
    def __init__(self, data=None, name=None, dtype=None, copy=False, fastpath=False):
        return super(Wavex, self).__init__(data=None, index=data, name=name, dtype=dtype, copy=copy, fastpath=fastpath)
    # @staticmethod
    # def dashed(x0,x1,dash=(1,1),phase=0.5,dx=1):
    #     # phase=0,0.5,1 → dash gap is right,center,left shifted within each dash cycle
    #     # dx determines sampling
    #     xs = np.linspace(x0,x1,int((x1-x0+0.5)/dx)+1)
    #     return Wavex(xs,'dash')
class Wx(Wavex):
    pass
class WW(Wave):
    pass
    def __init__(self, data=None, index=None, name=None, dtype=None, copy=False, fastpath=False, **kwargs):
        return super(WW, self).__init__(data=data, index=index, name=name, dtype=dtype, copy=copy, fastpath=fastpath, **kwargs)

def interpolate1d(x,xs,ys,kind='linear',extrapolate=None,checkvalidity=False):
    # kind: linear nearest nearest-up zero slinear quadratic cubic previous next
    # see scipy.interpolate.interp1d
    from scipy.interpolate import interp1d
    if not xs[0]<xs[-1]:
        xx = [x for x in xs if not np.isnan(x)]
        if not xx[0]<xx[-1]:
            xs,ys = xs[::-1],ys[::-1]
    def valid(xs): # 𝒪(1) monte carlo validity check
        from random import randrange
        if 1==len(xs):
            assert 0, 'need to implement interpolate1d for size one '
            return True
        i = randrange(0,len(xs)-1)
        # if not xs[i]<=xs[i+1]: print(i, xs[i], xs[i+1], xs[i]<=xs[i+1])
        return xs[i]<=xs[i+1] or np.isnan(xs[i]) or np.isnan(xs[i+1])
    def allvalid(xs): # 𝒪(N)
        return all([0<xi for xi in xs[1:]-xs[:-1]])
    assert valid(xs), 'failed monte carlo check, xs must be monotonically increasing or decreasing for interpolate1d (also use checkvalidity=True)'
    if checkvalidity: assert allvalid(xs),[xi for xi in xs[1:]-xs[:-1]]
    assert extrapolate in [None,'lin','linear','log','logarithmic','const','constant']
    if extrapolate in ['log','logarithmic']:
        ys = np.log(ys)
    fill_value = (ys[0],ys[-1]) if extrapolate in ['const','constant'] else (None if extrapolate is None else 'extrapolate')
    with np.errstate(invalid='ignore',divide='ignore'): # print(np.geterr())
        # suppress warnings when there are duplicate x values (eliminate dups instead?)
        f = interp1d(xs,ys,kind=kind,fill_value=fill_value,bounds_error=False)
        y = np.exp(f(x)) if extrapolate in ['log','logarithmic'] else f(x)
    # return y if hasattr(x,'__len__') else np.asscalar(y)
    return y if hasattr(x,'__len__') else y.item()
