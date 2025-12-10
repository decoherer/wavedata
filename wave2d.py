
import pandas as pd
import numpy as np
from numpy import sqrt,pi,nan,inf,sign,exp,log,sin,cos,floor,ceil
import scipy
from waveclass import Wave

class Wave2D(np.ndarray):
    def __new__(cls,array=None,xs=None,ys=None):
        array = np.array(array) if isinstance(array,(list,tuple)) else array
        if xs is not None and ys is not None:
            nx,ny = len(xs),len(ys)
            if array is not None:
                if array.size==nx*ny and not array.shape==(nx,ny):
                    print(f'Wave2D input has wrong shape: {array.shape}, reshaping: ({nx},{ny})')
                    array = array.reshape((nx,ny))
                assert (nx,ny)==array.shape, f'input array must have size (nx,ny), array:{array} nx,ny:{nx,ny} '
        else:
            nx,ny = array.shape
            xs,ys = np.linspace(0,nx-1,nx),np.linspace(0,ny-1,ny)
        if array is not None:
            obj = np.asarray(array).view(cls)
        else:
            obj = np.zeros((nx,ny)).view(cls)
        obj.xs,obj.ys = np.asarray(xs,dtype=float),np.asarray(ys,dtype=float)
        return obj
    @classmethod
    def fromtif(cls,filename,transpose=False):
        from PIL import Image
        im = Image.open(filename)
        a = np.array(im).T
        # print('im.size',im.size) # print('a.shape',a.shape) # print(a.min(),a.max())
        # pix = im.load() # print(pix[0,0],pix[1,0],pix[2,0])
        return Wave2D(a).transpose() if transpose else Wave2D(a)
    @classmethod
    def fromimage(cls,filename,transpose=False):
        from PIL import Image
        from numpy import asarray
        # if grayscale:
        #     image = Image.open(filename).convert('L')
        #     data = asarray(image)
        #     return Wave2D(data).transpose() if transpose else Wave2D(data)
        from PIL import Image
        from numpy import asarray
        image = Image.open(filename)
        rgbweights = [0.2989, 0.5870, 0.1140] # Rec. 601 Color Transform
        data = np.dot(asarray(image)[...,:3], rgbweights)
        # data = asarray(image)
        return Wave2D(data).transpose() if transpose else Wave2D(data)
    @classmethod
    def fromtiles(cls,a,d,xgrid,ygrid,dx,dy=None):
        # e.g. a = [[0,0,0],
        #           [0,1,0],
        #           [2,2,2]], d={0:1.4,1:1.6,2:1.5}, xs=[-2,-1,1,2], ys=[-2,-1,0,1]
        assert len(ygrid)==1+len(a)
        assert len(xgrid)==1+len(a[0])
        assert all([axy in d for ay in a for axy in ay])
        dy = dy if dy is not None else dx
        x0,x1,y0,y1 = (xgrid[0],xgrid[-1],ygrid[0],ygrid[-1])
        from wavedata import wrange
        ww = Wave2D(xs=wrange(x0,x1,dx),ys=wrange(y0,y1,dy))
        xs = [xgrid[0]-dx] + list(xgrid[1:-1]) + [xgrid[-1]+dx]
        ys = [ygrid[0]-dy] + list(ygrid[1:-1]) + [ygrid[-1]+dy]
        return sum([d[axy] * ww.rectangle(x0,x1,y0,y1) for y0,y1,ay in zip(ys[:-1],ys[1:],a[::-1]) for x0,x1,axy in zip(xs[:-1],xs[1:],ay)])
    def xslab(self,x0,x1,n=1):
        from wavedata import tophat
        return n * tophat(self.xx,x0,x1,self.dx)
    def yslab(self,y0,y1,n=1):
        from wavedata import tophat
        return n * tophat(self.yy,y0,y1,self.dy)
    def xslabs(self,xs,ns,debug=False):
        assert len(ns)==1+len(xs)
        from wavedata import tophat
        x0s,x1s = [self.xmin-self.dx]+list(xs),list(xs)+[self.xmax+self.dx]
        if debug:
            Wave.plot(*[Wave( 2*i+tophat(self.xs,x0,x1,self.dx), self.xs, f'{x0:g} {x1:g}' ) for i,(x0,x1) in enumerate(zip(x0s,x1s))],m='o',grid=1)
        return sum([n * tophat(self.xx,x0,x1,self.dx) for x0,x1,n in zip(x0s,x1s,ns)])
    def yslabs(self,ys,ns):
        assert len(ns)==1+len(ys)
        from wavedata import tophat
        y0s,y1s = [self.ymin-self.dy]+list(ys),list(ys)+[self.ymax+self.dy] # print('y0s,y1s',y0s,y1s)
        # Wave.plots([Wave(n*tophat(self.ys,y0,y1,self.dy),self.ys,m='o') for y0,y1,n in zip(y0s,y1s,ns)])
        return sum([n * tophat(self.yy,y0,y1,self.dy) for y0,y1,n in zip(y0s,y1s,ns)])
    def circle(self,x0,y0,r,n=1):
        from wavedata import tophat
        return n * tophat(sqrt(self.xx**2+self.yy**2),-r,r,self.dx)
    def rectangle(self,x0,x1,y0,y1,n=1):
        return n * self.xslab(x0,x1) * self.yslab(y0,y1)
    def centeredrectangle(self,w,h,x0=0,y0=0,n=1):
        return self.rectangle(x0=x0-w/2,x1=x0+w/2,y0=y0-h/2,y1=y0+h/2,n=n)
    # def __array_finalize__(self, obj):
    #     if obj is None: return
    #     self.xs,self.ys = getattr(obj, 'xs', None),getattr(obj, 'ys', None)
    def __array_wrap__(self,out,context=None,return_scalar=None):
        if return_scalar:
            return out.item()
        if hasattr(self,'xs') and hasattr(self,'ys') and out.shape==(len(self.xs),len(self.ys)):
            # return Wave2D(out,self.xs,self.ys)
            return type(self)(out,self.xs,self.ys)
        return np.asarray(out)
    # def __reduce__(self):  # for pickling, https://stackoverflow.com/a/26599346
    #     pickled_state = super(Wave2D, self).__reduce__() # Get the parent's __reduce__ tuple
    #     new_state = pickled_state[2] + (self.xs,self.ys) # Create our own tuple to pass to __setstate__
    #     return (pickled_state[0], pickled_state[1], new_state) # Return a tuple that replaces the parent's __setstate__ tuple with our own
    # def __setstate__(self, state): # for pickling
    #     self.xs,self.ys = state[-2],state[-1]  # Set the info attribute
    #     super(Wave2D, self).__setstate__(state[0:-2]) # Call the parent's __setstate__ with the other tuple elements.
    def __reduce__(self):
        parent_reconstructor, parent_args, parent_state = super(Wave2D, self).__reduce__()
        extras = {k: v for k, v in self.__dict__.items() if k not in {"xs", "ys"}}
        new_state = parent_state + (self.xs, self.ys, extras)
        return (parent_reconstructor, parent_args, new_state)
    def __setstate__(self, state):
        if isinstance(state[-1], dict): # new pickles
            parent_state   = state[:-3]
            xs, ys, extras = state[-3:]
        else:                           # legacy old pickles
            parent_state = state[:-2]
            xs, ys       = state[-2:]
            extras       = {}
        super(Wave2D, self).__setstate__(tuple(parent_state))
        self.xs, self.ys = xs, ys
        self.__dict__.update(extras)
    def __getitem__(self, key):
        if key == () or (isinstance(key, (tuple, list)) and len(key) == 0):
            return np.ndarray.__getitem__(self, ()) # Let ndarray handle it so we get a Python scalar back
        if isinstance(key, tuple) or isinstance(key, list):
            kx,ky = key
            if isinstance(kx, slice) and isinstance(ky, slice):
                return Wave2D(self.np[kx,ky],xs=self.xs[kx],ys=self.ys[ky])
            if isinstance(kx, int) and isinstance(ky, slice):
                return self.atxindex(kx)[ky]
            if isinstance(kx, slice) and isinstance(ky, int):
                return self.atyindex(ky)[kx]
            return self.np[kx,ky]
        if isinstance(key, int) or isinstance(key, slice):
            return self[key,:]
        raise TypeError('Index must be int, not {}'.format(type(key).__name__))
    def __str__(self):
        return str(self.np).replace('\n','') + ' xs' + str(np.array2string(self.xs, threshold=9)) + ' ys' + str(np.array2string(self.ys, threshold=9))
    @property
    def nx(self):
        return len(self.xs)
    @property
    def ny(self):
        return len(self.ys)
    @property
    def xmin(self):
        return self.xs[0]
    @property
    def xmax(self):
        return self.xs[-1]
    @property
    def ymin(self):
        return self.ys[0]
    @property
    def ymax(self):
        return self.ys[-1]
    @property
    def dx(self):
        dxs = np.diff(self.xs)
        assert np.all(np.isclose(dxs, dxs[0])), 'inconsistent delta x'
        return dxs[0]
    @property
    def dy(self):
        dys = np.diff(self.ys)
        assert np.all(np.isclose(dys, dys[0])), 'inconsistent delta y'
        return dys[0]
    # def limits(self):
    #     return (self.xmin,self.xmax,self.ymin,self.ymax)
    # def res(self):
    #     return self.step()
    def bounds(self):
        return (self.xmin,self.xmax,self.ymin,self.ymax)
    def step(self):
        assert np.allclose(self.dx,self.dy), 'inconsistent step'
        return self.dx
    @property
    def xx(self): # xx wave is a Wave2D, so that e.g. 1+w.xx is a Wave2D
        yy,xx = np.meshgrid(self.ys,self.xs)
        return type(self)(xx,self.xs,self.ys)
    @property
    def yy(self):
        yy,xx = np.meshgrid(self.ys,self.xs)
        return type(self)(yy,self.xs,self.ys)
    def grid(self):
        yy,xx = np.meshgrid(self.ys,self.xs)
        return type(self)(xx,self.xs,self.ys),type(self)(yy,self.xs,self.ys)
    def array(self):
        return np.asarray(self)
    @property
    def np(self):
        return np.asarray(self)
    def copy(self):
        return type(self)(np.copy(self),xs=self.xs,ys=self.ys)
    def isreal(self,tol=1e-9):
        return np.all(abs(self.imag)<tol)
    def real(self):
        return type(self)(np.real(self.array()),xs=self.xs,ys=self.ys)
    def imag(self):
        return type(self)(np.imag(self.array()),xs=self.xs,ys=self.ys)
    def T(self,mirrorx=False,mirrory=False):
        return self.transpose(mirrorx=mirrorx,mirrory=mirrory)
    def transpose(self,mirrorx=False,mirrory=False):
        # transpose: reflection along line y=x, transpose then mirror: equivalent to rotating the matrix by ±90°
        # mirrorx: rotate +90° (ccw), mirrory: rotate -90° (cw)
        # coordinates are unchanged, i.e. object at (x0,y0) remains at (x0,y0) (but new axis goes + to -)
        xs = self.ys[::-1 if mirrorx else 1]
        ys = self.xs[::-1 if mirrory else 1]
        zs = np.copy(self).T
        zs = np.flip(zs,axis=0) if mirrorx else zs
        zs = np.flip(zs,axis=1) if mirrory else zs
        return type(self)(zs,xs=xs,ys=ys)
    def mirrorx(self):
        zs = np.flip(np.copy(self),axis=0)
        return type(self)(zs,xs=self.xs,ys=self.ys)
    def mirrory(self):
        zs = np.flip(np.copy(self),axis=1)
        return type(self)(zs,xs=self.xs,ys=self.ys)
    def issymmetric(self,atol=1e-4,horizontal=True):
        if not horizontal: raise NotImplementedError
        return np.allclose(0*self,self - self.mirrorx(),atol=atol)
    def isasymmetric(self,atol=1e-4,horizontal=True):
        if not horizontal: raise NotImplementedError
        return np.allclose(0*self,self + self.mirrorx(),atol=atol)
    def x2p(self,x,nearest=False):
        return len(self.xs) if x==inf else Wave(self.xs,self.xs).x2p(x,nearest)
    def y2p(self,y,nearest=False):
        return len(self.ys) if y==inf else Wave(self.ys,self.ys).x2p(y,nearest)
    def xsum(self):
        return Wave(self.sum(axis=0),xs=self.ys)
    def ysum(self):
        return Wave(self.sum(axis=1),xs=self.xs)
    def xlen(self):
        return len(self.xs)
    def ylen(self):
        return len(self.ys)
    def xsubrange(self,xlim=(-inf,inf)):
        # if xlim is None: return self
        i0,i1 = [self.x2p(x,nearest=True) for x in xlim] # print('xlim',*xlim,'i0,i1',i0,i1)
        return self[i0:i1,:]
    def ysubrange(self,ylim=(-inf,inf)):
        # if ylim is None: return self
        j0,j1 = [self.y2p(y,nearest=True) for y in ylim] # print('ylim',*ylim,'j0,j1',j0,j1)
        return self[:,j0:j1]
    def subrange(self,xlim=(-inf,inf),ylim=(-inf,inf)):
        return self.ysubrange(ylim).xsubrange(xlim)
    def xwaves(self):
        return [ self[:,j] for j in range(self.ny) ]
    def ywaves(self):
        return [ self[i,:] for i in range(self.nx) ]
    def waterfall(self,scale=1):
        dy = self.max()-self.min()
        ys = [ self[i,:]+i*dy for i in range(self.nx) ]
        return ys
    def addxwave(self,w):
        assert len(w)==self.nx
        self += w.y[:,np.newaxis]
        return self
    def addywave(self,w):
        assert len(w)==self.ny
        self += w.y
        return self
    def addcircle(self,x,y,r,z):
        self += z*((self.xx-x)**2+(self.yy-y)**2<r)
        return self
    def addrectangle(self,x,y,dx,dy,z):
        self += z*((self.xx-x)**2<dx**2/4)*((self.yy-y)**2<dy**2/4)
        return self
    def shiftx(self,x,fill=None):
        fill = fill if fill is not None else np.nan
        # ww = Wave2D(np.copy(self),xs=self.xs,ys=self.ys)
        ww = np.array(self)
        ww[:,:] = 0
        for j,y in enumerate(self.ys):
            xslice = self.atyindex(j)
            dp = round(int(x/xslice.dx()))
            ww[:,j] = xslice.offsetp(dp,fill)
        return Wave2D(ww,self.xs,self.ys)
        # for i,x in enumerate(self.xs):
        #     # dy = angle*(x-self.xmin)/(self.xmax-self.xmin)
        #     dy = angle*(x-0.5*(self.xmin+self.xmax))
        #     yslice = self.atxindex(i)
        #     dp = round(int(dy/yslice.dx()))
        #     ww[i,:] = yslice.offsetp(dp) if fast else yslice.offsetx(-dy)
    def untilt(self,n=1,debug=False):
        if n<=0: return (0,0)
        yhists = Wave([w.histogram(dx=0.001).maxloc() for w in self.xwaves()],self.ys)
        ya,yb = yhists.linefit(coef=1)
        yfit = yhists.linefit(coef=4)
        self.addywave(-yfit)
        xhists = Wave([w.histogram(dx=0.001).maxloc() for w in self.ywaves()],self.xs)
        xa,xb = xhists.linefit(coef=1)
        xfit = xhists.linefit(coef=4)
        self.addxwave(-xfit)
        if debug:
            print(n,xa,ya,xa+xa0,ya+ya0)
            Wave.plots(yhists,yfit,m=0)
            Wave.plots(xhists,xfit,m=0)
        xa0,ya0 = self.untilt(n-1)
        return xa+xa0,ya+ya0
    def subtractquadraticfit(self,x=True,y=True):
        if y:
            wy = self.xaverage().quadraticfit(a=0,coef=0) # Wave.plots( self.xaverage(), wy )
            self -= wy # correct for surface height vs y
        if x:
            wx = self.yaverage().quadraticfit(a=0,coef=0) # Wave.plots( self.yaverage(), wx )
            self -= wx.y[:,np.newaxis] # correct for surface height vs x
        return self
    def clip(self,y0,y1,clipval=None):
        if clipval is None:
            return np.clip(self,y0,y1)
        return np.where(self<y1, np.where(self>y0, self, clipval), clipval)
    def atxy(self,x,y,method='linear'): # returns z(x,y) at given (x,y)
        if 'linear'==method:
            if not hasattr(self,'_sl'):
                self._sl = scipy.interpolate.RegularGridInterpolator((self.xs,self.ys),self) # access using ss((x,y))
            xb,yb = np.clip(x,self.xs.min(),self.xs.max()),np.clip(y,self.ys.min(),self.ys.max()) # clip to value at boundary
            try:
                val = self._sl((xb,yb))
            except ValueError:
                val = nan
        elif 'nearest'==method:
            if not hasattr(self,'_sn'):
                self._sn = scipy.interpolate.RegularGridInterpolator((self.xs,self.ys),self,method='nearest')
            val = self._sn((x,y))
        elif 'cubic'==method: # can handle out of bounds, gives value at edge
            if not hasattr(self,'_ss'):
                # self._ss = scipy.interpolate.RectBivariateSpline(self.xs,self.ys,self) # access using ss(x,y,grid=False)
                if 1==len(self.xs) or 1==len(self.ys): return self.atxy(x,y,method='linear')
                kx,ky = np.clip(len(self.xs)-1,1,3),np.clip(len(self.ys)-1,1,3)
                self._ss = scipy.interpolate.RectBivariateSpline(self.xs,self.ys,self,kx=kx,ky=ky) # use less than cubic spline if not enough points
            val = self._ss(x,y,grid=False)
        else:
            assert 0,f'{method} not defined for wave2D.atxy'
        return val
        # val = scipy.interpolate.interp2d(self._xx, self._yy, self, bounds_error=False)(x,y) # don't use interp2d: https://stackoverflow.com/a/37872172
        # val = scipy.interpolate.interp2d(self.xs, self.ys, self.T, kind='linear', bounds_error=False)(x,y) # seems to give same answer, doesn't hang
    def __call__(self,x,y,method='linear'):
        return self.atxy(x,y,method)
    def atx(self,x,method=None):
        if method is None:
            # return y-slice of 2d array at given x value, rounding to nearest grid location
            return self.atxindex(np.abs(self.xs-x).argmin())
        return Wave( [self.atxy(x,y,method) for y in self.ys], self.ys )
    def aty(self,y,method=None):
        if method is None:
            # return x-slice of 2d array at given y value, rounding to nearest grid location
            return self.atyindex(np.abs(self.ys-y).argmin())
        return Wave( [self.atxy(x,y,method) for x in self.xs], self.xs )
    def atxindex(self,p): # return y-slice of 2d array at given x index
        return Wave(self.array()[p,:],self.ys)
    def atyindex(self,q): # return x-slice of 2d array at given y index
        return Wave(self.array()[:,q],self.xs)
    def maxloc(self):
        return self.pmax()
    def maxindex(self):
        return self.pmax()
    def xaverage(self): # average over x-axis, returns y-slice
        return Wave( np.nanmean(self,axis=0), self.ys )
    def yaverage(self): # average over ys, returns x-slice
        return Wave( np.nanmean(self,axis=1), self.xs )
    def diagonal(self,axis='x'): # diagonal values vs x
        assert self.shape[0] == self.shape[1], "Wave2D must be square for diagonal extraction"
        return Wave( np.diagonal(np.array(self)), self.xs if 'x'==axis else self.ys )
    def xslicemax(self): # x-slice thru 2d array max
        return self.atyindex(self.maxindex()[1])
    def xslice(self,x='abs'):
        assert x in ['abs','max','min'], 'x indexing not yet implemented'
        ww = self.abs() if 'abs'==x else self.max() if 'max'==x else self.min() if 'min'==x else self
        return self.atyindex(ww.maxindex()[1])
    def yslicemax(self): # y-slice thru 2d array max
        return self.atxindex(self.maxindex()[0])
    def yslice(self,y='abs'):
        assert y in ['abs','max','min'], 'y indexing not yet implemented'
        ww = self.abs() if 'abs'==y else self.max() if 'max'==y else self.min() if 'min'==y else self
        return self.atxindex(ww.maxindex()[0])
    def perimeter(self):
        def listends(A):
            return list(A[:1])+list(A[-1:])
        xedge0,xedge1 = listends(self.xwaves())
        yedge0,yedge1 = listends(self.ywaves())
        return Wave(list(xedge0)+list(yedge1)+list(xedge1[::-1])+list(yedge0[::-1]))
    def yskew(self,angle,fast=True):
        ww = np.array(self.xx)
        for i,x in enumerate(self.xs):
            # dy = angle*(x-self.xmin)/(self.xmax-self.xmin)
            dy = angle*(x-0.5*(self.xmin+self.xmax))
            yslice = self.atxindex(i)
            dp = round(int(dy/yslice.dx()))
            ww[i,:] = yslice.offsetp(dp) if fast else yslice.offsetx(-dy)
            if 0==i%100 and not fast: print(i)
        return Wave2D(ww,self.xs,self.ys)
    def flatten(self):
        return np.hstack(self)
    def sdev(self):
        return Wave(self.flatten()).sdev()
    def nanmin(self):
        return np.nanmin(self.flatten())
    def nanmax(self):
        return np.nanmax(self.flatten())
    def histogram(self,*args,**kwargs):
        return Wave(self.flatten().tolist()).histogram(*args,**kwargs)
    # def plot(self,colormesh=False,contour=None,contourf=False,levels=None,vmin=None,vmax=None,xlim=(None,None),ylim=(None,None),x='µm',y='µm',origin='lower',aspect=False,scale=1,colormap=None,pause=True,save=None,**kwargs):
    def plot(self,colormesh=None,contour=None,contourf=None,**kwargs):
        contour = contour if isinstance(contour,Wave2D) or isinstance(contour,np.ndarray) else (self if contour is not None else None)
        colormesh = self if colormesh is not None else None
        contourf = self if contourf is not None else None
        image = self if colormesh is None and contour is None and contourf is None else None
        import plot
        plot.plot(contour=contour,colormesh=colormesh,contourf=contourf,image=image,**kwargs)
        return self
    def pmax(self):
        return np.unravel_index(self.argmax(), self.shape)
    def pmin(self):
        return np.unravel_index(self.argmin(), self.shape)
    def xymax(self):
        return self.xx[self.pmax()],self.yy[self.pmax()]
    def xymin(self):
        return self.xx[self.pmin()],self.yy[self.pmin()]
    def reduced(self,size=5):
        def subindices(zs): # indices of undersampled grid, first and last points are at grid edge
            return np.array([int(round(i/(size-1)*(len(zs)-1))) for i in range(size)])
        xg,yg = subindices(self.xs),subindices(self.ys)
        return Wave2D(self[xg[:,np.newaxis],yg[np.newaxis,:]], xs=self.xs[xg], ys=self.ys[yg])
        # dx,dy = len(self.xs)//size,len(self.ys)//size
        # return Wave2D(self[::dx,::dy], xs=self.xs[::dx], ys=self.ys[::dy])
    def findguide(self,width,filterwidth=None,xprop=True):
        filterwidth = filterwidth if filterwidth is not None else width
        def guidefilter(width,dx):
            n0 = round(int((width+filterwidth)/dx/2))
            xs = dx*np.linspace(-n0,n0,2*n0+1)
            return Wave( +1*(np.abs(2*xs)<width) + -1*(width<np.abs(2*xs)),xs)
        filter = guidefilter(width,self.dy)
        w = self.xaverage() if xprop else self.yaverage()
        return w.convolve(filter,abs=True) # returns convolution wave
    def downsampley(self,n=None,numsections=None,partial=True):
        return self.T().downsamplex(n=n,numsections=numsections,partial=partial)
    def downsamplex(self,n=None,numsections=None,partial=True):
        assert (n is None or numsections is None) and not (n is None and numsections is None)
        n = self.nx//numsections if numsections else n
        if n<2:
            return self
        def subsection(i):
            return self[i*n:(i+1)*n,:]
        def xavg(i):
            return np.mean(self.xs[i*n:(i+1)*n])
        # subsection(0).plot()
        # subsection(0).xaverage().plot()
        ws = [subsection(k//n).xaverage() for k in range(self.nx)[::(n if partial or 0==self.nx%n else n-1)]]
        xs = [xavg(k//n) for k in range(self.nx)[::(n if partial or 0==self.nx%n else n-1)]]
        return ws,xs

        # ww = np.array(self.xx())
        # for i,x in enumerate(self.xs):
        #     # dy = angle*(x-self.xmin)/(self.xmax-self.xmin)
        #     dy = angle*(x-0.5*(self.xmin+self.xmax))
        #     yslice = self.atxindex(i)
        #     dp = round(int(dy/yslice.dx()))
        #     ww[i,:] = yslice.offsetp(dp) if fast else yslice.offsetx(dy)

        # ys = [np.mean(np.array(self[i*n:i*n+n])) for i in range(len(self)//n)]
        # xs = [np.mean(np.array(self.index[i*n:i*n+n])) for i in range(len(self)//n)]
        # return Wave(ys,xs,name=self.name)

    def normalize(self,c=1):
        return c*self/self.nanmax()
    def volume(self):
        return np.sum(self)*self.dx*self.dy
    def norm(self):
        return np.sum(self**2)*self.dx*self.dy
    def abs(self):
        return np.abs(self)
    def sqr(self):
        return self.magsqr()
    def magsqr(self):
        return Wave2D(self.real()**2 + self.imag()**2,xs=self.xs,ys=self.ys)
    def schmidtnumber(self,res=400,keep=None,invspace=False):
        modes = [m for m,f,g in self.schmidtdecomposition(res=res,keep=keep,invspace=invspace)]
        return 1/sum([np.abs(m)**4 for m in modes])
    def schmidtdecomposition(ww,res=400,keep=None,invspace=False):
        import pyqentangle # don't use pyqentangle==3.1.7 (3.1.0 is ok)
        f,x0,x1,y0,y1 = ww, ww.xs[0], ww.xs[-1], ww.ys[0], ww.ys[-1]
        if invspace: f,x0,x1,y0,y1 = lambda x,y:ww(1/x,1/y), 1/ww.xs[-1], 1/ww.xs[0], 1/ww.ys[-1], 1/ww.ys[0]
        return pyqentangle.continuous_schmidt_decomposition(f,x0,x1,y0,y1, nb_x1=res, nb_x2=res, keep=keep)
    def indistinguishability(self): # https://arxiv.org/abs/2107.08070 Eq. 10
        if not np.allclose(self.xs,self.ys):
            return np.nan
        return np.sum(self * self.transpose().conj())/np.sum(self * self)
    def integrate(f,g): # computes ∫ f(x,y) g(y,z) dy
        assert f.dy==g.dx and np.allclose(f.ys,g.xs), f"f.dy:{f.dy:g} g.dx:{g.dx:g}"
        return type(f)(g.dx * np.array(f) @ np.array(g),xs=f.xs,ys=g.ys)
    def __matmul__(self,w):
        if isinstance(w,Wave2D):
            return self.integrate(w)
        return self @ w
    def overlap(self,ww):
        # overlap integral of both assuming they are e-fields
        assert (np.allclose(self.bounds(),ww.bounds(),rtol=1e-50) and 
                np.isclose(self.dx,ww.dx,rtol=1e-50) and 
                np.isclose(self.dy,ww.dy,rtol=1e-50) and 
                self.shape==ww.shape), f'overlap for different grid sizes not implemented: bounds():{self.bounds()},{ww.bounds()},dx:{self.dx},{ww.dx},dy:{self.dy},{ww.dy},shape:{self.shape},{ww.shape}'
        # return (np.sum(self*ww)*self.dx*self.dy)**2/self.norm()/ww.norm()
        # return np.sum(self*ww)**2/np.sum(self**2)/np.sum(ww**2)
        np.seterr(divide='ignore', invalid='ignore')
        return abs(np.sum(self*ww.conj()))**2 / np.sum(abs(self)**2) / np.sum(abs(ww)**2) # https://www.rp-photonics.com/mode_matching.html
    def amplitudeoverlap(self,ww): # suitable for decomposition
        return np.sum(self*ww.conj()) / sqrt( np.sum(abs(self)**2) * np.sum(abs(ww)**2) ) # https://www.rp-photonics.com/mode_matching.html
    def deadoverlap(self,w1,w2,w3):
        assert self.bounds()==w1.bounds()==w2.bounds()==w3.bounds() and self.dx==w1.dx==w2.dx==w3.dx and self.dy==w1.dy==w2.dy==w3.dy and self.shape==w1.shape==w2.shape==w3.shape, 'overlap for different grid sizes not implemented'
        # self = mask, 1 everywhere except 0 in dead region
        overlapintegral = self.dx*self.dy*np.sum(w1*w2*w3*self)
        norm1, norm2, norm3 = self.dx*self.dy*np.sum(w1**2), self.dx*self.dy*np.sum(w2**2), self.dx*self.dy*np.sum(w3**2)
        return norm1*norm2*norm3/overlapintegral**2
    def overlaparea(self,ww,ww2):
        assert self.bounds()==ww.bounds()==ww2.bounds() and self.dx==ww.dx==ww2.dx and self.dy==ww.dy==ww2.dy and self.shape==ww.shape==ww2.shape, 'overlap for different grid sizes not implemented'
        return self.dx*self.dy*np.sum(self**2)*np.sum(ww**2)*np.sum(ww2**2)/np.sum(self*ww*ww2)**2
    def propagatelightpipes(self,λ,z): # assumes xs,ys in µm, λ in nm, z in mm
        import LightPipes as lp # lightpipes units in m
        assert len(self.xs)==len(self.ys), 'LightPipes uses square grid'
        N = len(self.xs)
        F = lp.Begin(1e-6*self.dx*N,1e-9*λ,N=N) # print('F.siz,F.lam,F.N',F.siz,F.lam,F.N) # F.field
        F = lp.SubIntensity(F,self.magsqr())
        F = lp.SubPhase(F,np.angle(self.array()))
        F = lp.Fresnel(1e-3*z,F) # I = lp.Intensity(F);Wave(I[N//2],self.xs).plot()
        return Wave2D(F.field,xs=self.xs,ys=self.ys)
    def propagatediffractio(self,λ,z): # assumes xs,ys in µm, λ in nm, z in mm
        from diffractio.scalar_sources_XY import Scalar_source_XY # diffractio units in µm
        uu = Scalar_source_XY(x=self.xs, y=self.ys, wavelength=1e-3*λ)
        uu.u += self.array()
        vv = uu.RS(z=1e3*z)
        return Wave2D(vv.u,xs=self.xs,ys=self.ys)
    def lensdiffractio(self,λ,f): # assumes xs,ys in µm, λ in nm, f in mm
        from diffractio.scalar_sources_XY import Scalar_source_XY # diffractio units in µm
        from diffractio.scalar_masks_XY import Scalar_mask_XY # diffractio units in µm
        # lens_spherical, aspheric, fresnel_lens also available

        tt = Scalar_mask_XY(x=self.xs, y=self.ys, wavelength=1e-3*λ)
        tt.lens(r0=(0,0), focal=1e3*f, radius=0, angle=0)
        return self * tt.u

        uu = Scalar_source_XY(x=self.xs, y=self.ys, wavelength=1e-3*λ)
        uu.u += self.array()
        vv = uu * tt
        return Wave2D(vv.u,xs=self.xs,ys=self.ys)

        # u1 = Scalar_source_XY(x=x0, y=y0, wavelength=wavelength)
        # u1.plane_wave(A=1)
        # t1 = Scalar_mask_XY(x=x0, y=y0, wavelength=wavelength)
        # t1.slit(x0=0, size=50 * um, angle=0 * degrees)
        # u2 = u1 * t1

    def contours(self,val=None,singlewave=False):
        val = val if val is not None else 0.5*(self.min()+self.max())
        from skimage import measure
        contours = measure.find_contours(self, val, fully_connected='low', positive_orientation='low')
        px,py = np.arange(len(self.xs)),np.arange(len(self.ys))
        def contour2wave(contour):
            cx = np.interp(contour[:,0],px,self.xs) # convert array index to units
            cy = np.interp(contour[:,1],py,self.ys)
            return Wave(cy,cx)
        def contours2contour(contours): # convert to a single contour separated by nans
            c = contours[0] if contours else np.array([[nan,nan]])
            for contour in contours[1:]:
                c = np.concatenate((c,[[nan,nan]],contour))
            return c
        if singlewave: 
            return contour2wave(contours2contour(contours))
        return [contour2wave(contour) for contour in contours]
    def contour(self,val=None):
        return self.contours(val,singlewave=True)
    def replacenanwithmedian(self,size=3):
        ww = self.smoothed(smoothsize=size,medfilt=True)
        return Wave2D(np.where(np.isnan(self),ww,self), xs=self.xs, ys=self.ys)
    def smoothed(self,smoothsize=1,medfilt=False):
        if medfilt:
            # from scipy.signal import medfilt
            # return Wave2D( medfilt(self, kernel_size=smoothsize), xs=self.xs, ys=self.ys )
            from scipy.ndimage import median_filter
            return Wave2D( median_filter(self,size=(smoothsize,smoothsize),mode='nearest'), xs=self.xs, ys=self.ys )
        from scipy.ndimage.filters import gaussian_filter
        return Wave2D( gaussian_filter(self,sigma=smoothsize,mode='constant',cval=0.), xs=self.xs, ys=self.ys )
    def convolve(self,filter=None):
        from scipy import signal
        filter = filter if filter is not None else np.array(
            [[ -3, 0,  +3],
             [-10, 0, +10],
             [ -3, 0,  +3]]) # vertical edge filter
        out = signal.convolve2d(self, filter, boundary='symm', mode='same')
        return Wave2D(out,xs=self.xs,ys=self.ys)
    def localmax(self,size=3,smoothsize=0): # example at https://stackoverflow.com/a/3689710
        from scipy.ndimage.filters import maximum_filter,gaussian_filter
        smoothed = gaussian_filter(self,sigma=smoothsize,mode='constant',cval=0.) if smoothsize else self
        localmax2D = maximum_filter(smoothed, size=size, mode='constant')==smoothed
        mx,my = localmax2D.nonzero()
        # print(mx,my,all(mx>size),all(my>size),all(mx<self.shape[0]-1-size),all(my<self.shape[1]-1-size))
        # assert all(mx>size) and all(my>size) and all(mx<self.shape[0]-1-size) and all(my<self.shape[1]-1-size), 'local max found near edge, use larger size?'+str(mx)+str(my)
        return list(mx),list(my)
    def csv(self,save,gridinfo=True):
        s = f', gridx=({self.xmin:g},{self.xmax:g},{self.dx:g}), gridy=({self.ymin:g},{self.ymax:g},{self.dy:g})' if gridinfo else ''
        np.savetxt(save.replace('.csv','')+s+'.csv', self, delimiter=',')

class Mesh2D(np.ndarray):
    # Mesh2D is a Wave2D with values at cell centers instead of cell vertices
    def __new__(cls,array,xs=None,ys=None,names=None,toptobottom=True):
        def string2grid(s):
            seps = set([c for c in s if c not in '.0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ'])
            if 0==len(seps):
                assert len(s)==(len(xs)-1)*(len(ys)-1), f'array size mismatch, (nx-1)(ny-1)={len(xs)-1}x{len(ys)-1}!={len(s)}'
                return [list(map(int,s[i*(len(xs)-1):(i+1)*(len(xs)-1)])) for i in range(len(ys)-1)]
            assert 1==len(seps), f"expected one row separator, got {seps}"
            # return [list(map(int,c)) for c in s.split(list(seps)[0])]
            return [[nan if '.'==ci else ord(ci)-ord('0') for ci in c] for c in s.split(list(seps)[0])]
        array = string2grid(array) if isinstance(array,str) else array
        array = array[::-1] if toptobottom else array
        array = np.array(array) if isinstance(array,(list,tuple)) else array
        ys,xs = (np.arange(len(array)+1),np.arange(len(array[0])+1)) if xs is None else (ys,xs)
        nx,ny = len(xs)-1,len(ys)-1
        assert array.size==nx*ny and array.shape==(ny,nx), f'input array must have size nx*ny and shape (ny,nx), ny,nx:{ny},{nx}, shape:{array.shape}, size:{array.size}'
        obj = np.asarray(array).view(cls)
        obj.xs,obj.ys = np.asarray(xs,dtype=float),np.asarray(ys,dtype=float)
        obj.names = names if isinstance(names,dict) else {i:s for i,s in zip(obj.values(),names)} if names else {}
        return obj
    def __array_wrap__(self,out,context=None):
        if hasattr(self,'xs') and hasattr(self,'ys') and out.shape==(len(self.ys)-1,len(self.xs)-1):
            return type(self)(out,self.xs,self.ys)
        return np.asarray(out)
    def __reduce__(self):
        pickled_state = super(Mesh2D, self).__reduce__()
        new_state = pickled_state[2] + (self.xs,self.ys,self.names)
        return (pickled_state[0], pickled_state[1], new_state)
    def __setstate__(self, state):
        self.xs,self.ys,self.names = state[-3],state[-2],state[-1] 
        super(Mesh2D, self).__setstate__(state[0:-3])
    def values(self):
        return sorted(set(self.flatten()))
    def nameslist(self):
        return [self.names[i] for i in self.values()]
    def celshade(self,outeredges=False):
        a,xs,ys = self,self.xs,self.ys
        nx,ny = len(xs)-1,len(ys)-1
        assert len(a)==ny and len(a[0])==nx, f'array size mismatch, {len(a)}x{len(a[0])}!={ny}x{nx}'
        def hedge(i,j): # edge between horizontal cells (i,j) and (i+1,j)
            x,y0,y1 = xs[i+1],ys[j],ys[j+1]
            return (x,y0,x,y1)
        def vedge(i,j): # edge between vertical cells (i,j) and (i,j+1)
            x0,x1,y = xs[i],xs[i+1],ys[j+1]
            return (x0,y,x1,y)
        # hs = [hedge(i,j) for i in range(nx-1) for j in range(ny) if a[j][i]!=a[j][i+1]]
        # vs = [vedge(i,j) for i in range(nx) for j in range(ny-1) if a[j][i]!=a[j+1][i]]
        # x0,x1,y0,y1 = xs[0],xs[-1],ys[0],ys[-1]
        # bs = [(x0,y0,x1,y0),(x0,y1,x1,y1),(x0,y0,x0,y1),(x1,y0,x1,y1)]
        def issame(x,y):
            return (np.isnan(x) and np.isnan(y)) or x==y
        hs = [hedge(i,j) for i in range(nx-1) for j in range(ny) if not issame(a[j][i],a[j][i+1])]
        vs = [vedge(i,j) for i in range(nx) for j in range(ny-1) if not issame(a[j][i],a[j+1][i])]
        bs = ([vedge(i,j) for i in range(nx) for j in [-1] if not np.isnan(a[j+1][i])]
            + [vedge(i,j) for i in range(nx) for j in [ny-1] if not np.isnan(a[j][i])]
            + [hedge(i,j) for j in range(ny) for i in [-1] if not np.isnan(a[j][i+1])]
            + [hedge(i,j) for j in range(ny) for i in [nx-1] if not np.isnan(a[j][i])])
        return Wave.fromsegs(hs+vs+bs) if outeredges else Wave.fromsegs(hs+vs)
    def wireframe(self,*args,**kwargs):
        return self.celshade(*args,**kwargs)
    def celplot(self,*args,contour=None,outlinelinewidth=1,**kwargs):
        return self.plot(*args,contour=contour,outlinelinewidth=outlinelinewidth,**kwargs)
    def plot(self,*args,contour=None,outlinelinewidth=None,outeredges=0,**kwargs):
        import plot
        ws = [self.celshade(outeredges=outeredges).sp(c='k',lw=outlinelinewidth)] if outlinelinewidth else []
        plot.plot(waves=ws,contour=contour,colormesh=self,framelw=outlinelinewidth,capstyle='projecting',**kwargs)
        return self

if __name__ == '__main__':
    m = Mesh2D('000 010 111 121 222',[-3,-1,1,3],[0,1,2,3,4,5],names='ABC')
    print(m)
    print(m.xs)
    print(m.ys)
    print(m.shape)
    print('m.values',m.values())
    print('m.nameslist',m.nameslist())
    print('m.names',m.names)
    # m.plot()
    m.celplot()

    # import numpy as np
    # import matplotlib.pyplot as plt 
    # import matplotlib.patches as mpatches
    # vmin,vmax = 0,3
    # data = np.random.randint(vmin, vmax+1, size=(4,6))
    # fig, ax = plt.subplots()
    # pcm = plt.pcolormesh(data, cmap='viridis', vmin=vmin, vmax=vmax)

    # legend_elements = [mpatches.Rectangle((0, 0), 1, 1, facecolor=plt.cm.viridis(i/(vmax-vmin)), edgecolor='black', label=str(i)) for i in range(vmin,vmax+1)]
    # ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    # # plt.colorbar(pcm)
    # plt.tight_layout()
    # plt.show()