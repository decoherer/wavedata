
import pandas as pd
import numpy as np
from numpy import sqrt,pi,nan,inf,sign,exp,log,sin,cos,floor,ceil
import scipy
from waveclass import Wave

class Wave2D(np.ndarray):
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
    def __new__(cls,array=None,xs=None,ys=None):
        array = np.array(array) if isinstance(array,(list,tuple)) else array
        if xs is not None and ys is not None:
            nx,ny = len(xs),len(ys)
            if array is not None:
                if array.size==nx*ny:
                    array = array.reshape((nx,ny))
                assert (nx,ny)==array.shape, f'input array must have size (nx,ny), array:{array} nx,ny:{nx,ny} '
        else:
            nx,ny = array.shape
            xs,ys = np.linspace(0,nx-1,nx),np.linspace(0,ny-1,ny)
        if array is not None:
            # obj = np.asarray(array,dtype=float).view(cls)
            obj = np.asarray(array).view(cls)
        else:
            obj = np.zeros((nx,ny)).view(cls)
        obj.xs,obj.ys = np.asarray(xs,dtype=float),np.asarray(ys,dtype=float)
        obj.nx,obj.ny = obj.shape
        return obj
    # def __array_finalize__(self, obj):
    #     if obj is None: return
    #     self.xs,self.ys = getattr(obj, 'xs', None),getattr(obj, 'ys', None)
    def __array_wrap__(self,out,context=None):
        if hasattr(self,'xs') and hasattr(self,'ys') and out.shape==(len(self.xs),len(self.ys)):
            return Wave2D(out,self.xs,self.ys)
        return np.asarray(out)
    def __reduce__(self): # https://stackoverflow.com/a/26599346
        pickled_state = super(Wave2D, self).__reduce__() # Get the parent's __reduce__ tuple
        new_state = pickled_state[2] + (self.xs,self.ys) # Create our own tuple to pass to __setstate__
        return (pickled_state[0], pickled_state[1], new_state) # Return a tuple that replaces the parent's __setstate__ tuple with our own
    def __setstate__(self, state):
        self.xs,self.ys = state[-2],state[-1]  # Set the info attribute
        super(Wave2D, self).__setstate__(state[0:-2]) # Call the parent's __setstate__ with the other tuple elements.
    def __getitem__(self, key):
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
    def limits(self):
        return (self.xmin,self.xmax,self.ymin,self.ymax)
    @property
    def res(self):
        dxs,dys = np.diff(self.xs),np.diff(self.ys)
        return self.xs[1]-self.xs[0] if (np.all(dxs==dxs[0]) and np.all(dys==dys[0]) and dxs[0]==dys[0]) else None
    @property
    def dx(self):
        # assert np.equal.reduce( np.diff(self.xs) ), 'inconsistent delta x'
        return self.xs[1]-self.xs[0]
    @property
    def dy(self):
        # assert np.equal.reduce( np.diff(self.ys) ), 'inconsistent delta y'
        return self.ys[1]-self.ys[0]
    @property
    def xx(self): # xx wave is a Wave2D, so that e.g. 1+w.xx is a Wave2D
        yy,xx = np.meshgrid(self.ys,self.xs)
        return Wave2D(xx,self.xs,self.ys)
    @property
    def yy(self):
        yy,xx = np.meshgrid(self.ys,self.xs)
        return Wave2D(yy,self.xs,self.ys)
    def grid(self):
        yy,xx = np.meshgrid(self.ys,self.xs)
        return Wave2D(xx,self.xs,self.ys),Wave2D(yy,self.xs,self.ys)
    def array(self):
        return np.asarray(self)
    @property
    def np(self):
        return np.asarray(self)
    def copy(self):
        return Wave2D(np.copy(self),xs=self.xs,ys=self.ys)
    def isreal(self,tol=1e-9):
        return np.all(abs(self.imag)<tol)
    def real(self):
        return Wave2D(np.real(self.array()),xs=self.xs,ys=self.ys)
    def imag(self):
        return Wave2D(np.imag(self.array()),xs=self.xs,ys=self.ys)
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
        return Wave2D(zs,xs=xs,ys=ys)
    def mirrorx(self):
        zs = np.flip(np.copy(self),axis=0)
        return Wave2D(zs,xs=self.xs,ys=self.ys)
    def mirrory(self):
        zs = np.flip(np.copy(self),axis=1)
        return Wave2D(zs,xs=self.xs,ys=self.ys)
    def issymmetric(self,horizontal=True,atol=1e-4):
        if not horizontal: raise NotImplementedError
        return np.allclose(0*self,self - self.mirrorx(),atol=atol)
    def isasymmetric(self,horizontal=True,atol=1e-4):
        if not horizontal: raise NotImplementedError
        return np.allclose(0*self,self + self.mirrorx(),atol=atol)
    def x2p(self,x,nearest=False):
        return len(self.xs) if x==inf else Wave(index=self.xs).x2p(x,nearest)
    def y2p(self,y,nearest=False):
        return len(self.ys) if y==inf else Wave(index=self.ys).x2p(y,nearest)
    def xsum(self):
        return Wave(self.sum(axis=0),index=self.ys)
    def ysum(self):
        return Wave(self.sum(axis=1),index=self.xs)
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
            dp = rint(x/xslice.dx())
            ww[:,j] = xslice.offsetp(dp,fill)
        return Wave2D(ww,self.xs,self.ys)
        # for i,x in enumerate(self.xs):
        #     # dy = angle*(x-self.xmin)/(self.xmax-self.xmin)
        #     dy = angle*(x-0.5*(self.xmin+self.xmax))
        #     yslice = self.atxindex(i)
        #     dp = rint(dy/yslice.dx())
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
    def maxindex(self):
        return np.unravel_index(self.argmax(), self.shape)
    def xaverage(self): # average over x-axis, returns y-slice
        return Wave( np.nanmean(self,axis=0), self.ys )
    def yaverage(self): # average over ys, returns x-slice
        return Wave( np.nanmean(self,axis=1), self.xs )
    def xslicemax(self): # x-slice thru 2d array max
        return self.atyindex(self.maxindex()[1])
    def yslicemax(self): # y-slice thru 2d array max
        return self.atxindex(self.maxindex()[0])
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
            dp = rint(dy/yslice.dx())
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
            n0 = rint((width+filterwidth)/dx/2)
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
        #     dp = rint(dy/yslice.dx())
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
    def indistiguishability(self): # https://arxiv.org/abs/2107.08070 Eq. 10
        assert np.allclose(self.xs,self.ys)
        return np.sum(self * self.transpose().conj())/np.sum(self * self)
    def overlap(self,ww):
        # overlap integral of both assuming they are e-fields
        assert (np.allclose(self.limits,ww.limits,rtol=1e-50) and 
                np.isclose(self.dx,ww.dx,rtol=1e-50) and 
                np.isclose(self.dy,ww.dy,rtol=1e-50) and 
                self.shape==ww.shape), f'overlap for different grid sizes not implemented: limits:{self.limits},{ww.limits},dx:{self.dx},{ww.dx},dy:{self.dy},{ww.dy},shape:{self.shape},{ww.shape}'
        # return (np.sum(self*ww)*self.dx*self.dy)**2/self.norm()/ww.norm()
        # return np.sum(self*ww)**2/np.sum(self**2)/np.sum(ww**2)
        return abs(np.sum(self*ww.conj()))**2 / np.sum(abs(self)**2) / np.sum(abs(ww)**2) # https://www.rp-photonics.com/mode_matching.html
    def deadoverlap(self,w1,w2,w3):
        assert self.limits==w1.limits==w2.limits==w3.limits and self.dx==w1.dx==w2.dx==w3.dx and self.dy==w1.dy==w2.dy==w3.dy and self.shape==w1.shape==w2.shape==w3.shape, 'overlap for different grid sizes not implemented'
        # self = mask, 1 everywhere except 0 in dead region
        overlapintegral = self.dx*self.dy*np.sum(w1*w2*w3*self)
        norm1, norm2, norm3 = self.dx*self.dy*np.sum(w1**2), self.dx*self.dy*np.sum(w2**2), self.dx*self.dy*np.sum(w3**2)
        return norm1*norm2*norm3/overlapintegral**2
    def overlaparea(self,ww,ww2):
        assert self.limits==ww.limits==ww2.limits and self.dx==ww.dx==ww2.dx and self.dy==ww.dy==ww2.dy and self.shape==ww.shape==ww2.shape, 'overlap for different grid sizes not implemented'
        return self.dx*self.dy*np.sum(self**2)*np.sum(ww**2)*np.sum(ww2**2)/np.sum(self*ww*ww2)**2
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
            c = contours[0]
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

