
import pandas as pd
import numpy as np
from numpy import sqrt,pi,nan,inf,sign,exp,log,sin,cos,floor,ceil
import scipy
from waveclass import Wave
from wave2d import Wave2D

class Wave3D(np.ndarray):
    def __new__(cls,array=None,xs=None,ys=None,zs=None):
        array = np.array(array) if isinstance(array,(list,tuple)) else array
        if xs is not None and ys is not None:
            nx,ny,nz = len(xs),len(ys),len(zs)
            if array is not None:
                if array.size==nx*ny*nz:
                    array = array.reshape((nx,ny,nz))
                assert (nx,ny,nz)==array.shape, f'input array must have size (nx,ny,nz), array:{array} nx,ny,nz:{nx,ny,nz} '
        else:
            nx,ny,nz = array.shape
            xs,ys,zs = np.linspace(0,nx-1,nx),np.linspace(0,ny-1,ny),np.linspace(0,nz-1,nz)
        if array is not None:
            obj = np.asarray(array).view(cls)
        else:
            obj = np.zeros((nx,ny,nz)).view(cls)
        obj.xs,obj.ys,obj.zs = np.asarray(xs,dtype=float),np.asarray(ys,dtype=float),np.asarray(zs,dtype=float)
        obj.nx,obj.ny,obj.nz = obj.shape
        return obj
    def __array_wrap__(self,out,context=None):
        if hasattr(self,'xs') and hasattr(self,'ys') and hasattr(self,'zs') and out.shape==(len(self.xs),len(self.ys),len(self.zs)):
            return type(self)(out,self.xs,self.ys,self.zs)
        return np.asarray(out)
    def __reduce__(self):  # for pickling, https://stackoverflow.com/a/26599346
        pickled_state = super(Wave2D, self).__reduce__() # Get the parent's __reduce__ tuple
        new_state = pickled_state[2] + (self.xs,self.ys) # Create our own tuple to pass to __setstate__
        return (pickled_state[0], pickled_state[1], new_state) # Return a tuple that replaces the parent's __setstate__ tuple with our own
    def __setstate__(self, state): # for pickling
        self.xs,self.ys = state[-2],state[-1]  # Set the info attribute
        super(Wave2D, self).__setstate__(state[0:-2]) # Call the parent's __setstate__ with the other tuple elements.
    def __getitem__(self, key):
        assert len(key)==3, 'Wave3D key must be a 3-tuple'
        kx,ky,kz = key
        slicecount = isinstance(kx, slice) + isinstance(ky, slice) + isinstance(kz, slice)
        if 3==slicecount:
            return Wave3D(self.np[kx,ky,kz],xs=self.xs[kx],ys=self.ys[ky],zs=self.zs[kz])
        if 2==slicecount:
            # assert 0, 'return Wave2D slice not implemented yet'
            if isinstance(kx, int):
                return self.atxindex(kx)
            if isinstance(ky, int):
                return self.atyindex(ky)
            if isinstance(kz, int):
                return self.atzindex(kz)
        if 1==slicecount:
            assert 0, 'return Wave slice not implemented yet'
            # return self.atxindex(kx)[ky]
        if 0==slicecount:
            return self.np[kx,ky,kz]
        assert 0
    def __str__(self):
        return str(self.np).replace('\n','') + ' xs' + str(np.array2string(self.xs, threshold=9)) + ' ys' + str(np.array2string(self.ys, threshold=9)) + ' zs' + str(np.array2string(self.zs, threshold=9))
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
    def zmin(self):
        return self.zs[0]
    @property
    def zmax(self):
        return self.zs[-1]
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
    @property
    def dz(self):
        dzs = np.diff(self.zs)
        assert np.all(np.isclose(dzs, dzs[0])), 'inconsistent delta z'
        return dzs[0]
    def bounds(self):
        return (self.xmin,self.xmax,self.ymin,self.ymax,self.zmin,self.zmax)
    def step(self):
        assert np.allclose(self.dx,self.dy,self.dz), 'inconsistent step'
        return self.dx
    @property
    def xx(self): # xx wave is a Wave3D, so that e.g. 1+w.xx is a Wave3D
        xx,yy,zz = np.meshgrid(self.xs,self.ys,self.zs)
        return type(self)(xx,self.xs,self.ys,self.zs)
    @property
    def yy(self):
        xx,yy,zz = np.meshgrid(self.xs,self.ys,self.zs)
        return type(self)(yy,self.xs,self.ys,self.zs)
    @property
    def zz(self):
        xx,yy,zz = np.meshgrid(self.xs,self.ys,self.zs)
        return type(self)(zz,self.xs,self.ys,self.zs)
    def grid(self):
        xx,yy,zz = np.meshgrid(self.xs,self.ys,self.zs)
        return type(self)(xx,self.xs,self.ys,self.zs),type(self)(yy,self.xs,self.ys,self.zs),type(self)(zz,self.xs,self.ys,self.zs)
    def array(self):
        return np.asarray(self)
    @property
    def np(self):
        return np.asarray(self)
    def copy(self):
        return type(self)(np.copy(self),xs=self.xs,ys=self.ys,zs=self.zs)
    def isreal(self,tol=1e-9):
        return np.all(abs(self.imag)<tol)
    def real(self):
        return type(self)(np.real(self.array()),xs=self.xs,ys=self.ys,zs=self.zs)
    def imag(self):
        return type(self)(np.imag(self.array()),xs=self.xs,ys=self.ys,zs=self.zs)
    def mirrorx(self):
        zs = np.flip(np.copy(self),axis=0)
        return type(self)(zs,xs=self.xs,ys=self.ys,zs=self.zs)
    def mirrory(self):
        zs = np.flip(np.copy(self),axis=1)
        return type(self)(zs,xs=self.xs,ys=self.ys,zs=self.zs)
    def mirrorz(self):
        zs = np.flip(np.copy(self),axis=2)
        return type(self)(zs,xs=self.xs,ys=self.ys,zs=self.zs)
    def x2p(self,x,nearest=False):
        return len(self.xs) if x==inf else Wave(index=self.xs).x2p(x,nearest)
    def y2p(self,y,nearest=False):
        return len(self.ys) if y==inf else Wave(index=self.ys).x2p(y,nearest)
    def z2p(self,z,nearest=False):
        return len(self.zs) if z==inf else Wave(index=self.zs).x2p(z,nearest)
    def xsum(self):
        return Wave(self.sum(axis=0),index=self.ys)
    def ysum(self):
        return Wave(self.sum(axis=1),index=self.xs)
    def zsum(self):
        return Wave(self.sum(axis=2),index=self.zs)
    def xlen(self):
        return len(self.xs)
    def ylen(self):
        return len(self.ys)
    def zlen(self):
        return len(self.zs)
    def xsubrange(self,xlim=(-inf,inf)):
        # if xlim is None: return self
        i0,i1 = [self.x2p(x,nearest=True) for x in xlim] # print('xlim',*xlim,'i0,i1',i0,i1)
        return self[i0:i1,:,:]
    def ysubrange(self,ylim=(-inf,inf)):
        # if ylim is None: return self
        j0,j1 = [self.y2p(y,nearest=True) for y in ylim] # print('ylim',*ylim,'j0,j1',j0,j1)
        return self[:,j0:j1,:]
    def zsubrange(self,zlim=(-inf,inf)):
        k0,k1 = [self.z2p(z,nearest=True) for z in zlim]
        return self[:,:,k0:k1]
    def subrange(self,xlim=(-inf,inf),ylim=(-inf,inf),zlim=(-inf,inf)):
        return self.ysubrange(ylim).xsubrange(xlim).zsubrange(zlim)
    def xwaves(self):
        return [ self[:,j] for j in range(self.ny) ]
    def ywaves(self):
        return [ self[i,:] for i in range(self.nx) ]
    def addxwave(self,w):
        assert len(w)==self.nx
        self += w.y[:,np.newaxis,np.newaxis]
        return self
    def addywave(self,w):
        assert len(w)==self.ny
        self += w.y[np.newaxis,:,np.newaxis]
        return self
    def addzwave(self,w):
        assert len(w)==self.nz
        self += w.y[np.newaxis,np.newaxis,:]
        return self
    def clip(self,w0,w1,clipval=None):
        if clipval is None:
            return np.clip(self,w0,w1)
        return np.where(self<w1, np.where(self>w0, self, clipval), clipval)
    def __call__(self,x,y,method='linear'):
        return self.atxyz(x,y,z,method)
    def atxyz(self,x,y,z,method='linear'): # returns www(x,y,z) at given (x,y,z)
        if 'linear'==method:
            if not hasattr(self,'_sl'):
                self._sl = scipy.interpolate.RegularGridInterpolator((self.xs,self.ys,self.zs),self)
            xb,yb,zb = np.clip(x,self.xs.min(),self.xs.max()),np.clip(y,self.ys.min(),self.ys.max()),np.clip(z,self.zs.min(),self.zs.max())
            try:
                val = self._sl((xb,yb,zb))
            except ValueError:
                val = nan
        elif 'nearest'==method:
            if not hasattr(self,'_sn'):
                self._sn = scipy.interpolate.RegularGridInterpolator((self.xs,self.ys,self.zs),self,method='nearest')
            val = self._sn((x,y,z))
        elif 'cubic'==method:
            if not hasattr(self,'_ss'):
                self._ss = scipy.interpolate.RectBivariateSpline((self.xs,self.ys,self.zs),self)
            val = self._ss((x,y,z))
    # def atx(self,x,method=None):
    #     if method is None:
    #         # return y-slice of 2d array at given x value, rounding to nearest grid location
    #         return self.atxindex(np.abs(self.xs-x).argmin())
    #     return Wave( [self.atxy(x,y,method) for y in self.ys], self.ys )
    # def aty(self,y,method=None):
    #     if method is None:
    #         # return x-slice of 2d array at given y value, rounding to nearest grid location
    #         return self.atyindex(np.abs(self.ys-y).argmin())
    #     return Wave( [self.atxy(x,y,method) for x in self.xs], self.xs )
    def atxindex(self,p): # return 2d slice of 3d array at given x index
        return Wave2D(self.array()[p,:,:],xs=self.ys,ys=self.zs)
    def atyindex(self,p): # return 2d slice at given y index
        return Wave2D(self.array()[:,p,:],xs=self.xs,ys=self.zs)
    def atzindex(self,p): # return 2d slice at given z index
        return Wave2D(self.array()[:,:,p],xs=self.xs,ys=self.ys)
    def maxloc(self):
        return self.pmax()
    def maxindex(self):
        return self.pmax()
    # def xaverage(self): # average over x-axis, returns y-slice
    #     return Wave( np.nanmean(self,axis=0), self.ys )
    # def yaverage(self): # average over ys, returns x-slice
    #     return Wave( np.nanmean(self,axis=1), self.xs )
    # def xslicemax(self): # x-slice thru 2d array max
    #     return self.atyindex(self.maxindex()[1])
    # def yslicemax(self): # y-slice thru 2d array max
    #     return self.atxindex(self.maxindex()[0])
    # def flatten(self):
    #     return np.hstack(self)
    def sdev(self):
        return Wave(self.flatten()).sdev()
    def nanmin(self):
        return np.nanmin(self.flatten())
    def nanmax(self):
        return np.nanmax(self.flatten())
    def histogram(self,*args,**kwargs):
        return Wave(self.flatten().tolist()).histogram(*args,**kwargs)
    def pmax(self):
        return np.unravel_index(self.argmax(), self.shape)
    def pmin(self):
        return np.unravel_index(self.argmin(), self.shape)
    def xyzmax(self):
        return self.xx[self.pmax()],self.yy[self.pmax()],self.zz[self.pmax()]
    def xyzmin(self):
        return self.xx[self.pmin()],self.yy[self.pmin()],self.zz[self.pmin()]
    def normalize(self,c=1):
        return c*self/self.nanmax()
    def volume(self):
        return np.sum(self)*self.dx*self.dy*self.dz
    def norm(self):
        return np.sum(self**2)*self.dx*self.dy*self.dz
    def abs(self):
        return np.abs(self)
    def sqr(self):
        return self.magsqr()
    def magsqr(self):
        return Wave3D(self.real()**2 + self.imag()**2,xs=self.xs,ys=self.ys,zs=self.zs)
    def csv(self,save,gridinfo=True):
        s = f', gridx=({self.xmin:g},{self.xmax:g},{self.dx:g}), gridy=({self.ymin:g},{self.ymax:g},{self.dy:g}), gridz=({self.zmin:g},{self.zmax:g},{self.dz:g})' if gridinfo else ''
        np.savetxt(save.replace('.csv','')+s+'.csv', self, delimiter=',')

