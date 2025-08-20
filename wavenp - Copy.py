import numpy as np
class WWave(np.ndarray):
    def __new__(cls,a=10,xwave=None,name=''):
        if not isinstance(a,np.ndarray) and not isinstance(a,list) and not isinstance(a,tuple):
            a = np.zeros(a)
        obj = np.asarray(a).view(cls)
        obj.name = name
        if xwave is not None:
            obj.x = xwave
        elif isinstance(a,WWave):
            obj.x = a.x
        else:
            obj.x = np.arange(len(obj))
        return obj
    def __array_wrap__(self,out,context=None): # hopefully this never gets deprecated. need to implement using __array_ufunc__ if it does
        out.x = self.x # print('wrapping self:'+str(self)+' out:'+str(out))
        return out
    def __getslice__(self,i,j): # only needed in python 2.x
        w = super(WWave,self).__getslice__(i,j)
        w.x = self.x.__getslice__(i,j)
        return w
    def __getitem__(self,index):
        w = super(WWave,self).__getitem__(index)
        #if not isinstance(index,slice) and not isinstance(index,int) and not isinstance(index,tuple) and not isinstance(index,np.int64): print(index,type(index))
        #assert isinstance(index,slice) or isinstance(index,int) or isinstance(index,tuple) # print('index',index,'type',type(index))
        if isinstance(index,slice): # start,stop,step = index.indices(len(self)) # start,stop,step = index.start,index.stop,index.step
            w.x = self.x[index] # #print('slicing',w,w.x) # if index is a slice not a number then w is a Wave
        return w
    @property
    def p(self): # p wave is always returned as a Wave. It has w.x for its xwave, so you can do w = 10*w.p without losing w.x. Also w += 10*w.p and w = 10*w.p should be similar.
        return WWave(np.arange(len(self)),xwave=self.x)
    #def __array_finalize__(self,old): pass
    @property
    def xwave(self):
        return WWave(self.x,self.x)
    def setx(self,x0,xn):
        self.x = x0 + (xn-x0) * np.arange(len(self))/len(self)
        return self
    def setix(self,xleft,xright):
        self.x = xleft + (xright-xleft) * np.arange(len(self))/(len(self)-1)
        return self
    def setpx(self,x0,dx):
        self.x = x0 + dx * np.arange(len(self))
        return self
    def setdx(self,x0,dx):
        return self.setpx(x0,dx)
    def atx(self,x,left=None,right=None): # returns y(x) at given x, by default limited to x[0] <= x <= x[-1] unless otherwise specified by left,right
        if self.x[1]<self.x[0]: # if not increasing, assume x wave is strictly decreasing
            if left is not None or right is not None: raise ValueError("can't use np.interp left and right options if x wave is decreasing")
            return np.interp(-x,-self.x,self)
        return np.interp(x,self.x,self,left=left,right=right) # interp will not work if not strictly increasing
    def atxloglin(self,x): # similar to .atx but assuming y values are changing exponentially (varying linearly on a log scale)
        return np.exp(np.log(self).atx(x))
    def xmax(self):
        return self.x[np.argmax(self)]
    def xmin(self):
        return self.x[np.argmin(self)]
    def sinc(self):
        # def sinc(x): return np.sinc(x/np.pi)
        # assert np.allclose(np.sin(1)/1,sinc(1)) and np.allclose(np.sin(0.3)/0.3,sinc(0.3))
        return WWave( np.sinc(self.x/np.pi), xwave=self.x )
    def smooth(self,boxsize=5,savgol=True):
        if not savgol:
            w = np.convolve(self, np.ones(boxsize)/boxsize, mode='same') # assumes zeros beyond boundary
        else:
            from scipy.signal import savgol_filter
            w = savgol_filter(self, window_length=boxsize, polyorder=2)
        return WWave(w,self.x)
    # def __str__(self):
    #     return str(self.view(np.ndarray)) + 'x:' + str(self.x.view(np.ndarray))
    def printwave(self,s=''):
        print(s+' '+str(self)+' x'+str(self.x)+' p'+str(self.p))
    def plot(self,m=False,waves=[]):
        import matplotlib.pyplot as plt
        plt.rcParams['axes.facecolor']=plt.rcParams['savefig.facecolor']=plt.rcParams['figure.facecolor']='white'
        plt.rc('font',family='Arial')
        ms = 'osDx+'
        plt.plot(self.x,self,'darkred',marker=(ms[0] if m else ''))
        for w in waves:
            plt.plot(w.x,w,marker=(ms[0] if m else ''))
        if hasattr(self,'yname'): plt.ylabel(self.yname)
        if hasattr(self,'xname'): plt.xlabel(self.xname)
        x1,x2,y1,y2 = plt.axis()
        plt.axis((self.x[0],self.x[-1],min(self),max(self)))
        plt.show()
        return self
        