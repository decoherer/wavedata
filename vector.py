
import numpy as np
from numpy import sqrt,pi,nan,inf,sign,exp,log,sin,cos,floor,ceil
import scipy
from waveclass import Wave
from util import endlessshuffle

class Array(np.ndarray): # numpy subclass that normally returns the subclass, but returns ndarray for other (e.g. shape changing) operations
    # simple np.array subclass example
    # good examples:
    #   https://stackoverflow.com/a/51695181
    #   https://github.com/numpy/numpy/blob/v1.15.0/numpy/matrixlib/defmatrix.py#L70-L999
    #   https://docs.scipy.org/doc/numpy-1.15.0/user/basics.subclassing.html
    #   https://docs.scipy.org/doc/numpy/reference/arrays.classes.html
    #   https://stackoverflow.com/questions/5474844/
    def __new__(cls,a):
        try:
            obj = np.asarray(a).astype(float,casting='safe').view(cls)
        except TypeError: # catch complex numbers
            obj = np.asarray(a).view(cls)
        return obj
    for name in ['min','max','sum']:
        vars()[name] = (lambda attr: lambda self,*args,**kwargs: getattr(self.view(np.ndarray),attr)(*args,**kwargs))(name)
    def __str__(self):
        return str(np.array2string(self, precision=3, threshold=9)).replace('\n','')
    def __repr__(self):
        return str(self.__class__.__name__)+':'+str(self)
class V(Array):
    """
    Vector (x,y) pair data structure.
    """
    def __new__(cls,*a):
        obj = super().__new__(cls,*a) if 1==len(a) else super().__new__(cls,a)
        obj = obj.squeeze()
        # assert obj.valid(), 'V() must be length 2'
        return obj
    def valid(a):
        return a.shape==(2,)
    @property
    def x(self):
        return self[0]
    @property
    def y(self):
        return self[1]
    @x.setter
    def x(self,x):
        self[0] = x
    @y.setter
    def y(self,y):
        self[1] = y
    def __eq__(self,v):
        return all(vi==vj for vi,vj in zip(self,v))
    def __lt__(self, v):
        return tuple(self) < tuple(v)
    def __hash__(self):
        return hash(tuple(self))
    def magsqr(self):
        return self.dot(self)
    def mag(self):
        return np.sqrt(self.dot(self)) # np.sqrt(self@self)
    def rotate(self,θ,p0=[0,0]):
        x,y = self-p0
        return p0 + self.__class__(x*cos(θ)-y*sin(θ),x*sin(θ)+y*cos(θ))
    def cross(self,v):
        return self.__class__(np.cross(self,v))
    def unit(self):
        return self/self.mag()
    def norm(self):
        return self/self.mag()
    def direction(self): # angle that vector makes with x-axis
        return np.arctan2(self.y,self.x)
    def translation(self,d,v1,v0=(0,0)): # return translation of point (x,y) by distance d in direction of line passing through (x0,y0) and (x1,y1)
        return self+d*(v1-v0).unit()
    def reflection(self,v1,v0=(0,0)):
        (x,y),(x0,y0),(x1,y1) = self,v0,v1
        # return reflection of point (x,y) in line passing through (x0,y0) and (x1,y1)
        # https://stackoverflow.com/a/3307181
        if x0==x1:
            return V(2*x0-x, y)
        m,b = (y1-y0)/(x1-x0), (x1*y0-x0*y1)/(x1-x0) # define line as y=mx+b
        d = (x + (y-b)*m)/(1 + m*m)
        return V(2*d-x, 2*d*m+2*b-y)
    def originreflection(self,a): # reflection of self in a hyperplane (through origin) orthogonal to a
        return self - 2 * a * self.dot(a) / a.dot(a)
    def projection(self,a): # projection of self onto a hyperplane orthogonal to a
        return self - 1 * a * self.dot(a) / a.dot(a)
    def rotation(self,θ,axis0,axis1,p0=None): # rotate θ in (axis0,axis1) plane
        p0 = p0 if p0 is not None else 0*self
        v = self-p0
        if axis1<axis0:
            θ,axis0,axis1 = -θ,axis1,axis0
        x,y = v[axis0],v[axis1]
        xx,yy = x*cos(θ)-y*sin(θ),x*sin(θ)+y*cos(θ)
        vv = list(v[:axis0]) + [xx] + list(v[axis0+1:axis1]) + [yy] + list(v[axis1+1:])
        return p0 + self.__class__(vv)
    @staticmethod
    def lineintersection(p1,p2,p3,p4):
        (x1,y1),(x2,y2),(x3,y3),(x4,y4) = p1,p2,p3,p4
        denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator
        return V(px,py)
    @staticmethod
    def lineintersection2(p1,p2,p3,p4):
        c0,c1 =  V3.lineintersection(V3(*p1,0),V3(*p2,0),V3(*p3,0),V3(*p4,0))
        assert np.allclose(c0,c1), 'both points of closest approach should intersect in 2D'
        return V(c0[:2])
class V3(V):
    """
    Vector (x,y,z) data structure.
    """
    def __new__(cls,*a):
        obj = super().__new__(cls,*a) if 1==len(a) else super().__new__(cls,a)
        obj = obj.squeeze()
        assert obj.valid(), 'V() must be length 3'
        return obj
    def valid(a):
        return a.shape==(3,)
    @property
    def z(self):
        return self[2]
    @z.setter
    def z(self,z):
        self[2] = z
    def rotatez(self,θ,p0=[0,0,0]):
        x,y,z = self-p0
        return p0 + V3(x*cos(θ)-y*sin(θ),x*sin(θ)+y*cos(θ),z)
    def rotate(self,θ,p1,p0=(0,0,0)): # rotation of point p by θ radians about line p0 to p1
        from scipy.spatial.transform import Rotation as R
        r = R.from_rotvec(θ*(p1-V3(p0)).unit())
        return V3(r.apply(self-p0) + p0)
    def translate(self,d,p1,p0=(0,0,0)):
        return self + d * (V3(p1)-p0).unit()
    def sphericalinvert(self,r,p0=(0,0,0)):
        v = self - p0
        d,u = v.mag(),v.unit()
        return u*r*r/d + p0
    @staticmethod
    def lineintersection(p1,p2,p3,p4,floatprec=1e-8): # https://stackoverflow.com/a/2316934
        # returns two points, the point on line (p1,p2) closest to line (p3,p4) and vice versa
        p13,p21,p43 = p1-p3, p2-p1, p4-p3
        if p21.magsqr()<floatprec or p43.magsqr()<floatprec:
            return None
        d1343 = p13.x*p43.x + p13.y*p43.y + p13.z*p43.z
        d4321 = p43.x*p21.x + p43.y*p21.y + p43.z*p21.z
        d1321 = p13.x*p21.x + p13.y*p21.y + p13.z*p21.z
        d4343 = p43.x*p43.x + p43.y*p43.y + p43.z*p43.z
        d2121 = p21.x*p21.x + p21.y*p21.y + p21.z*p21.z
        denom = d2121 * d4343 - d4321 * d4321
        if np.abs(denom)<floatprec:
            return None
        numer = d1343 * d4321 - d1321 * d4343
        mua = numer / denom
        mub = (d1343 + d4321 * (mua)) / d4343
        return (p1 + mua*p21, p3 + mub*p43)

class Vs(Array):
    """
    Vector waveform data structure. Each element is an (x,y) pair.
    """
    def __new__(cls,*a,x=None,y=None,size=None,width=None,name=''):
        if size is not None:
            obj = cls(np.repeat([*a],size,axis=0)) if len(a) else cls(np.repeat([[0,0]],size,axis=0))
        elif len(a):
            obj = super().__new__(cls,*a)
        else:
            assert y is not None
            obj = cls(np.array([np.array(x).flatten(),np.array(y).flatten()]).T) if x is not None else cls(np.array([[i for i in range(len(y))],y]).T)
        # assert cls.valid(obj), 'cls() must have shape (n,2)'
        obj.width = getattr(a[0],'width',width) if len(a) else width
        obj.name = name
        return obj
    @classmethod
    def fromcomplex(cls,a):
        return Vs([(c.real,c.imag) for c in a])
    @staticmethod
    def valid(a):
        return len(a.shape)==2 and a.shape[-1]==2
    @property
    def x(self):
        return self[:,0]
    @property
    def y(self):
        return self[:,1]
    @x.setter
    def x(self,xs):
        self[:,0] = xs
    @y.setter
    def y(self,ys):
        self[:,1] = ys
    def __getitem__(self,index):
        if not isinstance(index,(int,slice,tuple)):
            p,f = int(floor(index)),index-floor(index)
            return self[p]*(1-f) + (self[p+1]*f if f else 0)
        # print('slice',index.start, index.stop, index.step) # TODO: implement non-integer slicing
        v = super().__getitem__(index)
        VV = type(self)
        return v if VV.valid(v) else (V(v) if V.valid(v) else np.asarray(v))
    def __pos__(self): # must use only with concatenation operator, a ++ b = a + +b = (b.__pos__()).__radd__(a)
        class Vsconcatenation: # helper class to allow ++ operator to act as a concatenator, works in association with x.__pos__ (+x) operator
            __array_priority__ = 1 # https://stackoverflow.com/a/37127087
            def __init__(self,val): self.val = val
            def __radd__(self,lhs):
                VV = type(self)
                return Vs(np.concatenate((lhs,self.val)))
        return Vsconcatenation(self)
    def __eq__(self,other):
        return np.array(self).shape==np.array(other).shape and np.allclose(self,other)
    def __ne__(self,other):
        return not self==other
    def rename(self,name):
        self.name = name
        return self
    def map(self,fx,fy):
        x,y = self.x,self.y
        VV = type(self)
        return VV(np.array([fx(x),fy(y)]).T)
    def scale(self,v,p0=[0,0]):
        x,y = (self-p0).x,(self-p0).y
        VV = type(self)
        return VV(np.array([x*v[0],y*v[1]]).T)
    def rotate(self,θ,p0=[0,0]):
        x,y = (self-p0).x,(self-p0).y
        VV = type(self)
        # return p0 + VV(x=x*cos(θ)-y*sin(θ),y=x*sin(θ)+y*cos(θ))
        return p0 + VV(np.array([x*cos(θ)-y*sin(θ),x*sin(θ)+y*cos(θ)]).T)
    def jitter(self,dx=None,dy=None,uniform=False):
        if not uniform: raise NotImplementedError
        dx = dx*np.random.random() if dx is not None else 0
        dy = dy*np.random.random() if dy is not None else 0
        return self + V(dx,dy)
    @staticmethod
    def concats(*vs,sep=False):
        from functools import reduce
        VV = type(vs[0])
        vnan = VV([[np.nan]*len(vs[0][0])])
        def concatfunc(x,y):
            if not sep:
                return np.concatenate((x,y))
            return np.concatenate((x,vnan,y))
        return VV( list(reduce(concatfunc,vs)) )
    def concat(self,*vs,sep=False):
        VV = type(self)
        return VV.concats(self,*vs,sep=sep)
        # if not len(vs): return self
        # if sep:
        #     vnan = Vs([[np.nan]*len(vs[0][0])])
        #     return Vs(np.concatenate((self,vnan,vs[0]))).concat(*vs[1:])
        # return Vs(np.concatenate((self,vs[0]))).concat(*vs[1:])
    def close(self):
        if np.all(self[0]==self[-1]):
            return self
        return self.concat(self[:1])
    @staticmethod
    def list2chain(vs): # list of Vs → nan separated Vs
        VV = type(vs[0])
        return VV.concats(*vs,sep=True)
    def chain2list(self,sep=(np.nan,np.nan)): # nan separated Vs → list of Vs
        inan = [i for i,n in enumerate(self) if np.isnan(n).any()] # print(inan)
        a = [self[i:j] for i,j in zip([0]+[i+1 for i in inan],inan+[len(self)])]
        return [ai for ai in a if 0<len(ai)]
    def sep(self): # add nan separator
        return self.concat(np.array([[np.nan,np.nan]]))
    def removenans(self):
        VV = type(self)
        return VV([p for p in self if not np.isnan(p).any()])
    def unsep(self): # remove nan separators
        return self.removenans()
    def reflection(self,v0,v1):
        VV = type(self)
        return VV([ v.reflection(v0,v1) for v in self ])
    def rotation(self,θ,axis0,axis1,p0=None): # rotate θ in (axis0,axis1) plane
        VV = type(self)
        return VV([ V(v).rotation(θ,axis0,axis1,p0) for v in self ])
    def shapelypolygon(self):
        from shapely.geometry import Polygon
        return Polygon(self)
    def intersects(self,vs):
        return self.shapelypolygon().intersects(vs.shapelypolygon())
    @staticmethod
    def shapelypolygon2Vs(Q):
        return Vs(list(zip(*Q.exterior.coords.xy)))
    def intersection(self,vs):
        Q = self.shapelypolygon().intersection(vs.shapelypolygon())
        return Vs.shapelypolygon2Vs(Q)
    def difference(self,vs):
        Q = self.shapelypolygon().difference(vs.shapelypolygon())
        return Vs.shapelypolygon2Vs(Q)
    def symmetric_difference(self,vs):
        Q = self.shapelypolygon().symmetric_difference(vs.shapelypolygon())
        return Vs.shapelypolygon2Vs(Q)
    def union(self,vs):
        Q = self.shapelypolygon().union(vs.shapelypolygon())
        return Vs.shapelypolygon2Vs(Q)
    def buffer(self,*args,**kwargs):
        Q = self.shapelypolygon().buffer(*args,**kwargs)
        return Vs.shapelypolygon2Vs(Q)
    def coordfilter(self,func,coord='x'):
        return sorted(list(set([getattr(v,coord) for v in self if func(v)])))
    def xmin(self):
        return min(self.x)
    def xmax(self):
        return max(self.x)
    def ymin(self):
        return min(self.y)
    def ymax(self):
        return max(self.y)
    def mean(self):
        return V(sum(self.x),sum(self.y))/len(self)
    def boundingbox(self):
        return [min(self.x),min(self.y),max(self.x),max(self.y)]
    def bb(self):
        return self.boundingbox()
    def aspect(self): # aspect = height/width
        x0,y0,x1,y1 = self.boundingbox()
        return (y1-y0)/(x1-x0)
    def list(self):
        return list([list(v) for v in self])
    def mag(self):
        return np.array([v.mag() for v in self])
    def unit(self):
        return self / self.mag()
    def length(self):
        return np.sum(self.segmentlengths())
    def cumlengths(self):
        cumlen = np.cumsum(self.segmentlengths(),axis=0) # shape = (n,1) for numpy broadcasting
        assert len(cumlen)==len(self)
        return cumlen.reshape(-1)
    def segmentlengths(self):
        def dist(v):
            return np.sqrt((v**2).sum(axis=-1,keepdims=True))
        return dist( self[:1].concat(self[:-1]) - self[:] )
    def distanceindex(self,d):
        return Wave(self.cumlengths()).xaty(d)
    def fractionallengths(self): # 0 for index 0, 1 for index len(vs)-1
        s = self.segmentlengths()
        f = np.cumsum(s,axis=0)
        return f/f[-1] # shape = (n,1) for numpy broadcasting
    def signedarea(self):
        assert np.allclose(self[0],self[-1]), 'curve not closed'
        return 0.5*(np.dot(self.x,np.roll(self.y,1))-np.dot(self.y,np.roll(self.x,1))) # https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
    def miters(self,widthcorrection=True): # vector pointing along seam of miter joint, pointing to the right, ⊥ for start and end points
        def dist(v):
            return np.sqrt((v**2).sum(axis=-1,keepdims=True))
        def norm(v):
            return v/dist(v)
        def extend(vs):
            return vs[:1].concat(vs,vs[-1:]) # double the first and last points
        def rightperp(vs):
            return Vs(x=vs[:,1:],y=-vs[:,:1]) # np.hstack((vs[:,1:],-vs[:,:1]))
        def leftperp(vs):
            return -rightperp(vs)
        segs = extend(norm(self[1:]-self[:-1])) # direction vector of each segment, segs[n] is segment before xys[n], segs[n+1] is dir of segment after xys[n]
        fperps = rightperp(segs[1:])  # fperps[n] = direction vector ⊥ to forward segment
        bperps = rightperp(segs[:-1]) # rperps[n] = direction vector ⊥ to backward segment
        def coshalftheta(cosθ):
            return np.sqrt((1+cosθ)/2)
        def rowwisedotproduct(x,y):
            return np.sum(x*y,axis=1,keepdims=True) # x dot y = cosθ
        jointwidth = 1/coshalftheta(rowwisedotproduct(fperps,bperps)) # relative width of joint needed to keep sides at constant ⊥ width (also constant area per unit length)
        return jointwidth * norm(fperps+bperps) if widthcorrection else norm(fperps+bperps)
    def thickribbons(self,widths,normal=None,cw=False,widthcorrection=True): # widths = list of ribbon widths and gaps
        widths = [(w(self.fractionallengths()) if callable(w) else w) for w in widths]
        offsets = [sum(widths[:i]) for i in range(len(widths)+1)]
        n0 = len(widths)//2 # index of the middle width, along which ribbons are centered
        offsets = [w-sum(offsets[n0:n0+2])/2 for w in offsets] # 
        sides = [self.thickside(offset,normal=normal,widthcorrection=widthcorrection) for offset in offsets]
        def curve(r,l,cw):
            return Vs(np.vstack((l,r[::-1],l[:1])) if cw else Vs(np.vstack((r,l[::-1],r[:1]))))
        return [curve(r,l,cw) for r,l in zip(sides[::2],sides[1::2])]
    def thickside(self,offset,scale=1,normal=None,widthcorrection=True):
        if callable(offset):
            t = offset(self.fractionallengths())
        else:
            t = np.array(offset)
            if not 1==t.size:
                t = t.reshape(-1,1)
                # print(t.shape,self.miters(widthcorrection=widthcorrection).shape,t,self.miters(widthcorrection=widthcorrection),t * self.miters(widthcorrection=widthcorrection))
                assert len(t)==len(self), 'offset (or width) array must be same length as Vs'
        if normal is None:
            return self + scale * t * self.miters(widthcorrection=widthcorrection)
        def dist(v):
            return np.sqrt((v**2).sum(axis=-1,keepdims=True))
        def norm(v):
            return v/dist(v)
        def rightperp(vs):
            return Vs(x=vs[:,1:],y=-vs[:,:1])
        fd = np.array(normal).reshape(1,-1)
        return self + scale * t * rightperp(norm(fd))
    @staticmethod
    def closesides(l,r,cw=False):
        return Vs(np.vstack((l,r[::-1],l[:1])) if cw else Vs(np.vstack((r,l[::-1 ],r[:1]))))
    def side(self,i=0): # return r or l assuming self is the result of closesides(l,r) with equal length l and r
        assert 1==len(self)%2, 'Vs has wrong length, unequal l and r'
        return self[:len(self)//2] if 0==i else self[len(self)//2:-1][::-1]
    def thickline(self,width=None,normal=None,cw=False,widthcorrection=True,lwidth=None,rwidth=None):
        if lwidth is None:
            width = width if width is not None else self.width
            assert width is not None
            r,l = [self.thickside(offset=width,scale=s,normal=normal,widthcorrection=widthcorrection) for s in [+0.5,-0.5]]
        else:
            assert rwidth is not None
            r,l = [self.thickside(offset=w,scale=1,normal=normal,widthcorrection=widthcorrection) for w in [lwidth,rwidth]]
        return Vs(np.vstack((l,r[::-1],l[:1])) if cw else Vs(np.vstack((r,l[::-1 ],r[:1]))))
        def curve(r,l,cw):
            return Vs(np.vstack((l,r[::-1],l[:1])) if cw else Vs(np.vstack((r,l[::-1 ],r[:1]))))
        return curve(r,l,cw)
    @classmethod
    def arc(cls,r,φ0,φ1,width=None,p0=(0,0),minangle=0.5):
        # used to be class Arc(Vs)
        # def __new__(cls,r,φ0,φ1,width=None,p0=[0,0],minangle=0.5):
        def arcsegs(r,φ0,φ1,minangle): # segments of arc will form a circumscribing regular polygon of the radius r circle
            def xy(r,φ):
                return np.array([r*cos(φ),r*sin(φ)])
            n = int(abs(φ1-φ0)/minangle)+1 # there will be n+1 triangle in circumscribing polygon, n-1 isosceles triangles plus two half-isosceles at start and end
            θ = (φ1-φ0)/n/2.
            R = r/cos(θ) # R = radius of circle that circumscribes the polygon
            a = np.array([xy(r,φ0)] + [xy(R,m*θ+φ0) for m in range(1,2*n,2)] + [xy(r,2*n*θ+φ0)]) # print(n,np.array([0]+list(range(1,2*n,2))+[2*n]))
            return np.array(p0) + a - a[0]
        # obj = super().__new__(cls,arcsegs(r,φ0,φ1,minangle),width=width)
        obj = cls(arcsegs(r,φ0,φ1,minangle),width=width)
        obj.r,obj.φ0,obj.φ1 = r,φ0,φ1 # pylint: disable=W0201
        return obj
    @classmethod
    def circle(cls,r,p0=(0,0),minangle=0.5):
        vs = cls.arc(r,0,2*pi,width=0,p0=p0,minangle=minangle)
        return vs[:-1].close()
    def upsample(self,num=10):
        ns = [n for n in np.linspace(0,len(self)-1,(len(self)-1)*num+1)]
        return Vs([self[n] for n in ns])
    def endnormals(self,n0,n1,delta=0.1):
        n0,n1 = n0/n0.mag()*(self[1]-self[0]).mag()*delta,n1/n1.mag()*(self[-2]-self[-1]).mag()*delta
        return self[:1].concat([self[0]+n0], self[1:-1], [self[-1]+n1], self[-1:])
    def cosangles(self): # returns cos(angle) formed by prior and latter vertices for each vertex
        # -1<=cos(angle)<=1
        # cos(θ)=-1 → flattest angle, cos(θ)=1 → sharpest angle
        # use the dot product of the vectors formed by the vertices to find cosθ
        def cosθ(a,b,c):
            v0,v1 = a-b,c-b
            return np.dot(v0,v1)/v0.mag()/v1.mag()
        return [np.nan]+[cosθ(a,b,c) for a,b,c in zip(self[:-2],self[1:-1],self[2:])] + [np.nan]
    def sharpestvert(self,closed=False):
        cosθs = self.cosangles()
        if closed:
            assert np.all(self[0]==self[-1]), 'curve not closed'
            cosθs = Vs(list(self)+[self[1]]).cosangles()
        return cosθs.index(np.nanmax(cosθs))
    def shiftclosedcurve(self,i):
        assert np.all(self[0]==self[-1]), 'curve not closed'
        assert 0<=i<len(self)-1, 'index out of range'
        return self[i:-1] ++ self[:i+1]
    def radiused(self,r,i=None,res=0.05,getc=False,debug=False): # r = radius, i = index of vertex to round off, res = angular resolution of arc
        assert np.all(self[0]==self[-1]), 'curve not closed'
        def dist(p,l): # distance of point p to line l
            return np.abs(np.cross(l[1]-l[0],p-l[0]))/np.linalg.norm(l[1]-l[0])
        def shorten(a,s): # shorten, but not too much
            if a[-2:].length()<=s:
                return a[:-1]
            aend = a[-1] + (a[-2]-a[-1]).unit()*s
            return Vs(list(a[:-1]) + [aend])
        assert np.all([(pi-pj).mag() for pi,pj in zip(self[:-1],self[1:])]), 'no duplicate vertices'
        i = i if i is not None else self.sharpestvert(closed=0)
        assert 0<i<=len(self)-1, 'index out of range'
        vv = self.copy()
        # a,b = vv[:i+1],vv[i:][::-1] # input,output curve, they meet at last point of a and last point of b
        di = (i+len(vv[:-1])//2)%len(vv[:-1])
        i,vv = ((i-di+len(vv[:-1]))%len(vv[:-1]),vv.shiftclosedcurve(di)) if i<0.25*len(vv) or 0.75*len(vv)<i else (i,vv) # shift so that i is away from the ends
        a,b = vv[:i+1],vv[i:][::-1] # input,output curve, they meet at last point of a and last point of b
        def ispointinwedge(p, u, v): # return True if point p is inside wedge formed by vectors u and v
            p,u,v = V([*p,0]).unit(),V([*u,0]).unit(),V([*v,0]).unit()
            return u@v<=u@p and v@u<=v@p
        # for f in [V(+1,+1),V(-1,+1),V(+1,-1),V(-1,-1)]: print(f'ispointinwedge({f},V(1,0),V(0,1))',ispointinwedge(f,V(1,0),V(0,1)))
        def linepairequidistantpoints(d,p1,p2,p3,p4,inner=True,outer=False): # d = distance, p1,p2 = line 1, p3,p4 = line 2
            v1,v2 = p2-p1,p4-p3
            v1,v2 = v1/np.linalg.norm(v1),v2/np.linalg.norm(v2)
            n1,n2 = np.array([-v1[1],v1[0]]),np.array([-v2[1],v2[0]])
            def intersection(d1,d2):
                A = np.vstack([n1,n2])
                b = np.array([d1*d+np.dot(p1,n1),d2*d+np.dot(p3,n2)])
                try: return np.linalg.solve(A,b)
                except np.linalg.LinAlgError: return None
            qs = [p for d1 in [-1,1] for d2 in [-1,1] if (p:=intersection(d1,d2)) is not None]
            # return qs if not inner else [p for p in qs if np.dot(p-p1,n1)*np.dot(p-p3,n2)<0]
            return [q for q in qs if (1 if not (inner or outer) else outer^(np.dot(q-p1,n1)*np.dot(q-p3,n2)<0))]
        # print(linepairequidistantpoints(r,V(0,0),V(0,1),V(1,0),V(1,1),inner=False))
        while 1: # shorten a,b until there is room to fit a tangent circle to their extrapolations
            # if len(a)<2 or len(b)<2:
            #     ws = [vv.wave().setplot(ms=5,mf='w'),vv[1:-1].wave().setplot(ms=5,mf='w'),a.wave().rename(f'A len {len(a)}'),b.wave().rename(f'B len {len(b)}')]
            #     Wave.plot(*ws,m='o',ms=2,seed=3,aspect=1,grid=1)
                
            c = V.lineintersection(a[-2],a[-1],b[-2],b[-1]) # intersection of the extrapolated segments a[-2:] and b[-2:]
            ua,ub = (a[-2]-a[-1]).unit(),(b[-2]-b[-1]).unit() # unit vectors for a and b pointing away from c
            qq = [q for q in linepairequidistantpoints(r,*a[-2:],*b[-2:],inner=0)] # 4 points r distant from lines a[-2:] and b[-2:]
            assert len([q for q in qq if ispointinwedge(q-c,ua,ub)])==1, 'expected one equidistant point inside wedge'
            q = [q for q in qq if ispointinwedge(q-c,ua,ub)][0] # only one is inside the wedge formed by ua and ub, q will be the center of the circle
            da,db = np.dot(a[-1]-c,ua),np.dot(b[-1]-c,ub) # distance of c from a[-2] along ua and and b[-2] along ub
            uq,dq = (q-c).unit(), (q-c).mag() # dq = distance of c from q # print('da,db,dq',da,db,dq)
            if da>dq and db>dq:
                break
            s = min(a[-2:].length(),b[-2:].length(),r)
            a,b = shorten(a,s),shorten(b,s)
            if 1<debug: Vs.plots(a.rename('A'),b.rename('B'),Vs([c]).rename('c'),Vs([q]).rename('q'),m='o',seed=3)
        cosϕ = np.sqrt(0.5*(1+ua@ub)) # ϕ = halfangle subtended by ua,ub
        a0,b0 =c+ua*dq*cosϕ,c+ub*dq*cosϕ # a0,b0 = points on a,b at distance r from q
        θa,θb = (a0-q).direction(),(b0-q).direction() # arc is from θa to θb, the short way around the circle
        isccw = (θb-θa+2*pi)%(2*pi)<pi
        θ0,θ1 = ((θa,θb) if θb>θa else (θa,θb+2*pi)) if isccw else ((θa,θb) if θa>θb else (θa+2*pi,θb))
        # print('isccw',isccw,'θa,θb',θa,θb,'θ0,θ1',θ0,θ1)
        arc = Vs([q+r*V(cos(θ),sin(θ)) for θ in np.linspace(θ0,θ1,int(abs(θ1-θ0)/res)+2)])
        assert np.allclose(arc[0],a0) and np.allclose(arc[-1],b0), 'arc end points should be a0 and b0'
        if debug:
            us = [Vs([q]).wave().rename('o').setplot(l=' ',mf='w') for q in linepairequidistantpoints(r,*a[-2:],*b[-2:],outer=1)]
            vs = [Vs([c]).wave().rename('c')]+[Vs([q]).wave().rename('i').setplot(l=' ',mf='1') for q in linepairequidistantpoints(r,*a[-2:],*b[-2:],inner=1) if ispointinwedge(q-c,a[-2]-a[-1],b[-2]-b[-1])]
            ws = [a.wave().rename('A'),b.wave().rename('B'),Vs(list(a)+[a0]).wave().rename('a').setplot(l='2'),Vs(list(b)+[b0]).wave().rename('b').setplot(l='2')]
            ts = [arc.wave().rename('arc').setplot(l='3',m=' '),Vs([a0]).wave().rename('a0').setplot(l=' ',mf='2'),Vs([b0]).wave().rename('b0').setplot(l=' ',mf='2')]
            Wave.plot(*ws,*us,*vs,*ts,m='o',seed=3,aspect=1,grid=1)
        # a,b = Vs(list(a)+[a0]),Vs(list(b)+[b0])
        return (a.concat(arc).concat(b[::-1]),c) if getc else a.concat(arc).concat(b[::-1])

    def bspline(self, n=101, degree=3, closed=False, keependnormals=False): # https://stackoverflow.com/a/35007804/12322780
        cv,count = np.asarray(self),self.shape[0]
        if closed:
            kv, (factor,fraction) = np.arange(-degree,count+degree+1), divmod(count+degree+1, count)
            cv = np.roll(np.concatenate((cv,) * factor + (cv[:fraction],)),-1,axis=0)
            degree = np.clip(degree,1,degree)
        else:
            degree = np.clip(degree,1,count-1)
            kv = np.clip(np.arange(count+degree+1)-degree,0,count-degree)
        vs = Vs(scipy.interpolate.BSpline(kv, cv, degree)(np.linspace(0,count-(degree*(1-closed)),n)))
        return vs if (closed or not keependnormals) else vs.endnormals(self[1]-self[0],self[-2]-self[-1])
    def voronoi(self,debug=False):
        from scipy.spatial import Voronoi, voronoi_plot_2d
        vor = Voronoi(self)
        regions = [(id,r) for id,r in enumerate(vor.regions) if r and -1 not in r]
        if debug:
            import matplotlib.pyplot as plt
            voronoi_plot_2d(vor)
            plt.show()
        sortedregions = [vor.regions[n] for n in vor.point_region] # in order of self
        def valid(r):
            return r and -1 not in r
        ns = [n for n,r in enumerate(sortedregions) if valid(r)]
        centers = [self[n] for n in ns]
        def close(l):
            return l+[l[0]]
        tiles = [[vor.vertices[i] for i in close(sortedregions[n])] for n in ns]
        tiles,centers = Vs.concats(*[Vs(t).sep() for t in tiles]),Vs(centers)
        return tiles,centers
    def array(self,vs):
        return Vs.list2chain([self+v for v in vs])
    def gridarray(self,nx,ny,dx=1,dy=1):
        return self.array(Vs.grid(nx,ny,dx,dy))
    def radialarray(self,nr,nθ,dr=1):
        return self.array(Vs.radialgrid(nr,nθ,dr))
    @staticmethod
    def grid(nx,ny,dx=1,dy=1): # list of points on nx x ny grid
        return Vs([(x,y) for x in dx*np.linspace(-nx/2,nx/2,nx) for y in dy*np.linspace(-ny/2,ny/2,ny)])
    @staticmethod
    def radialgrid(nr,nθ,dr=1):
        ps = [(-n*dr*np.sin(2*pi*x), n*dr*np.cos(2*pi*x)) for n in range(1,nr+1) for x in np.linspace(0,1,n*nθ,endpoint=False)]
        return Vs([(0,0)] + ps)
    @staticmethod
    def random(N,xlim=(-1,1),ylim=(-1,1),seed=None): # N random points within a box
        def r(lim):
            return lim[0]+(lim[1]-lim[0])*np.random.random(N)
        if seed is None:
            return Vs(np.array([(x,y) for x,y in zip(r(xlim),r(ylim))]))
        np.random.seed(seed)
        rr = Vs.random(N,xlim,ylim)
        np.random.seed()
        return rr
    @staticmethod
    def regularstarpolygon(r,N,step=2,end=True,q0=0):
        if step>1 and 0==N%step:
            pss = [Vs.regularstarpolygon(r,N//step,1,end,q0=2*np.pi*i/N) for i in range(step)]
            return Vs([p for ps in pss for p in ps+[(np.nan,np.nan)]][:-1])
        qs = np.linspace(0,2*np.pi*step,N+1) if end else np.linspace(0,2*np.pi*step,N,endpoint=False)
        return Vs([ (-r*np.sin(q+q0), r*np.cos(q+q0)) for q in qs ])
    @staticmethod
    def star(r0,r1,N,end=True,q0=0):
        qs = np.linspace(0,2*np.pi,2*N+1) if end else np.linspace(0,2*np.pi,2*N,endpoint=False)
        return Vs([ (-r*np.sin(q+q0), r*np.cos(q+q0)) for r,q in zip([r0,r1]*N+[r0],qs) ])
    @staticmethod
    def polygon(r,N,end=True):
        return Vs.regularstarpolygon(r,N,step=1,end=end)
    def wave(self,name=''):
        return Wave(self[:,1],self[:,0],name)
    def vs2wave(self,name=''):
        return self.wave(name)
    def plot(self,*args,**kwargs):
        return self.wave(name=self.name).plot(*args,aspect=(kwargs.pop('aspect',True)),**kwargs)
    @staticmethod
    def plots(*ls,**kwargs):
        return Wave.plots(*[l.wave(name=l.name) for l in ls],aspect=(kwargs.pop('aspect',True)),**kwargs)
class Polar(Wave): # self.x = θ/2π, self.y = r
    def valid(self):
        if not all(0<=x<=1 for x in self.x):
            return False
        return all(xi<xj for xi,xj in zip(self.x[:-1],self.x[1:]))
    @classmethod
    def polygon(cls,N=5,r=1):
        xs = np.linspace(0,1,N+1)
        return cls(1+0*xs,xs)
    @classmethod
    def star(cls,N=5,r0=0.5,r1=1):
        xs = np.linspace(0,1,2*N+1)
        return cls([r1,r0]*N+[r1],xs)
    @classmethod
    def triangle(cls,r=1):
        return cls.polygon(N=3,r=r)
    @classmethod
    def square(cls,r=1):
        return cls.polygon(N=4,r=r)
    @classmethod
    def circle(cls,r=1,num=40):
        return cls.polygon(N=num,r=r)
    @classmethod
    def flower(cls,N=5,r0=0.5,r1=1,m=1,num=50,saw=False):
        xs = np.linspace(0,1,num*N+1)
        # use ease funcs?
        def cosfunc(x):
            return np.cos(N*pi*x)**(2*m)
        def sawfunc(x):
            return np.abs(np.arcsin(np.cos(N*pi*x))/pi*2)**m
        func = sawfunc if saw else cosfunc
        return cls(r0+(r1-r0)*func(xs),xs)
    def upsample(self,num=10):
        xs = list(self.x)
        xxs = []
        for x0,x1 in zip(xs[:-1],xs[1:]):
            xxs += list(np.linspace(x0,x1,num=1 if x0==x1 else num,endpoint=False))
        xxs += [xs[-1]]
        yys = [self.radialinterpolate(x=x) for x in xxs]
        return Polar(yys,xxs)
    def radialinterpolate(self,*,θ=None,x=None): # θ = 2*pi*x
        x = x if x is not None else θ/2/pi
        # assert self.x.y[-1]==1 and self.x.y[0] <= x <= self.x.y[-1], str(x)+str(self.x.y) # too slow
        θs,rs = 2*pi*self.x,self.y
        def radiallerp(θ,θ0,θ1,r0,r1): # returns radius at θ
            # straight line interpolation between two points in polar coordinates
            x0,y0,x1,y1 = r0*np.cos(θ0),r0*np.sin(θ0),r1*np.cos(θ1),r1*np.sin(θ1)
            if x1==x0:
                return x0/np.cos(θ)
            m = (y1-y0)/(x1-x0)
            return (m*x0-y0)/(m*np.cos(θ)-np.sin(θ))
        from itertools import takewhile,dropwhile
        θrs = [(θ,r) for θ,r in zip(θs,rs)]
        # θ0 = takewhile(lambda x:x<θ,θs)[-1]
        # θ1 = dropwhile(lambda x:x<θ,θs)[0]
        θ0,r0 = list(takewhile(lambda z:z[0] <= 2*pi*x,θrs))[-1]
        θ1,r1 = list(dropwhile(lambda z:z[0] <  2*pi*x,θrs))[0]
        return radiallerp(2*pi*x,θ0,θ1,r0,r1)
    def radialsum(self,other):
        xx = sorted(list( set(self.x) | set(other.x) ))
        rr = [self.radialinterpolate(x=x)+other.radialinterpolate(x=x) for x in xx]
        return Polar(rr,xx)
    def radialaverage(self,other,f=0.5): # f = mix fraction of other
        return ((1-f)*self).radialsum(f*other)
    @classmethod
    def random(cls,t,ease=0):
        # cycles every integer, cycle order varies, also choose easing function
        def sinease(x):
            return np.sin(x*np.pi/2)**2
        def powerease(x):
            return 0.5*( 1 + np.sign(2*x-1) * np.abs(2*x-1)**(1/ease) )
        easefunc = sinease if 0==ease else powerease
        choices = ['triangle','square','circle','star','polygon','flower'] # e.g. cls.triangle
        n0,n1 = endlessshuffle(int(t),choices,norepeats=True)
        s0,s1 = getattr(cls,n0),getattr(cls,n1)
        return s0().radialaverage(s1(),easefunc(t-int(t)))
    def polar2wave(self):
        xs,ys = zip(*self.polar2xy())
        return Wave(ys,xs,self.name)
    def polar2xy(self):
        return [ (-r*np.sin(q), r*np.cos(q)) for r,q in zip(self.y,2*np.pi*np.array(self.x)) ]
    def polar2vs(self):
        return Vs(self.polar2xy())
    def plot(self,**kwargs):
        import plot
        plot.plot(waves=[self.polar2wave()],fill=True,aspect=(kwargs.pop('aspect',True)),**kwargs)
        return self
    @staticmethod
    def plots(*ws,**kwargs):
        ws = ws[0] if isinstance(ws[0],(list,tuple)) else ws
        wws = [w.polar2wave() for w in ws]
        import plot
        plot.plot(waves=wws,fill=True,aspect=(kwargs.pop('aspect',True)),**kwargs)
class V3s(Vs):
    @staticmethod
    def valid(a):
        return len(a.shape)==3 and a.shape[-1]==3
    @property
    def z(self):
        return self[:,2]
    @z.setter
    def z(self,zs):
        self[:,2] = zs
    def rotate(self,θ,p1,p0=(0,0,0)): # rotation of point p about line p0 to p1
        return Vs([V3(v).rotate(θ,p1,p0=p0) for v in self])
    def plot(self,*args,**kwargs):
        return Wave(self.y,self.x,self.name).plot(*args,aspect=(kwargs.pop('aspect',True)),**kwargs)
        return self
    @staticmethod
    def plots(*ls,**kwargs):
        return Wave.plots(*[Wave(l.y,l.x,l.name) for l in ls],aspect=(kwargs.pop('aspect',True)),**kwargs)
def piecewisebinop(a,b,op,extrapolate='const'):
    # must have a.x and a.y defined and a(x) for interpolation
    i,j = 0,0
    xs,ys = [],[]
    while i<len(a) or j<len(b):
        x = a.x[i] if len(b)==j else b.x[j] if len(a)==i else min(a.x[i],b.x[j])
        def multiplicity(x,w,k):
            return 0 if (len(w.x)==k or not x==w.x[k]) else 1+multiplicity(x,w,k+1)
        na,nb = multiplicity(x,a,i),multiplicity(x,b,j)
        n = max(na,nb)
        assert 1<=n
        assert n<4, f'n={n} duplicate x values not yet supported' # TODO: whichever one has more duplicates, use its y values and the interpolated y values of the other
        ya = n*[a(x,extrapolate=extrapolate)] if 0==na else n*[a.y[i]] if 1==na else list(a.y[i:i+na])
        yb = n*[b(x,extrapolate=extrapolate)] if 0==nb else n*[b.y[j]] if 1==nb else list(b.y[j:j+nb]) # print('ya,yb',ya,yb)
        ya = ya if n==len(ya) else ya[:1] + (n-2)*[0.5*(ya[0]+ya[-1])] + ya[-1:]
        yb = yb if n==len(yb) else yb[:1] + (n-2)*[0.5*(yb[0]+yb[-1])] + yb[-1:]
        # print('i',i,'j',j,'x',x,'na,nb',na,nb,'ya,yb',ya,yb);print()
        assert len(ya)==len(yb)==n
        xs += n*[x]
        ys += [op(za,zb) for za,zb in zip(ya,yb)]
        i,j = i+na,j+nb
    return ys,xs
class Piecewise(Wave):
    def binop(a,b,op,extrapolate='const'): # binary operator (as opposed to unary operator)
        if not isinstance(b,Piecewise):
            return Piecewise(op(a.y,b),a.x,a.name)
        # xs = sorted(list(a.x)+list(b.x)) # assume no duplicate x values in a.x or b.x (but ok if list(a.x)+list(b.x) has duplicates)
        # ys = [op(a(x,extrapolate=extrapolate),b(x,extrapolate=extrapolate)) for x in xs]
        ys,xs = piecewisebinop(a,b,op,extrapolate=extrapolate) # deals with (up to 3) duplicate x values correctly
        return Piecewise(ys,xs)
    def valid(self):
        return all(0<=xi for xi in self.x[1:]-self.x[:-1])
    def fourierintegrate(self,freq,norm=False):
        # exact fourier integral of piecewise linear function
        return fourierintegral(self.y,self.x,freq,norm=norm,returnwaves=returnwaves)
    def integrate(self):
        # exact integral of piecewise linear function
        return fourierintegral(self.y,self.x,freq=0).real
    def __add__(self,w): from operator import __add__; return self.binop(w,__add__)
    def __sub__(self,w): from operator import __sub__; return self.binop(w,__sub__)
    def __mul__(self,w): from operator import __mul__; return self.binop(w,__mul__)
    def __truediv__(self,w): from operator import __truediv__; return self.binop(w,__truediv__)
    def __floordiv__(self,w): from operator import __floordiv__; return self.binop(w,__floordiv__)
    def __pow__(self,w): from operator import __pow__; return self.binop(w,__pow__)
    def __mod__(self,w): from operator import __mod__; return self.binop(w,__mod__)
    def __iadd__(self,w): from operator import __iadd__; return self.binop(w,__iadd__)
    def __isub__(self,w): from operator import __isub__; return self.binop(w,__isub__)
    def __imul__(self,w): from operator import __imul__; return self.binop(w,__imul__)
    def __itruediv__(self,w): from operator import __itruediv__; return self.binop(w,__itruediv__)
    def __ifloordiv__(self,w): from operator import __ifloordiv__; return self.binop(w,__ifloordiv__)
    def __ipow__(self,w): from operator import __ipow__; return self.binop(w,__ipow__)
    def __imod__(self,w): from operator import __imod__; return self.binop(w,__imod__)


# ThickLine1 + ThickLine2 = ThickLine (error if 1.out!=2.in?)
# MultiThickLine / ThickBranch / ThickLineBunch
# see for tips:
# https://github.com/sbliven/geometry-simple/blob/master/geo.py
# http://toblerity.org/shapely/manual.html#introduction

if __name__ == '__main__':
    ...
