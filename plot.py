
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import multiprocessing
disablepause = 0
noshow = 0
# todo: check https://github.com/matplotlib/cheatsheets
# todo: add linegapcolor for dashed lines

def colorlist(waves,c):
    replacements = {'k':'#000000','r':'#bb0000','g':'#007700','b':'#0000bb','w':'#ffffff','y':'#555500'}
    c = [w.c if (hasattr(w,'c') and w.c is not None) else (c[n%len(c)] if hasattr(c,'__len__') else n) for n,w in enumerate(waves)]
    c = [replacements[ci] if ci in replacements else ci for ci in c]
    return c
def isindex(c):
    return (isinstance(c,str) and c in '0123456789') or (isinstance(c,int) and c<9999)
def linestylelist(waves,l):
    return [w.l if (hasattr(w,'l') and w.l is not None) else (l[n%len(l)] if hasattr(l,'__len__') else l) for n,w in enumerate(waves)]
def linewidthlist(waves,lw):
    return [w.lw if (hasattr(w,'lw') and w.lw is not None) else (lw[n%len(lw)] if hasattr(lw,'__len__') else lw) for n,w in enumerate(waves)]
def lineindexlist(waves,li):
    return [w.li if (hasattr(w,'li') and w.li is not None) else (li[n%len(li)] if hasattr(li,'__len__') else li) for n,w in enumerate(waves)]
def markersizelist(waves,ms):
    return [w.ms if (hasattr(w,'ms') and w.ms is not None) else (ms[n%len(ms)] if hasattr(ms,'__len__') else ms) for n,w in enumerate(waves)]
def markerfilllist(waves,mf):
    return [w.mf if (hasattr(w,'mf') and w.mf is not None) else (mf[n%len(mf)] if hasattr(mf,'__len__') else mf) for n,w in enumerate(waves)]
def markeredgewidthlist(waves,mew,lw):
    assert len(waves)==len(lw)
    return [w.mew if (hasattr(w,'mew') and w.mew is not None) else (mew[n%len(mew)] if hasattr(mew,'__len__') else mew if mew is not None else 0.6*lwi) for n,(w,lwi) in enumerate(zip(waves,lw))]
def markerlist(waves,m):
    from matplotlib.markers import MarkerStyle
    def symbol(a): # square symmetric symbol, points in first quadrant are zip(a,a[::-1]), other quadrants are symmetric
        aa = list(a) + list(map(lambda x:-x,a[::-1])) + list(map(lambda x:-x,a)) + list(a[::-1])
        return list(zip(aa+aa[:1],aa[-len(a):]+aa[:1-len(a)]))
    m = m if hasattr(m,'__len__') else 'o' if 1==m else m if m else '' # 'osD*x+' '$\\clubsuit$','$\\spadesuit$','$\\odot$','$\\oplus$'
    hmarker = MarkerStyle("d")
    hmarker._transform.rotate_deg(90) # or use hmarker._transform.scale(1.0, 0.6)
    replacements = {'"':'',' ':'',
        'v':MarkerStyle("d"),'h':hmarker,
        'V':MarkerStyle("d"),'H':hmarker,
        'C':'$\\clubsuit$','S':'$\\spadesuit$',
        '♣':'$\\clubsuit$','♠':'$\\spadesuit$','♦':'$\\diamondsuit$','♥':'$\\heartsuit$',
        '+':symbol([3,1,1]),'t':symbol([4,1,0]),'x':symbol([1,2,0]),}
    # m = list(map(lambda c,d=replacements:d[c] if c in d else c, [m] if ''==m else m))
    m = [m] if ''==m else m
    m = [w.m if (hasattr(w,'m') and w.m is not None) else m[n%len(m)] for n,w in enumerate(waves)]
    m = list(map(lambda c,d=replacements:d[c] if c in d else c, m))
    return m
def plot(waves=[],image=None,contour=None,contourf=None,colormesh=None,colorbar=False,lines=[],texts=[],
        m='',l=0,c=None,mf=1,li=1,ms=4,lw=1.5,mew=None,
        fill=False,g=None,seed=None,showseed=False,
        x=None,y=None,xlabel=None,ylabel=None,xlim=(None,None),ylim=(None,None),
        rightwaves=[],rightlabel=None,save=None,savefolder='figs',legendtext=None,legendreverse=False,pause=True,groupsize=None,
        swap=False,scale=1.0,size=None,corner=None,xphase=False,fewerticklabels=False,
        xticks=None,xticklabels=None,yticks=None,yticklabels=None,clip=True,
        aspect=None,show=True,grid=False,colormap=None,vmin=None,vmax=None,levels=None,
        sort=False,connectgaps=False,axes=True,disable=False,darken=0,clickcoords=False,
        zerox=False,zeroy=False,bar=False,barwidth=0.8,errorbars=[],abbrev=False,fixnans=False,legendalpha=0.0,**kwargs):
    # fork process so computations can continue while plot is shown
    if kwargs.pop('fork',True):
        # print('waves before fork',[hasattr(w,'m') for w in waves]) # if wave attributes don't survive pickling we must specify markers,colors,etc explicitly before forking process
        if disable:
            return
        args = locals()
        kwargs = args.pop('kwargs')
        multiprocessing.Process(target=plot, args=[], kwargs={'fork':False,**args,**kwargs}).start()
        return
    # print('waves after fork',[hasattr(w,'m') for w in waves])
    c = colorlist(waves,c)
    l = linestylelist(waves,l)
    lw = linewidthlist(waves,lw)
    li = lineindexlist(waves,li)
    m = markerlist(waves,m)
    ms = markersizelist(waves,ms)
    mf = markerfilllist(waves,mf)
    mew = markeredgewidthlist(waves,mew,lw)

    # unexplicit kwargs: font,fontsize,log,loglog,fewerticks,linewidth,legendfontsize
    log,loglog,logx = kwargs.pop('log',False), kwargs.pop('loglog',False), kwargs.pop('logx',False)
    legendfontsize = kwargs.pop('legendfontsize',kwargs.get('fontsize',12))
    block = kwargs.pop('block',True)
    if fixnans:
        waves = [w.removenans() for w in waves]
    for w in waves:
        assert not any(np.isinf(w)), 'plotted waves connot contain inf'
    # https://matplotlib.org/3.1.1/tutorials/text/mathtext.html#symbols
    # https://matplotlib.org/3.1.1/api/markers_api.html#module-matplotlib.markers # diamond = [(0,5),(5,0),(0,-5),(-5,0),(0,5)] (size is normalized)

    # matplotlib.use('Qt5Cairo') # plt.rcParams['backend'] = 'Qt5Cairo'
    plt.rcParams['keymap.quit'] = ['ctrl+w','cmd+w','q','escape']
    plt.rcParams['font.sans-serif'] = plt.rcParams['font.family'] = kwargs.pop('font') if 'font' in kwargs else FontProperties(fname='c:/windows/fonts/arialuni.ttf').get_name() # 'Arial' # 'DejaVu Sans'
    # print('font.sans-serif',matplotlib.rcParams['font.sans-serif'],'font.family',matplotlib.rcParams['font.family'])
    # plt.rcParams['axes.facecolor']=plt.rcParams['savefig.facecolor']=plt.rcParams['figure.facecolor']='white'; # plt.rc('font',family='Arial'); 
    plt.style.use('seaborn-deep'); plt.rcParams['font.size'] = kwargs.pop('fontsize') if 'fontsize' in kwargs else 12
    plt.ion()
    if pause and not disablepause: plt.ioff()
    plt.figure()
    ft = kwargs.pop('fewerticks',False)
    if ft: plt.locator_params(nbins=3+ft)

    # plot 2D
    # def levellist(z): return [x*z.max() for x in np.linspace(0.1,0.9,9)]
    def levellist(z): return [x*z.max() for x in np.linspace(0.0,1.0,11)]
    # cms = ['Accent','Accent_r','Blues','Blues_r','BrBG','BrBG_r','BuGn','BuGn_r','BuPu','BuPu_r','CMRmap','CMRmap_r','Dark2','Dark2_r','GnBu','GnBu_r','Greens','Greens_r','Greys','Greys_r','LUTSIZE','OrRd','OrRd_r','Oranges','Oranges_r','PRGn','PRGn_r','Paired','Paired_r','Pastel1','Pastel1_r','Pastel2','Pastel2_r','PiYG','PiYG_r','PuBu','PuBuGn','PuBuGn_r','PuBu_r','PuOr','PuOr_r','PuRd','PuRd_r','Purples','Purples_r','RdBu','RdBu_r','RdGy','RdGy_r','RdPu','RdPu_r','RdYlBu','RdYlBu_r','RdYlGn','RdYlGn_r','Reds','Reds_r','ScalarMappable','Set1','Set1_r','Set2','Set2_r','Set3','Set3_r','Spectral','Spectral_r','YlGn','YlGnBu','YlGnBu_r','YlGn_r','YlOrBr','YlOrBr_r','YlOrRd','YlOrRd_r','__builtins__','__cached__','__doc__','__file__','__loader__','__name__','__package__','__spec__','_cmapnames','_generate_cmap','_reverse_cmap_spec','_reverser','afmhot','afmhot_r','autumn','autumn_r','binary','binary_r','bone','bone_r','brg','brg_r','bwr','bwr_r','cbook','cmap_d','cmapname','collections','colors','cool','cool_r','coolwarm','coolwarm_r','copper','copper_r','cubehelix','cubehelix_r','datad','flag','flag_r','get_cmap','gist_earth','gist_earth_r','gist_gray','gist_gray_r','gist_heat','gist_heat_r','gist_ncar','gist_ncar_r','gist_rainbow','gist_rainbow_r','gist_stern','gist_stern_r','gist_yarg','gist_yarg_r','gnuplot','gnuplot2','gnuplot2_r','gnuplot_r','gray','gray_r','hot','hot_r','hsv','hsv_r','jet','jet_r','ma','mpl','nipy_spectral','nipy_spectral_r','np','ocean','ocean_r','os','pink','pink_r','prism','prism_r','rainbow','rainbow_r','register_cmap','revcmap','seismic','seismic_r','spec','spec_reversed','spectral','spectral_r','spring','spring_r','summer','summer_r','terrain','terrain_r','winter','winter_r']
    cmap = plt.cm.Spectral_r if colormap is None else colormap # Spectral_r,cubehelix,inferno,seismic

    def xxyyzz(ww): return ww.xx.array(),ww.yy.array(),ww.array()
    if image is not None:
        plt.imshow(image.array().T,extent=[image.xs[0],image.xs[-1],image.ys[0],image.ys[-1]],vmin=vmin,vmax=vmax,origin='lower',interpolation='bilinear',cmap=cmap)
    if colormesh is not None:
        plt.pcolormesh(*xxyyzz(colormesh),cmap=cmap,vmin=vmin,vmax=vmax,shading='auto')
    if contour is not None:
        plt.contour(*xxyyzz(contour),levels=levellist(contour) if levels is None else levels,linewidths=0.5,colors='black')
    if contourf is not None:
        plt.contourf(*xxyyzz(contourf),levels=levellist(contourf.array()) if levels is None else levels)
    for text in texts: # https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.text.html
        plt.text(**text) # e.g. texts = [{'x':0,'y':1,'s':'test'}] # text(x, y, s, bbox=dict(facecolor='red', alpha=0.5)) # text(0.5, 0.5, 'matplotlib', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    if colorbar:
        plt.colorbar()

    # plot waves
    waves = sorted(waves,key=lambda w:-w.mean()) if sort else waves
    if connectgaps: waves = [w.removenans() for w in waves]
    groupsize = groupsize if groupsize is not None else (g if g is not None else 0)

    l = '0'*groupsize+'3'*groupsize+'1'*groupsize+'2'*groupsize+'4'*groupsize if groupsize else l
    ls = [('solid', (0, ())), ('densely dashed', (0, (5, 1))), ('dashed', (0, (5, 5))), ('loosely dashed', (0, (5, 10))), ('densely dotted', (0, (1, 1))), ('dotted', (0, (1, 5))), ('loosely dotted', (0, (1, 10))), ('densely dashdotted', (0, (3, 1, 1, 1))), ('dashdotted', (0, (3, 5, 1, 5))), ('loosely dashdotted', (0, (3, 10, 1, 10))), ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1))), ('dashdotdotted', (0, (3, 5, 1, 5, 1, 5))), ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10)))]
    ls = [l for l in ls if 'loosely' not in l[0]]
    ls = {**{str(n):l[1] for n,l in enumerate(ls)},**{'—':ls[0][1],'–':ls[1][1],'-':ls[2][1],'.':ls[3][1],'"':'None',' ':'None','':'None',None:'None','None':'None'}}

    cs = seedcolors(seed,waves,showseed,darken)
    cs = (cs*groupsize)[:groupsize] if groupsize else cs
    # colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] if colors[0] is None else colors # usage: color=colors[n%len(colors)]
    def vline(x):
        y0,y1 = min(w.min() for w in waves),max(w.max() for w in waves)
        return {'xdata':(x,x),'ydata':(y0,y1),'color':'k','linestyle':(0, (1, 1))}
    lines = [x if isinstance(x,dict) else vline(x) for x in lines]
    for line in lines:
        plt.gca().add_line(line) if isinstance(line,matplotlib.lines.Line2D) else plt.gca().add_line(plt.Line2D(**line)) # line = {'xdata':(x0,x1),'ydata':(y0,y1),'color':'k','linestyle':(0, (1, 1))}
        # TODO:  check xlim,ylim

    def abbreviate(ss):
        ss = [str(s) for s in ss]
        if None in ss or not ss:
            return ss
        def hassameend(n): # n = length of end
            # all(s==ss[0] for s in ss) # all strings equal
            return len({s[-n:] for s in ss}) <= 1
        n,s0 = 0,ss[0]
        while hassameend(n+1) and n<len(s0): n += 1       # find common end
        while n>0 and not s0[len(s0)-n] in ', ': n -= 1   # abbreviate only whole words
        return ss[:1] + [s[:len(s)-n] for s in ss[1:]]    # remove ending from all but first
    labelinline = []
    names = [w.name if hasattr(w,'name') and w.name is not None else '' for w in waves]
    names = abbreviate(names) if abbrev else names
    for n,w in list(enumerate(waves))[::-1 if legendreverse else +1]:
        wx = w.x if hasattr(w,'x') else np.arange(len(w))
        # nx,ny = (n//groupsize,n%groupsize) if swap else (n%groupsize,n//groupsize)
        nx,ny = n,n
        c0 = (cs[int(c[nx%len(c)])%len(cs)] if isindex(c[nx%len(c)]) else c[nx%len(c)]) if c else cs[nx%len(cs)]
        l0 = ls[ str(l[n%len(l)]) if l else str(ny%len(ls)) ]
        lw0 = lw[ny%len(lw)] if hasattr(lw,'__len__') else lw
        li0 = int(li[n%len(li)]) if hasattr(li,'__len__') else li  # style 0=supress, 1=legend, 2=above line, 3=on line, 4=below line
        m0 = m[n%len(m)] if m else m # m0 = m[ny%len(m)] if m else m
        ms0 = int(ms[n%len(ms)]) if hasattr(ms,'__len__') else ms
        mf0 = mf[n%len(mf)] if hasattr(mf,'__len__') else mf
        mf0 = ('#ffffff'+2*mf0 if mf0 in '0123456789abcdef' else mf0) if isinstance(mf0,str) else addalpha(c0,mf0)
        mew0 = mew[n%len(mew)] if hasattr(mew,'__len__') else mew
        # m0,nl,mf0,li0 = m[n%len(m)] if m else '', int(l[n%len(l)]) if l else n%len(ls), int(mf[n%len(mf)]) if mf else 1, int(li[n%len(li)]) if li else 1 # print(c0,m0,l0,mf0,li0,lw0)
        if errorbars:
            line = plt.errorbar(wx,w,yerr=errorbars[n],capthick=0,label=names[n] if 1==li0 else'',marker=m0,linestyle=l0,color=c0,markersize=ms0,linewidth=lw0,mfc=mf0,mec=c0,mew=mew0,**kwargs)
        elif bar:
            line = plt.bar(wx,w,label=names[n] if 1==li0 else'',width=barwidth,color=c0,**kwargs) # ,mfc=mf0,mec=c0,mew=mew0
        elif fill:
            line = plt.fill(wx,w,facecolor=c0,label=names[n] if 1==li0 else'',linestyle=l0,edgecolor='k',linewidth=lw0)
        else:
            line = plt.plot(wx,w,label=names[n] if 1==li0 else'',clip_on=clip,zorder=3,
                marker=m0,linestyle=l0,color=c0,markersize=ms0,linewidth=lw0,mfc=mf0,mec=c0,mew=mew0,gapcolor=None,**kwargs)
        if 1<li0:
            labelinline += [[line[0],names[n],li0]]
        # TODO marker zorder: plt.scatter(wx,w,zorder=2.1,linestyle=ls[l0][1],marker=m0,label=None,color=c0,s=ms0,**(kwargs if mf0 else {**kwargs,'facecolors':'white'}))

    if x is not None or xlabel is not None: plt.xlabel(x if x is not None else xlabel)
    if y is not None or ylabel is not None: plt.ylabel(y if y is not None else ylabel)

    if labelinline:
        lines,labels,styles = zip(*labelinline)
        labelLines(lines, labels, styles, dy=0, align=True, zorder=2.5)
    if rightwaves:
        plt.tick_params('y', colors=cs[0])
        plt.ylabel(y if y is not None else ylabel, color=cs[0])
        rightwaves = rightwaves if hasattr(rightwaves, '__len__') else [rightwaves]
        ax2 = plt.twinx()
        for i,w in enumerate(rightwaves):
            # wave attributes don't survive pickling
            # workaround for now using fixed linestyle list
            # TODO: pickling now works, need to fix this
            wx = w.x if hasattr(w,'x') else np.arange(len(w))
            ax2.plot(wx,w,color='darkred',linestyle=ls[f"{i}"],label=w.name)
        ax2.tick_params('y', colors='darkred')
        if rightlabel: ax2.set_ylabel(rightlabel, color='darkred')

    if legendtext or any(hasattr(w,'name') and w.name for w in waves):
        hs,_ = plt.gca().get_legend_handles_labels()
        leg = plt.legend(handles=hs if hs else [matplotlib.patches.Patch(color='none', label=legendtext)],
            framealpha=legendalpha,title=legendtext if hs else '',loc=corner,
            prop={'size':(legendfontsize if legendfontsize else 12)}) #,loc='upper right',loc='center left',loc='upper center',loc='center',frameon=False
        leg._legend_box.align = 'left'
    if grid:
        if 2==grid:
            plt.grid(True, which='major', linestyle='-', color='0.85')
        else:
            plt.minorticks_on()
            plt.grid(True, which='major', linestyle='-', color='0.85')
            plt.grid(True, which='minor', linestyle='--', color='0.85')
    if zerox: plt.axvline(0,color='k')
    if zeroy: plt.axhline(0,color='k')
    if logx or loglog: plt.xscale('log')
    if log or loglog: plt.yscale('log')
    # xleft,xright = plt.xlim()  # return the current xlim
    xlim = (min([w.x.min() for w in waves]),max([w.x.max() for w in waves])) if xlim=='f' or xlim=='flush' else xlim
    ylim = (min([w.y.min() for w in waves]),max([w.y.max() for w in waves])) if ylim=='f' or ylim=='flush' else ylim
    if isinstance(xlim,float) and 0<abs(xlim)<1:
        plt.margins(x=xlim); xlim = None
    if isinstance(ylim,float) and 0<abs(ylim)<1:
        plt.margins(y=ylim); ylim = None
    plt.xlim(xlim); plt.ylim(ylim)
    if not (loglog or log or logx):
        plt.ticklabel_format(style='plain')
        plt.ticklabel_format(useOffset=False) # don't use "+1.55e3" notation at top of axis
    ax = plt.gca()
    if xphase:
        from matplotlib.ticker import FuncFormatter, MultipleLocator
        ax.xaxis.set_major_formatter(FuncFormatter(lambda val,pos: '{:.0g}$\pi$'.format(val/np.pi) if val !=0 else '0'))
        ax.xaxis.set_major_locator(MultipleLocator(base=np.pi))
    if 2==clip:
        (x0,x1),(y0,y1) = ax.get_xlim(),ax.get_ylim()
        clip_rect = matplotlib.patches.Rectangle((x0-0.05*(x1-x0), y0), 1.1*(x1-x0), y1-y0, transform=ax.transData)
        for artist in ax.get_children():
            # if isinstance(artist, (matplotlib.lines.Line2D, matplotlib.collections.PathCollection)):
            #     artist.set_clip_on(True)
                artist.set_clip_path(clip_rect)

    if fewerticklabels:
        for label in plt.gca().get_xticklabels()[0::2]: label.set_visible(False)
    if xticks is not None:
        plt.xticks(xticks,xticklabels,rotation=90)
    if yticks is not None:
        plt.yticks(yticks,yticklabels,rotation=0)
    if not axes:
        plt.axis('off')
        plt.tight_layout()
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    fig = plt.gcf()
    sx,sy = scale if hasattr(scale,'__len__') else (scale,scale)
    fig.set_size_inches((fig.get_size_inches()[0]*sx, fig.get_size_inches()[1]*sy) if size is None else size)
    if aspect: # or not (colormesh is None and image is None and contour is None and contourf is None):
        # aspect = +1, aspect fixed when zoomed
        # aspect = -1, aspect not fixed when zoomed
        plt.gca().set_aspect(abs(aspect))
        if aspect<0:
            fig.set_size_inches(fig.get_size_inches())
            plt.gca().set_aspect('auto')
    f = zoom_factory(plt.gca()) # enable mouse zoom

    if clickcoords:
        def onclick(event):
            # nonlocal coords
            # coords.append((event.xdata, event.ydata))
            print(f" ({event.xdata:g},{event.ydata:g})")
        cid = plt.gcf().canvas.mpl_connect('button_press_event', onclick)

    import os,time
    savefolder = savefolder if os.path.isdir(savefolder) else '.'
    if save:
        savename = save.replace('.png','')
        savefile = savefolder.rstrip('/')+'/'+savename+'.png'
        plt.savefig(savefile, bbox_inches='tight',dpi=300)
        # add_custom_watermark(savefile,savename,overwrite=True)
        # time.sleep(2)
        # os.utime(savefile, (time.time(), time.time()))
    else:
        savefile = savefolder.rstrip('/')+'/out.png'
        plt.savefig(savefile, bbox_inches='tight',dpi=300)
    if show and not noshow:
        plt.show(block=block); plt.pause(0.1)
    return savefile
def defaultcolors():
    seaborn = ['#4C72B0','#55A868','#C44E52','#8172B2','#CCB974','#64B5CD'] # plt.rcParams['axes.prop_cycle'].by_key()['color']
    msofficetheme1 = ['#4F81BD','#C0504D','#9BBB59','#8064A2','#4BACC6','#F79646','#404080','#804080']
    empusa_ = ['#c92a28','#e69301','#1f8793','#13652b','#48233b','#e3b3ac']
    roygbiv_warm_ = ['#705f84','#687d99','#6c843e','#fc9a1a','#dc383a']
    verena = ['#f1594a', '#f5b50e', '#14a160', '#2969de', '#885fa4']
    rag_mysore = ['#ec6c26', '#613a53', '#e8ac52', '#639aa0']
    roygbiv_toned_ = ['#817c77','#396c68','#89e3b7','#f59647','#d63644']
    tundra3_ = ['#87c3ca','#7b7377','#b2475d','#eb7f64','#d9c67a']
    iiso_daily_ = ['#e76c4a', '#7f8cb6', '#1daeb1', '#ef9640', '#f0d967']
    kov_04_ = ['#d03718', '#292b36', '#33762f', '#ce7028', '#689d8d']
    kov_06_ = ['#a87c2a','#f14616','#017724','#0e2733','#2b9ae9']
    tricolor = ['#ec643b', '#56b7ab', '#1f1e43', '#f8cb57']
    olympia = ['#ff3250', '#ffb33a', '#008c36', '#0085c6', '#4c4c4c']
    cc232 = ['#5c5f46', '#ff7044', '#66aeaa', '#ffce39']
    cc242 = ['#bbd444', '#fcd744', '#fa7b53', '#423c6f']
    coolors = [ # from coolors.co
    ["#9aa0a8","#a7c4b5","#a9d8b8","#72705b","#8b6220","#720e07","#45050c","#273c2c","#0d5c63","#236b71"],
    # ["#e5c1bd","#d2d0ba","#b6be9c","#7b9e87","#5e747f","#32373b","#4a5859","#c83e4d","#414535"],
    ["#090c08","#474056","#757083","#8a95a5","#b9c6ae","#f9eae1","#d1be9c","#ffbfb7","#ffd447","#a1cdf4"],
    ["#01161e","#124559","#598392","#aec3b0","#eff6e0","#d4c5c7","#dad4ef","#65532f","#736342","#807153"],
    ["#780116","#f7b538","#db7c26","#d8572a","#c32f27","#044389","#7cafc4","#5995ed","#9fb8ad","#475841"],
    ["#c9cba3","#ffe1a8","#e26d5c","#723d46","#472d30","#004777","#1d2f6f","#8390fa"],
    ["#d5573b","#885053","#777da7","#94c9a9","#c6ecae","#08090a","#f4f7f5","#222823"],
    ["#331832","#694d75","#1b5299","#9fc2cc","#f1ecce","#cea07e","#edd9a3","#e2e8c0"]]
    a = [seaborn, msofficetheme1,
    empusa_,roygbiv_warm_,verena,rag_mysore,roygbiv_toned_,tundra3_,iiso_daily_,kov_04_,kov_06_,tricolor,olympia,cc232,cc242,
    # ['#3f3d99','#993d71','#998b3d','#3d9956','#3d5a99','#993d90','#996d3d','#43993d','#3d7999','#843d99','#994e3d','#62993d','#3d9799','#653d99','#993d4b'], #seed=15
    ['#3f3d99','#993d71','#998b3d','#3d9956','#3d5a99','#993d90','#996d3d','#43993d','#3d7999','#843d99','#994e3d','#62993d'], #seed=15
    ['#575757','#a41b0b','#311b10','#cabb9c','#cf9807','#1b4129','#204f6d','#142551','#a89ab3','#856b74','#7c0818'], #seed=16
    ['#807f7f','#cb837f','#683e35','#cdaf9a','#493d33','#c1b493','#b2b990','#759381','#92a1ab','#575859','#435772'],
    # ['#9f2520','#a84e2c','#3e1e0f','#f1c0a2','#d67936','#d3b96f','#717f8c','#3f4f5f','#2b3d61','#4c2d32'],
    ['#740d06','#ad5000','#e7a312','#2a6a1e','#bad5dd','#1f82bb','#a38fc0','#251d2a','#f00002'],
    ['#681108','#dc6707','#efc600','#a3c1e5','#505ea5','#988dcf','#270d1a','#de88a3','#b8223f'],
    # ['#3b2d2a','#5d2f20','#88441d','#a48533','#dee3ad','#adc5ab','#69868c','#666769','#848283'],
    # ['#d2bbaf','#686d44','#9daf7c','#566f64','#869799','#6c757b','#647e91','#9b8f94'],
    # ['#be4d00','#e17e18','#d3cf74','#aeb0a5','#9cb057','#457357','#29475f','#26457b','#14295e','#a8a2ce'],
    # ['#bc7933','#81490c','#3f4c42','#658b6e','#8a80a2','#371e54','#5d4c70'],
    ['#66c2a5','#fc8d62','#8da0cb','#e78ac3','#a6d854','#ffd92f','#e5c494','#b3b3b3'],
    ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628','#f781bf','#999999'],
    ['#405e73','#80bfe6','#2e8c2e','#8f8c45','#d9994d','#d9b366','#fadb96'],
    ['#9e0242','#e85c47','#fdbf70',darker('#fffebe',0.85),'#bee59f','#54adaf','#5f4fa2'], # seed=23
    *coolors,
    ]
    return a
def seedcolors(seed,ws,showseed,darken):
    sd = [ord(c) for c in seed] if isinstance(seed,str) else seed
    sd = sum(sd) if hasattr(sd,'__len__') else sd
    n = int(sd) if sd is not None else len(ws) + (len(ws[0]) if len(ws) else 0)
    mrgb = defaultcolors()
    cs = mrgb[n % len(mrgb)]
    if showseed: print('seed:',n % len(mrgb))
    def extendcolors(cs,size):
        return cs if size<len(cs) else cs + [darker(c) for c in cs]*(size//len(cs))
    ecs = extendcolors(cs,len(ws))
    return [darker(c,0.9**darken) for c in ecs] if darken else ecs
def darker(rgb,x=0.75):
    def dark(s):
        h = hex(int(x*int(s,base=16))).replace('0x','')
        return '0'*(1==len(h)) + h
    s = '#'+''.join([dark(rgb[i:i+2]) for i in (1,3,5)])
    assert '#'==rgb[0] and 7==len(rgb), f'input color format {rgb} should be #RRGGBB'
    assert '#'==s[0] and 7==len(s), f'output color format {s} should be #RRGGBB'
    return s
def addalpha(c,x): # c is e.g. '#11aaff' and x is x float 0<=x<=1
    c = '#000000' if 'k'==c else c
    c = c[:7] if '#'==c[0] and 9==len(c) else c # remove alpha from color if present
    assert '#'==c[0] and 7==len(c), c
    assert 255==int(256-1e-9) and 0<=x<=1, x
    return c+f'{int(x*(256-1e-9)):02x}'
def zoom_factory(axis, scale_factor=1.4):
    # returns zooming functionality to axis. From C:\Python38\Lib\site-packages\phidl\quickplotter.py
    # which was based on https://gist.github.com/tacaswell/3144287
    def zoom_fun(event, ax, scale): # zoom when scrolling
        if event.inaxes == axis:
            scale_factor = np.power(scale,-event.step)
            xdata = event.xdata
            ydata = event.ydata
            x_left = xdata - ax.get_xlim()[0]
            x_right = ax.get_xlim()[1] - xdata
            y_top = ydata - ax.get_ylim()[0]
            y_bottom = ax.get_ylim()[1] - ydata
            ax.set_xlim([xdata - x_left * scale_factor,
                         xdata + x_right * scale_factor])
            ax.set_ylim([ydata - y_top * scale_factor,
                         ydata + y_bottom * scale_factor])
            ax.figure.canvas.draw()
            fig.canvas.toolbar.push_current() # Update toolbar so back/forward buttons work
    fig = axis.get_figure()
    fig.canvas.mpl_connect('scroll_event', lambda event: zoom_fun(
        event, axis, scale_factor))
def animate(func,num,xlim=None,ylim=None,lw=1,delay=0,save='',aspect=None,show=True):
    from matplotlib.animation import FuncAnimation
    plt.style.use('seaborn-pastel')
    fig = plt.figure()
    xlim = xlim if xlim is not None else (func(0)[0].min(),func(0)[0].max())
    ylim = ylim if ylim is not None else (func(0)[1].min(),func(0)[1].max())
    ax = plt.axes(xlim=xlim, ylim=ylim)
    if aspect: plt.gca().set_aspect(aspect)
    line, = ax.plot([], [], lw=lw)
    def f(i):
        # x = np.linspace(0, 4, 1000)
        # y = np.sin(2 * np.pi * (x - 0.01 * i))
        x, y = func(i)
        line.set_data(x, y)
        return line,
    anim = FuncAnimation(fig, f, frames=num, interval=delay, blit=True) # interval = ms delay
    if save: anim.save(save+'.gif') #, writer='imagemagick')
    if show: plt.show()

from math import atan2, degrees
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import date2num, DateConverter, num2date
from matplotlib.container import ErrorbarContainer
from datetime import datetime
def labelLine(line, x, dy=0, label=None, align=True, **kwargs): # https://github.com/cphyc/matplotlib-label-lines/blob/master/labellines/core.py
    ax = line.axes
    xdata = line.get_xdata()
    ydata = line.get_ydata()
    mask = np.isfinite(ydata)
    if mask.sum() == 0:
        raise Exception('The line %s only contains nan!' % line)
    # Find first segment of xdata containing x
    for i, (xa, xb) in enumerate(zip(xdata[:-1], xdata[1:])):
        if min(xa, xb) <= x <= max(xa, xb):
            break
    else:
        # raise Exception('x label location is outside data range!',x)
        print('x label location is outside data range!',x,label)
        i = len(xdata)//2
        xa,xb = xdata[i], xdata[i+1]
    def x_to_float(x):
        return date2num(x) if isinstance(x, datetime) else x
    xfa = x_to_float(xa)
    xfb = x_to_float(xb)
    ya = ydata[i]
    yb = ydata[i + 1]
    y = ya + (yb - ya) * (x_to_float(x) - xfa) / (xfb - xfa)
    if not (np.isfinite(ya) and np.isfinite(yb)):
        warnings.warn(("%s could not be annotated due to `nans` values. Consider using another location via the `x` argument.") % line, UserWarning)
        return
    if not label: label = line.get_label()
    if align:
        # Compute the slope and label rotation
        # print('ax.transData',ax.transData)
        screen_dx, screen_dy = ax.transData.transform((xfa, ya)) - ax.transData.transform((xfb, yb))
        rotation = (degrees(atan2(screen_dy, screen_dx)) + 90) % 180 - 90
    else:
        rotation = 0
    # print('align',align,'rotation',rotation,'screen_dx, screen_dy',screen_dx, screen_dy)
    # Set a bunch of keyword arguments
    if 'color' not in kwargs: kwargs['color'] = line.get_color()
    if ('horizontalalignment' not in kwargs) and ('ha' not in kwargs): kwargs['ha'] = 'center'
    if ('verticalalignment' not in kwargs) and ('va' not in kwargs): kwargs['va'] = 'center'
    # if 'backgroundcolor' not in kwargs: kwargs['backgroundcolor'] = ax.get_facecolor()
    if 'clip_on' not in kwargs: kwargs['clip_on'] = True
    if 'zorder' not in kwargs: kwargs['zorder'] = 2.5
    txt = ax.text(x, y+dy, label, rotation=rotation, **kwargs)
def labelLines(lines, labels, styles, dy=0, align=True, xvals=None, **kwargs):
    ax = lines[0].axes
    if xvals is None:
        xvals = ax.get_xlim()  # set axis limits as annotation limits, xvals now a tuple
    if type(xvals) == tuple:
        xmin, xmax = xvals
        xscale = ax.get_xscale()
        if xscale == "log":
            xvals = np.logspace(np.log10(xmin), np.log10(xmax), len(lines)+2)[1:-1]
        else:
            xvals = np.linspace(xmin, xmax, len(lines)+2)[1:-1]
        if isinstance(ax.xaxis.converter, DateConverter):
            # Convert float values back to datetime in case of datetime axis
            xvals = [num2date(x).replace(tzinfo=ax.xaxis.get_units())
                     for x in xvals]
    for line, x, label, style in zip(lines, xvals, labels, styles):
        d = {**kwargs, 'backgroundcolor':ax.get_facecolor()} if 3==style else kwargs
        label = (4<=style)*(style-3)*'\n' + str(label) + (2==style)*'\n'
        labelLine(line, x, dy, label, align, **d)

def create_qr_code(data, size=None):
    from PIL import Image
    import qrcode
    qr = qrcode.QRCode(
        version=15,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=1,
        border=0)
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    img = img.convert("RGBA")  # Convert the QR code to RGBA mode
    if size:
        img = img.resize((size, size), Image.ANTIALIAS)
    return img
def add_custom_watermark(image_path, watermark_data, overwrite=False):
    from PIL import Image
    watermark = create_qr_code(watermark_data)
    base_image = Image.open(image_path).convert("RGBA")  # Convert the base image to RGBA mode
    width, height = base_image.size
    watermark_width, watermark_height = watermark.size
    # print('watermark.size',watermark.size) # 77
    assert watermark_width==watermark_height==77
    position = (width - watermark_width, height - watermark_height)
    for x in range(watermark_width):
        for y in range(watermark_height):
            watermark_pixel = watermark.getpixel((x, y))
            is_white = (watermark_pixel[0] + watermark_pixel[1] + watermark_pixel[2]) % 2 == 1
            base_x, base_y = position[0] + x, position[1] + y
            base_pixel = base_image.getpixel((base_x, base_y))
            base_sum = sum(base_pixel[:-1])  # Ignore the alpha channel
            if is_white and base_sum % 2 == 0:
                new_pixel = tuple([value+1 if value<255 else 254 for value in base_pixel[:-1]] + [base_pixel[-1]])
            elif not is_white and base_sum % 2 == 1:
                new_pixel = tuple([value-1 if 0<value else 1 for value in base_pixel[:-1]] + [base_pixel[-1]])
            else:
                new_pixel = base_pixel
            # print('is_white',int(is_white),'sum',sum(new_pixel[:-1])%2,'base_sum',base_sum%2,'new_pixel',new_pixel)
            base_image.putpixel((base_x, base_y), new_pixel)
    if overwrite:
        base_image.save(image_path)
    return base_image
def read_qr_code(image, crop_size=None, debug=False):
    from PIL import Image
    from pyzbar.pyzbar import decode
    def convert_image_bw(image):
        width, height = image.size
        bw_image = Image.new('RGBA', (width, height))  # Create a new RGBA image for black and white conversion
        for x in range(width):
            for y in range(height):
                pixel = image.getpixel((x, y))
                is_white = (pixel[0] + pixel[1] + pixel[2]) % 2 == 1
                if is_white:
                    bw_image.putpixel((x, y), (255, 255, 255, pixel[3]))
                else:
                    bw_image.putpixel((x, y), (0, 0, 0, pixel[3]))
        if debug: bw_image.save('output_image_bw.png')
        return bw_image
    if crop_size:
        width, height = image.size
        cropped = image.crop((width - crop_size, height - crop_size, width, height))
        image = cropped.resize((2*crop_size,2*crop_size), Image.NEAREST)
    im = convert_image_bw(image) # im.show()
    return decode(im)

def multiplot(product=True,verbose=0):
    def decorator(f):
        import itertools
        def wrapper(*args, **kwargs):
            if verbose: print('args',args,'kwargs',kwargs)
            args = [a if isinstance(a,list) else [a] for a in args]
            ns = [len(a) for a in args] # print('ns',ns)
            # r,c,l,*_ = sorted(ns+3*[1],reverse=1) # 
            r,c,l,*_ = ns #  # print('r,c,l',r,c,l)print('r,c,l',r,c,l)
            assert product or all(max(ns)%n==0 for n in ns)
            runs = list(itertools.product(*args)) if product else list(zip(*[a*(max(ns)//len(a)) for a in args])) # list(zip(*[a*int(ceil(n/len(a))) for a in args]))
            N = len(runs)
            if verbose: print(runs)
            if 1==len(runs):
                return f(*runs[0], **kwargs, mp=dict())
            ws = [f(*ai, **kwargs, mp=dict(disable=1)) for ai in runs[1:]]
            from waves import Wave
            m = 1 if isinstance(ws[0],Wave) else len(ws[0])
            ws = ws if 1==m else [v for vs in ws for v in vs]
            cs = ''.join([s*(N//r) for s in '0123456789'[:r]]) # print('cs',cs)
            ls = '0123456789'[:N//r]*r # print('ls',ls)
            d = dict(show=1,waves=ws,c=cs,l=ls)
            w0 = f(*runs[0], **kwargs, mp=d)
            ws = [w0] if 1==m else [*w0] + ws
            return [w.setplot(c=c,l=l,override=False) for c,l,w in zip(m*cs,m*ls,ws)]
        return wrapper
    return decorator
def multiplotdemo():
    from waves import Wave
    @multiplot(verbose=True,product=True)
    def test(h,g,m,a=0,b=0,mp=None):
        xs = np.linspace(-2,2,5)
        u = Wave(+h*xs**2+g*xs+m,xs,f"u {h} {g} {m}")
        u.plot(x='x',y='y',grid=1,**mp)
        return u
    # test(1,2,3)
    # us = test(1,[-2,2],[3,4,5])
    # us = test([3,4,5],1,[-2,2])
    # us = test([3,4,5,6],[10,20],3,4,[-2,2])
    us = test([3,4,5,6],[10,20],[-2,2])
    Wave.plots(*us)
    ## not supported:
    # @multiplot(verbose=True,product=True)
    # def test2(h,g,m,*,mp=None):
    #     xs = np.linspace(-2,2,5)
    #     u = Wave(+h*xs**2+g*xs+m,xs,f"u {h} {g} {m}")
    #     v = Wave(-h*xs**2-g*xs-m,xs,f"v {-h} {-g} {-m}")
    #     Wave.plots(u,v,x='x',y='y',grid=1,**mp)
    #     return u,v
    # # test2(1,2,3)
    # # us = test2([3,4,5],1,[-2,2])
    # # us = test2([3,4,5],[-2,2],[10,20])
    # # Wave.plots(*us)

def multiplots(product=True,verbose=0):
    def decorator(f):
        import itertools
        def wrapper(*args, **kwargs):
            args = [a if isinstance(a,list) else [a] for a in args]
            ns = [len(a) for a in args] # print('ns',ns)
            r,c,l,*_ = ns #  # print('r,c,l',r,c,l)
            assert product or all(max(ns)%n==0 for n in ns)
            runs = list(itertools.product(*args)) if product else list(zip(*[a*(max(ns)//len(a)) for a in args])) # list(zip(*[a*int(ceil(n/len(a))) for a in args]))
            N = len(runs)
            cs = ''.join([s*(N//r) for s in '0123456789'[:r]]) # print('cs',cs)
            ls = '0123456789'[:N//r]*r # print('ls',ls)
            if verbose: print(runs)
            class aa:
                def __getitem__(self,i):
                    return dict(disable=1) if 1<N else dict()
            if 1==N:
                return f(*runs[0], **kwargs, mp=aa())
            vss = [f(*ai, **kwargs, mp=aa()) for ai in runs[1:]]
            assert hasattr(vss[0],'__len__')
            # nplots = len(vss[0]) # print('nplots',nplots)
            def transpose(listoflists):
                return [list(ll) for ll in zip(*listoflists)]
            ds = [dict(show=1,waves=ws,c=cs,l=ls) for ws in transpose(vss)] # for d in ds: print(d)
            w0s = f(*runs[0], **kwargs, mp=ds)
            wss = [[w]+ws for w,ws in zip(w0s,transpose(vss))]
            return [[w.setplot(c=c,l=l,override=False) for c,l,w in zip(cs,ls,ws)] for ws in wss]
        return wrapper
    return decorator
def multiplotsdemo():
    from waves import Wave
    @multiplots(verbose=True,product=True)
    def test(h,g,m,*,mp=None):
        xs = np.linspace(-2,2,5)
        u = Wave(+h*xs**2+g*xs+m,xs,f"u {h} {g} {m}")
        v = Wave(-h*xs**2-g*xs-m,xs,f"v {h} {g} {m}")
        u.plot(x='x',y='y',grid=1,**mp[0])
        v.plot(x='x',y='y',grid=1,**mp[1])
        return u,v
    # test(1,2,3)
    # us,vs = test([-2,2],1,[3,4,5])
    # us,vs = test([-2,2],[0,1],[3,4,5])
    us,vs = test([3,4,5],[-2,2],[10,20])
    # us,vs = test([3,4,5],[-2,2],1)
    # Wave.plots(*us,*vs)

def colortest():
    from waves import Wave
    nn = len(defaultcolors())
    print(nn)
    w = Wave([0,1,0,1])
    for i,j in [(0,8),(8,16),(16,24),(24,None)]:
        dc,ws = defaultcolors(),[]
        for n,cs in enumerate(dc[i:j if j is not None else len(dc)]):
            print(i+n,len(cs),' '.join(cs))
            ws += [(w-0.08*k+i+n).setplot(c=c) for k,c in enumerate(cs)]
        Wave.plots(ws,lw=4,xlim='f',fewerticks=9)
def watermarktest(file='pmc.png'):
    watermark_data = 'This is a machine-readable watermark.'
    image_with_watermark = add_custom_watermark(file, watermark_data)
    image_with_watermark.save('pmc-wm.png')
    decoded_data = read_qr_code(image_with_watermark,77)
    if decoded_data:
        print(f"Watermark: {decoded_data[0].data.decode('utf-8')}")
    else:
        print("No watermark found.")
def findwatermark(file):
    from PIL import Image
    decoded_data = read_qr_code(Image.open(file),77)
    print(decoded_data[0].data.decode('utf-8') if decoded_data else '-qr code not found-')


if __name__ == '__main__':
    # multiplotdemo()
    # multiplotsdemo()
    # colortest()
    print(len(defaultcolors()))
    watermarktest()
    # findwatermark("c:/py/work/sell/figs/modifieddiffusiontest.png")
