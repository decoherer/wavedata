# **wavedata**

Tools for processing waveform-like data such as time series or xy data.
Perform common scientific operations on xy data using numpy.
Store, analyze, transform, and display xy data.

## Installation

Clone the repository, navigate to the directory, and install the package
```
git clone https://github.com/decoherer/wavedata.git
cd wavedata
python3 setup.py install
```

Alternatively, install directly using pip:
```
pip install git+https://github.com/decoherer/wavedata.git
```

## Usage examples for the `Wave` class

The Wave class acts much like a numpy or pandas array.


```python
from wavedata import *
from plot import plot
import numpy as np

positions = [0,4,5,4]
times = [0,1,2,3]
w = Wave(positions,times,'w')
w
```




    Wave([0,4,5,4],[0,1,2,3],'w')



Plotting:


```python
w.plot(m='o',grid=1,xlabel='time (s)',ylabel='distance (m)',scale=(0.5,0.5),fork=False);
```


![png](README_files/README_7_0.png)


Some example operations:


```python
print([w.min(),w.max(),w.area(),w.mean(),w.sdev()])
```

    [0.0, 5.0, 13.0, 3.25, 1.920286436967152]
    

Some more examples:


```python
a,b,c = np.sqrt(w),0.2*w+1,(w+1).log()
plot(waves=[a.rename('a'),b.rename('b'),c.rename('c')],m='o',grid=1,xlabel='s',ylabel='',scale=(0.5,0.5),fork=False);
```


![png](README_files/README_11_0.png)



## Auto-generated Wave class usage examples

### 1. Using `maxloc`

The `maxloc` method identifies the location of the maximum value in the wave data.

```python
from wavedata import Wave

wave = Wave([1, 3, 4, 5, 4, 0], [0, 2, 3, 4, 6, 7])
max_location = wave.maxloc()
print("Location of maximum value:", max_location)
```

### 2. Using `xaty`

The `xaty` method retrieves the x-coordinate for a given y-value.

```python
x_value = wave.xaty(3)
print("X-coordinate for y=3:", x_value)
```

### 3. Using `scalex`

The `scalex` method scales the x-axis of the wave data.

```python
scaled_wave = wave.scalex(2)
print("Scaled Wave:", scaled_wave)
```

### 4. Using `quadmaxloc`

The `quadmaxloc` method finds the location of the maximum value using quadratic interpolation.

```python
quad_max_location = wave.quadmaxloc()
print("Quadratic Max Location:", quad_max_location)
```

### 5. Using `upsample`

The `upsample` method increases the sampling rate of the wave data.

```python
upsampled_wave = wave.upsample(num=4)
print("Upsampled Wave:", upsampled_wave)
```

### 6. Using `peakwidth`

The `peakwidth` method calculates the width of the peak at half maximum.

```python
width = wave.peakwidth(ylevel=0.5)
print("Peak width at half maximum:", width)
```

### 7. Using `integrate`

The `integrate` method computes the integral of the wave data.

```python
integral = wave.integrate()
print("Integral of the wave:", integral)
```




