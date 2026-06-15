# torched-tacaw workshop 24/10/2025
```
╺┳╸┏━┓┏━┓┏━╸╻ ╻┏━╸╺┳┓   ╺┳╸┏━┓┏━╸┏━┓╻ ╻   
 ┃ ┃ ┃┣┳┛┃  ┣━┫┣╸  ┃┃╺━╸ ┃ ┣━┫┃  ┣━┫┃╻┃   
 ╹ ┗━┛╹┗╸┗━╸╹ ╹┗━╸╺┻┛    ╹ ╹ ╹┗━╸╹ ╹┗┻┛   
=======================================
 LET'S IMPROVE THIS TUTORIAL TOGETHER !
```
if you have any suggestions, let's add them 
straight away into this file!

### topics 

1. what it is and where to find it
2. instalation, where to find it on dardel
3. github, file organization
4. general architecture
5. demo 
   - hBN planewave
   - STEM
   - postprocessing
7. what it can do now
8. possibility of extension (magnon tacaw)


## What is torched tacaw?
_import torched_tacaw as tt_

tt is a computational engine employing **torch** library 
and Hamish Brown's **py_multislice** which enables it 
to do large scale energy and momentum resolved STEM 
calculations. Thanks to torch, it can run on GPU's.

It was designed mainly to handle the huge data traffic 
that is needed for the computation at some point but can be 
discarded on the fly. For example for a good image quality 
we need to have quite large momentum space but most of it is
not needed for postprocessing (detection). This data is 
discarded on-the-fly. Since the problem we are solving
is in some aspect embarasingly paralel, tt uses this and 
enables for paralel computation of any computation chunks
at the same time. In my current workflow, 1 GPU corresponds 
to one calculator but in principle, there can be arbitrrily 
many calculators running in paralel.

---

## Instalation & Dardel

see [[../../README.md]] for installation guide

**note about venvs:** a venv can be created by `python -m venv <name>` 
and libraries installed by `pip install numpy torch scipy` you then 
source it by `source <name>/bin/activate`  


### Dardel

On dardel there is a virtual environment in
``
/cfs/klemming/home/o/osmera/emcdetal/orbvenvgpu/venv
``
in which torched tacaw is installed. (Venv can be activated by 
`source /cfs/klemming/home/o/osmera/emcdetal/orbvenvgpu/venv/bin/activate`)
Torched tacaw can be loaded by simple
```python
import torched_tacaw as tt
```
or sth like

```python
from torched_tacaw import Config, Dispatcher
from torched_tacaw import postprocessing as tp
```

### Lumi

I did not yet made a venv on lumi but i will make sure it works


---

## Github & File organization 

the repo's main branch is called `main` - it is the one 
accesible on dardel currently.

there is a development branch `dev` and further developments 
shall be done on its subbranches before merging into `dev` 
and eventually `main`. 

### Github

in order to comunicate with github, one currently needs 
to be able to autheticate themselves. The easiest way to do
it is using presonal token.

to generate prsonal token, in github go to 
`settings > developer settings > personal tokens`
and then generate a token (classic is simpler and enough), e.g.:
`
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
`
then when using git clone you do this:
instead of:
```bash
git clone https://github.com/Martexiwell/torched_tacaw.git
```
you do:
```bash
git clone https://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@github.com/Martexiwell/torched_tacaw.git
```
This is not the safe and good way to this but it works. Because


---

## General architecture & Ideas

The user shall interact with high level objects, namely 
- `Config` that is used for setting up the calculation
  and also doing some basic things like getting coordinates
  in the calculations or basic physics
- (`Calculator`) in parentheses because most of the time 
  the user actually does not use it directly. It actually 
  performs the whole calculation for one calculation chunk
- `Dispatcher` is like a director which keeps track of 
  what has been calculated and is ordering a calculation
  to be done. It actually has only one `Calcuator` running 
  at any moment

These three guys are implemented in `.core` submodule. 

Result of any calculation (so far) is a fully energy and 
momentum resolved STEM image - stored in `tacaw.zarray` 
-- which is a 6 dimensional array with dimensions:
  [dummy, energy, scan_x, scan_y, k_x, k_y]
-- such an array then often needs to be postprocessed by
`DetectorSet` from `.postprocessing` submodule.

### Submodules

there are several submodules:

- `.postprocessing`
this module is used basically to postprocess the 

- `.io`
I/O handler

- `.units`
for conversions of units - e.g. THz to meV etc.

- `.coordinates`
for working with coordinates

- `.tools`
usefule things like ensure_valid_path and rotation matrices


---

## Demo time!

I copied the trajectories we will need into subdirectories on
`dardel:`



(!) be careful that batch_shape should divide scanning_shape, 
otherwise results can behave unexpectedly

### hbN - planewave

### TiO2 - scanning

###


---

## Code & current state & how it can be improved

Let's look at the code!

I am using pycharm

Possibility of extension - how to implement magnon tacaw? 
I believe it's quite simple...

