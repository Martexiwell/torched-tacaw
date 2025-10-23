# torched-tacaw workshop 24/10/2025

### topics 

1. what it is and where to find it
2. instalation, where to find it on dardel
3. github, file organization
4. general architecture
5. usage 
   - simple
   - cookbook
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
from torched_tacaw import Config, Master
from torched_tacaw import postprocessing as tp
```

### Lumi

I did not yet made a venv on lumi but i will make sure it works


---

## Github & File organization 



