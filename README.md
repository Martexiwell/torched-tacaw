 
```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ ╺┳╸┏━┓┏━┓┏━╸╻ ╻┏━╸╺┳┓   ╺┳╸┏━┓┏━╸┏━┓╻ ╻ ┃
┃  ┃ ┃ ┃┣┳┛┃  ┣━┫┣╸  ┃┃╺━╸ ┃ ┣━┫┃  ┣━┫┃╻┃ ┃
┃  ╹ ┗━┛╹┗╸┗━╸╹ ╹┗━╸╺┻┛    ╹ ╹ ╹┗━╸╹ ╹┗┻┛ ┃ 
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛                                       
 ``` 
 
TACAW engine built around Hamish Brown's py_multislice. Running 
on torch, gpu accelerated.

We designed `torched-TACAW` to be user-friendly, 
resource and human-time efficient, and free-libre-open-source software (FLOSS).

\texttt{Torched-TACAW} was developed to be user-friendly at all levels: 
from the development of code, through on-boarding (installation), usage, 
including computation configuration as well as monitoring of progress, 
to data retrieval and postprocessing. Python was chosen as the basic 
platform for its friendliness, wide adoption, rich package availability 
and FLOSS nature. \texttt{Torched-TACAW} was also built to be transparent, 
where reasonable, not trying to hide the inner workings from the user.

Installation
===

## 0. Prep

In your virtual environment, it is a good idea to install torch 
and make sure that it is compatible with the current machine for 
it to work well with gpu. (This is relevant  mostly for 
clusters specifically. Since it may not be straightforward to install
environment in login node for a special node that you will use 
for your calculation).

## 1. Install py_multislice

Clone and install Hamish Brown's py_multislice. 
While in directory of your choice having virtual environment sourced:
```bash
git clone https://github.com/HamishGBrown/py_multislice.git
cd py_multislice
pip install -e .
```

## 2. Install torched_tacaw

Clone and install via pip:
```bash
git clone https://github.com/Martexiwell/torched_tacaw.git
pip install -e torched_tacaw
```

## Upgrade torched tacaw
When in torched_tacaw directory:
```bash
git pull                # pull all commits from remote for the current branch
git checkout <branch>   # changes current branch 
```

  
## branches
- **default branch:** main
- **development branch:** dev
  - other branches for feature development

The **main** branch hosts the stable version of the code.
All features are to be implemented on subrbranches of **dev**
named **dev-featureName**. When ready, they can be pulled
into **dev**. **dev** can be eventually pulled into main.


Good to keep in mind
===
- be mindful of the fence post problem - by convention 
  we don't add the "+1" point in our scanning grid


How to cite?
===
We are very happy you are using torched-TACAW. If you want to reference it, 
you are most welcome to cite **preprint xxxxxx** as well as this GitHub repo. 