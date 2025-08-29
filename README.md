Example Code - AI-Based Modeling for Energy-Efficient Buildings - Competition
===


Intro
---

This is a repository that can serve as an initialization for participating in the 
ELIAS AI-Based Modeling for Energy-Efficient Buildings competition. 
It contains simple versions of the data loading/preprocessing/training/submission building blocks of a method for that
competition. This allows
participants a quick way to get to the core method development. Note that all of the building blocks might have
to be modified to obtain well-performing, robust methods, in particular, these aspects are worth improving:
* take more or all of the covariate variables as predictors (there are hundreds), not just the current few example ones
* also include the 2024 data from source Zenodo, see `README.md` or competition page for details
* some timeseries may rather need linear interpolation instead of the forward fill (ffill) that is currently used
  (reason for the ffill is that, at least for some time series, the recodring is change-triggered).
* and of course the model and training which lies at the core of the competition, and is currently just a toy example

For further details see `main.py` where all functions (using PyTorch) as well as a sample execution are gathered and
described in more detail.

Purpose of the repository and disclaimer
---

This software is a research prototype, solely developed for and published as part of the aforementioned ELIAS competition.
It will neither be maintained nor monitored in any way.


Software setup
---

The package manager used to set up this repo and that we recommend to get a virtual environment to run the scripts 
is [uv](https://docs.astral.sh/uv/guides/install-python/). We recommend to install it.

Once uv is installed installed, we recommend the following steps:
1. go to root dir of this repo, 
2. `uv venv`,
3. then run `uv sync`.


Set up and run simple load/preprocess/train/submission pipeline
---

A simple sample pipeline for load/preprocess/train/submission is contained in `main.py`. Run it in the following way:

1. Download the competition data from Kaggle and put it into a (newly created) folder `data/kaggle_dl/` (there should then be folders like
`data/kaggle_dl/RBHU-2025-01/` etc., 
to make the file structure complies with the format required by the `main.py` script).
2. Do `uv run main.py`.

Again, be aware that this pipeline is just a starting point for building methods to participating in this competition.
In particular, consider using the additional 2024 building data from Zenodo: [part 1](https://zenodo.org/records/12590466), 
[part 2](https://zenodo.org/records/14591934).


License
---

This project is open-sourced under the MIT license. See the
[LICENSE](LICENSE) file for details.
