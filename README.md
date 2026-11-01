# TempFlash

Code for "TempFlash: GPU-Resident, Single-Launch Training of Temporal Graph Neural Networks".
The appendix of the paper is [appendix.pdf](appendix.pdf).

## Setup

    pip install -r requirements.txt

Prepare LastFM, Wiki-Talk, Stack-Overflow and GDELT with the scripts of [TGL](https://github.com/amazon-science/tgl)
and [ETC](https://github.com/eddiegaoo/ETC), then link the data directory:

    ln -s /path/to/DATA tempflash/DATA

## Run

    bash scripts/train_default.sh 0

`tempflash/layers.py`, `modules.py`, `memorys.py` and `utils.py` are adapted from TGL (Apache-2.0,
`tempflash/LICENSE-TGL`) and ETC.
