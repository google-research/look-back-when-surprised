# Look Back When Surprised: RER++


This repository serves to open-source the code used in the paper: "[Look Back When Surprised: Stabilizing Reverse Experience Replay for Neural Approximation](https://arxiv.org/abs/2206.03171)". **This is not an officially supported Google product.**

### Getting started

To avoid any conflict with your existing Python setup, it is suggested to work in a virtual environment with [`virtualenv`](https://docs.python-guide.org/dev/virtualenvs/). To install `virtualenv`:
```bash
pip install --upgrade virtualenv
```
Create a virtual environment, activate it and install the requirements in [`requirements.txt`](requirements.txt).
```bash
virtualenv env
source env/bin/activate
pip install -r requirements.txt
```

### Training & Testing

Our models can be trained as follows:
```bash
python -W ignore -m src.main --exp_name <name> --algo <algo> --replay_buffer_sampler <replay_buffer_choice> --env <env_name> --train --seed $1 --snapshot_dir $2
```

# Paper Citation

If you find our codes useful, do consider citing our [paper](https://arxiv.org/abs/2206.03171):
```
@article{kumar2022look,
  title={Look Back When Surprised: Stabilizing Reverse Experience Replay for Neural Approximation},
  author={Kumar, Ramnath and Nagaraj, Dheeraj},
  journal={arXiv preprint arXiv:2206.03171},
  year={2022}
}
```

# References

Our repository makes use of various open-source codes. Most of which have been documented at Garage. If you find the respective codes useful, do cite their respective papers as well:

```
@misc{garage,
 author = {The garage contributors},
 title = {Garage: A toolkit for reproducible reinforcement learning research},
 year = {2019},
 publisher = {GitHub},
 journal = {GitHub repository},
 howpublished = {\url{https://github.com/rlworkgroup/garage}},
 commit = {be070842071f736eb24f28e4b902a9f144f5c97b}
}
```
