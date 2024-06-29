#!/bin/bash

# Pendulum
python train.py --env pendulum --setup baseline --capture-video --video-freq 20 --track
python train.py --env pendulum --setup dylam --capture-video --video-freq 20 --track
# LunarLander
python train.py --env lunarlander --setup baseline --capture-video --video-freq 20 --track
python train.py --env lunarlander --setup dylam --capture-video --video-freq 20 --track
# Hopper
python train.py --env hopper --setup baseline --capture-video --video-freq 20 --track
python train.py --env hopper --setup dylam --capture-video --video-freq 20 --track
# Humanoid
python train.py --env humanoid --setup baseline --capture-video --video-freq 20 --track
python train.py --env humanoid --setup dylam --capture-video --video-freq 20 --track
# VSS
python train.py --env vss --setup baseline --capture-video --video-freq 20 --track
python train.py --env vss --setup dylam --capture-video --video-freq 20 --track