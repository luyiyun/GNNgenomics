#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from visdom import Visdom


class VisObj:

    def __init__(self, port, env):
        self.env_name = env
        self.vis = Visdom(port=port, env=env)

    def line(self, x, hist, phases=["train", "val"], metrics=["loss"]):
        for m in metrics:
            ys = [hist[p][m][-1] for p in phases]
            self.vis.line(
                Y=np.array([ys]).astype("float"),
                X=np.array([x]).astype("float"),
                win=m,
                update="append" if x > 0. else None,
                opts=dict(legend=phases, title=m)
            )

    def save_env(self):
        self.vis.save([self.env_name])
