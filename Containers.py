

class Trajectory(object):
    """
    Holds a time indexed trajectory
    """
    def __init__(self):
        self._t = []
        self._x = []

    @property
    def t(self):
        return self._t

    @property
    def x(self):
        return self._x
    
    def new_step(self, t, x):
        self._t.append(t)
        self._x.append(x)

    def __getitem__(self, idx):
        return (self._t[idx], self._x[idx])

    def extend(self, ts, xs):
        self._t.extend(ts)
        self._x.extend(xs)