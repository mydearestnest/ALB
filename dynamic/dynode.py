from base import BaseNode


class Dynode(BaseNode):
    def __init__(self, coords, dim=2):
        super(Dynode, self).__init__(coords, dim)
        self._pdxc = None
        self._pdyc = None
        self._pdxct = None
        self._pdyct = None

