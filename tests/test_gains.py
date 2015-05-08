import unittest
from autogain import gains

class TestGains(unittest.TestCase):

    def test_dumpnload(self):
        g = gains.Gains()
        key = ('a','b','c','d')       
        val = 1.1
        key = '.'.join(key)
        print key
        g.trace_gains = {key: val}
        print g.load(g.dump())

if __name__=='__main__':
    unittest.main()
