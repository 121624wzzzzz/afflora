import unittest
from common import canonical
from scoring import score,parse,aggregate
from prepare import grounded

class Checks(unittest.TestCase):
    def test_grounding(self):
        self.assertFalse(grounded({'text':'Do it tonight at 11 PM','target':[{'arguments':{'date':'2023-09-30 23:00'}}]}))
        self.assertTrue(grounded({'text':'Find 10 books in Paris','target':[{'arguments':{'n':10,'city':'Paris'}}]}))
        self.assertFalse(grounded({'text':'Find 100 books','target':[{'arguments':{'n':10}}]}))
        self.assertFalse(grounded({'text':'Find books','target':[]}))
    def test_occurrence(self):
        gold=[{'type':'name','text':'张三','start':4,'end':5}]
        row={'task':'cluener','text':'张三看见张三','gold_spans':gold}
        good=[{'type':'name','text':'张三','occurrence':1}]
        self.assertTrue(score(canonical(good),row)['content_correct'])
        wrong=[dict(good[0],occurrence=0)];s=score(canonical(wrong),row);self.assertEqual(s['span_tp'],0);self.assertEqual(s['text_tp'],1)
        self.assertFalse(score(canonical([dict(good[0],occurrence=True)]),row)['schema_valid'])
        s=score(canonical(good*2),row);self.assertEqual(s['span_tp'],1);self.assertEqual(s['span_pred'],2)
        self.assertEqual(aggregate([score('invalid',row)],'cluener')['primary'],0)
    def test_calls(self):
        tools=[{'name':'f','parameters':{'type':'object','properties':{'x':{'type':'integer'},'y':{'type':'integer','default':2}},'required':['x']}}]
        row={'task':'toolace','tools':tools,'target':[{'name':'f','arguments':{'x':3}}]}
        self.assertTrue(score('[{"name":"f","arguments":{"x":3,"y":2}}]',row)['content_correct'])
        self.assertFalse(score('[{"name":"f","arguments":{"x":"3"}}]',row)['content_correct'])
        self.assertIsNone(parse('[{"a":1,"a":2}]')[0])
        self.assertFalse(score('[]',row)['content_correct'])
if __name__=='__main__':unittest.main()
