import unittest
from common import canonical
from scoring import score,parse,aggregate
from prepare import convert_calls

class ScoringTests(unittest.TestCase):
    def test_calls(self):
        tools=[{'name':'Get Data','parameters':{'type':'dict','properties':{'x':{'type':'integer'}},'required':['x']}}]
        gold=[{'name':'Get Data','arguments':{'x':3}}];row={'task':'toolace','tools':tools,'target':gold}
        self.assertTrue(score(canonical(gold),row)['content_correct'])
        self.assertTrue(score('```json\n'+canonical(gold)+'\n```',row)['content_correct'])
        self.assertFalse(score(canonical([{'name':'Get Data','arguments':{'x':'3'}}]),row)['content_correct'])
        self.assertFalse(score(canonical([{'name':'Other','arguments':{'x':3}}]),row)['content_correct'])
        self.assertFalse(score(canonical(gold*2),row)['content_correct'])
        self.assertFalse(score('The answer is '+canonical(gold),row)['content_correct'])
        self.assertIsNone(parse('[{"x":1,"x":2}]')[0])
        self.assertEqual(convert_calls('[Get Data(x=3)]',['Get Data']),gold)
        self.assertEqual(convert_calls('[Get Data(x="Get Data(x=3)")]',['Get Data'])[0]['arguments']['x'],'Get Data(x=3)')
    def test_empty_is_not_parse_failure(self):
        row={'task':'toolace','tools':[],'target':[]}
        self.assertTrue(score('[]',row)['content_correct'])
        self.assertFalse(score('not json',row)['content_correct'])
    def test_spans_and_content(self):
        gold=[{'type':'name','text':'张三','start':0,'end':1}];row={'task':'cluener','text':'张三来了','target':gold}
        got=score(canonical(gold),row);self.assertEqual(aggregate([got],'cluener')['primary'],100)
        wrong=[dict(gold[0],start=1,end=2)];got=score(canonical(wrong),row)
        self.assertEqual(got['text_tp'],1);self.assertEqual(got['span_tp'],0);self.assertEqual(got['span_pred'],1)
        got=score(canonical(gold*2),row);self.assertEqual(got['span_tp'],1);self.assertEqual(got['span_pred'],2)
        failed=score('invalid',row);self.assertEqual(aggregate([failed],'cluener')['primary'],0)
if __name__=='__main__':unittest.main()
