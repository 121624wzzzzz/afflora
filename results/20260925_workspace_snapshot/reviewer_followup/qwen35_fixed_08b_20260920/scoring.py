import scoring_sql,scoring_trec
def score(text,row):
 assert row['task']=='wikisql'
 return scoring_sql.score(text,row)
def aggregate(records,task):
 return (scoring_sql if task=='wikisql' else scoring_trec).aggregate(records,task)
