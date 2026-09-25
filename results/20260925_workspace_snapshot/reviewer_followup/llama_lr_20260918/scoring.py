import scoring_ner,scoring_sql

def score(text,row):
    return (scoring_ner if row['task']=='cluener' else scoring_sql).score(text,row)
def aggregate(records,task):
    return (scoring_ner if task=='cluener' else scoring_sql).aggregate(records,task)
