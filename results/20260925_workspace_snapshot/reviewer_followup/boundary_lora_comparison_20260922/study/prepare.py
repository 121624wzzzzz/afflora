from common import *
import sys
def official_engine(split):
    sys.path[:0]=[str(HERE/"raw/reference_deps"),str(HERE/"raw/official_wikisql")]
    from lib.dbengine import DBEngine
    return DBEngine(str(HERE/f"raw/wikisql/data/{split}.db"))
