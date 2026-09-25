from common import *
OLD=HERE.parent/'stacking_sciq_placement_20260915'
SEAL=read(OLD/'ARTIFACT_MANIFEST.json')['sha256']

def checked(path):
    path=Path(path);assert sha(path)==SEAL[str(path)],str(path)
    return path

def original_specs():return read(checked(OLD/'CONFIRMATION_JOBS.json'))

def load_predictions(spec):
    path=Path(spec['checkpoint'])/'test_predictions.jsonl'
    return rows(checked(path))
