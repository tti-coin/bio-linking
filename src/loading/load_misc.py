import json
import pandas as pd
from typing import List

from .datatype import Instance

def load_json_corpus_file(fname:str, id_prefix="json") -> List[Instance]:
    """
    the given file name should be the json-dumped file's name whose dumped object is:
    [
        (id_1, repr_1, concept_1),
        (id_2, repr_2, concept_2),
        ...
    ]
    """
    with open(fname) as f:
        data = json.load(f)
    instances = [Instance(repr=repr_, concept=concept, id=f"{id_prefix}:{fname}@{id_}") for id_, repr_, concept in data]
    return instances

def load_csv_corpus_file(fname:str, id_prefix:str="csv", raw_id:bool=False) -> List[Instance]:
    """
    the given csv file should have fields of ["id", "repr", "concept"].
    """
    df = pd.read_csv(fname, keep_default_na=False)
    if raw_id:
        get_id = lambda record: record.id
    else:
        get_id = lambda record: f"{id_prefix}:{fname}@{record.id}"
    instances = [Instance(repr=record.repr, concept=record.concept, id=get_id(record)) for _, record in df.iterrows()]
    return instances
