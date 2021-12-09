from typing import List
from .datatype import Instance

def load_database_file(database_fname:str) -> List[Instance]:
    outs = list()
    with open(database_fname) as f:
        first_line = f.readline().rstrip()
        assert first_line in ["\t".join(["CUI", "STR"]), "\t".join(["MESH", "STR"])], f"first line shoud be column names: {first_line}"
        for l,line in enumerate(f):
            line = line.rstrip()
            if len(line) == 0:
                continue
            concept, repr_ = line.split("\t")
            out = Instance(repr=repr_, concept=concept, id=f"database:{database_fname}@{l}")
            outs.append(out)
    return outs

