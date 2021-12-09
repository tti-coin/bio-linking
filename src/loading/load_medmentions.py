import os
import gzip
import re
from typing import Tuple, Dict, List
from dataclasses import dataclass

from .datatype import Span, Instance, Document, Corpus
TrainCorpus = Corpus
DevCorpus = Corpus
TestCorpus = Corpus
DocId = str

@dataclass
class MedMentionAnnotation:
    doc_id: str
    start: int
    end: int
    surface: str
    semantic_type: str
    concept: str


FULL_ANNOTATION_FNAME = "full/data/corpus_pubtator.txt.gz"
ST21PV_ANNOTATION_FNAME = "st21pv/data/corpus_pubtator.txt.gz"
TRAIN_DOC_ID_FNAME = "full/data/corpus_pubtator_pmids_trng.txt"
DEV_DOC_ID_FNAME = "full/data/corpus_pubtator_pmids_dev.txt"
TEST_DOC_ID_FNAME = "full/data/corpus_pubtator_pmids_test.txt"


def load_medmentions_corpus(corpus_dir:str, is_st21pv:bool=False) -> Tuple[TrainCorpus, DevCorpus, TestCorpus]:
    train_id_fname = os.path.join(corpus_dir, TRAIN_DOC_ID_FNAME)
    dev_id_fname = os.path.join(corpus_dir, DEV_DOC_ID_FNAME)
    test_id_fname = os.path.join(corpus_dir, TEST_DOC_ID_FNAME)
    if is_st21pv:
        text_fname = os.path.join(corpus_dir, ST21PV_ANNOTATION_FNAME)
    else:
        text_fname = os.path.join(corpus_dir, FULL_ANNOTATION_FNAME)
    assert os.path.exists(text_fname), text_fname
    assert os.path.exists(train_id_fname), train_id_fname
    assert os.path.exists(dev_id_fname), dev_id_fname
    assert os.path.exists(test_id_fname), test_id_fname

    all_documents, doc_id_to_anns = parse_medmentions_txt(text_fname, is_st21pv=is_st21pv, is_text_gzipped=True)
    
    all_instances = transform_medmentions_corpus(all_documents=all_documents, doc_id_to_anns=doc_id_to_anns)

    train, dev, test = split_medmentions_corpus(
        all_documents=all_documents, all_instances=all_instances,
        train_id_fname=train_id_fname, dev_id_fname=dev_id_fname, test_id_fname=test_id_fname)

    return train, dev, test

def parse_medmentions_txt(fname, is_st21pv, is_text_gzipped=True) -> Tuple[Dict[DocId, Document], Dict[DocId, List[MedMentionAnnotation]]]:
    open_func = gzip.open if is_text_gzipped else open
    with open_func(fname) as f:
        txt = f.read()
        if is_text_gzipped:
            txt = txt.decode()
    lines = txt.rstrip().splitlines()

    ptn_entity_ann = re.compile("[0-9]+(\t[^\t]+){4,6}$")
    ptn_title = re.compile("[0-9]+\\s*\\|t")
    ptn_abst = re.compile("[0-9]+\\s*\\|a")
    ptn_relation_ann = re.compile("[0-9]+(\t[^\t]+){3}$")

    doc_id_to_anns = dict()
    all_documents = dict()
    buff = list()
    for line in lines:
        line = line.rstrip()
        if len(line) == 0:
            continue

        if ptn_entity_ann.match(line):
            line = line.split("\t")
            assert len(line) in [5,6,7] # [doc_id, start, end, surface, semantic_type] + [concept] + [conjunction]
            if len(line) == 5:
                line = line + [line[-1]] # use semantic_type as the concept id.
            elif len(line) == 7:
                line = line[:6] # cut conjunction information.
            assert line[0] == buff[0] # must be same doc_id
            ann = MedMentionAnnotation(*line)
            ann.start, ann.end = map(int, [ann.start, ann.end])
            if is_st21pv:
                assert ann.concept[:5] == "UMLS:"
                ann.concept = ann.concept[5:]
            buff.append(ann)
        elif ptn_title.match(line):
            if len(buff) > 0:
                doc_id, title, abst = buff[:3]
                anns = buff[3:]
                all_documents[doc_id] = Document(title=title, abst=abst, doc_id=doc_id)
                doc_id_to_anns[doc_id] = anns
                buff = list()
            line = line.split("|")
            doc_id, tag, *body = line
            assert len(body) >= 1
            body = "|".join(body)
            assert tag == "t"
            buff.append(doc_id)
            buff.append(body) # title
        elif ptn_abst.match(line):
            line = line.split("|")
            doc_id, tag, *body = line
            assert len(body) >= 1
            body = "|".join(body)
            assert tag == "a"
            assert doc_id == buff[0]
            buff.append(body) # abst
        elif ptn_relation_ann.match(line):
            pass
        else:
            raise ValueError(line)
    if len(buff) > 0:
        doc_id, title, abst = buff[:3]
        anns = buff[3:]
        all_documents[doc_id] = Document(title=title, abst=abst, doc_id=doc_id)
        doc_id_to_anns[doc_id] = anns
        buff = list()
    return all_documents, doc_id_to_anns

def transform_medmentions_corpus(all_documents, doc_id_to_anns) -> Dict[DocId, List[Instance]]:
    cat_documents = {doc_id:" ".join([doc.title, doc.abst]) for doc_id,doc in all_documents.items()}

    all_instances = dict()
    for doc_id, anns in doc_id_to_anns.items():
        instances = list()
        doc = cat_documents[doc_id]
        for a,ann in enumerate(anns):
            assert ann.surface == doc[ann.start:ann.end]
            instance = Instance(repr=ann.surface, concept=ann.concept, id=f"{doc_id}@{a}", doc_id=doc_id, context=ann.surface, context_start_on_doc=ann.start, spans_on_doc=[Span(start=ann.start, end=ann.end)])
            instances.append(instance)
        all_instances[doc_id] = instances

    return all_instances

def split_medmentions_corpus(all_documents, all_instances, train_id_fname, dev_id_fname, test_id_fname) -> Tuple[Corpus, Corpus, Corpus]:
    # input
    # all_documents:dict => all_documents[doc_id] = Document(title, abst)
    # all_instances:dict => all_instances[doc_id] = [instance0, instance1, ...]

    def read_doc_id_file(fname):
        with open(fname) as f:
            return [line.rstrip() for line in f.read().rstrip().splitlines()]
    train_doc_ids, dev_doc_ids, test_doc_ids = map(read_doc_id_file, [train_id_fname, dev_id_fname, test_id_fname])
    assert len(set(train_doc_ids) & set(dev_doc_ids)) == 0
    assert len(set(train_doc_ids) & set(test_doc_ids)) == 0
    assert len(set(dev_doc_ids) & set(test_doc_ids)) == 0

    def pick_instances(doc_ids):
        return [instance for doc_id in doc_ids for instance in all_instances[doc_id]]
    train_instances, dev_instances, test_instances = map(pick_instances, [train_doc_ids, dev_doc_ids, test_doc_ids])

    def pick_documents(doc_ids):
        return {doc_id:all_documents[doc_id] for doc_id in doc_ids}
    train_documents, dev_documents, test_documents = map(pick_documents, [train_doc_ids, dev_doc_ids, test_doc_ids])

    train = Corpus(train_instances, train_documents)
    dev = Corpus(dev_instances, dev_documents)
    test = Corpus(test_instances, test_documents)

    return train, dev, test


