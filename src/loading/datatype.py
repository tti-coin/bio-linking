import dataclasses
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Span:
    start: int
    end: int

@dataclass
class Instance:
    repr: str
    concept: str
    id: str

    # document information
    doc_id: Optional[str] = None
    context: Optional[str] = None
    context_start_on_doc: Optional[int] = None
    spans_on_doc: Optional[List[Span]] = None

@dataclass
class Document:
    title: str
    abst: str
    doc_id: str

@dataclass
class Corpus:
    instances: List[Instance]
    documents: Optional[List[Document]] = None # won't be used


@dataclass
class Record:
    repr: str
    concept: str

@dataclass
class Database:
    records: List[Record]


@dataclass
class Dataset:
    train: Corpus
    dev: Corpus
    test: Corpus

    db: Database


# In the program, each Input instance is imidiately converted into the list object by Input.to_list.
InputAsList = list
@dataclass
class Input:
    repr_ids: List[int]
    entity_start: int
    entity_end: int
    concept_id: int
    id: str = "none"

    def to_list(self) -> InputAsList:
        return [self.repr_ids, self.entity_start, self.entity_end, self.concept_id, self.id]
    @classmethod
    def from_list(cls, list_input:InputAsList):
        return Input(*list_input)

