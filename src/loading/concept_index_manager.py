from dataclasses import dataclass
from typing import List

ANNOTATION_OOV_CONCEPT_INDEX = 0
OOV_CONCEPT_INDEX = -1

@dataclass
class ConceptIndexData:
    finalized: bool
    reverse_mapping: List[str]


class ConceptIndexManager:
    def __init__(self, annotation_oov_concept:str):
        """
        annotation_oov_concept:
            When using MCN dataset (n2c2 2019 track 3), this should be "CUI-less". The corresponding index is ANNOTATION_OOV_CONCEPT_INDEX(=0).
            This is for the instance that is "annotated" as OOV, i.e., this is NOT for the dev/test instance that has a certain CUI/MESH id that is not covered by train+db splits.
            If the dev/test instances have some CUI/MESH but that is not covered by train+db instances so there is no corresponding index, then it will be indexed as OOV_CONCEPT_INDEX(=-1).
        """
        self._finalized = False
        self.mapping = {annotation_oov_concept:ANNOTATION_OOV_CONCEPT_INDEX}
        self.reverse_mapping = [annotation_oov_concept]

    def encode(self, concept:str, is_training:bool) -> int:
        assert is_training or self._finalized

        if is_training:
            if concept not in self.mapping:
                new_concept_id = self.num_concept
                self.mapping[concept] = new_concept_id
                self.reverse_mapping.append(concept)
            concept_id = self.mapping[concept]
        else:
            concept_id = self.mapping.get(concept, OOV_CONCEPT_INDEX)

        return concept_id

    def decode(self, concept_id:int) -> str:
        return self.reverse_mapping[concept_id]

    @property
    def num_concept(self):
        return len(self.reverse_mapping)

    def finalize(self) -> None:
        assert not self._finalized
        self._finalized = True

    def dump(self) -> ConceptIndexData:
        return ConceptIndexData(finalized=self._finalized, reverse_mapping=self.reverse_mapping)

    @classmethod
    def from_dumped(cls, dumped:ConceptIndexData):
        restored = cls(dumped.reverse_mapping[0])
        restored._finalized = dumped.finalized
        restored.mapping = {concept:concept_id for concept_id, concept in enumerate(dumped.reverse_mapping)}
        restored.reverse_mapping = dumped.reverse_mapping
        return restored
