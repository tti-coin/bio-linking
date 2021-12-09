# %%
import os
import transformers
import tqdm
import dataclasses
from typing import List

from .load_medmentions import load_medmentions_corpus
from .load_database import load_database_file
from .load_misc import load_csv_corpus_file
from .concept_index_manager import ConceptIndexManager, ConceptIndexData
from .datatype import Corpus, Instance, Input, InputAsList

@dataclasses.dataclass
class PreprocessConfig:
    version: str
    corpus_type: str
    corpus_dir: str
    database_fnames: List[str]
    additional_train_corpuses: str
    bert_tokenizer_name_or_path: str
    annotation_oov_concept: str = "CUI-less"
    max_token_length: int = 512
    bc7tr2_cv_size: int = 5
    bc7tr2_dev_index: int = 4
    bc7tr2_train_all: bool = False
    use_tqdm: bool = True

@dataclasses.dataclass
class PreprocessedDataset:
    version: str

    train_inputs: List[InputAsList]
    dev_inputs: List[InputAsList]
    test_inputs: List[InputAsList]
    db_inputs: List[InputAsList]

    additional_train_inputs: List[InputAsList]

    concept_index_data: ConceptIndexData
    num_concept: int

    bert_tokenizer_name_or_path: str
    pad_token_id: int
    vocab_size: int

    preprocess_config: dict # to store "dataclasses.asdict(PreprocessConfig)"

    def to_dict(self) -> dict:
        # dataclasses.asdict(preprocessed_dataset_instance) is very slow because of the large db_inputs, so we mannually convert this into dict.
        out = vars(self)
        out["concept_index_data"] = dataclasses.asdict(self.concept_index_data)
        return out
    @classmethod
    def from_dict(cls, src):
        src = dict(src)
        src["concept_index_data"] = ConceptIndexData(**src["concept_index_data"])
        return cls(**src)

def preprocess(config: PreprocessConfig) -> PreprocessedDataset:
    assert config.corpus_type in ["medmentions", "medmentions-st21pv", "mcn", "biocreative7track2", "csv"]

    if config.corpus_type == "medmentions":
        train_corpus, dev_corpus, test_corpus = load_medmentions_corpus(corpus_dir=config.corpus_dir, is_st21pv=False)
    elif config.corpus_type == "medmentions-st21pv":
        train_corpus, dev_corpus, test_corpus = load_medmentions_corpus(corpus_dir=config.corpus_dir, is_st21pv=True)
    elif config.corpus_type == "mcn":
        raise NotImplementedError("mcn")
    elif config.corpus_type == "biocreative7track2":
        # TODO: replace
        from .load_biocreative7track2 import load_biocreative7track2_corpus
        train_corpus, dev_corpus, test_corpus = load_biocreative7track2_corpus(corpus_dir=config.corpus_dir, bert_tokenizer_name_or_path=config.bert_tokenizer_name_or_path, cv_size=config.bc7tr2_cv_size, dev_index=config.bc7tr2_dev_index, train_all=config.bc7tr2_train_all)
    elif config.corpus_type == "csv":
        def load_and_pack(fname, prefix):
            instances = load_csv_corpus_file(fname, id_prefix=prefix)
            return Corpus(instances=instances)
        # assume corpus_dir/train.csv, dev.csv, test.csv exist.
        train_corpus, dev_corpus, test_corpus = [load_and_pack(os.path.join(config.corpus_dir, f"{split}.csv"), prefix=split) for split in ["train", "dev", "test"]]

    database_instances = [instance for fname in config.database_fnames for instance in load_database_file(fname)]

    additional_train_instances = [instance for fname in config.additional_train_corpuses for instance in load_csv_corpus_file(fname, id_prefix="add-train")]

    bert_tokenizer = transformers.AutoTokenizer.from_pretrained(config.bert_tokenizer_name_or_path)
    concept_index_manager = ConceptIndexManager(annotation_oov_concept=config.annotation_oov_concept)

    def instance_to_input(instance:Instance, is_training:bool) -> InputAsList:
        encoded = bert_tokenizer(instance.repr, truncation=True, max_length=config.max_token_length)
        input_ids = encoded["input_ids"]
        concept_id = concept_index_manager.encode(instance.concept, is_training=is_training)
        return Input(repr_ids=input_ids, entity_start=1, entity_end=len(input_ids)-1, concept_id=concept_id, id=f"{instance.id}").to_list()
    def instances_to_inputs(instances:List[Instance], is_training:bool, tqdm_desc=None) -> List[InputAsList]:
        if config.use_tqdm:
            instances = tqdm.tqdm(instances, desc=tqdm_desc)
        return [instance_to_input(instance, is_training=is_training) for instance in instances]

    train_inputs = instances_to_inputs(train_corpus.instances, is_training=True, tqdm_desc="train_inputs")
    db_inputs = instances_to_inputs(database_instances, is_training=True, tqdm_desc="db_inputs")
    additional_train_inputs = instances_to_inputs(additional_train_instances, is_training=True, tqdm_desc="additional_train_inputs")
    concept_index_manager.finalize()
    dev_inputs = instances_to_inputs(dev_corpus.instances, is_training=False, tqdm_desc="dev_inputs")
    test_inputs = instances_to_inputs(test_corpus.instances, is_training=False, tqdm_desc="test_inputs")

    preprocessed_dataset = PreprocessedDataset(
        version=config.version,
        train_inputs=train_inputs, dev_inputs=dev_inputs, test_inputs=test_inputs, db_inputs=db_inputs,
        additional_train_inputs=additional_train_inputs,
        concept_index_data=concept_index_manager.dump(), num_concept=concept_index_manager.num_concept,
        bert_tokenizer_name_or_path=config.bert_tokenizer_name_or_path, pad_token_id=bert_tokenizer.pad_token_id, vocab_size=bert_tokenizer.vocab_size,
        preprocess_config=dataclasses.asdict(config)
    )
    return preprocessed_dataset

def csv_to_eval_inputs(csv_path, max_token_length, use_tqdm, concept_index_data, bert_tokenizer_name_or_path) -> List[InputAsList]:
    csv_corpus = Corpus(instances=load_csv_corpus_file(csv_path, raw_id=True))

    bert_tokenizer = transformers.AutoTokenizer.from_pretrained(bert_tokenizer_name_or_path)
    concept_index_manager = ConceptIndexManager.from_dumped(concept_index_data)

    def instance_to_input(instance:Instance, is_training:bool) -> InputAsList:
        encoded = bert_tokenizer(instance.repr, truncation=True, max_length=max_token_length)
        input_ids = encoded["input_ids"]
        concept_id = concept_index_manager.encode(instance.concept, is_training=is_training)
        return Input(repr_ids=input_ids, entity_start=1, entity_end=len(input_ids)-1, concept_id=concept_id, id=f"{instance.id}").to_list()
    def instances_to_inputs(instances:List[Instance], is_training:bool, tqdm_desc=None) -> List[InputAsList]:
        if use_tqdm:
            instances = tqdm.tqdm(instances, desc=tqdm_desc)
        return [instance_to_input(instance, is_training=is_training) for instance in instances]

    csv_inputs = instances_to_inputs(csv_corpus.instances, is_training=False, tqdm_desc="csv_inputs")
    return csv_inputs

# %%
