# %%
import sys, os
import datetime
import time
import gzip
import json
import collections
import contextlib
from contextlib import ExitStack as NullContext
import tqdm

import dataclasses as D
from typing import Optional, List

import torch
import numpy as np
import pandas as pd

from src.loading.preprocessing import PreprocessConfig, PreprocessedDataset, preprocess, csv_to_eval_inputs
from src.loading.datatype import Input
from src.loading.concept_index_manager import OOV_CONCEPT_INDEX
from src.modeling import Model, ModelConfig
from src.utils.argparse import DataclassArgumentParser
from src.utils.misc import ClosedInterval
from src.utils.chunk import chunking
from src.utils.dataloader_util import SelectiveDataset, OversampleDatasetBuilder
import src.version_manager as V


# %%

PARAXLMR = "sentence-transformers/paraphrase-xlm-r-multilingual-v1"

@D.dataclass
class Config:
    mode: str = D.field(default="train", metadata={"choices":["train", "eval-dev", "eval-test", "eval-csv"]})
    config_from: Optional[str] = D.field(default=None, metadata={"help":"path/to/config.json"})
    save_dir: Optional[str] = D.field(default=None, metadata={"help":"if not set, neither trained models nor log files won't be saved. if \"none\" is set, treat as not set."})
    init_model_path: Optional[str] = D.field(default=None, metadata={"help":"path/to/model.pt"})

    do_overwrite_weight: bool = False

    corpus_type: str = D.field(default="medmentions", metadata={"choices":["medmentions", "medmentions-st21pv", "biocreative7track2", "csv"]})
    annotation_oov_concept: str = "CUI-less"
    corpus_dir: str = "data/medmentions/MedMentions/"
    additional_train_corpuses: str = D.field(default="", metadata={"help": "json files ([(key, repr, link), ...]). separate by comma."})
    databases: str = D.field(default="data/umls2017aa_full.sqlout", metadata={"help": "separate by comma."})
    # use_level0: bool = True # TODO: n2c2
    # use_level0_cui_less: bool = False # TODO: n2c2
    preprocess_cache_path: str = "data/medmentions/preprocess-medmentions_scibert_umls2017aa.json.gz"
    without_preprocess_cache: bool = False
    force_preprocess: bool = False
    bc7tr2_cv_size: int = 5
    bc7tr2_dev_index: int = 4
    bc7tr2_train_all: bool = False
    oversample_coeff: float = 1.0

    eval: bool = False # TODO: not implemented. maybe will be deleted this option.
    eval_output_dir: Optional[str] = None # TODO: not implemented. maybe will be deleted this option.
    eval_csv_path: Optional[str] = None
    eval_output: Optional[str] = None
    eval_topk: int = 64

    encoder: str = D.field(default="bert", metadata={"choices":["bert", "lstm"]})
    encoder_pool_target: str = D.field(default="entity", metadata={"choices":{"entity", "input"}})
    bert: str = "allenai/scibert_scivocab_uncased"
    feature_dim: int = D.field(default=128, metadata={"choices":[128, 232]})
    use_feature_bias: bool = True
    use_feature_following_linear: bool = False
    dropout_output_rate: float = D.field(default=0.0, metadata={"choices":[ClosedInterval(0.0, 0.5)]})
    output_layer: str = D.field(default="cosine", metadata={"choices":["cosine", "linear"]})
    cosine_scale: float = D.field(default=28.623, metadata={"choices":[ClosedInterval(0.0,np.inf)]})
    arcface_margin: float = D.field(default=0.612213, metadata={"choices":[ClosedInterval(0.0, 2.0)]})

    learning_rate: float = 4.98807e-5
    batch_size: int = D.field(default=128, metadata={"choices":[ClosedInterval(1, np.inf)]})
    max_epoch: int = D.field(default=1000, metadata={"choices": [ClosedInterval(1, np.inf)]})
    save_every_epoch: int = D.field(default=50, metadata={"help": "if set 0 (or smaller), only 'model_best.pt' will be saved."})
    max_token_length: int = D.field(default=27, metadata={"choices":[ClosedInterval(19,512)]})
    do_train_database: bool = True

    device: str = "cuda:0"
    chunk_batch: bool = True
    chunk_batch_size: int = D.field(default=64, metadata={"help": "The batch size per one forward process to calculate the cumulative loss. This is for controlling the GPU memory use. NOTE: The parameter of the effective batch size for one back propagation process is 'batch_size'."})

    def check_sum(self):
        assert (not self.chunk_batch) or (self.batch_size % self.chunk_batch_size == 0)
        if self.mode == "eval-csv":
            assert self.eval_csv_path is not None
        return self

config = DataclassArgumentParser(Config).parse_args()
force_run = False
if config.config_from is not None:
    with open(config.config_from) as f:
        default_config = Config(**json.load(f)) # load custom default values.
    config = DataclassArgumentParser(Config, default_config).parse_args() # redo parsing with custom default values.
    if (config.mode == "train") and (config.save_dir is not None) and (config.save_dir != "none") and (config.save_dir == default_config.save_dir):
        print("WARNING! config.save_dir is same as the loaded config's save_dir. This may cause overwriting the previous experimental results.")
        force_run = (input("[run/quit]=>").strip() in ["run", "r"])
        if not force_run:
            exit(0)
config.check_sum()

chunk_batch_size = config.chunk_batch_size if config.chunk_batch else config.batch_size
num_each_chunk = config.batch_size // chunk_batch_size

do_save = ((config.save_dir is not None) and (config.save_dir != "none"))

experiment_setting = {"config": D.asdict(config)}

# %% preprocess

def resolve_comma_split(string):
    out = [value.strip() for value in string.strip().split(",")]
    return [value for value in out if len(value) != 0]

print("loading dataset...", flush=True, file=sys.stderr)
if config.force_preprocess or config.without_preprocess_cache or (not os.path.exists(config.preprocess_cache_path)):
    database_fnames = resolve_comma_split(config.databases)
    additional_train_corpuses = resolve_comma_split(config.additional_train_corpuses)
    preprocess_config = PreprocessConfig(
        version=V.VERSION,
        corpus_type=config.corpus_type, corpus_dir=config.corpus_dir, database_fnames=database_fnames,
        additional_train_corpuses=additional_train_corpuses,
        bert_tokenizer_name_or_path=config.bert, annotation_oov_concept=config.annotation_oov_concept, max_token_length=config.max_token_length,
        bc7tr2_cv_size=config.bc7tr2_cv_size, bc7tr2_dev_index=config.bc7tr2_dev_index, bc7tr2_train_all=config.bc7tr2_train_all,
        use_tqdm=True
    )
    preprocessed_data_dic = preprocess(preprocess_config).to_dict()

    if not config.without_preprocess_cache:
        with gzip.open(config.preprocess_cache_path, "wt") as f:
            json.dump(preprocessed_data_dic, f)
else:
    with gzip.open(config.preprocess_cache_path, "rt") as f:
        preprocessed_data_dic = json.load(f)

preprocessed_data = PreprocessedDataset.from_dict(preprocessed_data_dic)
assert V.is_acceptable_version(preprocessed_data.version), f"program:{V.VERSION} data:{preprocessed_data.version}"
experiment_setting["version"] = V.VERSION
experiment_setting["date"] = str(datetime.datetime.today())
print("done.", flush=True, file=sys.stderr)

# %% load eval csv
if config.mode == "eval-csv":
    print("loading csv...", flush=True, file=sys.stderr)
    csv_eval_inputs = csv_to_eval_inputs(
        config.eval_csv_path,
        max_token_length=config.max_token_length, use_tqdm=True,
        concept_index_data=preprocessed_data.concept_index_data, bert_tokenizer_name_or_path=preprocessed_data.bert_tokenizer_name_or_path
    )
    print("done.", flush=True, file=sys.stderr)


# %% prepare dataloader

device = torch.device(config.device)

corpus_known_concept_ids = {Input.from_list(i).concept_id for i in preprocessed_data.train_inputs}
concept_index_to_concept: List[str] = preprocessed_data.concept_index_data.reverse_mapping

minibatch_fields = [
    {"name":"repr_ids", "origin":0, "dtype":torch.long, "device":device, "padding":True, "padding_value":preprocessed_data.pad_token_id, "padding_mask":True},
    {"name":"entity_start", "origin":1, "dtype": torch.long, "device":device},
    {"name":"entity_end", "origin":2, "dtype": torch.long, "device":device},
    {"name":"concept_id", "origin":3, "dtype": torch.long, "device":device},
    {"name":"id", "origin":4},
    {"name":"is_db_instance", "mapping":lambda instance:(instance[4].split(":")[0] == "database"), "dtype":torch.long},
    {"name":"is_corpus_known_class", "mapping":lambda instance:(instance[3] in corpus_known_concept_ids), "dtype":torch.long},
]

all_train_inputs = [input_ for inputs in [preprocessed_data.train_inputs, preprocessed_data.additional_train_inputs] for input_ in inputs]
if config.do_train_database:
    train_dataset_builder = OversampleDatasetBuilder(base_instances=all_train_inputs, supplemental_instances=preprocessed_data.db_inputs, selectors=minibatch_fields, oversample_coeff=config.oversample_coeff)
    get_next_train_dataloader = lambda keep_step=False: train_dataset_builder.next_dataset(keep_step=keep_step).dataloader(chunk_batch_size, shuffle=True)
    print(f"len(supp)/len(main): {train_dataset_builder.supp_vs_main_ratio}", file=sys.stderr, flush=True)
    experiment_setting["supp_vs_main_ratio"] = train_dataset_builder.supp_vs_main_ratio
    print(f"oversample_rate: {train_dataset_builder.oversample_rate}", file=sys.stderr, flush=True)
    experiment_setting["oversample_rate"] = train_dataset_builder.oversample_rate
else:
    train_dataloader = SelectiveDataset(all_train_inputs, minibatch_fields).dataloader(chunk_batch_size, shuffle=True)
    get_next_train_dataloader = lambda keep_step: train_dataloader

dev_dataloader = SelectiveDataset(preprocessed_data.dev_inputs, minibatch_fields).dataloader(chunk_batch_size, shuffle=False)
test_dataloader = SelectiveDataset(preprocessed_data.test_inputs, minibatch_fields).dataloader(chunk_batch_size, shuffle=False)
db_dataloader = SelectiveDataset(preprocessed_data.db_inputs, minibatch_fields).dataloader(chunk_batch_size, shuffle=False)


# %% prepare model

is_cosine_output_layer = {"cosine":True, "linear":False}[config.output_layer]
model_config = ModelConfig(
    encoder=config.encoder, encoder_pool_target=config.encoder_pool_target,
    feature_dim=config.feature_dim, num_class=preprocessed_data.num_concept,
    dropout_output_rate=config.dropout_output_rate, use_feature_bias=config.use_feature_bias, use_feature_following_linear=config.use_feature_following_linear,
    cosine=is_cosine_output_layer, cosine_scale=config.cosine_scale, arcface_margin=config.arcface_margin,
    bert_name_or_path=config.bert,
    vocab_size=None, emb_dim=None, pad_id=None, cell_size=None,
)
model = Model(model_config)
model.to(device)
ce_criterion = torch.nn.CrossEntropyLoss(ignore_index=OOV_CONCEPT_INDEX)
opt = torch.optim.Adam(model.parameters(), config.learning_rate)
print("built model.", flush=True, file=sys.stderr)

if config.init_model_path is not None:
    model.load_state_dict(torch.load(config.init_model_path, map_location=device))
    print(f"loaded from {config.init_model_path}.", flush=True, file=sys.stderr)

# %% overwrite the output layer's weight
def overwrite_weight(dataloader:torch.utils.data.DataLoader) -> None:
    model.eval()

    new_weights = collections.defaultdict(list)
    with torch.no_grad(), contextlib.closing(tqdm.tqdm(dataloader, desc="overwrite_weight (extract features)")) as dataloader_pbar:
        for minibatch in dataloader_pbar:
            arange_seq_len = torch.arange(minibatch["repr_ids"].shape[1]).to(minibatch["entity_start"])[None] # [1,T]
            entity_mask = (minibatch["entity_start"][:,None] <= arange_seq_len) * (arange_seq_len < minibatch["entity_end"][:,None])
            entity_mask = entity_mask.to(minibatch["repr_ids_mask"]) # [B,T]

            features = model.extract_features(inputs=minibatch["repr_ids"], input_mask=minibatch["repr_ids_mask"], entity_mask=entity_mask) # [B,feature_dim]
            for feature, concept_id in zip(features.cpu().numpy(), minibatch["concept_id"].tolist()):
                new_weights[concept_id].append(feature)
    with contextlib.closing(tqdm.tqdm(new_weights.keys(), desc="overwrite_weight (normalize)")) as key_pbar:
        for key in key_pbar:
            mean_feature = np.mean(new_weights[key], 0) # [feature_dim]
            new_weights[key] = mean_feature / np.linalg.norm(mean_feature)
    model.output.overwrite_weight(new_weights)

if (config.mode == "train") and config.do_overwrite_weight:
    overwrite_weight(db_dataloader)
    print("overwrote output layer's weight.", file=sys.stderr, flush=True)

# %% define iteration

class EachScore:
    def __init__(self):
        self.total_num = 0
        self.total_correct = 0
    def update(self, num, correct) -> None:
        self.total_num += num
        self.total_correct += correct
    @property
    def accuracy(self) -> float:
        return self.total_correct / max(1, self.total_num)

class Scores:
    def __init__(self):
        self.all = EachScore()
        self.corpus_known_class = EachScore()
        self.corpus_unseen_class = EachScore()
        self.corpus_instance = EachScore()
        self.db_instance = EachScore()
    def update(self, preds, minibatch):
        # preds: [B]
        correct = preds.eq(minibatch["concept_id"]).to(minibatch["is_corpus_known_class"]) # [B]
        self.all.update(num=len(preds), correct=correct.sum().item())
        self.corpus_known_class.update(num=minibatch["is_corpus_known_class"].sum().item(), correct=(minibatch["is_corpus_known_class"]*correct).sum().item())
        self.corpus_unseen_class.update(num=(1-minibatch["is_corpus_known_class"]).sum().item(), correct=((1-minibatch["is_corpus_known_class"])*correct).sum().item())
        self.corpus_instance.update(num=(1-minibatch["is_db_instance"]).sum().item(), correct=((1-minibatch["is_db_instance"])*correct).sum().item())
        self.db_instance.update(num=minibatch["is_db_instance"].sum().item(), correct=(minibatch["is_db_instance"]*correct).sum().item())

def run_dataloader(dataloader:torch.utils.data.DataLoader, do_update:bool, with_rank:bool=False, with_prob:bool=False, topk=config.eval_topk):
    total_loss = 0.0
    num_loss_computable_instances = 0
    scores = Scores()
    ids = list()
    predictions = list()
    golds = list()
    ranks = list()
    topk_probs = list()
    topk_indices = list()
    gold_probs = list()
    with NullContext() if do_update else torch.no_grad():
        with contextlib.closing(tqdm.tqdm(dataloader)) as dataloader_pbar:
            for chunk in chunking(dataloader_pbar, chunk_size=num_each_chunk):
                if do_update:
                    model.zero_grad()

                for minibatch in chunk:
                    arange_seq_len = torch.arange(minibatch["repr_ids"].shape[1]).to(minibatch["entity_start"])[None] # [1,T]
                    entity_mask = (minibatch["entity_start"][:,None] <= arange_seq_len) * (arange_seq_len < minibatch["entity_end"][:,None]) # [B,T]
                    entity_mask = entity_mask.to(minibatch["repr_ids_mask"]) # [B,T]

                    model_outputs = model.forward(inputs=minibatch["repr_ids"], input_mask=minibatch["repr_ids_mask"], entity_mask=entity_mask, golds=minibatch["concept_id"]) # [B,C]

                    step_loss = ce_criterion(model_outputs["logits"], minibatch["concept_id"])
                    step_num_loss_computable_instances = minibatch["concept_id"].ne(OOV_CONCEPT_INDEX).sum().item()
                    total_loss += step_loss.item() * step_num_loss_computable_instances
                    num_loss_computable_instances += step_num_loss_computable_instances

                    if do_update:
                        step_train_loss = ce_criterion(model_outputs["arcface_logits"], minibatch["concept_id"])
                        (step_train_loss / len(chunk)).backward()

                    step_preds = model_outputs["logits"].argmax(1) # [B]
                    ids.extend(list(minibatch["id"]))
                    predictions.extend(list(step_preds.tolist()))
                    golds.extend(minibatch["concept_id"].tolist())
                    scores.update(preds=step_preds, minibatch=minibatch)

                    if with_rank:
                        step_sort_inds = model_outputs["logits"].sort(-1, descending=True).indices # [B, L]
                        step_golds = minibatch["concept_id"] # [B]
                        step_ranks = ((step_sort_inds==step_golds[:,None])*torch.arange(step_sort_inds.shape[-1]).to(step_sort_inds)).sum(-1) + ((step_golds==-1)*step_sort_inds.shape[-1]).to(step_sort_inds) # [B]
                        ranks.extend(step_ranks.tolist())

                    if with_prob:
                        step_probs = model_outputs["logits"].softmax(-1) # [B, C]
                        step_pred_topks = step_probs.topk(topk, -1)
                        step_pred_topk_values = step_pred_topks.values # [B, topk]
                        step_pred_topk_indices = step_pred_topks.indices # [B, topk]
                        topk_probs.extend(step_pred_topk_values.tolist())
                        topk_indices.extend(step_pred_topk_indices.tolist())
                        step_gold_probs = step_probs[torch.arange(len(step_probs)), minibatch["concept_id"]] # [B]
                        gold_probs.extend(step_gold_probs.tolist())

                if do_update:
                    opt.step()
                dataloader_pbar.set_description(f"batchloss:{step_loss.item():.4e}  acc:{scores.all.accuracy:.4f} (known:{scores.corpus_known_class.accuracy:.4f} unseen:{scores.corpus_unseen_class.accuracy:.4f} corpus:{scores.corpus_instance.accuracy:.4f} db:{scores.db_instance.accuracy:.4f})")
    result = {
        "mean_loss": total_loss / num_loss_computable_instances,
        "accuracy": scores.all.accuracy,
        "accuracy_known": scores.corpus_known_class.accuracy,
        "accuracy_unseen": scores.corpus_unseen_class.accuracy,
        "accuracy_corpus": scores.corpus_instance.accuracy,
        "accuracy_db": scores.db_instance.accuracy,
        "ids": ids,
        "predictions": predictions,
        "golds": golds,
    }
    if with_rank:
        result["ranks"] = ranks
    if with_prob:
        result["topk_probs"] = topk_probs
        result["topk_indices"] = topk_indices
        result["gold_prob"] = gold_probs
    return result

# %%

def format_scores(scores):
    return " ".join([
        f'loss:{scores["mean_loss"]:.6e}',
        f'acc:{scores["accuracy"]:.4f}',
        f'(known:{scores["accuracy_known"]:.4f}',
        f'unseen:{scores["accuracy_unseen"]:.4f}',
        f'corpus:{scores["accuracy_corpus"]:.4f}',
        f'db:{scores["accuracy_db"]:.4f})'
    ])

def run_training():
    if do_save:
        # save configs
        with open(os.path.join(config.save_dir, "config.json"), "w") as f:
            json.dump(D.asdict(config), f)
        with open(os.path.join(config.save_dir, "experiment_setting.json"), "w") as f:
            json.dump(experiment_setting, f)

    best_dev_accuracy = -1.0
    for epoch in range(0, config.max_epoch+1):
        is_initial_epoch = (epoch == 0)
        epoch_start_time = time.time()
        epoch_log = [f"epoch:{epoch}" + ("(init)" if is_initial_epoch else "")]

        # run training
        model.train()
        train_result = run_dataloader(get_next_train_dataloader(keep_step=is_initial_epoch),do_update=(not is_initial_epoch))
        epoch_log.append("train " + format_scores(train_result))

        # run dev
        new_dev_best = False
        if len(preprocessed_data.dev_inputs) == 0:
            new_dev_best = True
        else:
            model.eval()
            dev_result = run_dataloader(dev_dataloader, do_update=False)
            epoch_log.append("dev " + format_scores(dev_result))

            if dev_result["accuracy"] > best_dev_accuracy:
                new_dev_best = True
                best_dev_accuracy = dev_result["accuracy"]

        # report
        epoch_running_time = time.time() - epoch_start_time
        epoch_log.append(f"[{int(epoch_running_time):d}s]")
        epoch_log = ",".join(epoch_log)
        print(epoch_log)

        # save
        if do_save:
            with open(os.path.join(config.save_dir, "log.txt"), "a") as f:
                print(epoch_log, file=f, flush=True)
            if (config.save_every_epoch > 0) and (epoch % config.save_every_epoch) == 0:
                torch.save(model.state_dict(), os.path.join(config.save_dir, f"model_{epoch}.pt"))
            if new_dev_best:
                torch.save(model.state_dict(), os.path.join(config.save_dir, "model_best.pt"))
                with open(os.path.join(config.save_dir, "best-score.txt"), "a") as f:
                    print(epoch_log, file=f, flush=True)

    return


# %% evaluate
def run_validation(dataloader):
    log = list()
    eval_start_time = time.time()
    log.append("eval ")
    model.eval()
    eval_result = run_dataloader(dataloader, do_update=False, with_rank=True, with_prob=True)
    log.append(format_scores(eval_result))
    eval_end_time = time.time()
    log.append(f"[{int(eval_end_time-eval_start_time):d}s]")

    # report
    log = ",".join(log)
    print(log)

    df = pd.DataFrame({"id":eval_result["ids"], "gold_idx":eval_result["golds"], "pred_idx":eval_result["predictions"], "rank":eval_result["ranks"], "gold_prob":eval_result["gold_prob"]})
    df["correct"] = df.apply(lambda row:row["gold_idx"]==row["pred_idx"], axis=1)
    id_to_concept = lambda x: preprocessed_data.concept_index_data.reverse_mapping[x] if x != -1 else "!UNK"
    df["gold"] = df["gold_idx"].map(id_to_concept)
    df["pred"] = df["pred_idx"].map(id_to_concept)
    for k, preds_k in enumerate(zip(*eval_result["topk_indices"]), start=1):
        df[f"pred_{k}"] = [id_to_concept(p) for p in preds_k]
    for k, probs_k in enumerate(zip(*eval_result["topk_probs"]), start=1):
        df[f"prob_{k}"] = probs_k

    return log, df, eval_result


# %%
if config.mode == "train":
    if do_save and (not force_run):
        os.makedirs(config.save_dir, exist_ok=False)
    run_training()

elif config.mode in ["eval-dev", "eval-test", "eval-csv"]:
    if config.mode == "eval-dev":
        target_dataloader = dev_dataloader
    elif config.mode == "eval-test":
        target_dataloader = test_dataloader
    elif config.mode == "eval-csv":
        target_dataloader = SelectiveDataset(csv_eval_inputs, minibatch_fields).dataloader(chunk_batch_size, shuffle=False)
    else:
        raise ValueError(config.mode)
    target_log, target_df, target_result = run_validation(dataloader=target_dataloader)

    if config.eval_output is not None:
        target_df.to_csv(config.eval_output, index=False)

