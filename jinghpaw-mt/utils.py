import transformers
import pandas as pd
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import gc
import torch
from collections import OrderedDict
import os
import torch.distributed as dist
import numpy as np
import re
import yaml
from typing import Tuple, List, Literal, Dict

# local
import utils


class DataFetcher:
    with open("datasets.yaml", "r") as f:
        ds = yaml.safe_load(f)
    PARADISEC_TRAIN = ds["paradisec"]["train"]
    PARADISEC_DEV = ds["paradisec"]["dev"]
    PARADISEC_TEST = ds["paradisec"]["test"]

    NLLB_TRAIN_ENG = ds["nllb"]["eng"]
    NLLB_TRAIN_KAC = ds["nllb"]["kac"]

    DICT_TRAIN_ENG = ds["dictionary"]["eng"]
    DICT_TRAIN_KAC = ds["dictionary"]["kac"]
    DICT_TRAIN_JPA = ds["dictionary"]["jpa"]

    HEADWORDS_TRAIN_ENG = ds["headwords"]["eng"]
    HEADWORDS_TRAIN_KAC = ds["headwords"]["kac"]
    HEADWORDS_TRAIN_JPA = ds["headwords"]["jpa"]

    COLLOCATION_TRAIN_KAC = ds["collocation"]["kac"]
    COLLOCATION_TRAIN_JPA = ds["collocation"]["jpa"]

    TEXTBOOK_TRAIN_ENG = ds["textbook"]["eng"]
    TEXTBOOK_TRAIN_KAC = ds["textbook"]["kac"]
    TEXTBOOK_TRAIN_JPA = ds["textbook"]["jpa"]

    FLORES_DEV_ENG = ds["flores"]["dev"]["eng"]
    FLORES_DEV_KAC = ds["flores"]["dev"]["kac"]

    CONV_DEV_ENG = ds["conv"]["eng"]
    CONV_DEV_KAC = ds["conv"]["kac"]

    def __init__(self,
                 src_lang: str,
                 tgt_lang: str,
                 batch_size: int,
                 japanese: bool,
                 bidirectional: bool,
                 curriculum_learning: bool):
        with open("datasets.yaml", "r") as f:
            self.ds = yaml.safe_load(f)
        self.ds = self.replace_homedir(self.ds)
        
        self.PARADISEC_TRAIN = self.ds["paradisec"]["train"]
        self.PARADISEC_DEV = self.ds["paradisec"]["dev"]
        self.PARADISEC_TEST = self.ds["paradisec"]["test"]

        self.NLLB_TRAIN_ENG = self.ds["nllb"]["eng"]
        self.NLLB_TRAIN_KAC = self.ds["nllb"]["kac"]

        self.DICT_TRAIN_ENG = self.ds["dictionary"]["eng"]
        self.DICT_TRAIN_KAC = self.ds["dictionary"]["kac"]
        self.DICT_TRAIN_JPA = self.ds["dictionary"]["jpa"]

        self.HEADWORDS_TRAIN_ENG = self.ds["headwords"]["eng"]
        self.HEADWORDS_TRAIN_KAC = self.ds["headwords"]["kac"]
        self.HEADWORDS_TRAIN_JPA = self.ds["headwords"]["jpa"]

        self.COLLOCATION_TRAIN_KAC = self.ds["collocation"]["kac"]
        self.COLLOCATION_TRAIN_JPA = self.ds["collocation"]["jpa"]

        self.TEXTBOOK_TRAIN_ENG = self.ds["textbook"]["eng"]
        self.TEXTBOOK_TRAIN_KAC = self.ds["textbook"]["kac"]
        self.TEXTBOOK_TRAIN_JPA = self.ds["textbook"]["jpa"]

        self.FLORES_DEV_ENG = self.ds["flores"]["dev"]["eng"]
        self.FLORES_DEV_KAC = self.ds["flores"]["dev"]["kac"]

        self.CONV_DEV_ENG = self.ds["conv"]["eng"]
        self.CONV_DEV_KAC = self.ds["conv"]["kac"]

        self.funcmap = {"paradisec": self.load_paradisec,
                   "nllb": self.load_nllb,
                   "dict": self.load_dict,
                   "headwords": self.load_headwords,
                   "collocation": self.load_collocation,
                   "textbook": self.load_textbook,
                   "flores": self.load_flores,
                   "conv": self.load_conv}
        self.batch_size = batch_size
        self.japanese = japanese
        self.bidirectional = bidirectional
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.curriculum_learning = curriculum_learning

        curriculum_order = ["textbook", "headwords", "collocation",
                           "dict", "paradisec", "nllb"]
        self.curriculum = {dataname: i for (i, dataname) in enumerate(curriculum_order)}
 
    def replace_homedir(self, ds: dict) -> dict:
        """Replace the relative path of the home directory (`~`) to the absolute path."""
        homedir = os.getenv("HOME") # <- this will only work on Mac/Linux OSs
        for dataname, content in ds.items():
            for split, path in content.items():
                if type(path) == dict:
                    for lang, val in path.items():
                        ds[dataname][split][lang] = ds[dataname][split][lang].replace("~", homedir)
                else:
                    ds[dataname][split] = ds[dataname][split].replace("~", homedir)
        return ds

    def get_dataloaders(self,
                        train: List[pd.DataFrame],
                        dev: Dict[str, pd.DataFrame]) -> Tuple[DataLoader,
                                                               Dict[str, DataLoader]]:
        """Make dataloaders for training and evaluation (dev).
        Return:
        - Tuple[List[List[DataLoader]]]: (train_dataloaders, dev_dataloaders)
          - List[List[DataLoader]]: datasets consisting of (bi/uni)directional dataloaders."""
        train_dl = self.get_train_dataloader(train)

        dev_dataloaders: Dict[str, DataLoader] = dict()
        for name, df in dev.items():
            print(name, df)
            ds = self.dataframe_to_dataset(df, self.src_lang, self.tgt_lang)
            dl = DataLoader(dataset=ds,
                            batch_size=self.batch_size,
                            shuffle=False)
            dev_dataloaders[name] = dl
        return train_dl, dev_dataloaders

    def get_train_dataloader(self,
                             train: Dict[str, pd.DataFrame]) -> DataLoader:
        """Make dataloaders from dataasets.
        If self.curriculum_learning, concatenate the datasets based on the order.
        Else, concatenate the datasets and shuffle.
        """
        ds_dict = dict()
        for dataname, df in train.items():
            print("Loading training data:", dataname)
            if df.empty:
                continue
            dss = self.make_datasets(df)
            print(dss)
            ds = ConcatDataset(dss)
            ds_dict[dataname] = ds

        if self.curriculum_learning:
            # sort
            sorted_ds_dict = OrderedDict()
            for dataname in self.curriculum.keys(): # the curriculum can just be a list
                try:
                    sorted_ds_dict[dataname] = ds_dict[dataname]
                except KeyError:
                    continue # the dataset in the curriculum does not exist in the training data
            shuffle = False
            dataset = ConcatDataset(list(sorted_ds_dict.values()))
        else:
            shuffle = True
            # concatenate
            dataset = ConcatDataset(list(ds_dict.values()))

        # dataloader
        train_dl = DataLoader(dataset=dataset,
                              batch_size=self.batch_size,
                              shuffle=shuffle)
        return train_dl

    def make_datasets(self,
                      df: pd.DataFrame) -> List[Dataset]:
        """Make a custom dataset.
        For training data,
        _____________________________________________
        |         |bidirectional    |unidirectional | 
        |---------|-----------------|---------------|
        |kac,ja,en|kac<->ja,kac<->en|ja->kac,en->kac|
        |kac,en   |kac<->en         |en->kac        |
        --------------------------------------------- 
        For evaluation data, en->kac
        """
        lang1, lang2, lang3 = "kac_Latn", "eng_Latn", "jpa_Jpan"
        dss = []
        if lang2 in df.columns: # exclude (kac, ja)-type: collocation
            en_kac_ds = self.dataframe_to_dataset(df, lang2, lang1)
            dss.append(en_kac_ds)
            if self.bidirectional:
                kac_en_ds = self.dataframe_to_dataset(df, lang1, lang2)
                dss.append(kac_en_ds)
        print(df.columns)
        if self.japanese and lang3 in df.columns:
            ja_kac_ds = self.dataframe_to_dataset(df, lang3, lang1)
            dss.append(ja_kac_ds)
            if self.bidirectional:
                kac_ja_ds = self.dataframe_to_dataset(df, lang1, lang3)
                dss.append(kac_ja_ds)
        return dss

    def make_dataloaders(self,
                         dataset: pd.DataFrame) -> List[DataLoader]:
        """DEPRECATED. Use `self.make_datasets() instead
        Make dataloaders.
        For training data,
        _____________________________________________
        |         |bidirectional    |unidirectional |
        |---------|-----------------|---------------|
        |kac,ja,en|kac<->ja,kac<->en|ja->kac,en->kac|
        |kac,en   |kac<->en         |en->kac        |
        ---------------------------------------------
        For evaluation data, en->kac
        """
        
        dataloaders: List[DataLoader] = []
        lang1, lang2, lang3 = "kac_Latn", "eng_Latn", "jpa_Latn"
        if lang2 in dataset.columns: # exclude (kac, ja)-type: collocation
            en_kac = self.dataframe_to_dataloader(dataset, lang2, lang1)
            dataloaders.append(en_kac)
            if self.bidirectional:
                kac_en = self.dataframe_to_dataloader(dataset, lang1, lang2)
                dataloaders.append(kac_en)
        if self.japanese and lang3 in dataset.columns:
            ja_kac = self.dataframe_to_dataloader(dataset, lang3, lang1)
            dataloaders.append(ja_kac)
            if self.bidirectional:
                kac_ja = self.dataframe_to_dataloader(dataset, lang1, lang3)
                dataloaders.append(kac_ja)
        return dataloaders

    def make_dev_dataloader(self,
                            df: pd.DataFrame) -> DataLoader:
        """Make dataloaders for the evaluation dataset (en -> kac only)"""
        ds = self.dataframe_to_dataset(df, self.src_lang, self.tgt_lang)
        dl = DataLoader(dataset=dataset,
                        batch_size=self.batch_size,
                        shuffle=shuffle)
        return dl

    def dataframe_to_dataloader(self,
                                df: pd.DataFrame,
                                src_lang: str,
                                tgt_lang: str,
                                shuffle: bool = True) -> DataLoader:
        """Get a data frame, convert it to CustomDataset,
        and make a DataLoader."""
        print(df)
        dataset = self.dataframe_to_dataloader(df,
                                               src_lang=src_lang,
                                               tgt_lang=tgt_lang)
        dataloader = DataLoader(dataset=dataset,
                                batch_size=self.batch_size,
                                shuffle=shuffle)
        return dataloader

    def dataframe_to_dataset(self,
                                df: pd.DataFrame,
                                src_lang: str,
                                tgt_lang: str) -> Dataset:
        dataset = CustomDataset(df,
                                src_lang=src_lang,
                                tgt_lang=tgt_lang)
        return dataset

    def fetch_data(self,
                   datanames: List[str]) -> Tuple[Dict[str, pd.DataFrame],
                                                  Dict[str, pd.DataFrame]]:
        """Fetch the required data."""
        train = self.fetch_multilingual(datanames)
        # dev data
        paradisec_dev = self.load_paradisec("dev")
        flores = self.load_flores()
        conv = self.load_conv()
        dev = {"paradisec": paradisec_dev,
               "flores": flores,
               "conv": conv}
        print("data overview:")
        print("paradisec_dev:", dev["paradisec"])
        print("flores:", dev["flores"])
        print("conv:", dev["conv"])
        return train, dev

    def fetch_multilingual(self,
                           datanames: list) -> Dict[str, pd.DataFrame]:
        """Fetch multilingual datasets with 2 or 3 languages."""
        train_tri = pd.DataFrame({"eng_Latn": [],
                                  "kac_Latn": [],
                                  "jpa_Jpan": []})
        train_enkac = pd.DataFrame({"eng_Latn": [],
		                    "kac_Latn": []})
        train_jakac = pd.DataFrame({"jpa_Jpan": [],
                                    "kac_Latn":	[]})
        df_dict = dict()
        for dataname in datanames:
            if dataname == "paradisec":
                train = self.load_paradisec("train")
                train_enkac = pd.concat([train_enkac, train])
                df_dict[dataname] = train_enkac
            else:
                train = self.funcmap[dataname]()
                if set(train.columns.tolist()) == {"jpa_Jpan", "kac_Latn"}:
                    if self.japanese:
                        train_jakac = pd.concat([train_jakac, train])
                        df_dict[dataname] = train_jakac
                    else:
                        continue
                elif set(train.columns.tolist()) == {"jpa_Jpan", "kac_Latn", "eng_Latn"}:
                    train_tri = pd.concat([train_tri, train])
                    df_dict[dataname] = train_tri
                elif set(train.columns.tolist()) == {"eng_Latn", "kac_Latn"}:
                    train_enkac = pd.concat([train_enkac, train])
                    df_dict[dataname] = train_enkac
                else:
                    print(dataname, train.columns)
                    raise Exception("Unknown language")

        return df_dict
                    
    def load(self, path: str) -> list:
        """Load the txt data from the path and format it."""
        with open(path) as f:
            data = f.readlines()
        data = [d.strip() for d in data]
        return data

    def load_paradisec(self, split: Literal["train", "dev"]) -> pd.DataFrame:
        """Load the training or dev split of the Paradisec dataset."""
        colname_map = {"Jinghpaw": "kac_Latn", "English": "eng_Latn"}
        split_map = {"train": self.PARADISEC_TRAIN, "dev": self.PARADISEC_DEV}
        df = pd.read_csv(split_map[split], index_col=0).dropna()
        df = df.rename(columns=colname_map)
        return df

    def load_nllb(self) -> pd.DataFrame:
        """Load the data from NLLB (for training only)."""
        eng = self.load(self.NLLB_TRAIN_ENG)
        kac = self.load(self.NLLB_TRAIN_KAC)
        train = pd.DataFrame({"eng_Latn": eng,
                             "kac_Latn": kac})
        train = self.clean_nllb(train)
        return train

    def clean_nllb(self,
                   df: pd.DataFrame,
                   src_lang: str = "eng_Latn",
                   tgt_lang: str = "kac_Latn") -> pd.DataFrame:
        """Clean the NLLB data."""
        nllb_filter = NLLBFilter(src_lang, tgt_lang)
        df = nllb_filter.clean_text(df) # remove sentence-initial numbers
        df = nllb_filter.length_filter(df, threshold=30) # remove sentences that are too short
        df = nllb_filter.length_ratio_filter(df, upper_w=0.7, lower_w=0.5) # remove pairs with an unusual length ratio
        df = nllb_filter.drop_duplicate_samples(df)
        return df

    def load_dict(self) -> pd.DataFrame:
        """Load the dictionary example sentences (for training only)."""
        eng = self.load(self.DICT_TRAIN_ENG)
        kac = self.load(self.DICT_TRAIN_KAC)
        jpa = self.load(self.DICT_TRAIN_JPA)
        train = pd.DataFrame({"eng_Latn": eng,
                              "kac_Latn": kac,
                              "jpa_Jpan": jpa})
        return train

    def load_headwords(self) -> pd.DataFrame:
        """Load the dictionary headwords (for training only)."""
        eng = self.load(self.HEADWORDS_TRAIN_ENG)
        kac = self.load(self.HEADWORDS_TRAIN_KAC)
        jpa = self.load(self.HEADWORDS_TRAIN_JPA)
        train = pd.DataFrame({"eng_Latn": eng,
                              "kac_Latn": kac,
                              "jpa_Jpan": jpa})
        return train

    def load_collocation(self) -> pd.DataFrame:
        """Load the dictionary collocations (for training only.)"""
        kac = self.load(self.COLLOCATION_TRAIN_KAC)
        jpa = self.load(self.COLLOCATION_TRAIN_JPA)
        train = pd.DataFrame({"kac_Latn": kac,
                              "jpa_Jpan": jpa})
        return train

    def load_textbook(self) -> pd.DataFrame:
        """Load the textbook example sentences (for training only.)"""
        eng = self.load(self.TEXTBOOK_TRAIN_ENG)
        kac = self.load(self.TEXTBOOK_TRAIN_KAC)
        jpa = self.load(self.TEXTBOOK_TRAIN_JPA)
        train = pd.DataFrame({"eng_Latn": eng,
                              "kac_Latn": kac,
                              "jpa_Jpan": jpa})
        return train          

    def load_flores(self) -> pd.DataFrame:
        """Load the dev data from Flores."""
        eng = self.load(self.FLORES_DEV_ENG)
        kac = self.load(self.FLORES_DEV_KAC)
        dev = pd.DataFrame({"eng_Latn": eng,
                            "kac_Latn": kac})
        return dev

    def load_conv(self) -> pd.DataFrame:
        """Load the dev data from the conversational text."""
        eng = self.load(self.CONV_DEV_ENG)
        kac = self.load(self.CONV_DEV_KAC)
        dev = pd.DataFrame({"eng_Latn": eng,
                            "kac_Latn": kac})
        return dev

### For distributed data parallel (DDP) training
def ddp_setup(rank: int, world_size: int) -> None:
    """Setup for DDP training."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def ddp_cleanup() -> None:
    """Clean-up function for DDP training."""
    dist.destroy_process_group()
    
    
def word_tokenize(text: str) -> str:
    """a very naive word tokenizer for languages with English-like orthography"""
    return re.findall(r'(\w+|[^\w\s])', text)


def filter_by_sent_length(df: pd.DataFrame, lang: str, length=5):
    """Filter by the number of tokens (not subword-tokenized but
    simply splitting by whitespace)
    `lang` (str): Languge name (e.g., "Quechua")
    """
    assert not df.isnull().values.any()
    assert lang in list(lang_map.keys()) + ["Spanish"]
    if lang == "Spanish":
        df["spa_Latn_length"] = df["spa_Latn"].apply(get_sent_length)
        return df[df["spa_Latn_length"] <= length]
    else:
        col_name = lang_map[lang]
        new_col_name = col_name + "_length"
        df[new_col_name] = df[col_name].apply(get_sent_length)
        return df[df[new_col_name] <= length]


def sort_and_filter(df: pd.DataFrame, src_lang: str) -> pd.DataFrame:
    """Sort by (subword-tokenized) length and filter out samples
    that are too long.
    `src_lang` (str): e.g., kac_Latn
    """
    tok_colname = src_lang + "_tok"
    tok_len_colname = tok_colname + "_length"
    df[tok_colname] = df[src_lang].apply(tokenizer.tokenize)
    df[tok_len_colname] = df[tok_colname].apply(len)
    df = df[df[tok_len_colname] <= args.max_src_subtok_length]
    df = df.sort_values(by=tok_len_colname)
    return df


class NLLBFilter:
    def __init__(self,
                 lang1: str = "eng_Latn",
                 lang2: str = "kac_Latn"):
        self.lang1 = lang1
        self.lang2 = lang2
        
    def clean_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """For the NLLB dataset, remove the number digits in the beginning of a sentence,
        because this is often just an index of the Bible."""
        def regex_filter(text: str):
            pattern = r"^\s*\d+\s*"
            return re.sub(pattern, "", text)
    
        df[self.lang1] = df[self.lang1].apply(regex_filter)
        df[self.lang2] = df[self.lang2].apply(regex_filter)
        return df

    def length_filter(self, df: pd.DataFrame, threshold=40) -> pd.DataFrame:
        """Filter out sentences that are too short."""
        return df[(df[self.lang1].str.len() > threshold) &
                  (df[self.lang2].str.len() > threshold)]

    def length_ratio_filter(self,
                            df: pd.DataFrame,
                            upper_w: float = 0.7,
                            lower_w: float = 0.5):
        """Filter out sentence pairs that have an unusual sentence length ratio."""
        ratio_list = [len(e) / len(k) for e, k in zip(df[self.lang1], df[self.lang2])]
        ratio_stdev = np.std(ratio_list)
        ratio_mean = np.mean(ratio_list)
        upper = ratio_mean + ratio_stdev * upper_w
        lower = ratio_mean + ratio_stdev * lower_w
        return df[(df[self.lang1].str.len() / df[self.lang2].str.len() < upper) &
                  (df[self.lang1].str.len() / df[self.lang2].str.len() > lower)]

    def drop_duplicate_samples(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop duplicate entries."""
        df = df.drop_duplicates(subset=[self.lang1])
        df = df.drop_duplicates(subset=[self.lang2])
        return df


def get_batch_pairs(batch_size: int, step: int, df: pd.DataFrame, src_lang: str, tgt_lang) -> tuple:
    """Get batch pairs without random sampling.
    Make sure that the training data is sorted by length.
    args:
    - src_lang (str): Source language code (usually spa_Latn)
    - tgt_lang (str): Target language code
    """
    xx, yy = [], []
    for i in range(batch_size):
        if batch_size * step + i >= len(data):
            break
        item = df.iloc[batch_size * step + i]
        xx.append(item[src_lang])
        yy.append(item[tgt_lang])
    return xx, yy


def fix_tokenizer(tokenizer: transformers.models.nllb.tokenization_nllb_fast.NllbTokenizerFast,
                  new_lang: str) -> None:
    """Fix the tokenizer to include the new language code.
    args:
    - new_lang (str): the language code of the new language, e.g., `cni_Latn`
    """
    new_special_tokens = tokenizer.additional_special_tokens + [new_lang]
    tokenizer.add_special_tokens({'additional_special_tokens': new_special_tokens})
    tokenizer.lang_code_to_id[new_lang] = len(tokenizer) - 1

    
def get_random_batch_pairs(batch_size: int, data: pd.DataFrame) -> tuple:
    """Randomly sample for a training batch."""
    (l1, lang1), (l2, lang2) = random.sample(LANGS, 2)
    xx, yy = [], []
    for _ in range(batch_size):
        item = data.iloc[random.randint(0, len(data)-1)]
        xx.append(item[l1])
        yy.append(item[l2])
    return xx, yy, lang1, lang2


def cleanup() -> None:
    """Clean up the memory."""
    gc.collect()
    torch.cuda.empty_cache()


class CustomDataset(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 src_lang: str,
                 tgt_lang: str):
        self.df = df
        self.src = df[src_lang]
        self.tgt = df[tgt_lang]
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """Add the language code in the first 8 characters.
        To extract the sentence, just do something like sample[8:]."""
        x = self.src_lang + self.src.iloc[idx]
        y = self.tgt_lang + self.tgt.iloc[idx]
        return x, y


class EarlyStopper:
    """From https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch""" 
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
