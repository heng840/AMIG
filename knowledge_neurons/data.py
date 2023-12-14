import json
import urllib.request
import collections
from tqdm import tqdm
from pathlib import Path
import os

from opencc import OpenCC
import json
from collections import Counter


cc = OpenCC('t2s')

PARAREL_RELATION_NAMES = [
    "P39",
    "P264",
    "P37",
    "P108",
    "P131",
    "P103",
    "P176",
    "P30",
    "P178",
    "P138",
    "P47",
    "P17",
    "P413",
    "P27",
    "P463",
    "P364",
    "P495",
    "P449",
    "P20",
    "P1376",
    "P1001",
    "P361",
    "P36",
    "P1303",
    "P530",
    "P19",
    "P190",
    "P740",
    "P136",
    "P127",
    "P1412",
    "P407",
    "P140",
    "P279",
    "P276",
    "P159",
    "P106",
    "P101",
    "P937",
]


def pararel(data_path: str = "datasets/pararel.json"):
    parent_dir = Path(data_path).parent
    os.makedirs(parent_dir, exist_ok=True)
    if os.path.exists(data_path):
        with open(data_path, "r") as f:
            return json.load(f)


def pararel_expanded(
        data_path: str = "datasets/source_data/pararel_expanded.json", obj_label_replacement=None
):
    parent_dir = Path(data_path).parent
    os.makedirs(parent_dir, exist_ok=True)
    if os.path.exists(data_path):
        with open(data_path, "r") as f:
            return json.load(f)
    else:
        PARAREL = pararel()
        PARAREL_EXPANDED = collections.defaultdict(dict)
        # expand relations into sentences, grouped by their uuid
        for key, value in tqdm(
                PARAREL.items(), "expanding pararel dataset into full sentences"
        ):
            for vocab in value["vocab"]:
                for graph in value["graphs"]:
                    if not PARAREL_EXPANDED.get(vocab["uuid"]):
                        PARAREL_EXPANDED[vocab["uuid"]] = {
                            "sentences": [],
                            "relation_name": key,
                            "obj_label": vocab["obj_label"],
                        }
                    sentence = graph["pattern"]
                    full_sentence = sentence.replace("[X]", vocab["sub_label"]).replace(
                        "[Y]", "[MASK]"
                    )
                    PARAREL_EXPANDED[vocab["uuid"]]["sentences"].append(full_sentence)
        with open(data_path, "w") as f:
            json.dump(PARAREL_EXPANDED, f)
        return PARAREL_EXPANDED


# 下面是新的处理数据的函数
def mono_one_prompt_pararel(data_path, template_json, sub_obj_dir, language='en'):
    """
    单语言、每个语义只有一个prompt。
    template_json:
    {"relation": "P19", "template": "[X] was born in [Y] ."}
    {"relation": "P20", "template": "[X] died in [Y] ."}
    ...
    sub_obj_dir:
    {"sub_uri": "Q4046943", "obj_uri": "Q5092", "obj_label": "Baltimore", "sub_label": "Pianos Become the Teeth", "lineid": 0, "uuid": "a37257ae-4cbb-4309-a78a-623036c96797"}
    {"sub_uri": "Q808861", "obj_uri": "Q1757", "obj_label": "Helsinki", "sub_label": "Barren Earth", "lineid": 1, "uuid": "bde94480-05d7-40fb-bfed-a944f9909157"}
    ...
    """
    PARAREL = collections.defaultdict(dict)
    templates = {}

    # load templates from template.jsonl directly
    with open(template_json, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # if line is not empty or just whitespace
                template_info = json.loads(line.strip())  # parse the line
                templates[template_info["relation"]] = template_info["template"]

    for r in tqdm(PARAREL_RELATION_NAMES, "loading local data"):
        if r in templates:
            PARAREL[r]["graphs"] = [{"pattern": templates[r]}]

        with open(f'{sub_obj_dir}/{r}.jsonl', 'r', encoding='utf-8') as f:
            if language == 'zh':
                vocab = [json.loads(cc.convert(line.strip())) for line in f if line.strip()]
            else:
                vocab = [json.loads(line.strip()) for line in f if line.strip()]
            PARAREL[r]["vocab"] = vocab
    data = collections.defaultdict(dict)
    # expand relations into sentences, grouped by their uuid
    for key, value in tqdm(
            PARAREL.items(), "expanding parallel dataset into full sentences"
    ):
        for vocab in value["vocab"]:
            for graph in value["graphs"]:
                if not data.get(vocab["uuid"]):
                    data[vocab["uuid"]] = {
                        "sentences": [],
                        "relation_name": key,
                        "obj_label": vocab["obj_label"],
                    }
                sentence = graph["pattern"]
                full_sentence = sentence.replace("[X]", vocab["sub_label"]).replace(
                    "[Y]", "[MASK]"
                )
                data[vocab["uuid"]]["sentences"].append(full_sentence)
    with open(data_path, "w", encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)


def multi_one_prompt_pararel(en_data_path: str, zh_data_path: str, out_data_path: str):
    """
    Bilingual, one prompt for each semantic
    :param en_data_path: English data path
    :param zh_data_path: Chinese data path
    :param out_data_path: Output data path
    :return:
    """
    # Load English data
    with open(en_data_path, "r", encoding="utf-8") as en_f:
        en_data = json.load(en_f)

    # Load Chinese data
    with open(zh_data_path, "r", encoding="utf-8") as zh_f:
        zh_data = json.load(zh_f)

    # Combine English and Chinese data
    multi_lang_data = {}
    for uuid in en_data.keys():
        if uuid in zh_data:  # Only add uuids that exist in both datasets
            multi_lang_data[uuid] = {
                'en': en_data[uuid],
                'zh': zh_data[uuid]
            }

    # Save combined data
    with open(out_data_path, "w", encoding='utf-8') as f:
        json.dump(multi_lang_data, f, ensure_ascii=False)


if __name__ == "__main__":

    # Load the JSON file
    with open('datasets/correspond_dataset/zh.json') as f:
        data = json.load(f)

    # Create a list of all relation_names in the data
    relation_names = []
    for item in data.values():
        if item['relation_name'] not in relation_names:
            relation_names.append(item['relation_name'])

    # Use a Counter to count the occurrences of each relation_name
    # counts = Counter(relation_names)
    # counts_dict = dict(counts)

    # Save to a JSON file
    # with open('datasets/monolingual/counts.json', 'w') as f:
    #     json.dump(counts_dict, f)
    with open('datasets/monolingual/relations.json', 'w') as f:
        json.dump(relation_names, f)
    # Now counts is a dictionary where the keys are relation_names and the values are the counts
    # for relation_name in counts:
    #     print(f'Relation name {relation_name} appears {counts[relation_name]} times.')