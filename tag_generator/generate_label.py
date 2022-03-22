import pickle
import pandas as pd
from tqdm import tqdm

from utils.preprocess_data import perfect_align


def process_labels(correct, incorrect):
    _, labels = perfect_align(incorrect, correct, 3)
    final_label = []
    skip_flag = False
    for idx in range(len(labels)):
        label = labels[idx]
        if idx + 1 < len(labels):
            next_label = labels[idx+1]
            if label[0].startswith("KEEP") and next_label[0].startswith("APPEND"):
                final_label.append([next_label[0], label[1]])
                skip_flag = True
                if idx + 1 == len(labels) - 1:
                    break
            elif label[0].startswith("APPEND") and next_label[0].startswith("APPEND"):
                if final_label:
                    tmp_label, tmp_e = final_label[-1]
                    tmp_label = tmp_label + next_label[0][-1]
                    final_label[-1] = [tmp_label, tmp_e]
                    if idx + 1 == len(labels) - 1:
                        break
                else:
                    tmp_label, tmp_e = label
                    tmp_label = tmp_label + next_label[0][-1]
                    final_label.append([tmp_label, tmp_e])
                    skip_flag = True
            elif skip_flag:
                skip_flag = False
            else:
                final_label.append(label)
                skip_flag = False
        else:
            final_label.append(label)

    if len(final_label) == len(incorrect) + 1:
        if final_label[0][1] != final_label[1][1]:
            print(f"Length labels not equal for sentence: {incorrect}")
            raise Exception

    elif len(final_label) != len(incorrect):
        print(f"Length labels not equal for sentence: {incorrect}")
        raise Exception

    return [label[0] for label in final_label]


def extract_general_tag(label):
    for i in range(len(label)):
        if label[i].startswith("REPLACE") or label[i].startswith("APPEND"):
            label[i] = label[i].split("_")[0]
    return label


def save_file(corrects, incorrects, edit_tags, general_tags, pkl_file=False):
    tmp_df = pd.DataFrame({"incorrect": incorrects, 
                            "correct": corrects, 
                            "edit_tag": edit_tags,
                            "general_tag": general_tags})
    if pkl_file:
        with open("./labeled_dataset.pkl", "wb") as f:
            pickle.dump(tmp_df, f)
    else:
        tmp_df.to_csv("labeled_dataset.csv", encoding="utf8")


def main():
    df = pd.read_csv("./data/augment_data.csv", encoding="utf8")
    df.correct = df.correct.str.strip()
    df.incorrect = df.incorrect.str.strip()

    corrects = df.correct.values.tolist()
    incorrects = df.incorrect.values.tolist()

    bar = tqdm(total=len(corrects), position=0, leave=True)
    edit_tags = []
    general_tags = []
    for correct, incorrect in zip(corrects[:5], incorrects[:5]):
        edit_tag = process_labels(correct, incorrect)
        general_tag = extract_general_tag(edit_tag.copy())
        edit_tags.append(edit_tag)
        general_tags.append(general_tag)
        bar.update(1)
    save_file(corrects[:5], incorrects[:5], edit_tags, general_tags)


if __name__ == "__main__":
    main()