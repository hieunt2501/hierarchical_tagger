import pickle
import pandas as pd
from tqdm import tqdm

from utils.preprocess_data import perfect_align


def process_labels(correct, incorrect):
    _, labels = perfect_align(incorrect, correct, 2)
    # print(labels)
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
                tmp_label, tmp_e = final_label[-1]
                tmp_label = tmp_label + next_label[0][-1]
                final_label[-1] = [tmp_label, tmp_e]
                if idx + 1 == len(labels) - 1:
                    break
            elif skip_flag:
                skip_flag = False
            else:
                final_label.append(label)
                skip_flag = False
        else:
            final_label.append(label)
    
    assert len(final_label)==len(incorrect), f"Length labels not equal for sentence: {incorrect}"
    # print(final_label)
    return [label[0] for label in final_label]


def save_file(corrects, incorrects, labels):
    tmp_df = pd.DataFrame({"incorrect": incorrects, "correct": corrects, "label": labels})
    with open("./labeled_dataset.pkl", "wb") as f:
        pickle.dump(tmp_df, f)


def main():
    df = pd.read_csv("./augment_data.csv", encoding="utf8")
    df.correct = df.correct.str.strip()
    df.incorrect = df.incorrect.str.strip()

    corrects = df.correct.values.tolist()
    incorrects = df.incorrect.values.tolist()

    bar = tqdm(total=len(corrects), position=0, leave=True)
    labels = []
    for correct, incorrect in zip(corrects[:5], incorrects[:5]):
        labels.append(process_labels(correct, incorrect))
        bar.update(1)
    save_file(corrects[:5], incorrects[:5], labels)


if __name__ == "__main__":
    main()