from operator import index
import os
import ast
import pandas as pd
from queue import Queue
from threading import Thread, Event
from tqdm import tqdm


from utils.helpers import *
from utils.preprocess_data import *


data = pd.read_csv("./data/augment_data.csv", encoding="utf8")
bar = tqdm(total=len(data), position=0, leave=True)


def process_labels(correct, incorrect):
    _, labels = perfect_align(incorrect, correct, 2)
        
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
    return final_label


def write_file(label, idx):
    with open("labels.csv", "a", encoding="utf8") as f:
        f.write(str(idx) + ',' + str(label) + '\n')


def worker(q:Queue, e:Event, w:Queue):
    while not e.is_set() or not q.empty():
        correct, incorrect, idx = q.get()        
        label = process_labels(correct, incorrect)
        write_file(label, idx)
        w.put(idx)
        bar.update(1)


def write(q:Queue, e:Event):
    with open("done.txt", "a") as f:
        while not q.empty() or not e.is_set():
            idx = q.get()
            f.write(str(idx) + "\n")


def main():
    
    corrects = data["correct"].values.tolist()
    incorrects = data["incorrect"].values.tolist()
    index_list = data.index.to_list()
    
    # for idx in index_list:
    #     correct = corrects[idx]
    #     incorrect = incorrects[idx]
    #     label = process_labels(correct, incorrect)
    #     write_file(label, idx)
    #     # w.put(idx)
        
    #     bar.update(1)
    #     break     
    
    
    input_queue = Queue(maxsize=10000)
    writer_queue = Queue(maxsize=10000)

    event = Event()
    event.clear()

    writer_event = Event()
    writer_event.clear()

    threads = [Thread(target=worker, args=(input_queue, event, writer_queue), daemon=True) for _ in range(30)]
    writer_threads = Thread(target=write, args=(writer_queue, writer_event), daemon=True)

    [thread.start() for thread in threads]
    writer_threads.start()

    if os.path.exists("done.txt"):
        done = set(open("done.txt", "r").readlines())
        if len(done)%100000 == 0:
            print(f"done {len(done)} id")
    else: 
        done = set()

    for index in index_list:
        if index not in done:
            input_queue.put((corrects[index].strip(), 
                             incorrects[index].strip(), 
                             index))

    event.set()
    [thread.join() for thread in threads]
    writer_event.set()
    writer_threads.join()
    
if __name__ == "__main__":
    main()