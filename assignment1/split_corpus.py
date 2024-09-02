import argparse
import random


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--corpus_file", default='reviews.txt', type=str,
                        help="Corpus file to split into train/dev/test (default reviews.txt)")
    args = parser.parse_args()
    return args


def split_corpus(corpus_file):
    """
    Splits the corpus into train, dev, and test files.
    The default split ratio is 80% train, 10% dev, 10% test.

    :param corpus_file: Path to the corpus file to be split
    :return:
    """

    with open(corpus_file, encoding='utf-8') as in_file:
        lines = in_file.readlines()

    total_lines = len(lines)
    train_size = int(total_lines * 0.8)
    dev_size = int(total_lines * 0.1)

    random.seed(42)
    random.shuffle(lines)

    train_lines = lines[:train_size]
    dev_lines = lines[train_size:train_size + dev_size]
    test_lines = lines[train_size + dev_size:]

    with open('train.txt', 'w', encoding='utf-8') as train_file:
        train_file.writelines(train_lines)

    with open('dev.txt', 'w', encoding='utf-8') as dev_file:
        dev_file.writelines(dev_lines)

    with open('test.txt', 'w', encoding='utf-8') as test_file:
        test_file.writelines(test_lines)

    print(f"Data split completed: {len(train_lines)} train, {len(dev_lines)} dev, {len(test_lines)} test")


if __name__ == "__main__":
    args = create_arg_parser()

    split_corpus(args.corpus_file)
