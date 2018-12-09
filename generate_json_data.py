import argparse, json
from collections import Counter


def generate_json_data(split_path, data_path, max_captions_per_image, min_word_count):
    split = json.load(open(split_path, 'r'))
    word_count = Counter()

    train_img_paths = []
    train_caption_tokens = []
    validation_img_paths = []
    validation_caption_tokens = []

    max_length = 0
    for img in split['images']:
        caption_count = 0
        for sentence in img['sentences']:
            if caption_count < max_captions_per_image:
                caption_count += 1
            else:
                break

            if img['split'] == 'train':
                train_img_paths.append(data_path + '/imgs/' + img['filepath'] + '/' + img['filename'])
                train_caption_tokens.append(sentence['tokens'])
            elif img['split'] == 'val':
                validation_img_paths.append(data_path + '/imgs/' + img['filepath'] + '/' + img['filename'])
                validation_caption_tokens.append(sentence['tokens'])
            max_length = max(max_length, len(sentence['tokens']))
            word_count.update(sentence['tokens'])

    words = [word for word in word_count.keys() if word_count[word] >= args.min_word_count]
    word_dict = {word: idx + 4 for idx, word in enumerate(words)}
    word_dict['<start>'] = 0
    word_dict['<eos>'] = 1
    word_dict['<unk>'] = 2
    word_dict['<pad>'] = 3

    with open(data_path + '/word_dict.json', 'w') as f:
        json.dump(word_dict, f)

    train_captions = process_caption_tokens(train_caption_tokens, word_dict, max_length)
    validation_captions = process_caption_tokens(validation_caption_tokens, word_dict, max_length)

    with open(data_path + '/train_img_paths.json', 'w') as f:
        json.dump(train_img_paths, f)
    with open(data_path + '/val_img_paths.json', 'w') as f:
        json.dump(validation_img_paths, f)
    with open(data_path + '/train_captions.json', 'w') as f:
        json.dump(train_captions, f)
    with open(data_path + '/val_captions.json', 'w') as f:
        json.dump(validation_captions, f)


def process_caption_tokens(caption_tokens, word_dict, max_length):
    captions = []
    for tokens in caption_tokens:
        token_idxs = [word_dict[token] if token in word_dict else word_dict['<unk>'] for token in tokens]
        captions.append(
            [word_dict['<start>']] + token_idxs + [word_dict['<eos>']] +
            [word_dict['<pad>']] * (max_length - len(tokens)))

    return captions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate json files')
    parser.add_argument('--split-path', type=str, default='data/coco/dataset.json')
    parser.add_argument('--data-path', type=str, default='data/coco')
    parser.add_argument('--max-captions', type=int, default=5,
                        help='maximum number of captions per image')
    parser.add_argument('--min-word-count', type=int, default=5,
                        help='minimum number of occurences of a word to be included in word dictionary')
    args = parser.parse_args()

    generate_json_data(args.split_path, args.data_path, args.max_captions, args.min_word_count)
