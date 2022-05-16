
import numpy as np
import re

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image
from imagefolder import get_images
from contextlib import ExitStack

import json


pad_word = "<pad>"
bos_word = "<s>"
eos_word = "</s>"
unk_word = "<unk>"
pad_id = 0
bos_id = 1
eos_id = 2
unk_id = 3

from random import sample

from transformers import LayoutLMv2Processor
from PIL import Image

class dataloader(Dataset):
    """
    Dataloader class used to load in data in an image folder.
    Made it so that it performs a fixed set of transformations to a pair of images in different folders
    """

    def __init__(self, file_lists, opts, class_names, size=224, train=True):

        self.file_lists = file_lists
        # a pandas dataframe that holds multiple_choice_answer, image_id, question_id, question
        # that will get used for the project
        self.opts = opts
        self.size = size #for resizing
        self.train = train
        self.class_names = class_names

        # image transformation pipeline
        self.transforms = self.transform_images()

        # Processor

        self.processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", revision="no_ocr")


    def load_bbox(self, file_name):

        temp_name_ocr = file_name.split('.')[0] + '_ocr' + '.txt'
        ocr_list = []

        with open(temp_name_ocr) as ocr:

            while True:
                temp_list = []
                line_ocr = ocr.readline()

                if not line_ocr:
                    break

                temp = line_ocr.replace('[', '').replace(']','')
                temp = temp.split(',')

                if len(temp) == 4:
                    for number in temp:
                        temp_list.append(int(number))
                    ocr_list.append(temp_list)

            ocr.close()

        return ocr_list

    def load_ocr_texts(self, file_name, bbox):

        temp_name_text = file_name.split('.')[0]  + '_text' + '.txt'

        content = []
        iterator = 0
        ocr_word_list = []

        with open(temp_name_text) as f:

            while True:

                line_text = f.readline()

                if not line_text:
                    break

                split_line = line_text.split(' ')

                temp_ocr = bbox[iterator]

                for i in split_line:

                    content.append(re.sub(r'\W+', '', i).lower())
                    ocr_word_list.append(temp_ocr)

                iterator += 1

            f.close()

        return content, ocr_word_list

    def transform_images(self):

        transforms_list = [
            transforms.ToTensor(),
                           ]

        if self.train:
            # flip horizontally
            transforms_list.append(transforms.RandomHorizontalFlip(self.opts.h_flip))
            # random crop
            transforms_list.append(transforms.RandomResizedCrop(self.opts.size, scale=(0.85, 1.1), ratio=(0.85, 1.1)))

        transforms_list.append(transforms.Resize(size=(self.opts.size, self.opts.size)))
        transforms_list.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))

        return transforms.Compose(transforms_list)

    def listToString(self, s):

        # initialize an empty string
        str1 = " "

        # return string
        return (str1.join(s))

    def __len__(self):

        return len(self.file_lists)

    def __getitem__(self, index):

        # select an image
        file_name = self.file_lists[index]

        # load the image
        image = Image.open(file_name).convert('RGB')

        # get the ocr'ed text
        bbox = self.load_bbox(file_name)
        words, boxes = self.load_ocr_texts(file_name, bbox)

        label = file_name.split('/')[2]

        encoding = self.processor(image, words, boxes=boxes, return_tensors="pt")

        #encoding.update({'label':label})
        #print(encoding)
        return {"encoding": encoding, "label": label}

        # return the triplet of image, question and answer.
        #return {"image": self.transforms(image), "text": self.listToString(ocr_txt), "label": self.class_names[label], "bbox": bbox}


def LayoutLM_collate_fn(batch):
    '''

    '''

    # Sort conv_ids based on decreasing order of the src_lengths.
    # This is required for efficient GPU computations.
    input_ids_list = []
    image_list = []
    attention_mask_list = []
    token_type_ids = []
    bbox_list = []
    labels_list = []

    for e in batch:
        image_list.append(e['encoding']["image"])
        input_ids_list.append(e['encoding']["input_ids"])
        token_type_ids.append(e['encoding']["token_type_ids"])
        attention_mask_list.append(e['encoding']["attention_mask"])
        bbox_list.append(e['encoding']["bbox"])
        labels_list.append(e["label"])

    # sort based on the length of input_id
    batch = list(zip(input_ids_list, token_type_ids, attention_mask_list, bbox_list, image_list, labels_list))
    batch.sort(key=lambda x: len(x[0]), reverse=True)

    sentences, answer, masks, images, file_name = [], [], [], [], []

    idx = 0
    for data in batch:
        input_ids_, token_type_, attention_mask_, bbox_, image_, labels_ = data

        # tokenize the sentence
        tokenizer_output = tokenizer(ocr_txt)
        # get the ids of the input words
        tokenized_sent = tokenizer_output['input_ids'][:tokenizer.max_len_single_sentence]
        # get the attention_masks
        mask = tokenizer_output['attention_mask'][:tokenizer.max_len_single_sentence]
        # append the sentences
        sentences.append(torch.tensor(tokenized_sent))

        answer.append(label)
        masks.append(torch.tensor(mask))

        if idx == 0:
            images = torch.FloatTensor(images_).unsqueeze(0)  # unsqueeze to create a batch dimension
        else:
            images = torch.cat((images, torch.FloatTensor(images_).unsqueeze(0)), dim=0)

        idx += 1

    sentences_pad = pad_sequence(sentences, batch_first=True, padding_value=bert_pad_token)
    masks = pad_sequence(masks, batch_first=True, padding_value=0.0)
    answer = torch.tensor(answer)

    #print(sentences_pad.shape)

    return {"image": images, "text": sentences_pad, "label": answer, "masks": masks}



def transformer_collate_fn(batch, tokenizer):
    '''

    '''

    # Sort conv_ids based on decreasing order of the src_lengths.
    # This is required for efficient GPU computations.
    images = [torch.FloatTensor(e["image"]) for e in batch]
    ocr_txt = [e["text"] for e in batch]
    label = [e["label"] for e in batch]

    #print(images)
    #print(ocr_txt)
    #print(label)

    #
    batch = list(zip(ocr_txt, label, images))
    batch.sort(key=lambda x: len(x[0]), reverse=True)

    bert_vocab = tokenizer.get_vocab()
    bert_pad_token = bert_vocab['[PAD]']
    bert_unk_token = bert_vocab['[UNK]']
    bert_cls_token = bert_vocab['[CLS]']

    sentences, answer, masks, images, file_name = [], [], [], [], []

    idx = 0
    for data in batch:
        ocr_txt, label, images_ = data

        # tokenize the sentence
        tokenizer_output = tokenizer(ocr_txt)
        # get the ids of the input words
        tokenized_sent = tokenizer_output['input_ids'][:tokenizer.max_len_single_sentence]
        # get the attention_masks
        mask = tokenizer_output['attention_mask'][:tokenizer.max_len_single_sentence]
        # append the sentences
        sentences.append(torch.tensor(tokenized_sent))

        answer.append(label)
        masks.append(torch.tensor(mask))

        if idx == 0:
            images = torch.FloatTensor(images_).unsqueeze(0)  # unsqueeze to create a batch dimension
        else:
            images = torch.cat((images, torch.FloatTensor(images_).unsqueeze(0)), dim=0)

        idx += 1

    sentences_pad = pad_sequence(sentences, batch_first=True, padding_value=bert_pad_token)
    masks = pad_sequence(masks, batch_first=True, padding_value=0.0)
    answer = torch.tensor(answer)

    #print(sentences_pad.shape)

    return {"image": images, "text": sentences_pad, "label": answer, "masks": masks}

