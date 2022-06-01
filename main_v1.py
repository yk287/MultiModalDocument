

import pandas as pd
import re

# module used to get names of folders or names of images
from imagefolder import get_folders, get_images

# functions that splits files in a list of lists, and also combines a list of lists into a list
from util import train_valid_test_split, combine_lists

# get the names of folders. Returns a list of lists
folder_names = get_folders('data/Tobacco')
# the first one that gets returned is an empty list
folder_names = folder_names[1:]

# a dictionary that holds class name and the corresponding number associated for classification.
label2id = {}

# since the names of the folders represent the names of classes, go through the folder names.
for index, i in enumerate(folder_names):
    class_name = i.split('/')[-1]
    label2id.update({class_name:index})

# create a dictionary that will get used to go from id2label
id2label = {}
for key, value in label2id.items():
   if value in id2label:
       id2label[value].append(key)
   else:
       id2label[value]=[key]

## load the options for testing out different configs
#from options import options
#options = options()
#opts = options.parse()

# get the file names
file_names = get_images('data/Tobacco')


# sometimes the first list in the list is empty. This happens when the folder has subfolders for different class of data.
if len(file_names[0]) == 0:
    file_names = file_names[1:]

# if the list is a list of lists, we need to deal with them differently compared to list of files names.
if isinstance(file_names[0], list):
    # get the list of files which gets pre-split
    train_list, valid_list, test_list = train_valid_test_split(file_names)

    # turn a list of list into a list
    train_list = combine_lists(train_list)
    valid_list = combine_lists(valid_list)
    test_list = combine_lists(test_list)

    train_data = pd.DataFrame(train_list, columns=['image_path'])
    valid_data = pd.DataFrame(valid_list, columns=['image_path'])
    test_data = pd.DataFrame(test_list, columns=['image_path'])

    # populate label field which are the names of folders.
    train_data['label'] = train_data.image_path.str.split('/', expand=True)[2]
    valid_data['label'] = valid_data.image_path.str.split('/', expand=True)[2]
    test_data['label'] = test_data.image_path.str.split('/', expand=True)[2]


####################
# import focalloss #
####################

from focalloss import reweight, FocalLoss

#print(train_data.groupby('label').count())

cls_num_dict = train_data.groupby('label').count().apply(list).to_dict()

cls_num_dict = cls_num_dict['image_path']

reweight_value = reweight(cls_num_dict)

from datasets import Dataset

# read dataframe as HuggingFace Datasets object
train_dataset = Dataset.from_pandas(train_data)
valid_dataset = Dataset.from_pandas(valid_data)
test_dataset = Dataset.from_pandas(test_data)


import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 4

def load_bbox(file_name):
    '''
    A funciton used to load bounding boxes form a corresponding file that has already been extracted
    :param file_name:
    :return:
    '''

    temp_name_ocr = file_name.split('.')[0] + '_ocr' + '.txt' # the ofr file has _ocr at the end of the file name
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

def load_ocr_texts(file_name, bbox):
    '''
    A function that returns the words and ocr of the words, it appears that the words should be correctly separated and
    there should be a bbox for each word.
    :param file_name:
    :param bbox:
    :return:
    '''

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

def encode_example(example, max_seq_length=512, pad_token_box=[0, 0, 0, 0]):

    # take a batch of images
    #images = [Image.open(path).convert("RGB") for path in example['image_path']]

    # get the ocr'ed text
    box_lists = []
    words_lists = []

    # get the words and bbox
    file_name = str(example['image_path'])
    bbox = load_bbox(file_name)
    words, boxes = load_ocr_texts(file_name, bbox)

    #words = words_lists
    normalized_word_boxes = boxes

    assert len(words) == len(normalized_word_boxes)

    #for word in word
    #words = ' '.join([word for word in words_lists if str(word) != 'nan'])

    token_boxes = []
    for word, box in zip(words, normalized_word_boxes):
        word_tokens = tokenizer.tokenize(word)
        token_boxes.extend([box] * len(word_tokens))

    # Truncation of token_boxes
    special_tokens_count = 2
    if len(token_boxes) > max_seq_length - special_tokens_count:
        token_boxes = token_boxes[: (max_seq_length - special_tokens_count)]

    # add bounding boxes of cls + sep tokens
    token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]

    encoding = tokenizer(' '.join(words), padding='max_length', truncation=True)
    # Padding of token_boxes up the bounding boxes to the sequence length.
    input_ids = tokenizer(' '.join(words), truncation=True)["input_ids"]
    padding_length = max_seq_length - len(input_ids)
    token_boxes += [pad_token_box] * padding_length
    #print(encoding.keys())
    #print(encoding.items())
    encoding['bbox'] = token_boxes
    encoding['label'] = label2id[example['label']]
    encoding['image_path'] = str(example['image_path'])

    assert len(encoding['input_ids']) == max_seq_length
    assert len(encoding['attention_mask']) == max_seq_length
    assert len(encoding['token_type_ids']) == max_seq_length
    assert len(encoding['bbox']) == max_seq_length

    return encoding

from datasets import Features, Sequence, ClassLabel, Value, Array2D, Array3D

import datasets
print(datasets.__version__)
cutoff = 128

# we need to define the features ourselves as the bbox of LayoutLM are an extra feature
features = Features({
    'input_ids': Sequence(feature=Value(dtype='int64')),
    'bbox': Array2D(dtype="int64", shape=(512, 4)),
    'attention_mask': Sequence(Value(dtype='int64')),
    'token_type_ids': Sequence(Value(dtype='int64')),
    'label': ClassLabel(names=['refuted', 'entailed']),
    'image_path': Value(dtype='string'),
    #'words': Sequence(feature=Value(dtype='string')),
})

from transformers import LayoutLMTokenizer, LayoutLMForSequenceClassification

tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")

encoded_train_dataset = train_dataset.map(lambda example: encode_example(example), features=features)
encoded_valid_dataset = valid_dataset.map(lambda example: encode_example(example), features=features)

encoded_train_dataset.set_format(type='torch', columns=['input_ids', 'bbox', 'attention_mask', 'token_type_ids', 'label'])
encoded_valid_dataset.set_format(type='torch', columns=['input_ids', 'bbox', 'attention_mask', 'token_type_ids', 'label'])

import torch

train_dataloader = torch.utils.data.DataLoader(encoded_train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = torch.utils.data.DataLoader(encoded_valid_dataset, batch_size=batch_size, shuffle=True)
#test_dataloader = torch.utils.data.DataLoader(encoded_test_dataset, batch_size=32)


from transformers import LayoutLMv2ForSequenceClassification

model = LayoutLMForSequenceClassification.from_pretrained("microsoft/layoutlm-base-uncased", num_labels=len(label2id))
model.to(device)

from transformers import AdamW
from tqdm import tqdm

optimizer = AdamW(model.parameters(), lr=5e-5)

global_step = 0
num_train_epochs = 25

t_total = len(train_dataloader) * num_train_epochs  # total number of training steps

reweight_value = reweight_value.to(device)
FL = FocalLoss(reweight_value)
FL = FL.to(device)



# put the model in training mode

for epoch in range(num_train_epochs):
    print("Epoch:", epoch)
    running_loss = 0.0
    correct = 0

    model.train()
    for batch in tqdm(train_dataloader):

        # forward pass
        outputs = model(input_ids=batch["input_ids"].to(device),
                        bbox=batch["bbox"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                        token_type_ids=batch["token_type_ids"].to(device),
                        labels=batch["label"].to(device),
                        )

        #loss = outputs.loss


        loss = FL(outputs.logits, batch['label'].to(device))

        predictions = outputs.logits.argmax(-1)
        correct += (predictions == batch['label'].to(device)).float().sum()

        running_loss += loss.item()
        # backward pass to get the gradients
        loss.backward()
        # update
        optimizer.step()
        optimizer.zero_grad()
        global_step += 1

    print("Loss:", running_loss / batch["input_ids"].shape[0])
    accuracy = 100 * correct / len(train_data)
    print("Training accuracy:", accuracy.item())

    model.eval()
    correct = 0
    for batch in tqdm(valid_dataloader):

        # forward pass
        outputs = model(input_ids=batch["input_ids"].to(device),
                        bbox=batch["bbox"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                        token_type_ids=batch["token_type_ids"].to(device),
                        labels=batch["label"].to(device),
                        )

        loss = outputs.loss
        running_loss += loss.item()
        predictions = outputs.logits.argmax(-1)

        correct += (predictions == batch['label'].to(device)).float().sum()

    print("Loss:", running_loss / batch["input_ids"].shape[0])
    accuracy = 100 * correct / len(valid_data)
    print("valid accuracy:", accuracy.item())





