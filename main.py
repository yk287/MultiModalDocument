

model_name = 'microsoft/layoutlm-base-uncased'

import pandas as pd

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

from datasets import Dataset

# read dataframe as HuggingFace Datasets object
train_dataset = Dataset.from_pandas(train_data)
valid_dataset = Dataset.from_pandas(valid_data)
test_dataset = Dataset.from_pandas(test_data)

from transformers import LayoutLMv2Processor

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

from PIL import Image
from datasets import Features, Sequence, ClassLabel, Value, Array2D, Array3D
import re

cutoff = 128

# we need to define custom features
features = Features({
    'image': Array3D(dtype="int64", shape=(3, 224, 224)),
    'input_ids': Sequence(feature=Value(dtype='int64')),
    'attention_mask': Sequence(Value(dtype='int64')),
    'token_type_ids': Sequence(Value(dtype='int64')),
    'bbox': Array2D(dtype="int64", shape=(512, 4)),
    'labels': ClassLabel(num_classes=len(label2id), names=list(label2id.keys())),
})

#processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", revision="no_ocr", num_labels=len(label2id))
processor = LayoutLMv2Processor.from_pretrained(model_name,  num_labels=len(label2id))

def preprocess_data(examples):
    # take a batch of images
    images = [Image.open(path).convert("RGB") for path in examples['image_path']]

    # get the ocr'ed text

    box_lists = []
    words_lists = []

    for file_name in examples['image_path']:
        bbox = load_bbox(file_name)
        words, boxes = load_ocr_texts(file_name, bbox)

        box_lists.append(boxes)
        words_lists.append(words)

    encoded_inputs = processor(images, words_lists, boxes=box_lists, padding="max_length", truncation=True)

    # add labels
    encoded_inputs["labels"] = [label2id[label] for label in examples["label"]]

    return encoded_inputs


def encode_dataset(dataset, features, batch_size, batched):

    encoded_dataset = dataset.map(preprocess_data,
                                              remove_columns=dataset.column_names,
                                              features=features,
                                              batched=batched, batch_size=batch_size)

    return encoded_dataset.set_format(type="torch", device=device)

encoded_train_dataset = train_dataset.map(preprocess_data, remove_columns=train_dataset.column_names, features=features,
                              batched=True, batch_size=batch_size)
encoded_valid_dataset = valid_dataset.map(preprocess_data, remove_columns=valid_dataset.column_names, features=features,
                              batched=True, batch_size=batch_size)

encoded_train_dataset.set_format(type="torch", device="cuda")
encoded_valid_dataset.set_format(type="torch", device="cuda")


#encoded_train_dataset = train_dataset.map(preprocess_data, remove_columns=train_dataset.column_names, features=features,
                              #batched=True, batch_size=32)

#encoded_train_dataset = encode_dataset(train_dataset, features, 32, True)
#encoded_valid_dataset = encode_dataset(valid_dataset, features, 32, True)
#encoded_test_dataset = encode_dataset(test_dataset, features, 32, True)

import torch

train_dataloader = torch.utils.data.DataLoader(encoded_train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = torch.utils.data.DataLoader(encoded_valid_dataset, batch_size=batch_size, shuffle=True)
#test_dataloader = torch.utils.data.DataLoader(encoded_test_dataset, batch_size=32)


from transformers import LayoutLMv2ForSequenceClassification

#model = LayoutLMv2ForSequenceClassification.from_pretrained("microsoft/layoutlmv2-base-uncased", revision="no_ocr", num_labels=len(label2id))
model = LayoutLMv2ForSequenceClassification.from_pretrained("microsoft/layoutlmv2-base-uncased", revision="no_ocr", num_labels=len(label2id))
model = model.to(device)

from transformers import AdamW
from tqdm.notebook import tqdm

optimizer = AdamW(model.parameters(), lr=5e-5)

global_step = 0
num_train_epochs = 25

t_total = len(train_dataloader) * num_train_epochs  # total number of training steps

# put the model in training mode

for epoch in range(num_train_epochs):
    print("Epoch:", epoch)
    running_loss = 0.0
    correct = 0

    model.train()
    for batch in tqdm(train_dataloader):
        '''
        print("nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn")
        for k, v in batch.items():

            print(k, v)
        print("nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn")
        '''
        # forward pass
        outputs = model(**batch)
        loss = outputs.loss
        running_loss += loss.item()
        predictions = outputs.logits.argmax(-1)
        correct += (predictions == batch['labels']).float().sum()
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
    for v_batch in tqdm(valid_dataloader):

        # forward pass
        outputs = model(**v_batch)
        loss = outputs.loss
        running_loss += loss.item()
        predictions = outputs.logits.argmax(-1)

        correct += (predictions == v_batch['labels']).float().sum()

    print("Loss:", running_loss / v_batch["input_ids"].shape[0])
    accuracy = 100 * correct / len(valid_data)
    print("valid accuracy:", accuracy.item())





