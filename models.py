from os.path import exists
import os
import torchvision.models as models

# for neural net
import torch
import torch.nn as nn
# for norm
from torch import linalg

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class QuestionAnswer(nn.Module):

    def __init__(self,
                 bertModel,
                 opts,
                 ):

        super(QuestionAnswer, self).__init__()

        self.opts = opts

        self.loss_function = nn.CrossEntropyLoss()

        # output of bert has 768 dimensions, which is unfortunately smaller than 1024 used in the paper
        combine_dim = 768

        # initialize the vision part of the model
        self.vision_model = VisionNet(model_name=self.opts.vision_model_name).to(device)

        # linear head that deals with features extracted using vgg19
        self.v_linear = nn.Sequential(
            nn.Linear(4096, combine_dim) # these dimensions are pretty much set in stone.
        )

        # if the model's supposed to use bert, use pre-trained bert encoder
        if self.opts.use_bert:
            self.word_embedding = bertModel

        if self.opts.use_bert:
            self.q_linear = nn.Sequential(
                nn.Tanh(),
                nn.Dropout(self.opts.dropout),
                nn.Linear(combine_dim, combine_dim)
            )

            # for the case where we are dealing with binary
            # the part that does binary classification
            self.decoder = nn.Sequential(
                nn.Tanh(),
                nn.Dropout(self.opts.dropout),
                nn.Linear(combine_dim, combine_dim),
                nn.Tanh(),
                nn.Dropout(self.opts.dropout),
                nn.Linear(combine_dim, opts.k),
                nn.LogSoftmax(-1)
            )

    def vision_forward(self, image):
        # separate function to deal with more complicated vision modules
        # such as attention
        vision_features = None
        if self.opts.vision_type == 'basic':
            # get the features from the images
            vision_features = self.vision_model(image)

        if self.opts.normed:
            # features are l2 normed
            # get the norm. Everything up to this part does not require training so detach to make sure gradient
            # does not propagate
            norm = linalg.vector_norm(vision_features, ord=2, dim=1, keepdim=True)
            # print(norm.shape)
            # normalize the features
            vision_features = vision_features / norm

        vision_features = self.v_linear(vision_features)

        return vision_features

    def question_forward(self, data):
        # function that deals with the questions

        # get the questions and answers
        #q = data['text'].to(device)

        if self.opts.use_bert:

            # feed the questions into word embedding

            word_embedding = self.word_embedding(data['text'].to(device), data['masks'].to(device))
            last_hidden_states = word_embedding.last_hidden_state
            question_features = last_hidden_states[:, 0, :]
        else:

            source_lengths = torch.sum(q != self.pad_id, axis=0).cpu()
            word_embedding = self.word_embedding(q)

            word_embedding = nn.Tanh()(word_embedding)

            # pack the sequence as we are told
            packed_embedded = nn.utils.rnn.pack_padded_sequence(word_embedding, source_lengths)

            # get the output and hidden dimensions
            encoder_output, encoder_hidden = self.rnn(packed_embedded)

            # get the encoder_mask which corresponds to where pad_token was used
            encoder_mask = torch.where(q == self.pad_id, 1, 0).cpu()
            # repack the output
            encoder_output, _ = nn.utils.rnn.pad_packed_sequence(encoder_output)

            question_features = None
            # deal with the 2 layer hidden layer by adding them

            # in the paper, the authors combine the hidden state and the cell state
            question_features = torch.cat((encoder_hidden[0], encoder_hidden[1]), dim=2)

            # for cases where number of layers is greater than 1, we have num_layers * 1 * BiDirection X Batch_Size X 1024
            question_features = question_features.transpose(0, 1) # Batch_Size X num_layers * 1 * BiDirection X 1024
            question_features = question_features.reshape(question_features.shape[0], -1) # Batch_Size X num_layers * 1 * BiDirection * 1024

            if len(question_features.shape) == 3:
                question_features = question_features.squeeze(0)

        # return features extracted from the questions and the mask
        question_features = self.q_linear(question_features)

        return question_features, data['masks']

    def combine_features(self, v_features, q_features):
        # a function that combines features from the images and the features from the questions

        combined = None

        if self.opts.combine == 'basic':

            combined = v_features * q_features
        #else:
            # trying out bilinear
            #combined = torch.bmm(v_features.unsqueeze(2), q_features.unsqueeze(1))
            #combined = combined.reshape(combined.shape[0], -1)

        return combined

    def predict(self, data):

        # get the images
        image = data['image']

        # get the features from images.
        vision_features = self.vision_forward(image.to(device))

        q_features, q_mask = self.question_forward(data)

        # check to make sure that the size of the features match up
        assert vision_features.shape == q_features.shape

        combined = self.combine_features(vision_features, q_features)

        pred_prob = None
        # get the probs for binary classes
        pred_prob = self.decoder(combined)

        return pred_prob

    def forward(self, data):

        a = data['label']

        pred_prob = self.predict(data)

        loss = None
        correct = None

        # need to deal with 'yes' and 'no' as those have their own weird indices
        # within the embedding layer
        # Also for binary cases, the answers are surrounded by BOS and EOS tokens
        # so need to get rid of that as well
        # get the answer
        if self.opts.use_bert:
            ground_truth = a
        else:
            ground_truth = a
        # if the ground_truth value corresponds to the index of 'yes' then set it to 1 else 0
        # change prediction to one hot vector

        #print(pred_prob.shape)
        #print(ground_truth.shape)

        loss = self.loss_function(pred_prob, ground_truth.to(device))

        best_pred = torch.argmax(pred_prob.detach(), dim=1)

        # count correct preds
        correct = torch.where(ground_truth.to(device) == best_pred, 1, 0)

        return loss, correct

class VisionNet(nn.Module):
    def __init__(self, model_name='vgg16'):
        '''
        A model that is used exclusively to extract features out of images. In the paper the features are of size 4096
        Paper also mentions using VGG19
        :param model_name:
        '''
        super(VisionNet, self).__init__()
        # load a CNN vision model that will extract features out of images
        # in the paper VGG is used
        # https://pytorch.org/vision/stable/models.html
        if model_name == 'vgg16':
            self.model = models.vgg16(pretrained=True)
            # the model used in the paper uses ouputs from the layer that outputs 4096 features
            self.model.classifier = torch.nn.Sequential(*list(self.model.classifier.children())[:-3])
        elif model_name == 'vgg19':
            self.model = models.vgg19(pretrained=True)
            # the model used in the paper uses ouputs from the layer that outputs 4096 features
            self.model.classifier = torch.nn.Sequential(*list(self.model.classifier.children())[:-1])

        finetune_vision = False

        if not finetune_vision:
            # freeze the weights so that the model does not get trained
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x):

        finetune_vision = False

        with torch.no_grad():
            # extract out the 4096 features using pretrained model
            features = self.model.forward(x)

        if not finetune_vision:
            features = features.detach()

        return features

class FaNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FaNetwork, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.tanh = nn.Tanh()
        self.sigm = nn.Sigmoid()

        self.ytilde_w = nn.Linear(self.input_dim, output_dim)
        self.g_w = nn.Linear(self.input_dim, output_dim)

    def forward(self, x):
        '''
        Given an input x, go through EQ. 12, 13 and 14 from
        the paper
        '''

        # EQ 12
        ytilde = self.ytilde_w(x)
        ytilde = self.tanh(ytilde)

        # EQ 13
        g = self.g_w(x)
        g = self.sigm(g)

        assert ytilde.shape == g.shape

        # EQ 14
        y = ytilde * g

        return y











