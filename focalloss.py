
import torch
import torch.nn as nn
import numpy as np

def reweight(cls_num_dict, beta=0.9999):
    '''
    Implement reweighting by effective numbers
    :param cls_num_list: a list containing # of samples of each class
    :param beta: hyper-parameter for reweighting, see paper for more details
    :return:
    '''
    per_cls_weights = None
    #############################################################################
    # TODO: reweight each class by effective numbers                            #
    #############################################################################

    # numerator is given as 1 - beta in the paper
    numerator = 1 - beta
    # denominator is given as 1 - beta ^ n_y in the paper. I think n_y is the number of
    # samples in the class y

    # turn dict values into list
    cls_num_list = list(cls_num_dict.values())

    denom = 1.0 - (beta ** (np.asarray(cls_num_list)))

    # calculate the class balance multiplier
    per_cls_weights = numerator / denom

    # conver the numpy array to torch tensor
    per_cls_weights = torch.from_numpy(per_cls_weights)

    # normalize the weights
    per_cls_weights = per_cls_weights / per_cls_weights.sum()

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    return per_cls_weights * len(per_cls_weights) # to make the weight normalize to C


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        '''
        Implement forward of focal loss
        :param input: input predictions
        :param target: labels
        :return: tensor of focal loss in scalar
        '''
        loss = None
        #############################################################################
        # TODO: Implement forward pass of the focal loss                            #
        #############################################################################

        # Based on https://arxiv.org/pdf/1901.05555.pdf
        # initialize the sigmoid function to be used to convert z_i into p_i
        # sigmoid = nn.Sigmoid() # apparently sigmoid shouldn't be used.

        # softmax
        softmax = nn.Softmax()

        # in the paper p^t_i is defined as sigmoid(z^t_i), so neg_one * input creates z^t_i and feeding that through
        # sigmoid creates p_i
        probs = softmax(input)

        # get the predicted probability of the target class
        p_i = probs.gather(1, target.reshape(-1, 1))

        # reshape p_i so that the arithmatic works out
        p_i = p_i.reshape(-1)

        loss = - self.weight[target] * (1 - p_i) ** self.gamma * p_i.log()

        # multiply by the sum by the weights and then sum them to create a scalar loss.
        loss = loss.mean()

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return loss