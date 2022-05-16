import argparse

class options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # experiment specifics

        # Training Options
        self.parser.add_argument('--epochs', type=int, nargs='?', default=51, help='total number of training episodes')
        self.parser.add_argument('--k', type=int, nargs='?', default=10, help='total number of answers')

        self.parser.add_argument('--vk', type=int, nargs='?', default=13, help='total number of visual features extracted')

        self.parser.add_argument('--use_bert', type=bool, nargs='?', default=True, help='Whether to use bert or not')

        self.parser.add_argument('--show_every', type=int, nargs='?', default=1000, help='How often to show images')
        self.parser.add_argument('--print_every', type=int, nargs='?', default=100, help='How often to print scores')

        self.parser.add_argument('--batch', type=int, nargs='?', default=1, help='batch size to be used')
        self.parser.add_argument('--seed', type=int, nargs='?', default=66, help='seeds')

        self.parser.add_argument('--num_workers', type=int, nargs='?', default=16, help='number of cpu cores to use')

        self.parser.add_argument('--save_model', type=bool, nargs='?', default=True, help='Whether to save model or not')
        self.parser.add_argument('--save_model_every', type=int, nargs='?', default=50,
                                 help='how often to save the model')

        self.parser.add_argument('--lr', type=float, nargs='?', default=0.001, help='learning rate')
        self.parser.add_argument('--beta1', type=float, nargs='?', default=0.5, help='values for beta1')
        self.parser.add_argument('--beta2', type=float, nargs='?', default=0.999, help='values for beta2')

        self.parser.add_argument('--lr_decay', type=float, nargs='?', default=0.5, help='decay rate for lr')

        # transformation
        self.parser.add_argument('--size', type=int, nargs='?', default=224, help='size of the transformed image')
        self.parser.add_argument('--h_flip', type=float, nargs='?', default=0.5, help='How often to flip images')


        # questions and answer location
        self.parser.add_argument('--train_dataset', type=str, nargs='?', default='data/train_1000_validans.csv', help='name of dataset used for training')
        self.parser.add_argument('--valid_dataset', type=str, nargs='?', default='data/valid_1000_validans.csv', help='name of dataset used for validation')
        self.parser.add_argument('--test_dataset', type=str, nargs='?', default='test.csv', help='name of dataset used for testing')

        # data name
        self.parser.add_argument('--dataset_name', type=str, nargs='?', default='real', help='name of dataset used, abstract OR real')


        # questions and answer location
        self.parser.add_argument('--train_image_dataset', type=str, nargs='?', default='train.csv',
                                 help='name of dir where training images used for training are located')
        self.parser.add_argument('--valid_image_dataset', type=str, nargs='?', default='valid.csv',
                                 help='name of dir where training images used for validation are located')
        self.parser.add_argument('--test_image_dataset', type=str, nargs='?', default='test.csv',
                                 help='name of dir where training images used for testing are located')

        # model configs
        self.parser.add_argument('--glove', type=bool, nargs='?', default=False, help='Whether to use glove embedding or not')
        self.parser.add_argument('--RNN_name', type=str, nargs='?', default='LSTM', help='Whether to use LSTM or not')
        self.parser.add_argument('--dropout', type=float, nargs='?', default=0.50, help='values for dropout')
        self.parser.add_argument('--embedding_dimension', type=int, nargs='?', default=300, help='embedding dimension')
        self.parser.add_argument('--hidden_dimension', type=int, nargs='?', default=512, help='hidden dimension RNN')
        self.parser.add_argument('--num_layers', type=int, nargs='?', default=2, help='number of hidden dimensions')
        self.parser.add_argument('--bi_dir', type=bool, nargs='?', default=False, help='whether to use bi-directional RNN or not')

        self.parser.add_argument('--vision_model_name', type=str, nargs='?', default='vgg19', help='name of pre-trained cnn model for feature extraction')
        self.parser.add_argument('--finetune_vision', type=bool, nargs='?', default=False,
                                 help='whether to finetune pre-trained vision models or not')
        self.parser.add_argument('--vocab_name', type=str, nargs='?', default='basic',
                                 help='type of vocab to use')
        self.parser.add_argument('--vision_type', type=str, nargs='?', default='basic',
                                 help='type of vision model to use')
        self.parser.add_argument('--combine', type=str, nargs='?', default='basic', help='how to combine vision and NLP features')
        self.parser.add_argument('--binary', type=bool, nargs='?', default=False, help='whether the model is binary classification or not')
        self.parser.add_argument('--normed', type=bool, nargs='?', default=True, help='whether to use L2 norm of vision features')

    def parse(self):
        self.initialize()
        self.opt = self.parser.parse_args()

        return self.opt