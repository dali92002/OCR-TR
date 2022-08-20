import argparse


class Configs():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        
        # Data Creation
        self.parser.add_argument('--train_words', type=int, help='The number of training words to be generated')
        self.parser.add_argument('--valid_words', type=int, help='The number of validation words to be generated')
        self.parser.add_argument('--test_words', type=int, help='The number of testing words to be generated')
        
        # Train
        self.parser.add_argument('--data_path', type=str, help='specify your data path, better ending with the "/" ')
        self.parser.add_argument('--img_height', type=int, default=32, help= "the size of the input images (height)")
        self.parser.add_argument('--img_width', type=int, default=256, help= "the size of the input images (width)")
        self.parser.add_argument('--vit_patch_size', type=int, default=16 , help=" better be a multiple of 2 like 8, 16 etc ..")
        self.parser.add_argument('--max_text_len', type=int, default=12, help='the maximum word lenght, in our case 10 + 2, the added 2 are start token, end token')
        self.parser.add_argument('--train_type', type=str , choices=['normal','stn','htr_Augm'], help="specify the desired transformation to be applied during training")
        self.parser.add_argument('--batch_size', type=int, default=8)

        # Test 
        self.parser.add_argument('--test_model', type=str, help='specify the path of the model to be loaded for testing')

    def parse(self):
        return self.parser.parse_args()





