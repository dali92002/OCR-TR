import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import cv2
import random
from tqdm import tqdm
import os
from Config import Configs

cfg = Configs().parse()

n_train = cfg.train_words
n_valid = cfg.valid_words
n_test = cfg.test_words
min_size = 3 # minimum number of characters per word
max_size = 10 # maximum number of characters per word

# Download your data
train_dataset = datasets.EMNIST(root=".", split="bymerge", download=True, train=True, 
                transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.RandomAffine(degrees=20, translate=(0.2, 0.1), scale=(0.7, 1.15)),
                       lambda img: transforms.functional.rotate(img, -90),
                       lambda img: transforms.functional.hflip(img)
                   ]))
test_dataset = datasets.EMNIST(root=".", split="bymerge", download=True, train=False, 
                transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.RandomAffine(degrees=20, translate=(0.2, 0.1), scale=(0.7, 1.15)),
                       lambda img: transforms.functional.rotate(img, -90),
                       lambda img: transforms.functional.hflip(img)
                   ]))
 
def create_random_word(dataset,l, min_l = 3, max_l=10):
    """
    A method to create a word from the isolated characters, randomely.
    
    Parameters:
            dataset: the dataset of isolates characters/digits
            l: lenght of the dataset
            min_l: minimum length of a word (number of characters)
            max_l: maximum length of a word (number of characters) 
    """
    label = ''
    for i in range(random.randint(min_l,max_l)):
        chosen = random.randint(0,l-1)
        char = dataset[chosen][0][0]
        if i > 0:
            img = torch.cat((img,char),1)
        else:
            img = char
        label += str(dataset[chosen][1]) + ' '

    return img, label[:-1]



# Create our training, validation and testing datasets
if __name__ == "__main__":

    if not os.path.exists('./data/words'):
        os.makedirs('./data/words')

    l_train = len(train_dataset)
    l_test = len(test_dataset)

    f = open('./data/train.txt','w')
    for i in tqdm(range(n_train)):
        img, label = create_random_word(train_dataset, l_train, min_size, max_size)
        plt.imsave('./data/words/train_'+str(i)+'.png',img, cmap='gray')
        f.write('train_'+str(i)+' '+label+'\n')
    f.close()

    f = open('./data/valid.txt','w')
    for i in tqdm(range(n_valid)):
        img, label = create_random_word(train_dataset, l_train, min_size, max_size)
        plt.imsave('./data/words/valid_'+str(i)+'.png',img, cmap='gray')
        f.write('valid_'+str(i)+' '+label+'\n')
    f.close()

    f = open('./data/test.txt','w')
    for i in tqdm(range(n_test)):
        img, label = create_random_word(test_dataset, l_test, min_size, max_size)
        plt.imsave('./data/words/test_'+str(i)+'.png',img, cmap='gray')
        f.write('test_'+str(i)+' '+label+'\n')
    f.close()
