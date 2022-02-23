# train.py

from utils import *
from model import *
from config import Config
import numpy as np
import sys
import torch.optim as optim
from torch import nn
import torch

if __name__=='__main__':
    config = Config()
    train_file = '../data/ag_news.train'
    if len(sys.argv) > 2:
        train_file = sys.argv[1]
    test_file = '../data/ag_news.test'
    if len(sys.argv) > 3:
        test_file = sys.argv[2]
    w2v_file = './cc.te.300.vec'
    if len(sys.argv) > 4:
      w2v_file=sys.argv[3]

    
    dataset = Dataset(config)
    dataset.load_data(w2v_file, train_file, test_file)
    
    # Create Model with specified optimizer and loss function
    ##############################################################
    model = fastText(config, len(dataset.vocab), dataset.word_embeddings)
    if torch.cuda.is_available():
        model.cuda()
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=config.lr)
    NLLLoss = nn.NLLLoss()
    model.add_optimizer(optimizer)
    model.add_loss_op(NLLLoss)
    ##############################################################
    
    train_losses = []
    val_accuracies = []
    
    for i in range(config.max_epochs):
        print ("Epoch: {}".format(i))
        train_loss,val_accuracy = model.run_epoch(dataset.train_iterator, dataset.val_iterator, i)
        train_losses.append(train_loss)
        val_accuracies.append(val_accuracy)

    train_acc, train_prf = evaluate_model(model, dataset.train_iterator)
    val_acc, val_prf = evaluate_model(model, dataset.val_iterator)
    test_acc, test_prf = evaluate_model(model, dataset.test_iterator)

    print ('Final Training Accuracy: {:.4f}'.format(train_acc))
    print ('Final Training prf: ', train_prf)
    print ('Final Validation Accuracy: {:.4f}'.format(val_acc))
    print ('Final Validation prf: ', val_prf)
    print ('Final Test Accuracy: {:.4f}'.format(test_acc))
    print ('Final Test prf: ', test_prf)
