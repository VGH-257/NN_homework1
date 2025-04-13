import numpy as np
from mlp import MyModel
from data_loader import load_cifar10
from utils import softmax, cross_entropy_loss, accuracy, sgd
import os
from datetime import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse

# log_dir = "logs"
# os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
# log_path = os.path.join(log_dir, f"train_log_{timestamp}.txt")
# writer = SummaryWriter(log_dir=f"logs/{timestamp}")

def log(msg):
    with open(log_path, 'a') as f:
        f.write(msg + "\n")

def adjust_learning_rate(base_lr, epoch, decay_rate=0.5, decay_every=10):
    return base_lr * (decay_rate ** (epoch // decay_every))

def train(args, model, train_loader, val_loader):

    best_val_acc = 0
    for epoch in range(args.epochs):
        lr = adjust_learning_rate(args.lr, epoch, args.decay_rate, args.decay_every)
        total_loss = 0 
        step=0
        for X_batch, y_batch in tqdm(train_loader, desc="training"):
            X_batch = X_batch.view(X_batch.size(0), -1).numpy()
            y_batch = y_batch.numpy()
            logits = model.forward(X_batch)
            loss, probs = cross_entropy_loss(logits, y_batch, model, args.reg)
            grads = model.backward(probs, y_batch, args.reg)
            total_loss += loss 
            sgd(model.params, grads, lr, args.reg)
            # writer.add_scalar("Loss/train_step", loss, step)
            step+=1
            

        print(f"Epoch {epoch+1}: train_loss = {total_loss/len(train_loader):.4f}")
        # Validation
        val_loss = 0
        val_acc = 0
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.view(X_batch.size(0), -1).numpy()
            y_batch = y_batch.numpy()
            val_scores = model.forward(X_batch)

            _, val_probs = cross_entropy_loss(val_scores, y_batch)
            val_acc += accuracy(val_probs, y_batch)
            val_loss += cross_entropy_loss(val_scores, y_batch)[0]
        val_acc /= len(val_loader)
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}: val_acc = {val_acc:.4f}, val_loss = {val_loss:.4f}")
        log(f"Epoch {epoch+1}: val_acc = {val_acc:.4f}, val_loss = {val_loss:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save(args.output_path)
    #     writer.add_scalar("Loss/train_epoch", total_loss / len(train_loader), epoch)
    #     writer.add_scalar("Loss/val", val_loss, epoch)
    #     writer.add_scalar("Accuracy/val", val_acc, epoch)
    #     writer.add_scalar("lr", lr, epoch)
    # writer.close()
            
def test(model, test_loader):
    total_acc = 0
    total_samples = 0

    for X_batch, y_batch in test_loader:
        X_batch = X_batch.view(X_batch.size(0), -1).numpy()
        y_batch = y_batch.numpy()
        scores = model.forward(X_batch)
        probs = softmax(scores)
            
        acc = accuracy(probs, y_batch)
            
        total_acc += acc * X_batch.shape[0]
        total_samples += X_batch.shape[0]

    avg_acc = total_acc / total_samples
    log(f"Test Accuracy: {avg_acc:.4f}")
    print(f"Test Accuracy: {avg_acc:.4f}")
    return avg_acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a 3-layer neural network on CIFAR-10")

    parser.add_argument('--lr', type=float, default=1e-2, help='initial learning rate')
    parser.add_argument('--reg', type=float, default=1e-3, help='L2 regularization strength')
    parser.add_argument('--epochs', type=int, default=30, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='mini-batch size')
    parser.add_argument('--hidden', type=int, nargs=2, default=[512, 256], help='two hidden layer sizes')
    parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'sigmoid', 'tanh'], help='activation function')
    parser.add_argument('--decay_every', type=int, default=10, help= 'decay learning rate every N epochs')
    parser.add_argument('--decay_rate', type=float, default=0.5, help='decay factor')
    parser.add_argument('--output_path', type=str, default="../ckpt/best_model.npz", help='path of saved model')
    parser.add_argument('--log_path', type=str, default=None)
    
    args = parser.parse_args()

    global log_path
    log_path = args.log_path
    log(f"[INFO] Training with hyperparameters: {vars(args)}")
    train_loader, val_loader, test_loader = load_cifar10(batch_size=args.batch_size)  
    model = MyModel(3072, args.hidden, 10, activation=args.activation)   
    train(args, model, train_loader, val_loader)

    model = MyModel(3072, args.hidden, 10)
    model.load(args.output_path)
    test(model, test_loader)