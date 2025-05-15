import os
from training.train import train_model
from evaluation.evaluate import evaluate_model

if __name__ == '__main__':
    print("Training model...")
    train_model()

    print("Evaluating model...")
    evaluate_model()
