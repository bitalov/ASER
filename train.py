# Numerical and Data Handling
import numpy as np
import pandas as pd
import os

# Audio Processing and Visualization
import librosa
import librosa.display

# Plotting
import matplotlib.pyplot as plt

import seaborn as sns
   


from utils import *
from model import  EmotionRecognitionModel, train_epoch, validate_epoch
from transformer import EmotionRecognitionModelTransfomer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report, f1_score



# Hyperparameters for the model 
kernel_size = 10
padding = 4



def test_model(model, X_test, Y_test, loss_fn, batch_size, device):
     
    test_dataset = TensorDataset(torch.tensor(X_test).float(), torch.tensor(Y_test).long())
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    model.eval()
    total_loss = 0
    correct_preds = 0
    total_samples = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for X_batch, Y_batch in loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            outputs = model(X_batch)
            loss = loss_fn(outputs, Y_batch)

            # Accumulate loss and accuracy
            total_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct_preds += (predicted == Y_batch).sum().item()
            total_samples += Y_batch.size(0)

            # Store predictions and labels for further analysis if needed
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(Y_batch.cpu().numpy())

    avg_loss = total_loss / total_samples
    accuracy = (correct_preds / total_samples) * 100

    return avg_loss, accuracy, all_predictions, all_labels


def test_report_kfold(k=5):
    test_accs = []
    test_losses = []
    all_predictions = []
    all_labels = []
    
    for i in range(k):
       fold_num = i + 1
       _, X_test, _, Y_test = load_datasets(f"dataset_fold{fold_num}")  # Assuming load_datasets is defined elsewhere
       device = 'cuda' if torch.cuda.is_available() else 'cpu'
       BATCH_SIZE = 32
       num_emotions = np.unique(Y_test).shape[0]
       input_height = 128  # Height of your spectrograms
       model = EmotionRecognitionModel(num_emotions, input_height, kernel_size=kernel_size, padding=padding).to(device)  
       model.load_state_dict(torch.load(f'best_model{fold_num}.pth'))
              
       loss_fn = nn.CrossEntropyLoss()
       test_loss, test_acc, predictions, labels = test_model(model=model, X_test=X_test, Y_test=Y_test, 
                                                              loss_fn=loss_fn, batch_size=BATCH_SIZE, device=device) 
       print(f"fold {fold_num}, test accuracy {test_acc}%, test loss {test_loss}")      
       
       test_accs.append(test_acc)
       test_losses.append(test_loss)
       all_predictions.extend(predictions)
       all_labels.extend(labels)  

    # Calculate the average confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    cm_normalized = cm // len(test_accs)
    print(cm_normalized)

    # Define emotion labels
    emotion_labels = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Angry', 'Fear']

    # Plot the averaged confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='d', cmap='Blues', xticklabels=emotion_labels, yticklabels=emotion_labels)
    plt.title('Averaged Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save the confusion matrix as an image file (e.g., PNG)
    plt.savefig('averaged_confusion_matrix.png')  # Saves the figure as 'normalized_confusion_matrix.png'
    plt.close()  # Close the figure to free up memory

    # Calculate averages
    avg_test_acc = np.mean(test_accs)
    avg_test_loss = np.mean(test_losses)

    print("\nAverage Results:")
    print(f"Average Test Accuracy: {avg_test_acc:.2f}%")
    print(f"Average Test Loss: {avg_test_loss:.3f}")

    # Compute overall confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Overall Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')  
    plt.savefig('confusion_matrix.png') 
    plt.close() 

    # Print overall classification report
    emotion_labels = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Angry', 'Fear']
    print("\nOverall Classification Report:")
    print(classification_report(all_labels, all_predictions, target_names=emotion_labels))

    # Calculate overall F1 scores
    micro_f1 = f1_score(all_labels, all_predictions, average='micro')
    macro_f1 = f1_score(all_labels, all_predictions, average='macro')
    weighted_f1 = f1_score(all_labels, all_predictions, average='weighted')

    print("\nOverall F1 Scores:")
    print(f"Micro F1: {micro_f1:.3f}")
    print(f"Macro F1: {macro_f1:.3f}")
    print(f"Weighted F1: {weighted_f1:.3f}")


def train_model(save_id, model, X_train, Y_train, X_val, Y_val, device, # Hyperparameters
    EPOCHS = 100,
    BATCH_SIZE = 32,
    LEARNING_RATE = 1e-4,
    WEIGHT_DECAY = 1e-5,
    EARLY_STOPPING_PATIENCE = 20
    ):
   
    print(f'Selected device is {device}')


    # Initialize model
    num_emotions = np.unique(Y_train).shape[0]

    #model = EmotionRecognitionModel(num_emotions,input_height=128, kernel_size=kernel_size, padding=padding).to(device)
    #model = EmotionRecognitionModelTransfomer(num_emotions,input_height=128).to(device)
    
    print('Number of trainable parameters:', sum(p.numel() for p in model.parameters()))

    # Optimizer and loss function
    #optimizer = ADOPT(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()


    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # Prepare datasets and data loaders
    train_dataset = TensorDataset(torch.tensor(X_train).float(), torch.tensor(Y_train).long())
    val_dataset = TensorDataset(torch.tensor(X_val).float(), torch.tensor(Y_val).long())

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    

    # Training loop
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # Early stopping variables
    best_val_loss = float('inf')
    best_val_acc = 0
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_acc = validate_epoch(model, val_loader, loss_fn, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        # Learning rate scheduling
        scheduler.step(val_loss)

        print(f'Epoch {epoch}/{EPOCHS}: '
            f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%,'
            f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"new best val_loss {val_loss}")
            epochs_no_improve = 0
            # Save the best model
            torch.save(model.state_dict(), f'best_model{save_id}.pth')
       
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered!")
                break

def train_emotion_model_with_k_fold(use_saved_files=False, include_augmentation=True):
    """
    Trains a model using k-fold cross-validation.
    Parameters:
    use_saved_files (bool): If True, load pre-saved datasets for each fold. If False, load and preprocess the KSU emotions dataset.
    include_augmentation (bool): If True, include data augmentation in the training data. 
    Returns:
    None
    The function performs the following steps:
    1. Initializes the number of folds for cross-validation.
    2. Loads the datasets for each fold either from saved files or by preprocessing the KSU emotions dataset.
    3. If data augmentation is not included, uses only the first third of the training data.
    4. Trains the model for each fold using the specified parameters.
    5. Prints the test accuracies for each fold and the average accuracy across all folds.
    """
    
    num_folds = 5

    if not use_saved_files:   
        load_and_preprocess_ksu_emotions(num_folds=num_folds,include_augmentation=include_augmentation)        
     
        
        
    
    for f in range(num_folds):
        fold_num = f + 1        
        X_train, X_test, Y_train, Y_test = load_datasets(filename=f"dataset_fold{fold_num}")


        # this is only true if the saved files included 2 augmented versions of the original data, comment out if not (note to self: remove later maybe)
        if not include_augmentation and use_saved_files:
            X_train = X_train[:int(X_train.shape[0] / 3)]
            Y_train = Y_train[:int(Y_train.shape[0] / 3)]        

        print(f"Fold {fold_num}, {X_train.shape, X_test.shape}:")        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        BATCH_SIZE = 32
        input_height = X_train.shape[2] # Height of your spectrograms
        print(f"input_height {input_height}")
        num_emotions = np.unique(Y_test).shape[0]

        model = EmotionRecognitionModel(num_emotions, input_height, kernel_size=kernel_size, padding=padding).to(device)


        train_model(model=model, save_id=fold_num, X_train=X_train, Y_train=Y_train, X_val=X_test, Y_val=Y_test, 
              device=device, 
              BATCH_SIZE=32,
              EPOCHS = 100,              
              LEARNING_RATE = 1e-4,
              WEIGHT_DECAY = 1e-5,
              EARLY_STOPPING_PATIENCE = 20)
        


        
        
        model.load_state_dict(torch.load(f'best_model{fold_num}.pth'))          
       
        loss_fn = nn.CrossEntropyLoss()
        test_loss, test_acc, predictions, labels = test_model(model=model, X_test=X_test, Y_test=Y_test, 
                                                              loss_fn=loss_fn, batch_size=BATCH_SIZE, device=device)
        print(f"fold {fold_num}, test accuracy {test_acc}, test loss {test_loss}")  


    
   

