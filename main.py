from train import train_emotion_model_with_k_fold, test_report_kfold
import argparse

    
def main(args):
    print(f"saved files used {args.use_saved_files}, augementation included {args.include_augmentation}, test only {args.test_only}")
    if not args.test_only:      
        train_emotion_model_with_k_fold(use_saved_files=args.use_saved_files, include_augmentation=args.include_augmentation)      
    
    test_report_kfold()   
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and test emotion recognition model.')
    parser.add_argument('--use_saved_files', action='store_true', help='Use saved files for training and testing')
    parser.add_argument('--include_augmentation', action='store_true', help='Include data augmentation during training')
    parser.add_argument('--test_only', action='store_true', help='Run only the test phase')
    

    args = parser.parse_args()
    
    main(args)
    



    

    