import argparse


def arg_parser():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--predict_txt', default="/home/joo/name_data/data/test_label_2.csv",type=str)
    parser.add_argument('--raw_txt', default="/home/joo/name_data/data/all_data.csv",type=str)                    
    parser.add_argument('--lr', default=2e-5,type=float)
    parser.add_argument('--eps', default=1e-5,type=float)
    parser.add_argument('--epochs', default=5,type=int) 
    parser.add_argument('--batch', default=32,type=int) 
    parser.add_argument('--num_labels', default=18,type=int) 
    
    return parser.parse_args()
