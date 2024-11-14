import argparse
import sys

def set_params():
    my_parser = argparse.ArgumentParser(description='Train a deep-learning model with the given data')

    # Add the arguments
    my_parser.add_argument('--path',
                           type=str,
                           help='The path to json containining the parameters',
                           default="./best_model_json/trace_params.json")
    my_parser.add_argument('--device',
                           type=str,
                           help='The device',
                           default="cpu")

    
    my_parser.add_argument('--att_lambda',
                           type=float,
                           help='required to assign the contribution of the atention loss')

    my_parser.add_argument('--batch_size',
                           type=int,
                           help='batch size',
                           default=8,
                           required=False)
    my_parser.add_argument('--batch_size_test',
                           type=int,
                           help='batch size test',
                           default=8,
                           required=False)
    my_parser.add_argument('--max_length',
                           type=int,
                           help='batch size',
                           default=512,
                           required=False)
    my_parser.add_argument('--codetopk',
                           type=int,
                           help='how many files in commit code to consider',
                           default=1,
                           required=False)
    my_parser.add_argument('--epochs',
                           type=int,
                           help='how many files in commit code to consider',
                           default=4,
                           required=False)
    my_parser.add_argument('--pooling_method',
                           type=str,
                           help='pooling method',
                           default="max",
                           required=False) # pooling method根本没啥影响，codetopk超过1总是不行
    my_parser.add_argument('--attfirst',
                           action='store_true',
                       help='whether att code snippets are sorted first')
    my_parser.add_argument('--is_augment',
                       type=str,
                       default="code",
                       help='Whether code snippets are sorted first')
    my_parser.add_argument('--model_name',
                       type=str,
                       default="microsoft/codebert-base",
                       help='the model name')


    my_parser.add_argument("--data_dir", type=str, required=False)
    my_parser.add_argument('--num_samples',
metavar='--number_of_samples',
type=int,
default=100,
help='number of samples each instance of the data to pass in lime')
    
    
    # args = my_parser.parse_args()
    
    return my_parser
