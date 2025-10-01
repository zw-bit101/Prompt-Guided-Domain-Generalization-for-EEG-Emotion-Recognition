from cross_validation import CrossValidation
from prepare_data_DEAP import *
import argparse
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ######## Data ########

    parser.add_argument('--algorithm', type=str, default='DRM')

    parser.add_argument('--domain_num', type=int, default=14)
    parser.add_argument('--subjects', type=int, default=15)
    parser.add_argument('--num_class', type=int, default=4, choices=[2, 3, 4])

    parser.add_argument('--trial-duration', type=int, default=60, help='trial duration in seconds')
    parser.add_argument('--init-param', type=float, default=0.03)
    parser.add_argument('--unif-init', action='store_true', default=True)
    parser.add_argument('--meta-init', type=str, default=None)
    parser.add_argument('--zero-init', action='store_true', default=False)
    parser.add_argument('--prompt-size', type=int, default=4)

    parser.add_argument('--dataset', type=str, default='seediv', choices=['seed', 'seediv'])
    parser.add_argument('--use-kld', action='store_true', default=True)
    parser.add_argument('--norm-experts', action='store_true', default=True)
    parser.add_argument('--use-softmax', action='store_true', default=True)
    parser.add_argument('--use-tanh', action='store_true', default=True)
    parser.add_argument('--tuning', type=str, default='classifier', choices=['no', 'classifier', 'all'])
    parser.add_argument('--tuning_prompt', type=str, default='yes', choices=['no', 'yes'])

    parser.add_argument('--use-rand', action='store_true', default=False)
    parser.add_argument('--use-meta', action='store_true', default=False)
    parser.add_argument('--use-zero', action='store_true', default=False)
    parser.add_argument('--use-unif', action='store_true', default=True)
    parser.add_argument('--bands', type=str, default='allband',
                        choices=['band5', 'band1', 'band2', 'band3', 'band4', 'allband'])
    parser.add_argument('--region', type=str, default='allregion',
                        choices=['frontal6', 'frontal10', 'temporal6', 'temporal9', 'allregion'])
    ######## Training Process ########
    parser.add_argument('--random-seed', type=int, default=2024)
    parser.add_argument('--max-epoch', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=14 * 167)
    parser.add_argument('--learning-rate', type=float, default=3e-3)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--aug-pt', type=int, default=None)
    parser.add_argument('--save-path', default='./save/')
    parser.add_argument('--load-path', default='./save/max-acc.pth')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--save-model', type=bool, default=True)
    ######## Model Parameters ########
    parser.add_argument('--model', type=str, default='PGDG')
    parser.add_argument('--sub_dependent_or_not', type=str, default='cross_sub_train_model',
                        choices=['cross_sub_train_experts', 'not_cross_sub', 'test', 'cross_sub_attention_generalization', 'cross_sub_train_model'])

    # Reproduce the result using the saved model ###### args.domain_num +
    parser.add_argument('--reproduce', action='store_true')
    args = parser.parse_args()

    sub_to_run = np.arange(1, args.subjects)

    cv = CrossValidation(args)
    seed_all(args.random_seed)

    for sub in range(1, 2):
        cv.cross_sub_train_model(subject=sub, domain_num=args.domain_num, fold=5, reproduce=args.reproduce)

    for sub in range(1, 2):
        cv.domain_generalization_train_experts(
            subject=sub, domain_num=args.domain_num, fold=5, reproduce=args.reproduce)

    for sub in range(1, 2):
        cv.attention_based_generalization(subject=sub, domain_num=args.domain_num, fold=5, reproduce=args.reproduce)
