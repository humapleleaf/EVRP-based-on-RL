import argparse

if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='Reinforcement Learning for solving TSP/VRP')
    parse.add_argument('--seed', default=520, type=int)
    parse.add_argument('--task', default='gvrp', type=str)
    parse.add_argument('--capacity', default=60, type=int, help='fuel capacity, unit: gallon')
    parse.add_argument('--velocity', default=40, type=int, help='unit: miles/h')
    parse.add_argument('--cons_rate', default=0.2, type=float, help='fuel consumption rate, unit: gallon/mile')
    parse.add_argument('--t_limit', default=11, type=int, help='tour duration time limitation, unit: h')
    parse.add_argument('--num_afs', default=3, type=int)
    parse.add_argument('--static_features', default=2, type=int)
    parse.add_argument('--dynamic_features', default=3, type=int)
    parse.add_argument('--train_size', default=10000, type=int)
    parse.add_argument('--valid_size', default=1000, type=int)
    parse.add_argument('--test_size', default=1000, type=int)
    parse.add_argument('--batch_size', default=64, type=int)
    parse.add_argument('--embedding_dim', default=128, type=int)
    parse.add_argument('--num_nodes', default=20, type=int)
    parse.add_argument('--hidden_size', default=128, type=int)
    parse.add_argument('--beam_width', default=3, type=int)
    parse.add_argument('--exploring_c', default=10, type=int)
    parse.add_argument('--n_processing', default=3, type=int, help='num of process blocks used in critic network')
    parse.add_argument('--epochs', default=20, type=int)
    # parse.add_argument('--train', default=True, action='store_false')
    parse.add_argument('--actor_lr', default='1e-4', type=float)
    parse.add_argument('--critic_lr', default='1e-4', type=float)
    parse.add_argument('--max_grad', default=2, type=float)

    parse.add_argument('--checkpoint', default=False, action='store_true', help='load a trained model from specified dir')
    parse.add_argument('--save_dir', default='trained', type=str)
    parse.add_argument('--data_dir', default='GVRP_Instances', type=str, help='data for test')

    args = parse.parse_args()
    if args.task == 'gvrp':
        import GVRP
        GVRP.train(args)
    elif args.task == 'gvrp2':
        import GVRP2
        GVRP2.train(args)
    else:
        raise ValueError(f'not support task {args.task}')