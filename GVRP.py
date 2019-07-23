from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
import os
from models import RLAgent
import time
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GVRPDataset(Dataset):
    def __init__(self, train_size, num_nodes, t_limit, capacity, num_afs=3, data_dir=None, seed=520):
        super().__init__()
        self.size = train_size
        if data_dir:
            depot = set()
            afs = set()
            self.static = torch.zeros(10, 2, 1+num_afs+num_nodes, device=device)

            for i in range(10):
                filename = f'20c3sU{i + 1}.txt'
                x = []
                y = []
                with open(os.path.join(data_dir, filename), 'r') as f:
                    lines = f.readlines()[2:26]
                    depot.add((float(lines[0].split()[2]), float(lines[0].split()[3])))
                    for k in range(1, num_afs+1):
                        afs.add((float(lines[k].split()[2]), float(lines[k].split()[3])))
                    for line in lines[num_afs+1:]:
                        line = line.strip()
                        if not line.startswith('C'):
                            print(line)
                            raise ValueError(f'the format of {filename} is not consistent with the others')
                        line = line.split()
                        x.append(float(line[2]))
                        y.append(float(line[3]))
                self.static[i, 0, num_afs+1:] = torch.tensor(x)
                self.static[i, 1, num_afs+1:] = torch.tensor(y)

            assert len(depot) == 1 and len(afs) == num_afs
            self.static[:, :, 0] = torch.tensor(list(depot)).unsqueeze(0)
            self.static[:, :, 1:num_afs+1] = torch.tensor(sorted(list(afs), reverse=True)).transpose(1, 0).unsqueeze(0)
        else:
            torch.manual_seed(seed)
            # # left bottom: (-79.5, 36); top right: (-75.5, 39.5)
            afs = torch.tensor([[-76.338677, -77.08760885, -79.156076], [36.796046, 39.45787498, 37.383343]])
            afs = torch.cat((torch.tensor([-77.49439265, 37.60851245]).unsqueeze(1), afs), dim=1).to(device)  # add depot
            # afs[0] = (afs[0] + 79.5)/4
            # afs[1] = (afs[1] - 36)/3.5
            customers = torch.rand(train_size, 2, num_nodes, device=device)
            customers[:, 0, :] = customers[:, 0, :] * 4 - 79.5
            customers[:, 1, :] = customers[:, 1, :] * 3.5 + 36
            self.static = torch.cat((afs.unsqueeze(0).repeat(train_size, 1, 1), customers), dim=2).to(device)  # (train_size, 2, num_nodes+4)

        self.dynamic = torch.ones(train_size, 3, 1+num_afs+num_nodes, device=device)   # time duration, capacity, demands
        self.dynamic[:, 0, :] *= t_limit
        self.dynamic[:, 1, :] *= capacity
        self.dynamic[:, 2, :num_afs+1] = 0
        # self.dynamic[:, 1, :self.num_afs+1] = 0

        seq_len = self.static.size(2)
        self.distances = torch.zeros(train_size, seq_len, seq_len, device=device)
        for i in range(seq_len):
            self.distances[:, i] = cal_dis(self.static[:, 0, :], self.static[:, 1, :], self.static[:, 0, i:i+1], self.static[:, 1, i:i+1])

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.static[idx], self.dynamic[idx], self.distances[idx]    # dynamic: None


def cal_dis(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # degrees to radians
    lon1, lat1, lon2, lat2 = map(lambda x: x/180*np.pi, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = torch.pow(torch.sin(dlat / 2), 2) + torch.cos(lat1) * torch.cos(lat2) * torch.pow(torch.sin(dlon / 2), 2)
    c = 2 * 4182.44949 * torch.asin(torch.sqrt(a))    # miles
    # c = 2 * atan2(sqrt(a), sqrt(1 - a)) * 4182.44949
    return c


def train(args):
    agent = RLAgent(args.static_features,
                    args.dynamic_features,
                    args.embedding_dim,
                    args.hidden_size,
                    args.exploring_c,
                    args.n_processing,
                    update_fn,
                    args.beam_width,
                    args.capacity,
                    args.velocity,
                    args.cons_rate,
                    args.t_limit,
                    args.num_afs,
                    args.task).to(device)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    path = os.path.join(args.save_dir, f'gvrp{args.num_nodes}_{args.beam_width}.pt')
    # path = os.path.join(args.save_dir, f'gvrp{args.num_nodes}_{10}.pt')
    if args.checkpoint:
        agent.load_state_dict(torch.load(path, device))
        print('loaded a trained model')
    else:
        ## train model
        train_data = GVRPDataset(args.train_size, args.num_nodes, args.t_limit, args.capacity, args.num_afs, seed=args.seed)
        valid_data = GVRPDataset(args.valid_size, args.num_nodes, args.t_limit, args.capacity, args.num_afs, seed=args.seed+1)

        actor_optim = torch.optim.Adam(agent.ptnet.parameters(), lr=args.actor_lr)
        critic_optim = torch.optim.Adam(agent.critic.parameters(), lr=args.critic_lr)

        train_loader = DataLoader(train_data, args.batch_size, True, num_workers=0)
        valid_loader = DataLoader(valid_data, args.batch_size, False, num_workers=0)

        min_reward = np.inf

        for epoch in range(args.epochs):
            agent.train()
            start_time = time.time()
            reward_epoch = 0

            for batch_id, (static, dynamic, distances) in enumerate(train_loader):
                # static = static.to(device)
                # dynamic = dynamic.to(device)
                tours, prob_log, vals = agent(static, dynamic, distances)
                rewards, _ = reward_func(tours, static, distances)
                advantage = rewards - vals

                actor_loss = torch.mean(advantage.detach()*prob_log)
                actor_optim.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(agent.ptnet.parameters(), args.max_grad)
                actor_optim.step()

                # weight = (advantage.lt(0)*4 + 1).float().detach()
                # critic_loss = torch.mean(advantage**2*weight)
                critic_loss = torch.mean(advantage**2)
                critic_optim.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(agent.critic.parameters(), args.max_grad)
                critic_optim.step()

                reward_epoch += rewards.sum().item()  # watch out numerical overflow if reward is large
                if (batch_id + 1) % 100 == 0:
                    print(f'Batch %d/%d, reward: %2.3f, loss: %2.4f' % (batch_id+1, len(train_loader),
                                                                        rewards.mean().item(), actor_loss.item()))
            reward_epoch /= args.train_size
            end_time = time.time()
            print('Epoch %d/%d, reward: %2.3f, took: %2.2fs' %
                  (epoch+1, args.epochs, reward_epoch, end_time-start_time))

            valid_reward = validate(valid_loader, agent.ptnet, args=args)
            if valid_reward < min_reward:
                torch.save(agent.state_dict(), path)
                min_reward = valid_reward
                print('Epoch %d/%d: the reward of validation data is %2.3f' % (epoch+1, args.epochs, min_reward))
            print('-----------------------------------')
        if valid_reward != min_reward:
            torch.save(agent.state_dict(), path[:-3]+'_final.pt')

    ## test
    print('start testing......')
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)
    path = os.path.join(args.data_dir, 'Table 2')
    if not os.path.exists(path):
        test_data = GVRPDataset(args.test_size, args.num_nodes, args.t_limit, args.capacity, args.num_afs, seed=args.seed+2)
        test_loader = DataLoader(test_data, args.batch_size, False, num_workers=0)
    else:
        test_data = GVRPDataset(10, args.num_nodes, args.t_limit, args.capacity, args.num_afs, path, seed=args.seed + 2)
        test_loader = DataLoader(test_data, 1, False, num_workers=0)

    path = os.path.join(args.save_dir, f'gvrp{args.num_nodes}c{args.num_afs}s')
    test_reward = validate(test_loader, agent.ptnet, render=True, args=args, save_dir=path)
    print('the reward of test data is %2.3f' % test_reward)


def validate(data_loader, actor, render=False, args=None, save_dir=None):
    """
    :param data_loader:
    :param actor: PointerNet
    :param render: Boolean, whether to plot
    :param save_dir: string
    :return: a python num, average reward of all data from data_loader
    """
    actor.eval()
    avg_reward = []
    for i, (static, dynamic, distances) in enumerate(data_loader):
        # static = static.to(device)
        # dynamic = dynamic.to(device)
        with torch.no_grad():
            tours, _ = actor(static, dynamic, distances)
        reward, locs = reward_func(tours, static, distances, args.beam_width)
        avg_reward.append(reward.sum())
        if render:
            render_func(locs, static, args.num_afs, save_dir+f'U{i+1}.png')
            filename = os.path.join(args.save_dir, f'{args.task}_log.txt')
            with open(filename, 'a') as f:
                f.write('the reward of %dc%dsU%d is %2.3f\n' % (args.num_nodes, args.num_afs, i+1, reward))

    if render:
        with open(filename, 'a') as f:
            f.write(f'beam_width: {args.beam_width}, train_size: {args.train_size}, '
                    f'epochs: {args.epochs}, actor_lr_rate: {args.actor_lr}\n')
            f.write(time.strftime('%Y-%m-%d %H:%M:%S\n\n', time.localtime(time.time())))

    avg_reward = np.sum(avg_reward)/data_loader.dataset.size
    actor.train()

    return avg_reward


def reward_func(tours, static, distances, beam_width=1):
    """
    :param tours: LongTensor, (batch*beam, seq_len)
    :param static: (batch, 2, num_nodes)
    :param distances: (batch, num_nodes, num_nodes)
    :param beam_width: set beam_width=1 when training
    :return: reward: Euclidean distance between each consecutive point， (batch)
    :return: locs: (batch, 2, seq_len)
    """
    bb_size, seq_len = tours.size()
    batch_size = static.size(0)
    depot = torch.zeros(bb_size, 1, dtype=torch.long, device=device)
    tours = torch.cat((depot, tours, depot), dim=1)         # start from depot, end at depot(although some have ended at depot)
    id0 = torch.arange(bb_size).unsqueeze(1).repeat(1, seq_len+1)
    reward = distances.repeat(beam_width, 1, 1)[id0, tours[:, :-1], tours[:, 1:]].sum(1)    # (batch*beam)
    # (batch*beam) -> (batch), choose the best reward
    reward, id_best = torch.cat(torch.chunk(reward.unsqueeze(1), beam_width, dim=0), dim=1).min(1)  # (batch)
    bb_idx = torch.arange(batch_size, device=device) + id_best * batch_size
    tours = tours[bb_idx]
    # print(tours)
    tours = tours.unsqueeze(1).repeat(1, static.size(1), 1)
    locs = torch.gather(static, dim=2, index=tours)  # (batch, 2, seq_len+)
    return reward, locs


def render_func(locs, static, num_afs, save_dir):
    """
    :param locs: (batch, 2, sel_len)
    :param static: (batch, 2, num_nodes+1)
    :param num_afs: scalar
    :param save_dir: path to save figure
    :return: None
    """
    plt.close('all')
    data = locs[-1].cpu().numpy()   # (2, num_nodes+1), just plot the last one
    # demands = dynamic[-1, 1, 1:].cpu().numpy()*capacity   # (num_nodes), depot excluded
    # coords = static[-1][:, 1:].cpu().numpy().T     # (num_nodes, 2)
    plt.plot(data[0], data[1], zorder=1)
    origin_locs = static[-1, :, :].cpu().numpy()
    plt.scatter(origin_locs[0], origin_locs[1], s=4, c='r', zorder=2)
    plt.scatter(origin_locs[0, 0], origin_locs[1, 0], s=20, c='k', marker='*', zorder=3)  # depot
    plt.scatter(origin_locs[0, 1:num_afs+1], origin_locs[1, 1:num_afs+1], s=20, c='b', marker='+', zorder=4)    # afs
    for i, coords in enumerate(origin_locs.T[1:]):
        plt.annotate('%d' % (i+1), xy=coords, xytext=(2, 2), textcoords='offset points')  # mark numbers
    if os.path.isfile(save_dir):
        os.remove(save_dir)
    plt.axis('equal')
    plt.savefig(save_dir, dpi=400)


def update_fn(old_idx, idx, mask, dynamic, distances, dis_by_afs, capacity, velocity, cons_rate, t_limit, num_afs):
    """
    :param old_idx: (batch*beam, 1)
    :param idx: ditto
    :param mask: (batch*beam, seq_len)
    :param dynamic: (batch*beam, dynamic_features, seq_len)
    :param distances: (batch*beam, seq_len, seq_len)
    :param dis_by_afs: (batch*beam, seq_len)
    :param capacity, velocity, cons_rate, t_limit, num_afs: scalar
    :return: updated dynamic
    """
    dis = distances[torch.arange(distances.size(0)), old_idx.squeeze(1), idx.squeeze(1)].unsqueeze(1)
    depot = idx.eq(0).squeeze(1)
    afs = (idx.gt(0) & idx.le(num_afs)).squeeze(1)
    fs = idx.le(num_afs).squeeze(1)
    customer = idx.gt(num_afs).squeeze(1)    # TODO: introduce num_afs
    time = dynamic[:, 0, :].clone()
    fuel = dynamic[:, 1, :].clone()
    demands = dynamic[:, 2, :].clone()

    time -= dis/velocity
    time[depot] = t_limit
    time[afs] -= 0.25
    time[customer] -= 0.5

    fuel -= cons_rate * dis
    fuel[fs] = capacity
    demands.scatter_(1, idx, 0)

    dynamic = torch.cat((time.unsqueeze(1), fuel.unsqueeze(1), demands.unsqueeze(1)), dim=1).to(device)

    mask.scatter_(1, idx, float('-inf'))
    # forbid passing by afs if leaving depot, allow if returning to depot; forbid from afs to afs, not necessary but convenient
    mask[fs, 1:num_afs+1] = float('-inf')
    mask[afs, 0] = 0

    mask[customer, :num_afs+1] = 0
    mask[fs] = torch.where(demands[fs] > 0, torch.zeros(mask[fs].size(), device=device), mask[fs])

    # path1: ->Node->Depot
    dis1 = distances[torch.arange(distances.size(0)), idx.squeeze(1)].clone()
    fuel_pd0 = cons_rate * dis1
    time_pd0 = dis1 / velocity
    dis1[:, num_afs+1:] += distances[:, 0, num_afs+1:]
    fuel_pd1 = cons_rate * dis1
    time_pd1 = (distances[torch.arange(distances.size(0)), idx.squeeze(1)] + distances[:, 0, :]) / velocity
    time_pd1[:, 1:num_afs + 1] += 0.25
    time_pd1[:, num_afs + 1:] += 0.5

    # path2: ->Node-> Station-> Depot(choose the station making the total distance shortest)
    dis2 = distances[:, 1:num_afs+1, :].gather(1, dis_by_afs[1].unsqueeze(1)).squeeze(1)
    dis2[:, 0] = 0
    dis2 += distances[torch.arange(distances.size(0)), idx.squeeze(1)]
    fuel_pd2 = cons_rate * dis2
    time_pd2 = (distances[torch.arange(distances.size(0)), idx.squeeze(1)] + dis_by_afs[0]) / velocity
    time_pd2[:, 1:num_afs + 1] += 0.25
    time_pd2[:, num_afs + 1:] += 0.5

    # path3: ->Node-> Station-> Depot(choose the closest station to the node), ignore this path temporarily
    # the next node should be able to return to depot with at least one way; otherwise, mask it
    mask[~((fuel >= fuel_pd1) & (time >= time_pd1) | (fuel >= fuel_pd2) & (time >= time_pd2))] = float('-inf')

    mask[(fuel < fuel_pd0) | (time < time_pd0)] = float('-inf')

    all_masked = mask[:, num_afs+1:].eq(0).sum(1).le(0)
    mask[all_masked, 0] = 0  # unmask the depot if all nodes are masked

    return dynamic
