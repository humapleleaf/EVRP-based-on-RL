from torch import nn
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Encoder(nn.Module):
    def __init__(self, num_features, embedding_dim):
        super().__init__()
        self.embedding = nn.Conv1d(num_features, embedding_dim, kernel_size=1)

    def forward(self, inputs):
        return self.embedding(inputs)   # (batch, embedding_dim, seq_len)


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, inputs, hidden=None):
        """
        :param Tensor inputs: (batch_size, embedding_dim, 1)
        :param tuple hidden: (h_t, c_t), h_t/c_t: (1, batch, hidden_size)
        :return tuple: ditto
        """
        inputs = inputs.permute(0, 2, 1)
        _, hidden = self.lstm(inputs, hidden)
        return hidden


class Attention(nn.Module):
    def __init__(self, embedding_dim, hidden_size, num_afs=3, use_tanh=False, exploring_c=10, task='gvrp'):
        super().__init__()
        self.tanh = nn.Tanh()
        self.use_tanh = use_tanh
        self.exploring_c = exploring_c
        self.V0 = nn.Parameter(torch.Tensor(1, hidden_size), requires_grad=True)
        self.V1 = nn.Parameter(torch.Tensor(1, hidden_size), requires_grad=True)
        self.V2 = nn.Parameter(torch.Tensor(1, hidden_size), requires_grad=True)
        self.num_afs = num_afs
        self.context_linear1 = nn.Conv1d(embedding_dim, hidden_size, kernel_size=1)
        if task == 'gvrp' or task == 'gvrp2':
            self.context_linear2 = nn.Conv1d(embedding_dim, hidden_size, kernel_size=1)
        self.linear = nn.Linear(embedding_dim, hidden_size)
        nn.init.xavier_uniform_(self.V0)
        nn.init.xavier_uniform_(self.V1)
        nn.init.xavier_uniform_(self.V2)

    def forward(self, query, ref, dynamic=None):
        """
        :param query: Tensor, (batch, hidden_size)
        :param ref: embedded inputs, Tensor, (batch, embedding_dim, seq_len)
        :param dynamic: dynamic embedded inputs
        :return e: Tensor, (batch, hidden_size, seq_len)
        :return logits: Tensor, (batch, 1, seq_len)
        """
        batch_size = ref.size(0)
        v0 = self.V0.expand(batch_size, -1).unsqueeze(1)  # (batch, 1, hidden_size)
        v1 = self.V1.expand(batch_size, -1).unsqueeze(1)   # (batch, 1, hidden_size)
        v2 = self.V2.expand(batch_size, -1).unsqueeze(1)   # (batch, 1, hidden_size)
        e = self.context_linear1(ref)      # (batch, hidden_size, seq_len)
        e_plus = self.linear(query).unsqueeze(2).expand_as(e)
        if dynamic is not None:
            e_dyna = self.context_linear2(dynamic)
            e_sum = e + e_dyna + e_plus
            # logits = v.bmm(self.tanh(e + e_dyna + e_plus))  # (batch, 1, seq_len)
        else:
            e_sum = e + e_plus
            # logits = v.bmm(self.tanh(e + e_plus))  # (batch, 1, seq_len)
        logits = torch.cat((v0.bmm(self.tanh(e_sum)[:, :, 0:1]),
                            v1.bmm(self.tanh(e_sum)[:, :, 1:self.num_afs + 1]),
                            v2.bmm(self.tanh(e_sum)[:, :, self.num_afs + 1:])), dim=2)

        if self.use_tanh:
            logits = self.exploring_c*self.tanh(logits)

        return e, logits


class PointerNet(nn.Module):
    def __init__(self, static_features, dynamic_features, embedding_dim, hidden_size,
                 update_fn, beam_width=3, capacity=60,
                 velocity=40, cons_rate=0.2, t_limit=11, num_afs=3, task='gvrp'):
        super().__init__()
        self.encoder1 = Encoder(static_features, embedding_dim)
        self.decoder = Decoder(embedding_dim, hidden_size)
        if task == 'gvrp' or task == 'gvrp2':
            self.encoder2 = Encoder(dynamic_features, embedding_dim)
            self.attention = Attention(embedding_dim, hidden_size, use_tanh=False)
        else:
            pass    # to be continued
        self.linear = nn.Linear(embedding_dim + hidden_size, hidden_size)
        self.linear_single = nn.Linear(1, 1)
        self.softmax = nn.Softmax(dim=1)
        self.update_fn = update_fn
        self.beam_width = beam_width
        self.capacity = capacity
        self.velocity = velocity
        self.cons_rate = cons_rate
        self.t_limit = t_limit
        self.num_afs = num_afs

    def forward(self, static, dynamic, distances):
        """
        :param static: raw static inputs, (batch, 2, seq_len)
        :param dynamic: raw dynamic inputs,
        :param distances: (batch, seq_len, seq_len)
        :return tours: LongTensor, (batch*beam, seq_len-1)
        :return prob_log: (batch*baem)
        """
        batch_size, _, seq_len = static.shape
        max_times = 100
        embedding_static = self.encoder1(static)
        decoder_input = embedding_static[:, :, :1]  # (batch, embedding_dim, 1), always start from depot, i.e., the 1st point
        hidden = None
        mask = torch.zeros(batch_size, seq_len, device=device)
        mask[:, 0] = float('-inf')  # mask the depot
        dis_by_afs = [distances[:, i:i + 1, 0:1] + distances[:, i:i + 1, :] for i in range(1, self.num_afs+1)]
        dis_by_afs = torch.cat(dis_by_afs, dim=1).to(device)
        dis_by_afs = torch.min(dis_by_afs, dim=1)    # tuple: (batch, seq_len), ()
        dis_by_afs[0][:, 0] = 0
        dis_max = self.capacity / self.cons_rate

        # the customers that could not be served directly with a visit to one AFS are masked
        # mask = torch.where((distances[:, 0, :] > dis_max/2) & (dis_by_afs > dis_max), torch.ones(mask.size())*float('-inf'), mask)
        mask[(distances[:, 0, :] > dis_max/2) & (dis_by_afs[0] > dis_max)] = float('-inf')
        dynamic[:, 2, :] = torch.where((distances[:, 0, :] > dis_max/2) & (dis_by_afs[0] > dis_max),
                                       torch.zeros(batch_size, seq_len, device=device), dynamic[:, 2, :])
        old_idx = torch.zeros(batch_size, 1, dtype=torch.long, device=device)

        if self.training or self.beam_width == 1:
            tours = []
            prob_log = torch.zeros(batch_size, device=device)

            for _ in range(max_times):
                if (dynamic[:, 2, :] == 0).all():   # all demands have been satisfied
                    break
                h_t, c_t = self.decoder(decoder_input, hidden)
                h_t = h_t.squeeze(0)           # (batch*beam_width, hidden)
                embedding_dynamic = self.encoder2(dynamic)
                _, att = self.attention(h_t, embedding_static, dynamic=embedding_dynamic)       # (batch*beam_width, 1, seq_len)
                # _, att = self.attention(h_t, embedding_static)
                att = att.squeeze(1) + mask
                alphas = self.softmax(att)     # (batch*beam_width, seq_len)

                if self.training:
                    idx = torch.multinomial(alphas, 1)         # (batch*beam_width, 1)
                    probs = torch.gather(alphas, 1, idx).view(-1)
                else:
                    probs, idx = torch.max(alphas, 1)     # beam_width=1, a.k.a Greedy
                    idx = idx.unsqueeze(1)

                dynamic = self.update_fn(old_idx, idx, mask, dynamic, distances, dis_by_afs, self.capacity,
                                         self.velocity, self.cons_rate, self.t_limit, self.num_afs)  # update mask and dynamic
                old_idx = idx
                tours.append(idx)
                prob_log += torch.log(probs)

                ctx = embedding_static.bmm(alphas.unsqueeze(2))    # (batch*beam_width, embedding_dim, 1)
                h_t = self.linear(torch.cat((h_t, ctx.squeeze(2)), dim=1))
                hidden = (h_t.unsqueeze(0), c_t)
                decoder_input = embedding_static[torch.arange(batch_size), :, idx.squeeze(1)].unsqueeze(2)

            tours = torch.cat(tours, dim=1).to(device)
        else:
            ''' -------beam_width > 1--------'''
            embedding_static = embedding_static.repeat(self.beam_width, 1, 1)
            dynamic = dynamic.repeat(self.beam_width, 1, 1)
            decoder_input = decoder_input.repeat(self.beam_width, 1, 1)   # (batch*beam_width, embedding_dim, 1)
            mask = mask.repeat(self.beam_width, 1)    # (batch*beam_width, seq_len)
            batch_idx = torch.LongTensor([i for i in range(batch_size)]).to(device)
            batch_idx = batch_idx.repeat(self.beam_width)  # (batch*beam_width)
            old_idx = old_idx.repeat(self.beam_width, 1)
            distances = distances.repeat(self.beam_width, 1, 1)
            dis_by_afs = (dis_by_afs[0].repeat(self.beam_width, 1), dis_by_afs[1].repeat(self.beam_width, 1))

            for i in range(max_times):
                if (dynamic[:, 2, :] == 0).all():  # all demands have been satisfied
                    break
                h_t, c_t = self.decoder(decoder_input, hidden)
                h_t = h_t.squeeze(0)  # (batch*beam_width, hidden)
                embedding_dynamic = self.encoder2(dynamic)
                _, att = self.attention(h_t, embedding_static, dynamic=embedding_dynamic)  # (batch*beam_width, 1, seq_len)
                # _, att = self.attention(h_t, embedding_static)  # (batch*beam_width, 1, seq_len)
                att = att.squeeze(1) + mask
                alphas = self.softmax(att)  # (batch*beam_width, seq_len)

                if i == 0:
                    probs, idx = torch.topk(alphas[:batch_size], self.beam_width, dim=1)  # both: (batch, beam_width)
                    idx = idx.transpose(1, 0).contiguous().view(-1, 1)  # (batch*beam_width, 1)
                    probs = probs.transpose(1, 0).contiguous().view(-1, 1)  # ditto
                    prob_log = torch.log(probs)     # （batch*beam_width, 1)
                    tours = idx
                else:
                    prob_log_all = prob_log + torch.log(alphas)   # (batch*beam, seq_len)
                    prob_log_all = torch.cat(prob_log_all.chunk(self.beam_width, dim=0), dim=1)  # (batch, seq_len*beam)
                    prob_log, idx = torch.topk(prob_log_all, self.beam_width, dim=1)  # both: (batch, beam_width)
                    prob_log = prob_log.transpose(1, 0).contiguous().view(-1, 1)  # (batch*beam_width, 1)

                    hpt = (idx / seq_len).transpose(1, 0).contiguous().view(-1)    # from which beam, (batch*beam)
                    idx = idx % seq_len                                            # which node
                    idx = idx.transpose(1, 0).contiguous().view(-1, 1)  # (batch*beam_width, 1)
                    bb_idx = batch_idx + hpt*batch_size
                    tours = torch.cat((tours[bb_idx], idx), dim=1)
                    alphas = alphas[bb_idx]
                    prob_log = prob_log[bb_idx]
                    h_t = h_t[bb_idx]
                    c_t = c_t.squeeze(0)[bb_idx].unsqueeze(0)
                    old_idx = old_idx[bb_idx]
                    distances = distances[bb_idx]
                    dis_by_afs = (dis_by_afs[0][bb_idx], dis_by_afs[1][bb_idx])
                    mask = mask[bb_idx]
                    dynamic = dynamic[bb_idx]

                dynamic = self.update_fn(old_idx, idx, mask, dynamic, distances, dis_by_afs, self.capacity,
                                         self.velocity, self.cons_rate, self.t_limit, self.num_afs)  # update mask and dynamic
                old_idx = idx

                ctx = embedding_static.bmm(alphas.unsqueeze(2))  # (batch*beam_width, embedding_dim, 1)
                h_t = self.linear(torch.cat((h_t, ctx.squeeze(2)), dim=1))
                hidden = (h_t.unsqueeze(0), c_t)

                # idx = idx.unsqueeze(2).expand_as(decoder_input)
                # decoder_input = embedding_static.gather(2, idx)
                decoder_input = embedding_static[torch.arange(batch_size*self.beam_width), :, idx.squeeze(1)].unsqueeze(2)

            # tours = tours[:batch_size]
            # prob_log = prob_log[:batch_size]

        return tours, prob_log


class CriticNet(nn.Module):
    def __init__(self, static_features, dynamic_features, embedding_dim, hidden_size, exploring_c, n_processing=3):
        super().__init__()
        # if dynamic_features:
        #     self.embedding = Encoder(static_features + dynamic_features, hidden_size)
        # else:
        #     self.embedding = Encoder(static_features, hidden_size)
        self.hidden_size = hidden_size
        self.n_processing = n_processing
        self.encoder = Encoder(static_features+dynamic_features, embedding_dim)
        self.attention = Attention(embedding_dim, hidden_size, use_tanh=True, exploring_c=exploring_c)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.softmax = nn.Softmax(dim=2)

    def forward(self, static, dynamic):
        """
        :param: static: raw inputs, (batch, 2, seq_len)
        :param: dynamic: ditto
        :return: (batch), the valuations for each of examples
        """
        inputs = torch.cat((static, dynamic), dim=1)
        embedding_inp = self.encoder(inputs)
        batch_size = static.size(0)
        hidden = torch.zeros(batch_size, self.hidden_size, device=device)
        for _ in range(self.n_processing):
            ref, logits = self.attention(hidden, embedding_inp)
            hidden = torch.bmm(ref, self.softmax(logits).transpose(1, 2)).squeeze(2)  # (batch, hidden)
        vals = self.decoder(hidden)
        return vals


class RLAgent(nn.Module):
    def __init__(self, static_features, dynamic_features, embedding_dim, hidden_size, exploring_c, n_processing,
                 update_fn, beam_width, capacity, velocity, cons_rate, t_limit, num_afs, task='gvrp'):
        super().__init__()
        self.ptnet = PointerNet(static_features, dynamic_features, embedding_dim, hidden_size, update_fn,
                                beam_width, capacity, velocity, cons_rate, t_limit, num_afs, task)
        self.critic = CriticNet(static_features, dynamic_features, embedding_dim, hidden_size, exploring_c, n_processing)

    def forward(self, static, dynamic, distances):
        """
        :param static: raw data, tensor, (batch, 2, seq_len)
        :param dynamic: (batch, 3, seq_len)     TODO: add time windows
        :param distances: tensor, (batch, seq_len, seq_len)
        :return tours: tensor, (batch, seq_len-1)
        :return prob_log: (batch)
        :return vals: (batch)
        """
        tours, prob_log = self.ptnet(static, dynamic, distances)
        vals = self.critic(static, dynamic)
        return tours, prob_log, vals
