

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
import time, math
plt.switch_backend('agg')
import collections
import re
from torch import optim
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
from sklearn.preprocessing import OneHotEncoder
import os, math, glob, argparse

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

def _sample_gumbel(shape, eps=1e-20, out=None):
    U = out.resize_(shape).uniform_() if out is not None else torch.rand(shape)
    return - torch.log(eps - torch.log(U + eps))


def _gumbel_softmax_sample(logits, temp=1, eps=1e-20):
    dims = logits.dim()
    gumbel_noise = _sample_gumbel(logits.size(), eps=eps, out=logits.data.new())
    y = logits + Variable(gumbel_noise)
    return F.softmax(y / temp, dims - 1)


def gumbel_softmax(logits, temp=1, hard=False, eps=1e-20):
    shape = logits.size()
    assert len(shape) == 2
    y_soft = _gumbel_softmax_sample(logits, temp=temp, eps=eps)
    if hard:
        _, k = y_soft.data.max(-1)
        y_hard = logits.data.new(*shape).zero_().scatter_(-1, k.view(-1, 1), 1.0)
        y = Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y

class ResBlock(nn.Module):
    def __init__(self, hidden):
        super(ResBlock, self).__init__()
        self.res_block = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden, hidden, 5, padding=2, groups = 1),#nn.Linear(DIM, DIM),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden, hidden, 5, padding=2, groups = 1),#nn.Linear(DIM, DIM),
        )

    def forward(self, input):
        output = self.res_block(input)
        return input + (0.3*output)


class ResBlockD(nn.Module):
    def __init__(self, hidden):
        super(ResBlockD, self).__init__()
        self.res_block = nn.Sequential(
            nn.Conv1d(hidden, int(hidden/4), 5, padding=2, groups = 1),#nn.Linear(DIM, DIM),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(int(hidden/4), hidden, 5, padding=2, groups = 1),#nn.Linear(DIM, DIM),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input):
        output = self.res_block(input)
        return input + (0.1*output)


class CapsuleBlock(nn.Module):
    def __init__(self, hidden):
        super(CapsuleBlock, self).__init__()
        num_capsules=8
        self.res_block = nn.Sequential(        
            nn.Conv1d(hidden, int(hidden/4), 5, padding=2, groups = 1),#nn.Linear(DIM, DIM),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ModuleList([nn.Conv1d(int(hidden/4), hidden, 5, padding=2, groups = 1) 
                            for _ in range(num_capsules)]),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input):
        output = self.res_block(input)
        return input + (0.1*output)


class ResBlockG(nn.Module):
    def __init__(self, hidden):
        super(ResBlockG, self).__init__()
        self.res_block = nn.Sequential(
            nn.Conv1d(hidden, int(hidden/2), 5, padding=2, groups = 1),#nn.Linear(DIM, DIM),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(int(hidden/2), hidden, 5, padding=2, groups = 1),#nn.Linear(DIM, DIM),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input):
        output = self.res_block(input)
        return input + (0.1*output)

class Generator_lang(nn.Module):
    def __init__(self, n_chars, seq_len, batch_size, hidden):
        super(Generator_lang, self).__init__()
        self.fc1 = nn.Linear(128, hidden*seq_len)
        self.block = nn.Sequential(
            ResBlockG(hidden),
            ResBlockG(hidden),
            ResBlockG(hidden),
            ResBlockG(hidden),
            ResBlockG(hidden),
            ResBlockG(hidden),
        )
        self.conv1 = nn.Conv1d(hidden, n_chars, 1)
        self.n_chars = n_chars
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.hidden = hidden

    def forward(self, noise):
        output = self.fc1(noise)
        output = output.view(-1, self.hidden, self.seq_len) # (BATCH_SIZE, DIM, SEQ_LEN)
        output = self.block(output)
        output = self.conv1(output)
        output = output.transpose(1, 2)
        shape = output.size()
        output = output.contiguous()
        output = output.view(self.batch_size*self.seq_len, -1)
        output = gumbel_softmax(output, 0.5)
        return output.view(shape) # (BATCH_SIZE, SEQ_LEN, len(charmap))

class Discriminator_lang(nn.Module):
    def __init__(self, n_chars, seq_len, batch_size, hidden):
        super(Discriminator_lang, self).__init__()
        self.n_chars = n_chars
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.hidden = hidden
        capsule_net = CapsNet()
        self.block = nn.Sequential(
            ResBlockD(hidden),
            ResBlockD(hidden),
            ResBlockD(hidden),
            ResBlockD(hidden),
            ResBlockD(hidden),
            CapsuleBlock(hidden),
        )
        self.conv1d = nn.Conv1d(n_chars, hidden, 1)
        self.linear = nn.Linear(seq_len*hidden, 1)

    def forward(self, input):
        output = input.transpose(1, 2) # (BATCH_SIZE, len(charmap), SEQ_LEN)
        output = self.conv1d(output)
        output = self.block(output)
        output = output.view(-1, self.seq_len*self.hidden)
        output = self.linear(output)
        return output

def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))

def splitTrainTestValLists(dataset, percent_train, percent_val = 0):
    num_train = int(percent_train*len(dataset))
    num_val = int(percent_val*len(dataset))
    train = dataset[:num_train]
    val = dataset[num_train:(num_train + num_val)]
    test = dataset[(num_train + num_val):]
    if num_val == 0: return train, test
    return train, val, test

def plot_losses(losses_list, legends_list, file_out):
    assert len(losses_list) == len(legends_list)
    for i, loss in enumerate(losses_list):
        plt.plot(loss, label=legends_list[i])
    plt.legend()
    plt.savefig(file_out)
    plt.close()

def decode_one_seq(img, letter_dict = {'A':0, 'C':1, 'G':2, 'T':3}):
    seq = ''
    for row in range(len(img)):
        on = np.argmax(img[row,:])
        seq += letter_dict[on]
    return seq

def splitTrainTestVal(dataset, percent_train, percent_val = 0):
    num_train = int(percent_train*len(dataset))
    num_val = int(percent_val*len(dataset))
    train = dataset[:num_train, :]
    val = dataset[num_train:(num_train + num_val), :]
    test = dataset[(num_train + num_val):, :]
    if num_val == 0: return train, test
    return train, val, test

def one_hot_encode(seqs, letter_dict = {'A':0, 'C':1, 'G':2, 'T':3}):
    seqs_nums = np.array([[letter_dict[base] for base in seq.rstrip()] for seq in seqs], dtype=np.float32)
    seqs_one_hot = (np.arange(4) == seqs_nums[...,None]).astype(np.float32)
    #seqs_one_hot = np.expand_dims(seqs_one_hot, axis=3)
    seqs_one_hot = np.swapaxes(seqs_one_hot, 1, 2)
    return seqs_one_hot

def findMotif(seq, motif):
    if motif in seq:
        return True
    return False

def tokenize_string(sample):
    return tuple(sample.lower().split(' '))

class NgramLanguageModel(object):
    def __init__(self, n, samples, tokenize=False):
        if tokenize:
            tokenized_samples = []
            for sample in samples:
                tokenized_samples.append(tokenize_string(sample))
            samples = tokenized_samples

        self._n = n
        self._samples = samples
        self._ngram_counts = collections.defaultdict(int)
        self._total_ngrams = 0
        for ngram in self.ngrams():
            self._ngram_counts[ngram] += 1
            self._total_ngrams += 1

    def ngrams(self):
        n = self._n
        for sample in self._samples:
            for i in range(len(sample)-n+1):
                yield sample[i:i+n]

    def unique_ngrams(self):
        return set(self._ngram_counts.keys())

    def log_likelihood(self, ngram):
        if ngram not in self._ngram_counts:
            return -np.inf
        else:
            return np.log(self._ngram_counts[ngram]) - np.log(self._total_ngrams)

    def kl_to(self, p):
        # p is another NgramLanguageModel
        log_likelihood_ratios = []
        for ngram in p.ngrams():
            log_likelihood_ratios.append(p.log_likelihood(ngram) - self.log_likelihood(ngram))
        return np.mean(log_likelihood_ratios)

    def cosine_sim_with(self, p):
        # p is another NgramLanguageModel
        p_dot_q = 0.
        p_norm = 0.
        q_norm = 0.
        for ngram in p.unique_ngrams():
            p_i = np.exp(p.log_likelihood(ngram))
            q_i = np.exp(self.log_likelihood(ngram))
            p_dot_q += p_i * q_i
            p_norm += p_i**2
        for ngram in self.unique_ngrams():
            q_i = np.exp(self.log_likelihood(ngram))
            q_norm += q_i**2
        return p_dot_q / (np.sqrt(p_norm) * np.sqrt(q_norm))

    def precision_wrt(self, p):
        # p is another NgramLanguageModel
        num = 0.
        denom = 0
        p_ngrams = p.unique_ngrams()
        for ngram in self.unique_ngrams():
            if ngram in p_ngrams:
                num += self._ngram_counts[ngram]
            denom += self._ngram_counts[ngram]
        return float(num) / denom

    def recall_wrt(self, p):
        return p.precision_wrt(self)

    def js_with(self, p):
        log_p = np.array([p.log_likelihood(ngram) for ngram in p.unique_ngrams()])
        log_q = np.array([self.log_likelihood(ngram) for ngram in p.unique_ngrams()])
        log_m = np.logaddexp(log_p - np.log(2), log_q - np.log(2))
        kl_p_m = np.sum(np.exp(log_p) * (log_p - log_m))

        log_p = np.array([p.log_likelihood(ngram) for ngram in self.unique_ngrams()])
        log_q = np.array([self.log_likelihood(ngram) for ngram in self.unique_ngrams()])
        log_m = np.logaddexp(log_p - np.log(2), log_q - np.log(2))
        kl_q_m = np.sum(np.exp(log_q) * (log_q - log_m))

        return 0.5*(kl_p_m + kl_q_m) / np.log(2)

def load_dataset(max_length, max_n_examples, tokenize=False, max_vocab_size=2048, data_dir=''):
    print ("loading dataset...")

    lines = []

    finished = False

    for i in range(1):
        path = data_dir
        with open(path, 'r') as f:
            for line in f:
                line = line[:-1]
                if tokenize:
                    line = tokenize_string(line)
                else:
                    line = tuple(line)

                if len(line) > max_length:
                    line = line[:max_length]

                lines.append(line + ( ("P",)*(max_length-len(line)) ) )

                if len(lines) == max_n_examples:
                    finished = True
                    break
        if finished:
            break

    np.random.shuffle(lines)

    import collections
    counts = collections.Counter(char for line in lines for char in line)

    charmap = {'P':0}
    inv_charmap = ['P']

    for char,count in counts.most_common(max_vocab_size-1):
        if char not in charmap:
            charmap[char] = len(inv_charmap)
            inv_charmap.append(char)

    filtered_lines = []
    for line in lines:
        filtered_line = []
        for char in line:
            if char in charmap:
                filtered_line.append(char)
            else:
                filtered_line.append('P')
        filtered_lines.append(tuple(filtered_line))

    for i in range(1):
        print(filtered_lines[i])

    print("loaded {} lines in dataset".format(len(lines)))
    return filtered_lines, charmap, inv_charmap

class WGAN_LangGP():
    def __init__(self, batch_size=32, lr=0.0005, num_epochs=12, seq_len = 160, data_dir='', \
        run_name='test'):
        self.hidden = hidden
        self.batch_size = batch_size
        self.lr = lr
        #self.test_n_epochs = num_epochs
        self.n_epochs = num_epochs
        self.seq_len = seq_len
        self.d_steps = d_steps
        self.g_steps = 1
        self.lamda = 10 #lambda
        self.checkpoint_dir = './checkpoint/' + run_name + "/"
        self.sample_dir = './samples/test/' + run_name + "/"
        self.load_data(data_dir)
        #self.load_test_data(test_data_dir)
        if not os.path.exists(self.checkpoint_dir): os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.sample_dir): os.makedirs(self.sample_dir)
        self.use_cuda = True if torch.cuda.is_available() else False
        self.build_model()

    def build_model(self):
        self.G = Generator_lang(len(self.charmap), self.seq_len, self.batch_size, self.hidden)
        self.D = Discriminator_lang(len(self.charmap), self.seq_len, self.batch_size, self.hidden)
        if self.use_cuda:
            self.G.cuda()
            self.D.cuda()
        self.G_optimizer = optim.RMSprop(self.G.parameters(), lr=self.lr, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
        self.D_optimizer = optim.RMSprop(self.D.parameters(), lr=self.lr, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)

    def load_data(self, datadir):
        max_examples = 1e6
        lines, self.charmap, self.inv_charmap = load_dataset(
            max_length=self.seq_len,
            max_n_examples=max_examples,
            data_dir=datadir
        )
        self.data = lines

    def save_model(self, epoch):
        torch.save(self.G.state_dict(), self.checkpoint_dir + "G_weights_{}.pth".format(epoch))
        torch.save(self.D.state_dict(), self.checkpoint_dir + "D_weights_{}.pth".format(epoch))

    def load_model(self, directory = ''):
        if len(directory) == 0:
            directory = self.checkpoint_dir
        list_G = glob.glob(directory + "G*.pth")
        list_D = glob.glob(directory + "D*.pth")
        if len(list_G) == 0:
            print("[*] Checkpoint not found! Starting from scratch.")
            return 1 #file is not there
        G_file = max(list_G, key=os.path.getctime)
        D_file = max(list_D, key=os.path.getctime)
        epoch_found = int( (G_file.split('_')[-1]).split('.')[0])
        print("[*] Checkpoint {} found at {}!".format(epoch_found, directory))
        self.G.load_state_dict(torch.load(G_file),strict=True)
        self.D.load_state_dict(torch.load(D_file),strict=True)
        return epoch_found

    def calc_gradient_penalty(self, real_data, fake_data):
        alpha = torch.rand(self.batch_size, 1, 1)
        alpha = alpha.view(-1,1,1)
        alpha = alpha.expand_as(real_data)
        alpha = alpha.cuda() if self.use_cuda else alpha
        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        interpolates = interpolates.cuda() if self.use_cuda else interpolates
        interpolates = autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates = self.D(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).cuda() \
                                  if self.use_cuda else torch.ones(disc_interpolates.size()),
                                  create_graph=True, retain_graph=True)[0]

        gradients = gradients.contiguous().view(self.batch_size, -1)
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        #gradient_penalty = ((gradients.norm(2, dim=1).norm(2,dim=1) - 1) ** 2).mean() * self.lamda
        return self.lamda * ((gradients_norm - 1) ** 2).mean()

    def disc_train_iteration(self, real_data):
        self.D_optimizer.zero_grad()

        fake_data = self.sample_generator(self.batch_size)
        d_fake_pred = self.D(fake_data)
        d_fake_err = d_fake_pred.mean()
        d_real_pred = self.D(real_data)
        d_real_err = d_real_pred.mean()

        gradient_penalty = self.calc_gradient_penalty(real_data, fake_data)

        d_err = d_fake_err - d_real_err + gradient_penalty
        d_err.backward(retain_graph=True)
        self.D_optimizer.step()

        return d_fake_err.data, d_real_err.data, gradient_penalty.data

    def sample_generator(self, num_sample):
        z_input = Variable(torch.randn(num_sample, 128))
        if self.use_cuda: z_input = z_input.cuda()
        generated_data = self.G(z_input)
        return generated_data

    def gen_train_iteration(self):
        self.G.zero_grad()
        z_input = to_var(torch.randn(self.batch_size, 128))
        g_fake_data = self.G(z_input)
        dg_fake_pred = self.D(g_fake_data)
        g_err = -torch.mean(dg_fake_pred)
        g_err.backward()
        self.G_optimizer.step()
        return g_err

    def train_model(self, load_dir):
        init_epoch = self.load_model(load_dir)
        total_iterations = 1000
        losses_f = open(self.checkpoint_dir + "losses.txt",'a+')
        d_fake_losses, d_real_losses, grad_penalties = [],[],[]
        G_losses, D_losses, W_dist = [],[],[]

        table = np.arange(len(self.charmap)).reshape(-1, 1)
        one_hot = OneHotEncoder()
        one_hot.fit(table)

        counter = 0
        for epoch in range(self.n_epochs):
            n_batches = int(len(self.data)/self.batch_size)
            for idx in range(n_batches):
                _data = np.array(
                    [[self.charmap[c] for c in l] for l in self.data[idx*self.batch_size:(idx+1)*self.batch_size]],
                    dtype='int32'
                )
                data_one_hot = one_hot.transform(_data.reshape(-1, 1)).toarray().reshape(self.batch_size, -1, len(self.charmap))
                real_data = torch.Tensor(data_one_hot)
                d_fake_err, d_real_err, gradient_penalty = self.disc_train_iteration(real_data)

                d_err = d_fake_err - d_real_err + gradient_penalty

                # Append things for logging
                d_fake_np, d_real_np, gp_np = d_fake_err.cpu().numpy(), \
                d_real_err.cpu().numpy(), gradient_penalty.cpu().numpy()
                grad_penalties.append(gp_np)
                d_real_losses.append(d_real_np)
                d_fake_losses.append(d_fake_np)
                D_losses.append(d_fake_np - d_real_np + gp_np)
                W_dist.append(d_real_np - d_fake_np)

                if counter % self.d_steps == 0:
                    g_err = self.gen_train_iteration()
                    G_losses.append((g_err.data).cpu().numpy())

                if counter % 100 == 99:
                    self.sample(counter)

                if counter % 4000 == 3999:
                    self.save_model(counter)    
                #g_err_lst = []
                #d_err_lst = []
                if counter % 100 == 99:
                    summary_str = 'Iteration [{}/{}] - loss_d: {}, loss_g: {}, w_dist: {}, grad_penalty: {}'\
                        .format(counter, total_iterations, (d_err.data).cpu().numpy(),
                        (g_err.data).cpu().numpy(), ((d_real_err - d_fake_err).data).cpu().numpy(), gp_np)
                    losses_f.write(summary_str)
                counter += 1
            np.random.shuffle(self.data)

    def sample(self, epoch):
        z = to_var(torch.randn(self.batch_size, 128))
        self.G.eval()
        torch_seqs = self.G(z)
        seqs = (torch_seqs.data).cpu().numpy()
        decoded_seqs = [decode_one_seq(seq, self.inv_charmap)+"\n" for seq in seqs]
        with open(self.sample_dir + "sampled_{}.txt".format(epoch), 'w+') as f:
            f.writelines(decoded_seqs)
        self.G.train()

def main():
    load_dir=""
    model = WGAN_LangGP()
    model.train_model(load_dir)

if __name__ == '__main__':
    main()
