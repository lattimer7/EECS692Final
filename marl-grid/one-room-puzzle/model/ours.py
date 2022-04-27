from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import pickle
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from model.a3c_template import A3CTemplate, take_action, take_comm_action
from model.init import normalized_columns_initializer, weights_init

class ConvVAE(nn.Module):
  def __init__(self, z_size=32, learning_rate=0.0001, kl_tolerance=0.5, is_training=False, reuse=False, gpu_mode=False):
    super().__init__()
    self.z_size = z_size
    self.learning_rate = learning_rate
    self.is_training = is_training
    self.kl_tolerance = kl_tolerance

    self.encoder = nn.Sequential(
        nn.Conv2d(3, 32, 4, stride=2, padding=2),
        nn.ReLU(True),
        nn.Conv2d(32, 64, 4, stride=2, padding=2),
        nn.ReLU(True),
        nn.Conv2d(64, 128, 4, stride=2, padding=2),
        nn.ReLU(True),
        nn.Conv2d(128, 256, 4, stride=2),
        nn.ReLU(True)
    )
    self.mu = nn.Linear(2*2*256, self.z_size)
    self.logvar = nn.Linear(2*2*256, self.z_size)

    self.decoderdense = nn.Linear(self.z_size, 256*4)

    self.decoder = nn.Sequential(
        nn.ConvTranspose2d(4*256, 128, 5, stride=2),
        nn.ReLU(True),
        nn.ConvTranspose2d(128, 64, 5, stride=2, padding=2),
        nn.ReLU(True),
        nn.ConvTranspose2d(64, 32, 6, stride=2, padding=2),
        nn.ReLU(True),
        nn.ConvTranspose2d(32, 3, 6, stride=2)
    )
    self.N = torch.distributions.Normal(0, 1)
    self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
    self.N.scale = self.N.scale.cuda()
    # self.sampling = Lambda(self.sample)

#   def calculate_loss(self, mean, log_variance, predictions, targets):
#         mse = F.mse_loss(predictions, targets, reduction='mean')
#         log_sigma_opt = 0.5 * mse.log()
#         r_loss = 0.5 * torch.pow((targets - predictions) / log_sigma_opt.exp(), 2) + log_sigma_opt
#         r_loss = r_loss.sum()
#         kl_loss = self._compute_kl_loss(mean, log_variance)
#         return r_loss, kl_loss, log_sigma_opt.exp()

  def calculate_loss(self, mean, log_variance, predictions, targets):
    # compute the average MSE error, then scale it up i.e. simply sum on all axes
    r_loss = torch.sum((predictions - targets).pow(2), dim=(1, 2, 3))
    r_loss = torch.mean(r_loss)
    # compute the KL loss
    kl_loss = -0.5 * torch.sum(1 + log_variance - mean.pow(2) - log_variance.exp(), dim=-1)
    kl_loss = torch.maximum(kl_loss, torch.tensor(self.kl_tolerance * self.z_size))
    # return the average loss over all images in batch
    total_loss = r_loss + torch.mean(kl_loss)
    return total_loss

  def _compute_kl_loss(self, mean, log_variance): 
      return -0.5 * torch.sum(1 + log_variance - mean.pow(2) - log_variance.exp())

  # def sample(self, args):
  #       mu, log_variance = args

  #       std = torch.exp(log_variance / 2.0)
  #       # define a distribution q with the parameters mu and std
  #       q = torch.distributions.Normal(mu, std)
  #       # sample z from q
  #       z = q.rsample()

  #       return z

  def forward(self, x):
    h = self.encoder(x)
    h = torch.reshape(h, [-1, 2*2*256])
    mu = self.mu(h)
    logvar = self.logvar(h)
    sigma = torch.exp(logvar / 2.0)
    epsilon = self.N.sample(mu.shape)

    z = mu + sigma * epsilon

    d = self.decoderdense(z)
    d = torch.reshape(d, [-1, 4*256, 1, 1])
    d = self.decoder(d)
    d = torch.sigmoid(d)
    
    # r_loss, kl_loss, log_sigma_opt= self.calculate_loss(mu, logvar, d, x)
    # loss = r_loss+kl_loss
    loss = self.calculate_loss(mu, logvar, d, x)
    # print(loss)
    # if torch.isnan(loss).any():
    #     print('loss')
    return d, z, loss

def tf_lognormal(y, mean, logstd):
      return -0.5 * ((y - mean) / torch.exp(logstd)) ** 2 - logstd - (np.log(np.sqrt(2.0 * np.pi)))

def mdn_loss_fn(y, logmix, mean, logstd):
    v = logmix + tf_lognormal(y, mean, logstd)
    v = torch.logsumexp(v, 1, keepdims=True)
    return -v.mean()

def criterion(y, logmix, mean, logstd):
    return mdn_loss_fn(y, logmix, mean, logstd)


class MDNRNN(nn.Module):
    def __init__(self, in_z_size, out_z_size, n_hidden=256, n_gaussians=5, n_layers=1):
        super(MDNRNN, self).__init__()

        self.out_z_size = out_z_size
        self.in_z_size = in_z_size
        self.n_hidden = n_hidden
        self.n_gaussians = n_gaussians
        self.n_layers = n_layers
        
        self.lstm = nn.LSTM(in_z_size, n_hidden, n_layers, batch_first=True)
        self.fc1 = nn.Linear(n_hidden, n_gaussians*out_z_size*3)

        
    # def get_mixture_coef(self, y):
    #     rollout_length = y.size(1)
    #     pi, mu, sigma = self.fc1(y), self.fc2(y), self.fc3(y)
        
    #     pi = pi.view(-1, rollout_length, self.n_gaussians, self.out_z_size)
    #     mu = mu.view(-1, rollout_length, self.n_gaussians, self.out_z_size)
    #     sigma = sigma.view(-1, rollout_length, self.n_gaussians, self.out_z_size)
        
    #     pi = F.softmax(pi, 2)
    #     sigma = torch.exp(sigma)
    #     return pi, mu, sigma
        
    # def get_mixture_coef(self, y):
    #     rollout_length = y.size(1)
    #     logmix, mu, sigma = self.fc1(y), self.fc2(y), self.fc3(y)
        
    #     logmix = logmix.view(-1, rollout_length, self.n_gaussians, self.out_z_size)
    #     mu = mu.view(-1, rollout_length, self.n_gaussians, self.out_z_size)
    #     sigma = sigma.view(-1, rollout_length, self.n_gaussians, self.out_z_size)
        
    #     logmix = logmix - torch.logsumexp(logmix, 1, keepdim=True)

    #     return logmix, mu, sigma
    def get_mixture_coef(self, output):
        logmix, mean, logstd = torch.split(output, self.n_gaussians, dim=1)
        logmix = logmix - torch.logsumexp(logmix, 1, keepdims=True)
        return logmix, mean, logstd

        
        
    def forward(self, x, h, targets):
        # Forward propagate LSTM
        y, (h, c) = self.lstm(x, h)
        y = y.view(-1, self.n_hidden)
        output = self.fc1(y)

        output = output.view(-1, self.n_gaussians*3)

        logmix, mean, logstd = self.get_mixture_coef(output)

        flat_target = targets.view(-1, 1)

        loss = criterion(flat_target, logmix, mean, logstd)
        return (logmix, mean, logstd), (h, c), loss
    
    def init_hidden(self, bsz):
        return (torch.zeros(self.n_layers, bsz, self.n_hidden).cuda(),
                torch.zeros(self.n_layers, bsz, self.n_hidden).cuda())


class AENetwork(A3CTemplate):
    """
    An network with AE comm.
    """
    def __init__(self, obs_space, act_space, num_agents, comm_len,
                 discrete_comm, ae_pg=0, ae_type='', hidden_size=256,
                 img_feat_dim=64, load_comm=True):
        super().__init__()

        # assume action space is a Tuple of 2 spaces
        self.env_action_size = act_space[0].n  # Discrete
        self.action_size = self.env_action_size
        self.ae_pg = ae_pg
        self.comm_len=comm_len
        self.counter = 0
        
        self.num_agents = num_agents
        #comm_len is 96
        self.conv_vae = ConvVAE(z_size=comm_len, kl_tolerance=0.5)

        # individual memories
        self.feat_dim =  self.comm_len + self.action_size
        self.head = nn.ModuleList([MDNRNN(self.feat_dim, self.comm_len) for _ in range(num_agents)])
        self.is_recurrent = True

        if load_comm:
            model = AENetwork(obs_space, act_space, num_agents, comm_len, discrete_comm, ae_pg=0, ae_type='', hidden_size=256, img_feat_dim=64, load_comm=False)
            chkpt = torch.load('commnet.pth')
            model.load_state_dict(chkpt['net'])
            self.conv_vae = model.conv_vae
            model2 = AENetwork(obs_space, act_space, num_agents, comm_len, discrete_comm, ae_pg=0, ae_type='', hidden_size=256, img_feat_dim=64, load_comm=False)
            chkpt = torch.load('lstmnet.pth')
            model2.load_state_dict(chkpt['net'])
            self.head = model2.head

        # if not load_comm:
        # self.LinearShrink = nn.Linear(hidden_size, img_feat_dim)

        # # separate AC for env action and comm action
        # self.env_critic_linear = nn.ModuleList([nn.Linear(
        #     hidden_size+comm_len+img_feat_dim, 1) for _ in range(num_agents)])

        # self.env_actor_linear = nn.ModuleList([nn.Linear(
        #     hidden_size+comm_len+img_feat_dim, self.env_action_size) for _ in range(num_agents)])
        # else:
        self.LinearShrink = nn.Linear(hidden_size, img_feat_dim)

        # separate AC for env action and comm action
        self.env_critic_linear = nn.ModuleList([nn.Linear(
            hidden_size+comm_len+hidden_size, 1) for _ in range(num_agents)])

        self.env_actor_linear = nn.ModuleList([nn.Linear(
            hidden_size+comm_len+hidden_size, self.env_action_size) for _ in range(num_agents)])

        self.reset_parameters()
        return

    def reset_parameters(self):
        for m in self.env_actor_linear:
            m.weight.data = normalized_columns_initializer(
                m.weight.data, 0.01)
            m.bias.data.fill_(0)

        for m in self.env_critic_linear:
            m.weight.data = normalized_columns_initializer(
                m.weight.data, 1.0)
            m.bias.data.fill_(0)
        return

    def init_hidden(self):
        return [head.init_hidden(1) for head in self.head]

    def take_action(self, policy_logit, comm_out):
        act_dict = {}
        act_logp_dict = {}
        ent_list = []
        all_act_dict = {}
        for agent_name, logits in policy_logit.items():

            act, act_logp, ent = super(AENetwork, self).take_action(logits)

            act_dict[agent_name] = act
            act_logp_dict[agent_name] = act_logp
            ent_list.append(ent)
            #TODO: REMOVE   
            comm_act = (comm_out[int(agent_name[-1])]).cpu().numpy()
            all_act_dict[agent_name] = [act, comm_act]
        return act_dict, act_logp_dict, ent_list, all_act_dict

    def forward(self, inputs, hidden_state=None, env_mask_idx=None):
        assert type(inputs) is dict
        assert len(inputs.keys()) == self.num_agents + 1  # agents + global


        pov = []
        for i in range(self.num_agents):
            pov.append(inputs[f'agent_{i}']['pov'])
        xs = torch.cat(pov, dim=0)

        decoded, comm_out, ae_loss = self.conv_vae(xs)
        comm_out = comm_out.clone().detach()

        # (3) predict policy and values separately
        env_actor_out, env_critic_out = {}, {}
        ys = []
        for i, agent_name in enumerate(inputs.keys()):
            if agent_name == 'global':
                continue
            
            env_act = F.one_hot(
                inputs[f'agent_{i}']['self_env_act'].to(torch.int64),
                num_classes=self.action_size)
            env_act = torch.reshape(env_act, (1, self.action_size))
            cat_feat = torch.cat([inputs[f'agent_{i}']['comm'][-1].unsqueeze(0),env_act],
                                    dim=-1).unsqueeze(0)

            #insert zt-1 and at-1 with ht-1 to model zt and output ht with loss
            x, hidden_state[i], lstm_loss = self.head[i](cat_feat, hidden_state[i], comm_out[i].view(1, -1, self.comm_len))
            ys.append(x)

        ds = []
        for l in range(5):
            y_preds = torch.cat([torch.normal(mu, torch.exp(sigma))[:, l].unsqueeze(0) for pi, mu, sigma, in ys], dim=0)
            print(comm_out.shape)
            print(y_preds.shape)

            d = self.conv_vae.decoderdense(y_preds)
            d = torch.reshape(d, [-1, 4*256, 1, 1])
            d = self.conv_vae.decoder(d)
            d = torch.sigmoid(d)
            ds.append(d)


        mapped_hs = torch.cat([p[0].clone().detach() for p in hidden_state], dim=0).cuda()
        #current_hs = [p[0].clone().detach() for p in hidden_state]
        
        # mapped_hs = self.LinearShrink(mapped_hs)
        # self.counter += 1
        # if self.counter > 4000:
        #     print("SAVING")
        save_image(xs.cpu(), 'og.png')
        save_image(decoded.cpu(), 'decoded.png')
            # self.counter = 0
        for l in range(5):
            save_image(ds[l].clone().detach().cpu(), f'lstm_{l}.png')

        for i, agent_name in enumerate(inputs.keys()):
            otheragent = 1 if i==0 else 0
            if agent_name == 'global':
                continue
            # predict next action at with ht, zt, and commht-1
            handz = torch.cat([mapped_hs[i].squeeze(), comm_out[i], mapped_hs[otheragent].squeeze()]).unsqueeze(0)
            env_actor_out[agent_name] = self.env_actor_linear[i](handz)
            env_critic_out[agent_name] = self.env_critic_linear[i](handz)
            # mask logits of unavailable actions if provided
            if env_mask_idx and env_mask_idx[i]:
                env_actor_out[agent_name][0, env_mask_idx[i]] = -1e10
        return env_actor_out, env_critic_out, hidden_state, \
               comm_out.detach(), (ae_loss, lstm_loss)
