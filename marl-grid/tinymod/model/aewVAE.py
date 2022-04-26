from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.a3c_template import A3CTemplate, take_action, take_comm_action
from model.init import normalized_columns_initializer, weights_init
from model.model_utils import LSTMhead, ImgModule
class Encoder(nn.Module):
    def __init__(self, input_size, last_fc_dim=64):  
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(input_size[2], 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, stride=2, padding=0)
        self.linear1 = nn.Linear(1024, 128)
        self.linear2 = nn.Linear(128, last_fc_dim)
        self.linear3 = nn.Linear(128, last_fc_dim)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu =  self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z      

class Decoder(nn.Module):
    
    def __init__(self, input_size, last_fc_dim=64):
        super().__init__()

        self.decoder_lin = nn.Sequential(
            nn.Linear(last_fc_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 256*2*2),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(256, 2, 2))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, output_padding=0),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, input_size[2], 3, stride=2, padding=1, output_padding=1)
        )
        
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        return x

class EncoderDecoder(nn.Module):
    def __init__(self, obs_space, comm_len, discrete_comm, num_agents,
                 ae_type='', img_feat_dim=64):
        super(EncoderDecoder, self).__init__()
        in_size = obs_space['pov'].shape
        self.encoder = Encoder(obs_space['pov'].shape, last_fc_dim=img_feat_dim)
        self.decoder = Decoder(obs_space['pov'].shape, last_fc_dim=img_feat_dim)

    def decode(self, x):
        """
        input: inputs[f'agent_{i}']['comm'] (num_agents, comm_len)
            (note that agent's own state is at the last index)
        """
        return self.decoder(x)  # (num_agents, in_size)

    def forward(self, x):
        z = self.encoder(x)
        decoded = self.decoder(z)
        loss = F.mse_loss(decoded, x) + self.encoder.kl
        return z.detach(), loss


class AENetwork(A3CTemplate):
    """
    An network with AE comm.
    """
    def __init__(self, obs_space, act_space, num_agents, comm_len,
                 discrete_comm, ae_pg=0, ae_type='', hidden_size=256,
                 img_feat_dim=64):
        super().__init__()

        # assume action space is a Tuple of 2 spaces
        self.env_action_size = act_space[0].n  # Discrete
        self.action_size = self.env_action_size
        self.ae_pg = ae_pg

        self.num_agents = num_agents

        self.comm_ae = EncoderDecoder(obs_space, comm_len, discrete_comm,
                                      num_agents, ae_type=ae_type,
                                      img_feat_dim=comm_len)
        # individual memories
        self.feat_dim =  comm_len + self.action_size
        self.head = nn.ModuleList(
            [LSTMhead(self.feat_dim, hidden_size, comm_len, num_layers=1
                      ) for _ in range(num_agents)])
        self.is_recurrent = True

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
        return [head.init_hidden() for head in self.head]

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

            comm_act = (comm_out[int(agent_name[-1])]).cpu().numpy()
            all_act_dict[agent_name] = [act, comm_act]
        return act_dict, act_logp_dict, ent_list, all_act_dict

    def forward(self, inputs, hidden_state=None, env_mask_idx=None):
        assert type(inputs) is dict
        assert len(inputs.keys()) == self.num_agents + 1  # agents + global

        # WARNING: the following code only works for Python 3.6 and beyond

        # (1) pre-process inputs
        #comm_feat = []
        #for i in range(self.num_agents):
            # TODO: How do I hijack the comm vector?
            #cf = self.comm_ae.decode(inputs[f'agent_{i}']['comm'][:-1])
            #if not self.ae_pg:
            #    cf = cf.detach()
            #comm_feat.append(cf)

        #cat_feat = self.input_processor(inputs, comm_feat)
        # (2) generate AE comm output and reconstruction loss
        #with torch.no_grad():
        #x = self.input_processor(inputs)
        #x = torch.cat(x, dim=0)
        pov = []
        for i in range(self.num_agents):
            pov.append(inputs[f'agent_{i}']['pov'])
        xs = torch.cat(pov, dim=0)
        
        comm_out, comm_ae_loss = self.comm_ae(xs)
        comm_out = comm_out.detach()
        # (3) predict policy and values separately
        env_actor_out, env_critic_out = {}, {}
        for i, agent_name in enumerate(inputs.keys()):
            if agent_name == 'global':
                continue
            
            env_act = F.one_hot(
                inputs[f'agent_{i}']['self_env_act'].to(torch.int64),
                num_classes=self.action_size)
            env_act = torch.reshape(env_act, (1, self.action_size))
            cat_feat = torch.cat([inputs[f'agent_{i}']['comm'][-1].unsqueeze(0),env_act],
                                    dim=-1)
            
            #insert zt-1 and at-1 with ht-1 to model zt and output ht with loss
            x, hidden_state[i], lstm_loss = self.head[i](cat_feat, hidden_state[i], comm_out[i])
        
        otherhiddenstates = [p[0].clone().detach() for p in hidden_state]
        
        for i, agent_name in enumerate(inputs.keys()):
            otheragent = 1 if i==0 else 0
            if agent_name == 'global':
                continue
            # predict next action at with ht, zt, and commht-1
            handz = torch.cat([otherhiddenstates[i].squeeze(), comm_out[i], otherhiddenstates[otheragent].squeeze()]).unsqueeze(0)
            env_actor_out[agent_name] = self.env_actor_linear[i](handz)
            env_critic_out[agent_name] = self.env_critic_linear[i](handz)

            # mask logits of unavailable actions if provided
            if env_mask_idx and env_mask_idx[i]:
                env_actor_out[agent_name][0, env_mask_idx[i]] = -1e10

        return env_actor_out, env_critic_out, hidden_state, \
               comm_out.detach(), comm_ae_loss, lstm_loss
