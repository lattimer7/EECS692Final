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

from model.a3c_template import A3CTemplate, take_action, take_comm_action
from model.init import normalized_columns_initializer, weights_init
from model.model_utils import LSTMhead, ImgModule

class Encoder(nn.Module):
    def __init__(self, in_channels=3):
        super(Encoder, self).__init__()

        ##############################################################################
        # TODO: Build an encoder with the architecture as specified above.           #
        ##############################################################################
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 12, (4,4), stride=(2,2), padding=(1,1)),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(12, 24, (4,4),stride=(2,2), padding=(1,1)),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(24, 48, (4,4), stride=(2,2), padding=(1,1)),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(48, 24, (4,4), stride=(1,1), padding=(1,1)),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(24, 12, (4,4), stride=(2,2), padding=(1,1)),
            torch.nn.ReLU(inplace=True)
        )

        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

    def forward(self, x):
        '''
        Given an image x, return the encoded latent representation h.

        Args:
            x: torch.tensor

        Return: 
            h: torch.tensor
        '''
        
        h = self.encoder(x)
        h = torch.flatten(h, start_dim=1)
        return h

class Decoder(nn.Module):
    def __init__(self, out_channels=3, feat_dim=64):
        super(Decoder, self).__init__()

        ##############################################################################
        # TODO: Build the decoder as specified above.                                #
        ##############################################################################
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(12, 24, (4,4), stride=(2,2), padding=(1,1)),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(24, 48, (4,4), stride=(1,1), padding=(1,1)),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(48, 24, (4,4), stride=(2,2), padding=(1,1)),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(24, 12, (4,4),stride=(2,2), padding=(1,1)),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(12, 3, (4,4), stride=(2,2), padding=(1,1))
        )
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(12,2,2))
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

    def forward(self, h):
        '''
        Given latent representation h, reconstruct an image patch of size 64 x 64.

        Args:
            h: torch.tensor

        Return: 
            x: torch.tensor
        '''
        h = self.unflatten(h)
        x = self.decoder(h)
        x = torch.sigmoid(x)
        return x

class Autoencoder(nn.Module):
    def __init__(self, in_channels=3, feat_dim=64):
        super(Autoencoder, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        '''
        Compress and reconstruct the input image with encoder and decoder.

        Args:
            x: torch.tensor

        Return: 
            x_: torch.tensor
        '''

        z = self.encoder(x)
        decoded = self.decoder(z)
        loss = F.mse_loss(decoded, x)
        return z.detach(), loss

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
        
        self.num_agents = num_agents
        #comm_len is 96
        self.comm_ae = Autoencoder()
        if load_comm:
            model = AENetwork(obs_space, act_space, num_agents, comm_len, discrete_comm, ae_pg=0, ae_type='', hidden_size=256, img_feat_dim=64, load_comm=False)
            chkpt = torch.load('latest.pth')
            model.load_state_dict(chkpt['net'])
            self.comm_ae = model.comm_ae
            self.comm_ae.eval()

        self.LinearShrink = nn.Linear(hidden_size, img_feat_dim)

        # individual memories
        self.feat_dim =  comm_len + self.action_size
        self.head = nn.ModuleList(
            [LSTMhead(self.feat_dim, hidden_size, comm_len, num_layers=1
                      ) for _ in range(num_agents)])
        self.is_recurrent = True

        # separate AC for env action and comm action
        self.env_critic_linear = nn.ModuleList([nn.Linear(
            hidden_size+comm_len+img_feat_dim, 1) for _ in range(num_agents)])

        self.env_actor_linear = nn.ModuleList([nn.Linear(
            hidden_size+comm_len+img_feat_dim, self.env_action_size) for _ in range(num_agents)])

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
            #TODO: REMOVE   
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
        #    x = self.input_processor(inputs)
        #x = torch.cat(x, dim=0)
        #print(f'Max: {x.max()}, Min: {x.min()}')
        pov = []
        for i in range(self.num_agents):
            pov.append(inputs[f'agent_{i}']['pov'])
        xs = torch.cat(pov, dim=0)
        #xs = xs/255 if xs.max() > 2 else ((xs+1)/2)
        with open('lep.pkl', 'wb') as filz:
            pickle.dump(xs, filz)
        comm_out, comm_ae_loss = self.comm_ae(xs)
        comm_out = comm_out.detach()
        with open('lepord.pkl', 'wb') as filz:
            pickle.dump(self.comm_ae.decoder(comm_out), filz)
        #comm_ae_loss = comm_ae_loss.detach()

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
        
        mapped_hs = torch.cat([p[0].clone().detach() for p in hidden_state], dim=0).cuda()
        current_hs = [p[0].clone().detach() for p in hidden_state]
        
        mapped_hs = self.LinearShrink(mapped_hs)

        for i, agent_name in enumerate(inputs.keys()):
            otheragent = 1 if i==0 else 0
            if agent_name == 'global':
                continue
            # predict next action at with ht, zt, and commht-1
            handz = torch.cat([current_hs[i].squeeze(), comm_out[i], mapped_hs[otheragent].squeeze()]).unsqueeze(0)
            env_actor_out[agent_name] = self.env_actor_linear[i](handz)
            env_critic_out[agent_name] = self.env_critic_linear[i](handz)
            # mask logits of unavailable actions if provided
            if env_mask_idx and env_mask_idx[i]:
                env_actor_out[agent_name][0, env_mask_idx[i]] = -1e10

        return env_actor_out, env_critic_out, hidden_state, \
               comm_out.detach(), comm_ae_loss, lstm_loss
