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


class InputProcessor(nn.Module):
    """
    Pre-process the following individual observations:
        - pov (ImgModule)
        - self_env_act
        - selfpos
    """
    def __init__(self, obs_space, comm_feat_len, num_agents, last_fc_dim=64):
        super(InputProcessor, self).__init__()

        self.obs_keys = list(obs_space.spaces.keys())
        self.num_agents = num_agents

        # image processor
        assert 'pov' in self.obs_keys
        self.conv = ImgModule(obs_space['pov'].shape, last_fc_dim=last_fc_dim)
        feat_dim = last_fc_dim

        # state inputs processor
        state_feat_dim = 0

        if 'self_env_act' in self.obs_keys:
            # discrete value with one-hot encoding
            self.env_act_dim = obs_space.spaces['self_env_act'].n
            state_feat_dim += self.env_act_dim

        if 'selfpos' in self.obs_keys:
            self.discrete_positions = None
            if obs_space.spaces['selfpos'].__class__.__name__ == \
                    'MultiDiscrete':
                # process position with one-hot encoder
                self.discrete_positions = obs_space.spaces['selfpos'].nvec
                state_feat_dim += sum(self.discrete_positions)
            else:
                state_feat_dim += 2

        if state_feat_dim == 0:
            self.state_feat_fc = None
        else:
            # use state_feat_fc to process concatenated state inputs
            self.state_feat_fc = nn.Linear(state_feat_dim, 64)
            feat_dim += 64

        if self.state_feat_fc:
            self.state_layer_norm = nn.LayerNorm(64)
        self.img_layer_norm = nn.LayerNorm(last_fc_dim)

        # all other agents' decoded features, if provided
        self.comm_feat_dim = comm_feat_len * (num_agents - 1)
        feat_dim += self.comm_feat_dim

        self.feat_dim = feat_dim

    def forward(self, inputs, comm=None):
        # WARNING: the following code only works for Python 3.6 and beyond

        # process images together if provided
        if 'pov' in self.obs_keys:
            pov = []
            for i in range(self.num_agents):
                pov.append(inputs[f'agent_{i}']['pov'])
            x = torch.cat(pov, dim=0)
            print(x.shape)
            x = self.conv(x)  # (N, img_feat_dim)
            xs = torch.chunk(x, self.num_agents)

        # concatenate observation features
        cat_feat = [self.img_layer_norm(xs[i]) for i in range(self.num_agents)]

        if self.state_feat_fc is None:
            if comm is not None:
                for i in range(self.num_agents):
                    # concat comm features for each agent
                    c = torch.reshape(comm[i], (1, self.comm_feat_dim))
                    cat_feat[i] = torch.cat([cat_feat[i], c], dim=-1)
            return cat_feat

        for i in range(self.num_agents):
            # concatenate state features
            feats = []

            # concat last env act if provided
            if 'self_env_act' in self.obs_keys:
                env_act = F.one_hot(
                    inputs[f'agent_{i}']['self_env_act'].to(torch.int64),
                    num_classes=self.env_act_dim)
                env_act = torch.reshape(env_act, (1, self.env_act_dim))
            # concat agent's own position if provided
            if 'selfpos' in self.obs_keys:
                sp = inputs[f'agent_{i}']['selfpos'].to(torch.int64)  # (2,)
                if self.discrete_positions is not None:
                    spx = F.one_hot(sp[0],
                                    num_classes=self.discrete_positions[0])
                    spy = F.one_hot(sp[1],
                                    num_classes=self.discrete_positions[1])
                    sp = torch.cat([spx, spy], dim=-1).float()
                    sp = torch.reshape(sp, (1, sum(self.discrete_positions)))
                else:
                    sp = torch.reshape(sp, (1, 2))
                feats.append(sp)

            if len(feats) > 1:
                feats = torch.cat(feats, dim=-1)
            elif len(feats) == 1:
                feats = feats[0]
            else:
                raise ValueError('?!?!?!', feats)

            feats = self.state_feat_fc(feats)
            feats = self.state_layer_norm(feats)
            cat_feat[i] = torch.cat([cat_feat[i], feats], dim=-1)

            if comm is not None:
                # concat comm features for each agent
                c = torch.reshape(comm[i], (1, self.comm_feat_dim))
                cat_feat[i] = torch.cat([cat_feat[i], c], dim=-1)

        return cat_feat




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
        x = torch.sigmoid(x)
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
        loss = F.mse_loss(decoded, x)
        return z.detach(), loss


class AENetwork(A3CTemplate):
    """
    An network with AE comm.
    """
    def __init__(self, obs_space, act_space, num_agents, comm_len,
                 discrete_comm, ae_pg=0, ae_type='', hidden_size=256,
                 img_feat_dim=64):
        super().__init__()
        print(obs_space)
        self.id = obs_space['identity']
        # assume action space is a Tuple of 2 spaces
        self.env_action_size = act_space[0].n  # Discrete
        self.action_size = self.env_action_size
        self.ae_pg = ae_pg

        self.num_agents = num_agents

        self.comm_ae = EncoderDecoder(obs_space, comm_len, discrete_comm,
                                      num_agents, ae_type=ae_type,
                                      img_feat_dim=img_feat_dim)

        self.preprocessor = InputProcessor(obs_space, 0, num_agents,
                                           last_fc_dim=img_feat_dim)
        feat_dim = self.preprocessor.feat_dim

        if ae_type == '':
            self.input_processor = InputProcessor(
                obs_space,
                feat_dim,
                num_agents,
                last_fc_dim=img_feat_dim)
        else:
            self.input_processor = InputProcessor(
                obs_space,
                comm_len,
                num_agents,
                last_fc_dim=img_feat_dim)

        # individual memories
        self.feat_dim = self.input_processor.feat_dim + comm_len
        self.head = nn.ModuleList(
            [LSTMhead(128, hidden_size, num_layers=1
                      ) for _ in range(num_agents)])
        self.is_recurrent = True

        # separate AC for env action and comm action
        self.env_critic_linear = nn.ModuleList([nn.Linear(
            hidden_size, 1) for _ in range(num_agents)])
        self.env_actor_linear = nn.ModuleList([nn.Linear(
            hidden_size, self.env_action_size) for _ in range(num_agents)])

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
        comm_feat = []
        for i in range(self.num_agents):
            cf = self.comm_ae.decode(inputs[f'agent_{i}']['comm'][:-1])
            if not self.ae_pg:
                cf = cf.detach()
            comm_feat.append(cf)

        cat_feat = self.input_processor(inputs)

        # (2) generate AE comm output and reconstruction loss
        with torch.no_grad():
            x = self.input_processor(inputs)
        x = torch.cat(x, dim=0)
        pov = []
        for i in range(self.num_agents):
            pov.append(inputs[f'agent_{i}']['pov'])
        xs = torch.cat(pov, dim=0)
        comm_out, comm_ae_loss = self.comm_ae(xs)

        # (3) predict policy and values separately
        env_actor_out, env_critic_out = {}, {}

        for i, agent_name in enumerate(inputs.keys()):
            if agent_name == 'global':
                continue

            cat_feat[i] = torch.cat([cat_feat[i], comm_out[i].unsqueeze(0)],
                                    dim=-1)

            x, hidden_state[i] = self.head[i](cat_feat[i], hidden_state[i])

            env_actor_out[agent_name] = self.env_actor_linear[i](x)
            env_critic_out[agent_name] = self.env_critic_linear[i](x)

            # mask logits of unavailable actions if provided
            if env_mask_idx and env_mask_idx[i]:
                env_actor_out[agent_name][0, env_mask_idx[i]] = -1e10

        return env_actor_out, env_critic_out, hidden_state, \
               comm_out.detach(), comm_ae_loss
