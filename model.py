# coding=utf-8

import torch
import torch.nn as nn

import torch.nn.functional as F
from torch.autograd import Variable

import torch.backends.cudnn as cudnn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.clip_grad import clip_grad_norm_
import math
import numpy as np
from collections import OrderedDict

def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X"""
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X

def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X"""
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

def cosine_sim(x1, x2, dim=-1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()

class EncoderImage(nn.Module):
    """
    Build local region representations by common-used FC-layer.
    Args: - images: raw local detected regions, shape: (batch_size, 36, 2048).
    Returns: - img_emb: finial local region embeddings, shape:  (batch_size, 36, 1024).
    """
    def __init__(self, img_dim, embed_size, no_imgnorm=False):
        super(EncoderImage, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = nn.Linear(img_dim, embed_size)

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer"""
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized
        # 2048—>1024
        img_emb = self.fc(images)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            img_emb = l2norm(img_emb, dim=-1)

        return img_emb

    def load_state_dict(self, state_dict):
        """Overwrite the default one to accept state_dict from Full model"""
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImage, self).load_state_dict(new_state)

class EncoderText(nn.Module):
    """
    Build local word representations by common-used Bi-GRU or GRU.
    Args: - images: raw local word ids, shape: (batch_size, L).
    Returns: - img_emb: final local word embeddings, shape: (batch_size, L, 1024).
    """
    def __init__(self, vocab_size, word_dim, embed_size, num_layers,
                 use_bi_gru=False, no_txtnorm=False):
        super(EncoderText, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)
        self.dropout = nn.Dropout(0.4)

        # caption embedding
        self.use_bi_gru = use_bi_gru
        self.cap_rnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True, bidirectional=use_bi_gru)

        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, captions, lengths):
        # print captions
        """Handles variable size captions"""
        # embed word ids to vectors
        cap_emb = self.embed(captions)

        cap_emb = self.dropout(cap_emb)

        # pack the caption
        packed = pack_padded_sequence(cap_emb, lengths, batch_first=True)

        # forward propagate RNN
        out, _ = self.cap_rnn(packed)

        # reshape output to (batch_size, hidden_size)
        cap_emb, _ = pad_packed_sequence(out, batch_first=True)

        if self.use_bi_gru:
            cap_emb = (cap_emb[:, :, :cap_emb.size(2)/2] + cap_emb[:, :, cap_emb.size(2)/2:])/2

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            cap_emb = l2norm(cap_emb, dim=-1)

        return cap_emb


class MHAtt(nn.Module):
    def __init__(self,sim_dim):
        super(MHAtt, self).__init__()
        self.sim_dim = sim_dim
        self.linear_v = nn.Linear(sim_dim, sim_dim)
        self.linear_k = nn.Linear(sim_dim, sim_dim)
        self.linear_q = nn.Linear(sim_dim, sim_dim)
        self.linear_merge = nn.Linear(sim_dim, sim_dim)

        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.init_weights()

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = self.linear_v(v).view(n_batches,-1,8,self.sim_dim/8).transpose(1, 2)

        k = self.linear_k(k).view(n_batches,-1,8,self.sim_dim/8).transpose(1, 2)

        q = self.linear_q(q).view(n_batches,-1,8,self.sim_dim/8).transpose(1, 2)

        atted = self.att(v, k, q, mask)

        atted = atted.transpose(1, 2).contiguous().view(n_batches,-1,self.sim_dim)

        atted = self.linear_merge(atted)

        atted=self.relu(atted)

        return atted

    def init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)

class EncoderSimilarity(nn.Module):
    def __init__(self, embed_size, sim_dim):
        super(EncoderSimilarity, self).__init__()

        self.glob_v_w = nn.Linear(embed_size,1024)
        self.glob_t_w = nn.Linear(embed_size,1024)
        self.loc_v_w = nn.Linear(embed_size,1024)
        self.loc_t_w = nn.Linear(embed_size,1024)

        self.sim_eval_w = nn.Linear(sim_dim, 1)
        self.sigmoid = nn.Sigmoid()

        self.MHAttMFB_module = MHAtt(sim_dim)

        self.init_weights()

    def forward(self, img_emb, cap_emb, cap_lens):
        sim_all = []
        n_image = img_emb.size(0)
        n_caption = cap_emb.size(0)


        img_glo = torch.mean(img_emb, 1)
        img_glo = l2norm(img_glo, dim=-1)

        for i in range(n_caption):

            # get the i-th sentence
            n_word = cap_lens[i]
            cap_i = cap_emb[i, :n_word, :].unsqueeze(0)
            cap_i_expand = cap_i.repeat(n_image, 1, 1)


            cap_glo_i = torch.mean(cap_i, 1)

            Context_img = SCAN_attention(cap_i_expand, img_emb, smooth=9.0)

            context_img_loc = self.loc_v_w(Context_img)
            context_loc = self.loc_t_w(cap_i_expand)


            loc_contimg_sim = torch.mul(context_img_loc, context_loc)

            loc_contimg_sim= loc_contimg_sim.view(loc_contimg_sim.size(0), loc_contimg_sim.size(1), 256, 4)
            loc_contimg_sim = torch.sum(loc_contimg_sim, 3)  # sum pool
            loc_contimg_sim = torch.sqrt(F.relu(loc_contimg_sim)) - torch.sqrt(F.relu(-loc_contimg_sim))  # signed sqrt
            sim_loc = l2norm(loc_contimg_sim, dim=-1)


            img_glob = self.glob_v_w(img_glo)
            cap_glob = self.glob_t_w(cap_glo_i)
            glob_contimg_sim = torch.mul(img_glob, cap_glob)

            glob_contimg_sim = glob_contimg_sim.view(glob_contimg_sim.size(0), 256, 4)
            glob_contimg_sim = torch.sum(glob_contimg_sim, dim=2)  # sum pool
            glob_contimg_sim = torch.sqrt(F.relu(glob_contimg_sim)) - torch.sqrt(F.relu(-glob_contimg_sim))  # signed sqrt
            sim_glo = l2norm(glob_contimg_sim, dim=-1)


            sim_emb = torch.cat([sim_glo.unsqueeze(1), sim_loc], 1)


            sim_emb = self.MHAttMFB_module(sim_emb, sim_emb, sim_emb, None)

            sim_vec = sim_emb[:, 0, :]

            sim_i = self.sigmoid(self.sim_eval_w(sim_vec))
            sim_all.append(sim_i)

        sim_all = torch.cat(sim_all, 1)
        return sim_all

    def init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def SCAN_attention(query, context, smooth, eps=1e-8):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    # --> (batch, d, queryL)
    queryT = torch.transpose(query, 1, 2)

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    attn = torch.bmm(context, queryT)

    attn = nn.LeakyReLU(0.1)(attn)
    attn = l2norm(attn, 2)

    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch, queryL, sourceL
    attn = F.softmax(attn*smooth, dim=2)


    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, attnT)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)
    weightedContext = l2norm(weightedContext, dim=-1)

    return weightedContext

class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self, margin=0, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.max_violation = max_violation

    def forward(self, scores):
        # compute image-sentence score matrix
        diagonal = scores.diag().view(scores.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        if torch.cuda.is_available():
            I = mask.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
        return cost_s.sum() + cost_im.sum()

class BCELoss(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self, max_violation=True):
        super(BCELoss, self).__init__()
        self.max_violation = max_violation

    def forward(self, scores):
        # scores = scores.transpose(1, 0)
        eps = 0.000001

        scores = scores.clamp(min=eps, max=(1.0 - eps))
        de_scores = 1.0 - scores
        label = Variable(torch.eye(scores.size(0))).cuda()

        de_label = 1 - label

        scores = torch.log(scores) * label
        de_scores = torch.log(de_scores) * de_label

        if self.max_violation:

            loss = -(scores.sum() + scores.sum() + de_scores.min(1)[0].sum() + de_scores.min(0)[0].sum())
        else:
            loss = -(scores.diag().mean() + de_scores.mean())

        return loss

class MGLMN(object):

    def __init__(self, opt):
        # Build Models
        self.grad_clip = opt.grad_clip
        self.img_enc = EncoderImage(opt.img_dim, opt.embed_size,
                                    no_imgnorm=opt.no_imgnorm)
        self.txt_enc = EncoderText(opt.vocab_size, opt.word_dim,
                                   opt.embed_size, opt.num_layers, 
                                   use_bi_gru=opt.bi_gru,  
                                   no_txtnorm=opt.no_txtnorm)

        self.sim_enc = EncoderSimilarity(opt.embed_size, opt.sim_dim,
                                         opt.module_name, opt.sgr_step)

        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            self.sim_enc.cuda()
            cudnn.benchmark = True

        # Loss and Optimizer
        # if opt.loss == "rank":
        #     self.criterion = ContrastiveLoss(margin=opt.margin, max_violation=opt.max_violation)
        # else:
        #     self.criterion = BCELoss(max_violation=opt.max_violation)
        # self.criterion = BCELoss(max_violation=opt.max_violation)
        self.criterion = ContrastiveLoss(margin=opt.margin, max_violation=opt.max_violation)
        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.parameters())
        params += list(self.sim_enc.parameters())
        self.params = params

        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)

        self.Eiters = 0

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict(), self.sim_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])
        self.sim_enc.load_state_dict(state_dict[2])

    def train_start(self):
        """switch to train mode"""
        self.img_enc.train()
        self.txt_enc.train()
        self.sim_enc.train()

    def val_start(self):
        """switch to evaluate mode"""
        self.img_enc.eval()
        self.txt_enc.eval()
        self.sim_enc.eval()

    def forward_emb(self, images, captions, lengths):
        """Compute the image and caption embeddings"""
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()

        # Forward feature encoding
        # 2048->1024
        img_embs = self.img_enc(images)
        # 300->1024
        cap_embs = self.txt_enc(captions, lengths)
        return img_embs, cap_embs, lengths

    def forward_sim(self, img_embs, cap_embs, cap_lens):
        # Forward similarity encoding
        sims = self.sim_enc(img_embs, cap_embs, cap_lens)
        return sims

    def forward_loss(self, sims, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        loss = self.criterion(sims)
        self.logger.update('Loss', loss.item(), sims.size(0))
        return loss

    def train_emb(self, images, captions, lengths, ids=None, *args):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        img_embs, cap_embs, cap_lens = self.forward_emb(images, captions, lengths)
        sims = self.forward_sim(img_embs, cap_embs, cap_lens)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(sims)


        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()

