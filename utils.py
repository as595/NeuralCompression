import torch
from torch.autograd import Function

import argparse
import configparser as ConfigParser
import ast
import itertools
import numpy as np

# ----------------------------------------------------------

def parse_args():
    """
        Parse the command line arguments
	"""

    parser = argparse.ArgumentParser()
    parser.add_argument('-C','--config', default="myconfig.txt", required=True, help='Name of the input config file')
    parser.add_argument('-S','--seed', dest='random_state', default=42, type=int)
    
    args, __ = parser.parse_known_args()
    
    return vars(args)

# -----------------------------------------------------------

def parse_config(filename):
    
    config = ConfigParser.SafeConfigParser(allow_no_value=True)
    config.read(filename)
    
    # Build a nested dictionary with tasknames at the top level
    # and parameter values one level down.
    taskvals = dict()
    for section in config.sections():
        
        if section not in taskvals:
            taskvals[section] = dict()
        
        for option in config.options(section):
            # Evaluate to the right type()
            try:
                taskvals[section][option] = ast.literal_eval(config.get(section, option))
            except (ValueError,SyntaxError):
                err = "Cannot format field '{0}' in config file '{1}'".format(option,filename)
                err += ", which is currently set to {0}. Ensure strings are in 'quotes'.".format(config.get(section, option))
                raise ValueError(err)

    return taskvals, config

# -----------------------------------------------------------

def pairwise(iterable):
    # https://github.com/tilman151/ae_bakeoff/blob/master/src/utils.py
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

# -----------------------------------------------------------

class VectorQuantization(Function):

    """
    Compute the Euclidean distance between the embedding and the codebook:
    d_ij^2 = ||x_i - x_j||^2
           = (x_i - x_j)^T(x_i - x_j)
           = x_i^2 + x_j^2 - 2 x_i^T x_j
           = torch.addmm(x_i^2 + x_j^2, x_j, x_i^T, alpha=-2.0, beta=1.0)
           
    where:
     - x_i : codebook, e_j
     - x_j : inputs, z_e_x
    """

    @staticmethod
    def forward(ctx, inputs, codebook):
        with torch.no_grad():
        
            embedding_size = codebook.size(1)
            inputs_size = inputs.size()
            inputs_flatten = inputs.view(-1, embedding_size)

            # x_i^2 ; x_j^2
            codebook_sqr = torch.sum(codebook ** 2, dim=1)
            inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)

            # Euclidean distance calculation:
            # d_ij^2 = x_i^2 + x_j^2 - 2 x_i^T x_j
            distances = torch.addmm(codebook_sqr + inputs_sqr,
                inputs_flatten, codebook.t(), alpha=-2.0, beta=1.0)

            # find argmin_j d_ij^2
            _, indices_flatten = torch.min(distances, dim=1)
            
            # reshape output
            indices = indices_flatten.view(*inputs_size[:-1])
            ctx.mark_non_differentiable(indices)

            return indices

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError('Trying to call `.grad()` on graph containing '
            '`VectorQuantization`. The function `VectorQuantization` '
            'is not differentiable. Use `VectorQuantizationStraightThrough` '
            'if you want a straight-through estimator of the gradient.')

# -----------------------------------------------------------

class VectorQuantizationStraightThrough(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        indices = vq(inputs, codebook)
        indices_flatten = indices.view(-1)
        ctx.save_for_backward(indices_flatten, codebook)
        ctx.mark_non_differentiable(indices_flatten)

        codes_flatten = torch.index_select(codebook, dim=0,
            index=indices_flatten)
        codes = codes_flatten.view_as(inputs)

        return (codes, indices_flatten)

    @staticmethod
    def backward(ctx, grad_output, grad_indices):
        grad_inputs, grad_codebook = None, None

        if ctx.needs_input_grad[0]:
            # Straight-through estimator
            grad_inputs = grad_output.clone()
        if ctx.needs_input_grad[1]:
            # Gradient wrt. the codebook
            indices, codebook = ctx.saved_tensors
            embedding_size = codebook.size(1)

            grad_output_flatten = (grad_output.contiguous()
                                              .view(-1, embedding_size))
            grad_codebook = torch.zeros_like(codebook)
            grad_codebook.index_add_(0, indices, grad_output_flatten)

        return (grad_inputs, grad_codebook)

# -----------------------------------------------------------
# -----------------------------------------------------------

vq = VectorQuantization.apply
vq_st = VectorQuantizationStraightThrough.apply
__all__ = [vq, vq_st]
