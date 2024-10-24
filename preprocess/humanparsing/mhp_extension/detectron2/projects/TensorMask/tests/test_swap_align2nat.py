#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest
import torch
from torch.autograd import gradcheck

from tensormask.layers.swap_align2nat import SwapAlign2Nat

# if torch.backends.cpu.is_available():
#     device = torch.device("cpu")
#     print("Using device: cpu")
# elif torch.cuda.is_available():
#     device = torch.device("cpu")
#     print("Using device: cuda")
# else:
#     device = torch.device("cpu")
#     print("Using device: cpu")

device = torch.device("cpu")


class SwapAlign2NatTest(unittest.TestCase):
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_swap_align2nat_gradcheck_cuda(self):
        dtype = torch.float64
        device = torch.device("cpu")
        m = SwapAlign2Nat(2).to(dtype=dtype, device=device)
        x = torch.rand(2, 4, 10, 10, dtype=dtype, device=device, requires_grad=True)

        self.assertTrue(gradcheck(m, x), "gradcheck failed for SwapAlign2Nat CUDA")

    def _swap_align2nat(self, tensor, lambda_val):
        """
        The basic setup for testing Swap_Align
        """
        op = SwapAlign2Nat(lambda_val, pad_val=0.0)
        input = torch.from_numpy(tensor[None, :, :, :].astype("float32"))
        output = op.forward(input.to(device)).cpu().numpy()
        return output[0]


if __name__ == "__main__":
    unittest.main()
