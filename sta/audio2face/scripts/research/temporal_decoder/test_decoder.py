import unittest
import torch
from decoder import Decoder


class DecoderTests(unittest.TestCase):
    def model(self):
        torch.manual_seed(12)
        net=Decoder(torch.zeros(96),torch.ones(96),'temporal')
        torch.nn.init.normal_(net.output.weight,std=.02)
        return net

    def test_stream_history_matches_complete_track(self):
        net=self.model();x=torch.randn(1,65,96)
        joined=torch.cat((torch.zeros(1,7,96),x),1)
        expected=net(torch.stack([joined[:,i:i+65] for i in range(8)],2))
        history=torch.zeros(1,7,96);parts=[];cursor=0
        for count in [1,4,30,30]:
            block=torch.zeros(1,30,96);block[:,:count]=x[:,cursor:cursor+count]
            pred,history=net.stream(block,history,torch.tensor([count]))
            parts.append(pred[:,:count]);cursor+=count
        torch.testing.assert_close(torch.cat(parts,1),expected,rtol=1e-5,atol=1e-6)
        torch.testing.assert_close(history,x[:,-7:])

    def test_future_features_cannot_change_past(self):
        net=self.model();x=torch.randn(1,30,96);history=torch.zeros(1,7,96)
        a,_=net.stream(x,history,torch.tensor([30]));x[:,20:]+=100
        b,_=net.stream(x,history,torch.tensor([30]));torch.testing.assert_close(a[:,:20],b[:,:20])

    def test_zero_initial_correction_and_reset(self):
        net=Decoder(torch.zeros(96),torch.ones(96),'temporal')
        x=torch.randn(1,30,96);a,h=net.stream(x,torch.zeros(1,7,96),torch.tensor([30]))
        self.assertEqual(float(a.abs().max()),0.)
        torch.testing.assert_close(h,x[:,-7:])
        b,_=net.stream(x,torch.zeros(1,7,96),torch.tensor([30]));torch.testing.assert_close(a,b)

    def test_empty_block_does_not_advance_history(self):
        net=self.model();history=torch.randn(1,7,96)
        _,next_history=net.stream(torch.randn(1,30,96),history,torch.tensor([0]))
        torch.testing.assert_close(history,next_history)

    def test_control_is_independent_of_previous_frames(self):
        net=self.model();net.mode='control';x=torch.randn(2,30,8,96)
        a=net(x);x[:,:,:7]+=50;b=net(x);torch.testing.assert_close(a,b)


if __name__=='__main__':unittest.main()
