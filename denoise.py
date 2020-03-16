import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.autograd import Function 

class LinearFunction(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = input/weight
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output*weight
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output*input
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias 


class Linear(nn.Module):
    def __init__(self, batch_dim, bias=False):
        super(Linear, self).__init__()
        self.batch_dim = batch_dim

        # nn.Parameter is a special kind of Tensor, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters require gradients by default.
        self.weight = nn.Parameter(torch.Tensor(batch_dim,1,1))
        #self.weight = nn.Parameter(torch.Tensor(1))
        '''
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)
        '''
        self.weight.data = self.weight.data.normal_(mean=0,std = 0.01)+nn.Parameter(torch.ones(batch_dim,1,1))
        #if bias is not None:
        #    self.bias.data.uniform_(-0.1, 0.1)

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return LinearFunction.apply(input, self.weight, None)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )



class torch_sigma(nn.Module):
    def __init__(self, batch_size=64, f1=512, x_dim=3):
        super(torch_sigma, self).__init__()
        self.batch_size = batch_size
        self.f1 = f1
        self.D_K_1 = 512
        self.D_K_2 = 1024

        self.feature_extractor_part1 = nn.Sequential(
            nn.Linear(x_dim, self.f1),
            nn.ReLU()
        )

        self.deep_kernel = nn.Sequential(
            nn.Linear(self.f1,self.D_K_1),
            nn.ReLU(),
            nn.Linear(self.D_K_1,self.D_K_2),
            nn.ReLU()   
        )


        self.transform_1 = nn.Sequential(
            nn.Linear(self.f1,self.f1),
            nn.ReLU(),
            #nn.Linear(self.f1,self.f1),
            #nn.ReLU()
        )

        self.transform_2 = nn.Sequential(
            nn.Linear(self.f1,self.f1),
            nn.ReLU(),
            #nn.Linear(self.f1,self.f1),
            #nn.ReLU()
        )
        self.transform_3 = nn.Sequential(
            nn.Linear(self.f1,self.f1),
            nn.ReLU(),
            #nn.Linear(self.f1,self.f1),
            #nn.ReLU()
        )
        self.transform_4 = nn.Sequential(
            nn.Linear(self.f1,self.f1),
            nn.ReLU()
        )
        self.transform_5 = nn.Sequential(
            nn.Linear(self.f1,self.f1),
            nn.ReLU()
        )
        
        self.transform_6 = nn.Sequential(
            nn.Linear(self.f1,self.f1),
            nn.ReLU()
        )

        self.sigma = Linear(self.batch_size,bias=False)




        self.classifier = nn.Sequential(
            nn.Linear(self.f1,self.f1),
            nn.ReLU(),
            nn.Linear(self.f1, 40)
        )

        self.ro = nn.Sequential(
            # nn.Dropout(p=0.5),
            nn.Linear(self.f1,self.f1),
            nn.ReLU(),
            nn.Linear(self.f1, 5),
        )


        # Weight Initilization
        with torch.no_grad():
            self.transform_1[0].weight.data = self.transform_1[0].weight.data+nn.Parameter(torch.eye(512))
            self.transform_2[0].weight.data = self.transform_2[0].weight.data+nn.Parameter(torch.eye(512))
            self.transform_3[0].weight.data = self.transform_3[0].weight.data+nn.Parameter(torch.eye(512))
            #self.transform_1[2].weight.data = self.transform_1[2].weight.data+nn.Parameter(torch.eye(512))
            #self.transform_2[2].weight.data = self.transform_2[2].weight.data+nn.Parameter(torch.eye(512))
            #self.transform_3[2].weight.data = self.transform_3[2].weight.data+nn.Parameter(torch.eye(512))
            self.transform_4[0].weight.data = self.transform_4[0].weight.data+nn.Parameter(torch.eye(512))
            self.transform_5[0].weight.data = self.transform_5[0].weight.data+nn.Parameter(torch.eye(512))
            self.transform_6[0].weight.data = self.transform_6[0].weight.data+nn.Parameter(torch.eye(512))


    def message_passing(self,x,y,z,norm):
        xx = torch.sum(x*x,2).unsqueeze(1).repeat(1,x.shape[1],1).transpose(1,2)
        yy = torch.sum(y*y,2).unsqueeze(1).repeat(1,x.shape[1],1)
        xy = torch.matmul(x,y.transpose(1,2))
        k = -(xx+yy-2*xy)
        k = self.sigma(k)
        k = k - torch.mean(k,2).unsqueeze(2)
        if norm==1:
            sm = nn.Softmax(2)
            k = sm(k)
        z = torch.matmul(k,z)
        return z

    def forward(self, H):

        
        H = self.feature_extractor_part1(H)
        # Deep kernel feature extraction
        d_k_f = self.deep_kernel(H)
          



        # Update our features 
        H1 = self.message_passing(d_k_f,d_k_f,H,1)
        H2 = (H + H1)/2
        H2 = self.transform_1(H2)
        H3 = self.message_passing(d_k_f,d_k_f,H2,1)
        H4 = (H3 + H2)/2
        H4 = self.transform_2(H4)
        H5 = self.message_passing(d_k_f,d_k_f,H4,1)
        H6 = (H5 + H4)/2
        H6 = self.transform_3(H6)
        '''
        H7 = self.message_passing(d_k_f,d_k_f,H6,1)
        H8 = (H7 + H6)/2
        H8 = self.transform_4(H8)
        H9 = self.message_passing(d_k_f,d_k_f,H8,1)
        H10 = (H9 + H8)/2
        H10 = self.transform_5(H10)
        H11 = self.message_passing(d_k_f,d_k_f,H10,1)
        H12 = (H11 + H10)/2
        H12 = self.transform_6(H12)
        '''
        #M = H.reshape(1,len(H))
        # M = torch.max(H6,1)[0]
        # y_prob = self.classifier(M)
        # y_hat = torch.argmax(y_prob,1)
        return self.ro(H)










