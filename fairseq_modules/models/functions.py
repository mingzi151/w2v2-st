from torch.autograd import Function


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        # print("1" * 100)
        # breakpoint()
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # breakpoint()
        #print("----- passing through gradient reversal layer ----")
        output = grad_output.neg() * ctx.alpha
        return output, None
