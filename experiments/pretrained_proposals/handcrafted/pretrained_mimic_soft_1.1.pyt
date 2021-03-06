��
l��F� j�P.�M�.�}q (X   protocol_versionqM�X   little_endianq�X
   type_sizesq}q(X   shortqKX   intqKX   longqKuu.�(X   moduleq cpg_methods.policies
CategoricalPolicy
qXH   /project/6002510/zafarali/policy-gradient-methods/pg_methods/policies.pyqX  class CategoricalPolicy(Policy):
    """
    Used to pick from a range of actions.
    ```
    fn_approximator = MLP_factory(input_size=4, output_size=3)
    policy = policies.MultinomialPolicy(fn_approximator)
    the actions will be a number in [0, 1, 2]
    ```
    """
    def forward(self, state):
        policy_log_probs = self.fn_approximator(state)
        probs = F.softmax(policy_log_probs, dim=1)
        stochastic_policy = Categorical(probs)

        # sample discrete actions
        actions = stochastic_policy.sample()

        # get log probs
        log_probs = stochastic_policy.log_prob(actions)

        return actions, log_probs

    def log_prob(self, state, action):
        policy_log_probs = self.fn_approximator(state)
        probs = F.softmax(policy_log_probs, dim=1)
        stochastic_policy = Categorical(probs)
        return stochastic_policy.log_prob(action)
qtqQ)�q}q(X   _backendqctorch.nn.backends.thnn
_get_thnn_function_backend
q)Rq	X   _parametersq
ccollections
OrderedDict
q)RqX   _buffersqh)RqX   _backward_hooksqh)RqX   _forward_hooksqh)RqX   _forward_pre_hooksqh)RqX   _modulesqh)RqX   fn_approximatorq(h ctorch.nn.modules.container
Sequential
qXp   /home/zafarali/projects/def-dprecup/zafarali/venvs/RVI/lib/python3.6/site-packages/torch/nn/modules/container.pyqX�	  class Sequential(Module):
    r"""A sequential container.
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.

    To make it easier to understand, here is a small example::

        # Example of using Sequential
        model = nn.Sequential(
                  nn.Conv2d(1,20,5),
                  nn.ReLU(),
                  nn.Conv2d(20,64,5),
                  nn.ReLU()
                )

        # Example of using Sequential with OrderedDict
        model = nn.Sequential(OrderedDict([
                  ('conv1', nn.Conv2d(1,20,5)),
                  ('relu1', nn.ReLU()),
                  ('conv2', nn.Conv2d(20,64,5)),
                  ('relu2', nn.ReLU())
                ]))
    """

    def __init__(self, *args):
        super(Sequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def _get_item_by_idx(self, iterator, idx):
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError('index {} is out of range'.format(idx))
        idx %= size
        return next(islice(iterator, idx, None))

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return Sequential(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx, module):
        key = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)

    def __len__(self):
        return len(self._modules)

    def __dir__(self):
        keys = super(Sequential, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def forward(self, input):
        for module in self._modules.values():
            input = module(input)
        return input
qtqQ)�q}q(hh	h
h)Rqhh)Rqhh)Rq hh)Rq!hh)Rq"hh)Rq#(X   Inputq$(h ctorch.nn.modules.linear
Linear
q%Xm   /home/zafarali/projects/def-dprecup/zafarali/venvs/RVI/lib/python3.6/site-packages/torch/nn/modules/linear.pyq&X%  class Linear(Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, in\_features)` where :math:`*` means any number of
          additional dimensions
        - Output: :math:`(N, *, out\_features)` where all but the last dimension
          are the same shape as the input.

    Attributes:
        weight: the learnable weights of the module of shape
            `(out_features x in_features)`
        bias:   the learnable bias of the module of shape `(out_features)`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
    """

    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
q'tq(Q)�q)}q*(hh	h
h)Rq+(X   weightq,ctorch.nn.parameter
Parameter
q-ctorch._utils
_rebuild_tensor_v2
q.((X   storageq/ctorch
FloatStorage
q0X   22523840q1X   cpuq2K Ntq3QK KK�q4KK�q5�Ntq6Rq7��q8Rq9X   biasq:h-h.((h/h0X   19715328q;h2KNtq<QK K�q=K�q>�Ntq?Rq@��qARqBuhh)RqChh)RqDhh)RqEhh)RqFhh)RqGX   trainingqH�X   in_featuresqIKX   out_featuresqJKubX   linear_0qKh%)�qL}qM(hh	h
h)RqN(h,h-h.((h/h0X   23094912qOh2M NtqPQK KK�qQKK�qR�NtqSRqT��qURqVh:h-h.((h/h0X   23135600qWh2KNtqXQK K�qYK�qZ�Ntq[Rq\��q]Rq^uhh)Rq_hh)Rq`hh)Rqahh)Rqbhh)RqchH�hIKhJKubX   h_act_0_ReLUqd(h ctorch.nn.modules.activation
ReLU
qeXq   /home/zafarali/projects/def-dprecup/zafarali/venvs/RVI/lib/python3.6/site-packages/torch/nn/modules/activation.pyqfX�  class ReLU(Threshold):
    r"""Applies the rectified linear unit function element-wise
    :math:`\text{ReLU}(x)= \max(0, x)`

    .. image:: scripts/activation_images/ReLU.png

    Args:
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    Examples::

        >>> m = nn.ReLU()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def __init__(self, inplace=False):
        super(ReLU, self).__init__(0, 0, inplace)

    def extra_repr(self):
        inplace_str = 'inplace' if self.inplace else ''
        return inplace_str
qgtqhQ)�qi}qj(hh	h
h)Rqkhh)Rqlhh)Rqmhh)Rqnhh)Rqohh)RqphH�X	   thresholdqqK X   valueqrK X   inplaceqs�ubX   Outqth%)�qu}qv(hh	h
h)Rqw(h,h-h.((h/h0X   23144272qxh2K NtqyQK KK�qzKK�q{�Ntq|Rq}��q~Rqh:h-h.((h/h0X   24399824q�h2KNtq�QK K�q�K�q��Ntq�Rq���q�Rq�uhh)Rq�hh)Rq�hh)Rq�hh)Rq�hh)Rq�hH�hIKhJKubuhH�ubshH�ub.�]q (X   19715328qX   22523840qX   23094912qX   23135600qX   23144272qX   24399824qe.       �4?�j�?{h�=�=9�ޙ<?���.�>�_ƾ*� ?z״=UQL?���?qk�?�����55�s���        s��P�%@�Ҽ��@�l?q����'�>�5�<�2�>��?]���;���3�=�����?n�R?�G�>�B�?}j=����j��~LC?�0*?�RP�ٞ� I�@{4?�U&��[�>G�i��?��+�       ������P7ٽp�]<���=ɝ�>Ӻ$?m�0�Ϻ[����=z\G>/9o�|־�m�=�&,��hp�=�<6Ľ�Wu=-�=D�i�������ﾔ��=/K���=}@�Er�b����?��[>��=����v���>_�$>|3�=bҩ>/?Z?�k����s�|=�D�>vF�����U����a����=�'B>OӞ?����=�T�����C���泼����n����\껫�3�O?z.�B-[�n�Z<��	�\?���!����.�i��<p�=G5�2�Q���}�r��yk��ub>[���/�k�=�ަ<�m>�V?G꼼g =���=�TF� ������ْ�=��6���0>H���T?�P>�4	>����� _>y#Y>�����4Q�I]J��`�=�����p�=�Y>`�*=�lK�z�&��x�=�u�<v�w��0>�]>8`�	�/��}�=�=K=FY�=<3>8��[KO��'9���<�پ<B�'�I�0>@;�=��D,�=Y���{�{<��ܽ��=�?�-+�?�Z>Z�1�JXl>wf�=��@>"l��v�>C���7���=Kś�5����M��%>u��f"�,�k>`�>�_��0Y�����鶡>�M/��x���k,��b�>u]?}S�s�սl��=��$�*��_���m�aʾ���Y�F�m?��1>p�Y>g��=v��=�=�J#�����0뜼\�;�8���$���ۙ��i=9�q=,����H伷�?>Po>�U��&�=s��=�n�b9 =�Ug�/��	�z��aH�G\P�y�-�s�=��I��0*�5fK>�0��p��?� ���=�r�=�_i�����oH�<(�{>�N=�O�Ax����
;[�ѧX��Z#�a��n.Ͼ���=IET>�ڇ��ڼq*'?�&���e�9��=A������;mS���d>�ˊ��\}=̎=Ư=�i��{����B�p2�<�C�>'(l��US=1�`>[e������䠾�K�<��%=��d=       �������˴黺��?/�'>5�>?����5>P?x;>�,��|��څ��V���r	>        b��=蔔�aZ>R]%��!����>@��=� �=�w��X�<��:>�&ս��O�/'>̃@>�\1>�s>]�0?H7�>Y%�>^�.�Iꈾ%xx>�
��%��_�b=�rǾЅ����`<��>�[(�;3�       J��>�2�