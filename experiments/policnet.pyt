��
l��F� j�P.�M�.�}q (X   little_endianq�X
   type_sizesq}q(X   longqKX   intqKX   shortqKuX   protocol_versionqM�u.�(X   moduleq cpg_methods.utils.policies
MultinomialPolicy
qXI   /home/ml/zahmed8/dev/policy-gradient-methods/pg_methods/utils/policies.pyqX�  class MultinomialPolicy(Policy):
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
qtqQ)�q}q(X   _buffersqccollections
OrderedDict
q)Rq	X   _parametersq
h)RqX   _forward_pre_hooksqh)RqX   _modulesqh)RqX   fn_approximatorq(h ctorch.nn.modules.container
Sequential
qXR   /home/ml/zahmed8/zaf-tmp/lib/python3.5/site-packages/torch/nn/modules/container.pyqXn  class Sequential(Module):
    r"""A sequential container.
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.

    To make it easier to understand, given is a small example::

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

    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __len__(self):
        return len(self._modules)

    def forward(self, input):
        for module in self._modules.values():
            input = module(input)
        return input
qtqQ)�q}q(hh)Rqh
h)Rqhh)Rqhh)Rq(X   Inputq(h ctorch.nn.modules.linear
Linear
qXO   /home/ml/zahmed8/zaf-tmp/lib/python3.5/site-packages/torch/nn/modules/linear.pyqX<  class Linear(Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = Ax + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, in\_features)` where `*` means any number of
          additional dimensions
        - Output: :math:`(N, *, out\_features)` where all but the last dimension
          are the same shape as the input.

    Attributes:
        weight: the learnable weights of the module of shape
            (out_features x in_features)
        bias:   the learnable bias of the module of shape (out_features)

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = autograd.Variable(torch.randn(128, 20))
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

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) + ')'
qtqQ)�q }q!(hh)Rq"X   in_featuresq#KX   out_featuresq$K h
h)Rq%(X   weightq&ctorch.nn.parameter
Parameter
q'ctorch._utils
_rebuild_tensor
q(((X   storageq)ctorch
FloatStorage
q*X   85411584q+X   cpuq,K@Ntq-QK K K�q.KK�q/tq0Rq1�q2Rq3��N�q4bX   biasq5h'h(((h)h*X   78950576q6h,K Ntq7QK K �q8K�q9tq:Rq;�q<Rq=��N�q>buhh)Rq?hh)Rq@X   _forward_hooksqAh)RqBX   _backendqCctorch.nn.backends.thnn
_get_thnn_function_backend
qD)RqEX   trainingqF�X   _backward_hooksqGh)RqHubX   linear_0qIh)�qJ}qK(hh)RqLh#K h$K h
h)RqM(h&h'h(((h)h*X   78579488qNh,M NtqOQK K K �qPK K�qQtqRRqS�qTRqU��N�qVbh5h'h(((h)h*X   83856672qWh,K NtqXQK K �qYK�qZtq[Rq\�q]Rq^��N�q_buhh)Rq`hh)RqahAh)RqbhChEhF�hGh)RqcubX   h_act_0_ReLUqd(h ctorch.nn.modules.activation
ReLU
qeXS   /home/ml/zahmed8/zaf-tmp/lib/python3.5/site-packages/torch/nn/modules/activation.pyqfX  class ReLU(Threshold):
    r"""Applies the rectified linear unit function element-wise
    :math:`{ReLU}(x)= max(0, x)`

    Args:
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    Examples::

        >>> m = nn.ReLU()
        >>> input = autograd.Variable(torch.randn(2))
        >>> print(input)
        >>> print(m(input))
    """

    def __init__(self, inplace=False):
        super(ReLU, self).__init__(0, 0, inplace)

    def __repr__(self):
        inplace_str = 'inplace' if self.inplace else ''
        return self.__class__.__name__ + '(' \
            + inplace_str + ')'
qgtqhQ)�qi}qj(hh)RqkX	   thresholdqlK h
h)Rqmhh)RqnX   valueqoK hh)RqphAh)RqqhChEX   inplaceqr�hF�hGh)RqsubX   Outqth)�qu}qv(hh)Rqwh#K h$Kh
h)Rqx(h&h'h(((h)h*X   85370368qyh,K@NtqzQK KK �q{K K�q|tq}Rq~�qRq���N�q�bh5h'h(((h)h*X   86250752q�h,KNtq�QK K�q�K�q�tq�Rq��q�Rq���N�q�buhh)Rq�hh)Rq�hAh)Rq�hChEhF�hGh)Rq�ubuhAh)Rq�hChEhF�hGh)Rq�ubshAh)Rq�hChEhF�hGh)Rq�ub.�]q (X   78579488qX   78950576qX   83856672qX   85370368qX   85411584qX   86250752qe.       6E����=���ޙ�<][o=��~<����<H�<�z���Z���g~����=�7����=ՙ�=s�,����׽|_�M>Wr0>yݼ
V/>u�y=��#��)����f.�A�����@=Ř���I<�O]=vM6>�h�Qck=@f�;�x�����=�>��z=��k�c���v��M�9=��F=}��=�x��<��=B�يq=F�7>�2���c>+�=*�>��h>_�H�^pK>�Ei���8學��񽕳!=J�q=|�=u�,�'�+��=�=2�M�8��<"���k���`9;���*ʲ=�+=�"�=�W���x>�>zT��A��=T28=[�>�>>+_�=/�v;hM�= �=��ۼ�>�нJ����x�<��c=�r�=\ˎ9N~v=^���7>�3н�,�=䫞=�> �(>�R1���	>����>{ɽ[���l߽[@>>�=��	�o�����>=��X��K�==��={X���i4��D<[}>���=��;$� ;�=�L�=!�U��>��q�>�3>��'>hÈ=�7>t'0��M�<���<���=�_(=���=���ʸ��)=g�>��+=���=LH��ڻ�">�~�=l>P�"�pE$�)!��&>�6j=I�W=ӃŽ���>8ֽ�K=>tst�|�	���P=O^`=1������=d��<�i�XvA��Cv�G���X,�icR�s2N��+�=�'߽)QJ��P=�L��V����'��뙚�DE�<�=N$(>5|�=55�wU�<(i���4�=Z,>_�v����r@>��]=��=��%������a��7�s�2=6�����ռNw��м�I;½�.�����=� �=+��<�y�&����; �=��`<��=��=��/�@a��@�z>�c��vE�שl=��vI�M`L=(��=�o½�J�;Q�a<�]t=�u��V�=;�>��=,���ǽ��d=�`�Ngu��������R=h�^��8�=��W<�,��w`����ֽ�> �;��p&Լ��^#�pY��&�񜫽���=�i�=-%����=�L��t�� />��T;= TU=l�b=o1��)>��V�z���4>�1%�I"��2$��3�����=̼�[���m�=����o�潸��ٰ9>�+�<z��=�4a<�v�=��S���#H�1�罝�ۼ �2=�O�=-�L= ��<�Y�=��>=]���������L��w>o�>n��<�3�=���G��=��=�#�2$��Q`��,'���">}���=pO_<Q�����o�ż ����	>�	���><�ս�꼦��=̌�����OX�a�=�����;�>��q=y����P4>�۽���<0����>������*>Kl>L�)YԼ�g�=?ĝ�����߼�n�c=�5�=�� >��@�*��vy����q�|J=��=�ji=)��g��Lg=�(
��Z�1l�2�0�����=o�=��>1���g��K�<
�>_x�==�u��.�=��O�(!�(f�=�q��u�;�|�=M�)��9�=�Q8��ּϯ=.">�߉=��>=?�=v<<�So�<�oo���ؽ��!>~����Խ��=��м �+<�̰���ǽQ��eW0��ޜ=c��N=)>�h>W�*C!��$I����D!>Qѳ���>8�f����=.f½݈��3>��(�|=>��*�	�C>�Qc=%��<ׇ=>��=���<4&(�u>�1<Ǻ>�*�<��<m�H�
�=IP���fS�{�q��q>z*&��+�<��ɽq��<1+>��B֩�1Ǔ=+�X=T�=���fj�� >��<ȓ���������j�:����<n,ʺ>��=������<��;<cJ�Ryཅ�ټyN>0�	��2�=�J:��{B=�TC=��[���)�L�I=I��=��L�>
��<��=yS��^F�=be�=N���A4>~�+=���=G�<�,�<�-(=��>bD�=r�Ľ <J��<E%����
��= ��w2|�C�˽�F=�#�=�w~>Z���g>d��K�>��=v!��=�C�� �zT�=j%5��O>�E=�8L=�M�2�=^���/ٽ�{^�mM9�}�;>H�=+,>���;��	=���=�S何�>k�彷���q=Q�=bs���4=e�=F^�<�����=����$4>�"��D��l�����<�m<�o���y/>$:��J�;=	K�ˬ�<�᪼��=ʽ����zկ����N#�ѸQ=Yb�==���m�]=��+��A>+
C����.p�=��=�M��o�=�[>%�>g��=Y�=K-齡����_U="�/>�Ľ��������E��	�=|�]���	>#n0��{��x��S�>Iv�=��޽4�m���	>b=��>��W=6_�=�ɽ!� ��AQr=�#_�Ng,>�?����=I���t�\ =pІ=�j>1�q=������;<�Ž)�0>�����C>컵�ƽ��Ố�*�������=j�;�:�uC;��\ؼ�N=���<�g�=˛<��>� =ӥ�MS=��P���}=�E;�Z:�<�=�����s���o�4���]L =N�=M�ͽxT޽O:>���@<�=
H8>�]�=��<�	w�<s���G�=��<:䏽�B>%3<jý�>Lm(>�ν�r�x��ʰ��D��<�]$�$ !�|�N�Pض<�����w�=���'�>&پ<XS�=e���^��;�WԽ���p ;s*;��0�=���= �м�>�.<���<��<k��=��
���c=5@���>-��<�\>�Q`=G�仅	� �=�>��=r�N��eJ�ם��5����>�0 �����
>�1�=fw<��6>��R=ی6�m��=l�>�2p=����^ʽ�<� ��z<ɽǰa=J!,>��a���<�d6������>�!=f��;��=��
P~��= >Q�R���=d4�=rRʽ��Ƚ���=��t='��;�(=u=b�=��-��~h���=��<����<_W>Ar��(�r߹�o�=Bhý=����?V=�э��=9���>���<A��=��½l	����2�J�ҽ���='��=㠫=���=��ͽ�|�=DG/=�Y�=t�>�H=�4=@u	>t.<�� >$�=�B�=����^�=M4>d>.7�=#_ �C`E>BA�=��>�"<2��<ק����>�8�=B�=&����c��
�=p�=�ݽ�>�Y�m���0�何5%�g�½,~�<��|�Q<�=|��<<0��R��l�j=��>q��<��=B�I���=�F,>̼=�1G>e�=5����>��ڻ�d>=�=��*�j���QD�=���k�=�Q˽�I=)��=ťX����]Z��$��$�=�D��b��=�W>=�=i�o=���==)>g9)>g�=��J>J�>>�,?�� �>_0�����}9>oٽ��:>s�5������=#ͽ�ѽ��c� ��=�'��>�6�����Xme<1|G��.I�_��q�)�=��G���=�N��o;��vv�0e=��U�r�F�D��=��p=	>�Z���M�d�<;%�=�|��'�`�b;3w�=@������-m>�ᏽ����%�?���)����=��>~�}=l��=�Tv=�n/�얽�`�=�u�a9��V)��X-��Բ�:=��I�=A�=`b>>!�ۅ���>M�>�9�?ڴ� ���C���W�>mJؽ��G=e�B>�A�=�_^�W�>�M�=��
>t>��S[=�=rސ����;�=�a�=.>��y�<�S=�ࢼ����/��=^����v�lv*��d�=�U�=o����=uE����Pc=��=�k�</��MV�=s�2���K=$��<�H�=�Y�:�_#����=�{7=�#=�>�VK>�۽�Wf�7t=D�;���:J��=L�>k1y>>=�:&>q��aMD>���՜=@��<        �����	���P���>���>*+�=@�?�h���%<���>�A�>�>�n?�N>���>.���+!��-��^�<i?�����>�E!��E���5g�!�,>����I'?/x�=^�>        L��Vj�=��=N�<Zț��Q����=���<��$���4��@�k�=v*���l����!���VE>��{�f=�9S<?`��*�X�G�:\ᒽĢ�<�o�>u��<���=OCl���=@       3c=Iƭ� hY�� >]��r�Ӽ�>u/>0 =�2)=н����V=�̫�e45� ���m*t=e��=���v>�&�2�=����}�=���=��=����:=c��e9>>J�;�"�=jı�s��<�#>�+�=<�=@)`�	'k�G�<�ۥ=��>��=z<;��p;����e�<$�n���=�|��oҽV_>��$�?c���6<�#��D1�=Z�Lh>��=���=�}+��s/=��=�F3�@       ��+��)��*����)�@b��b:���>{`�>�� >-6?�@����>�<�u�d�����;��iྣ�#?���>���>�8>t;�c�"?�(�K�Ծ�b׾����>�����`t��| �>�X����!�����}���r�M�0�sg��~3��
�=�s��W������>��5��� ��>*���۽]e/?�L�>�p�A��2?
��>ɵ>5�k�(?�Ո>$ޥ���>7�L=����6��       �>����