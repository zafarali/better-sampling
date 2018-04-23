��
l��F� j�P.�M�.�}q (X   protocol_versionqM�X   little_endianq�X
   type_sizesq}q(X   shortqKX   intqKX   longqKuu.�(X   moduleq cpg_methods.policies
CategoricalPolicy
qXE   /Users/zaf/development/policy-gradient-methods/pg_methods/policies.pyqX�  class CategoricalPolicy(Policy):
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
qXW   /Users/zaf/anaconda/envs/py36/lib/python3.6/site-packages/torch/nn/modules/container.pyqXn  class Sequential(Module):
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
qtqQ)�q}q(hh	h
h)Rqhh)Rqhh)Rq hh)Rq!hh)Rq"hh)Rq#(X   Inputq$(h ctorch.nn.modules.linear
Linear
q%XT   /Users/zaf/anaconda/envs/py36/lib/python3.6/site-packages/torch/nn/modules/linear.pyq&X<  class Linear(Module):
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
q'tq(Q)�q)}q*(hh	h
h)Rq+(X   weightq,ctorch.nn.parameter
Parameter
q-ctorch._utils
_rebuild_tensor
q.((X   storageq/ctorch
FloatStorage
q0X   140562109237248q1X   cpuq2K@Ntq3QK K K�q4KK�q5tq6Rq7�q8Rq9��N�q:bX   biasq;h-h.((h/h0X   140562113382624q<h2K Ntq=QK K �q>K�q?tq@RqA�qBRqC��N�qDbuhh)RqEhh)RqFhh)RqGhh)RqHhh)RqIX   trainingqJ�X   in_featuresqKKX   out_featuresqLK ubX   linear_0qMh%)�qN}qO(hh	h
h)RqP(h,h-h.((h/h0X   140562113194960qQh2M NtqRQK K K �qSK K�qTtqURqV�qWRqX��N�qYbh;h-h.((h/h0X   140562113238176qZh2K Ntq[QK K �q\K�q]tq^Rq_�q`Rqa��N�qbbuhh)Rqchh)Rqdhh)Rqehh)Rqfhh)RqghJ�hKK hLK ubX   h_act_0_ReLUqh(h ctorch.nn.modules.activation
ReLU
qiXX   /Users/zaf/anaconda/envs/py36/lib/python3.6/site-packages/torch/nn/modules/activation.pyqjX  class ReLU(Threshold):
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
qktqlQ)�qm}qn(hh	h
h)Rqohh)Rqphh)Rqqhh)Rqrhh)Rqshh)RqthJ�X	   thresholdquK X   valueqvK X   inplaceqw�ubX   linear_1qxh%)�qy}qz(hh	h
h)Rq{(h,h-h.((h/h0X   140562113097584q|h2M Ntq}QK K K �q~K K�qtq�Rq��q�Rq���N�q�bh;h-h.((h/h0X   140562108963984q�h2K Ntq�QK K �q�K�q�tq�Rq��q�Rq���N�q�buhh)Rq�hh)Rq�hh)Rq�hh)Rq�hh)Rq�hJ�hKK hLK ubX   h_act_1_ReLUq�hi)�q�}q�(hh	h
h)Rq�hh)Rq�hh)Rq�hh)Rq�hh)Rq�hh)Rq�hJ�huK hvK hw�ubX   Outq�h%)�q�}q�(hh	h
h)Rq�(h,h-h.((h/h0X   140562108955952q�h2K@Ntq�QK KK �q�K K�q�tq�Rq��q�Rq���N�q�bh;h-h.((h/h0X   140562115133440q�h2KNtq�QK K�q�K�q�tq�Rq��q�Rq���N�q�buhh)Rq�hh)Rq�hh)Rq�hh)Rq�hh)Rq�hJ�hKK hLKubuhJ�ubshJ�ub.�]q (X   140562108955952qX   140562108963984qX   140562109237248qX   140562113097584qX   140562113194960qX   140562113238176qX   140562113382624qX   140562115133440qe.@       v�Z��?�!���Ƒ�0��Z��?�a½��ý��^;��ͼ�C��S�����>�U�=%����~�><��Ť>�~:	�=�x9>8=�=KW"��8���~>-]��q��>N�������b�j���0�ü��A?]雿��u>wH�>�6�>�c��Ӏ�>{�J��,��5�@Y?�}��Cr�AȽ�#�=��o�UH�ށQ=46������ �� >�q8?�s�J����":>�ľ�h�?�@��L�h=�2�zi�        ���=�@��">�6���> ���%>�@;�*����f�eA<N�4��������9�5=���惽#�,y<��=���� tʽ��;�޹=�>�|>�E�=�eN��qr�آ�=�y�@       Yƽe�R��?���?����w�vъ�W��?,e�:���Qt>~Ϯ?��>��?Ĳ�> ��i"��, ��y>��D��gϾk�?�`��5���������?j����?� ?�Q ������R������?�F��L�^:�=Y���S��G��3Z>�ɿ�Y�;��@R4>h=��UV�=�Ѻ����>빡��O��o�=P?)��O������Z:>pc@KS�7��?���>��}?�(�� 7��       �k���V;6w?MĜ=H��4k>�Ig>q>���s�)
�=h��=Ǖ=RB�>ͯ��i�D�/������l=�l?8�[��t�>�ɼ��>'=rVھ1I�?�>9�6��>K1ӽ �R=}�O>�=nLj���׾�;]����&#�U����?>��@����M���3�=�w?ǖ�x����,Ƽ����yr>>�+ͻ����*�<�����[���w�6b����=�� �$��<� >��.�jiy�@Fǽ�x���[�=����}F��G�<����->烹�X
���"�f��=��>j��:8=���=��#=$�<r>i3�˶���ս�=v�ǣ==�F�T1���k=���=g@�=X$��C�Y%����V�o|=4��>#j:=��伫�>����V�=���崽��=*����>����^k��A�=�j�����Z�>�*b�;'.>ll����>��^=<k9��{?2p�qD�<���=az�=A�t��a�>2?5�d���Yn�>�0�S�==҂�=8�p=�����Z�=��½��<���<��P�`���g�=k�w=��b��"?B�ܼ,eZ>��>@L�>9؄���2���D?�����m���rK>�9�����=w�>4K�ln�=U�=͎�<�.��r���ۼ@Lg>�@ㄺ;Y����ǽ�5?�"��#,�3T]>�+U=ߋ��'��]����
<5!��$ �[HK=��ľ 3���B>@gq�g/����>\� ;�Et�Q�����=�Z�>��������<�꼽S���Ԛx>�ێ�ޏL=4��=����8pԽ�C�=��N>�4ݽP.?�d3=��>��=��>>T
%>Z�<��dG?�3�=n��=���:����q�7{>+H�=2�G=P��<U�;�5��B��g�𽚨
>=�>�\~�.���0x���k<�-�=(ǣ�t'[=��1�U���>�5p9��0i���<6�;�&�=y�~�6ֵ�����-:�<���=�o���9�k��=윳=� �rK>_���W�>�<6�>|�?A�5��ݦ=^����M>��y>w�ռ�>�E�=;��=F58�*�=� ������f4�l�:>���*�H���=�Y>k�"��t*>�Y��T�;;�/�)>��:�Y�a�&S�;�a����=��c>�X�?y�)�Go�=,�=|��>~���y�=E��0�:=�7�;�����>]d���i�=.��ʖL��W������{�>��F�)��ڽ^�c>���=��S=��Q>O�1?J�9A;���<>G�Z�d�=m2.�b�Q>[䂽�a�=��T<�/v����=ƽ6�z�.��<�>_��=�*�>�@���V&?Ù�=�چ���?>0.��*:��n��4?>�|�;]�>l�2�%Z$��E<^�=��Ľ ��?]�x��<=���xTżWT�3�>�.=�ƽ1����\�x��<vП=��=\��*,뽐�t<�C�Zʒ=p��<�."��tm=�l��MⰽX{"� ߼0�?<E@�P9=�p>�9�=wx/>��>�9<��ػ�?�?v�#�O�M�+#�=_p�>��f��� >���� ��=h�d���6
>���::����4j�����$>��@�W���
q>���<)�=�yG�p��!=�{\=�_�	9�=3��=2��4%��B��#�� 2.��3C�H�=���x$���Q������u��6�=`�=����$e�l3�=�H&��B>�ʑ�^%$��Ž��=ٹh�D�=�⣾�F>Ҟ�t[=&�]=8��>柽�U.��Ve��b���>��>P���1D�"�O�44>u<�=	j�<m�����9�>:󿽵G�>��?�۽�+\>�P=�Zƾ��,�.�ý���=w�=8�g�9���T>U"�B��Ɇ=E>�>E�?�3�=��-�f�=��>�h����?=6��=������<���x��p�e;ǆ�=��=�%�@<_�m���	=:��V��� >�!-=��%�n�<[��=�>|�M=��#���㽓zX>���=FI������V+>�'K>��=j�E=�9]��~b�<�N=W�)= 8?L��=��=h�\>�A�>蒟=�x�,�:?Y�"э��e󼫣�=b�=v�R=���y�L%>�إ=��r>q�Ľ	��O}[>+�h=ک#�$�#>h�=�!�=KE*�h,?��?��~=�pڽ�Q�=�3�w�=��>��˼�?6��\6>͈%>�>�|���>�vU>l�3>�u7>�->I��lF�=t"��5�=���<�!��\�Gߪ�f��=�˶�m(�=PA�=����W�>Ez>��������=��<�5.��J=�'ҽ0}.�B31�9���nI�~�=�ƃ��1���A=��2>�.��$�=��J}+<���<�&�C��=��=X=~?3��=W;��<�=��kq<�D=��>�jӼ��ռ*[���n=�ѽ��>B�����=eG6=v8���==�#�% �<�l=;:s1>���Yh��mw�=�H�"������>���=�k�=2s��@����E=O�k<��I:ϣ���7�;Ρ�;�B�=�k�>�&=T���s̽�13�Vs<�6���{Z�-��>�;侧�h��F�=9n�,g)�V�>=���=V��=ZJ���!>�,>$j�6��2W{���=0G��(ѽӋ�������E�=�C_��.�Lh��	�����Q>�vX��) >�C��v���LǻN>E�o|��_�2z�=�ܴ=
qn�hn=u0��� �E�>�>L2�g�=Ff>�s�<�}y=���	V8>%PV����<y'�>���6(���z>���<Aԛ=�C?�7�d�>(��=��?����Gƽ5q�? "�B�H���>�;)�}K�=�R�>� =)����m�=�eнcJ/�j�.a��<�	��8�&�\�%�7H`�'�Q�u�=F,��D��=U�==O�мm��=���;lK�=��=x8��(���
>���Ro�=% 0��;�=���=�r���ż�ۿ�G{��4�=�0�<r2�:��(�O�G��<��>�� =��R�o��=�Ȓ>b��?�H���p>�ޖ�����/�+�=}��_r�{`���=��?��#4���0���ٽɈ��}�=��PD�<��'>�I�=GH]�Q��=6��=p󍽨�>៽�\|��'�P~9=kS>v0>����=g�=ω>{�^>���=�Z�����~"V�螺�/i��^�;)�n>k�yſ�Hc�<�ɥ=_(��g>�h�x�׽ �5>���>��������ؽ�H>���>�9?]���o��s�4��u�>���' ��>�=�a�=���
��|3p�Y�*>E���=�q3���;S桽���UTP�I�ҽ��T�̺��󔽑!>Y�?�u9Ʈ]��3>n�C=�t]�$ҭ�	��=Q�J�}��>fA���O1�"jD>s�g=�Ď=܁~?������>�1�H�?ثf�ކ��?'��Hˁ�nx?̘>�׵�[Ҷ>��Y�ړ�=�vd���h����=[�<jR�� �0�O���=}�ʽ�C���V<�彆��=r�=ÄM��jŽ���=J�L��o��>Vt���`�������=?��=�<m��.0�ؗ>p���穨���������o>����� >y�|>�� �<��=(�N�����
=�~>1o�%��>�=��=���<�	���9�=v	?�v�;�N�=��1=�**=ˬU�C�0��&�>��T=��I�J�>8}>+L�������=_d0>��=��=�:�|)����:��ֽ�W�?>�=��ȶ=6> �>�(�%��=�o�����=����A>q�$�jB��I���uE�<ϲ����Z�2XN>�D��e�ݾU# =~c��4����╼lԀ=Y�ɽ���=X�r��*�=/E�r��=��=Ǌ����<���=�h �Y������f����;��ǽ�*��D!���V���=薻�������]*=���pmK<�,� �9�`����n�       /�<��z#>�-���켍9�Z	Q;N�+N��hF9=�!�O�)=t~ܽ���>���k�=[>Ľ��=���=�-W=r>�d�<�co?}�8���=����b/�������%پ�>4s:>���%<Q��$>�D��vq=;K0��#R<��%����{��<����ヽlm=l����>�������Kxý=��P>[6>,�����g�=��;ѷ���x��E=dem?�p��*�d�-���J=�C>�!V���9>�����u�=뽥�<��R=�3&>��P;�٣���=nN���"=9>�L���ӽ��ս~q>�TW>��=�G��6bh=��i>��=��>mҒ��M@�#����b=�˭���>�ŽYU	�X��~�!>!����=m �<��<B����~���>�"ĽW�h������;Ӣ =�%W<����v->G$��>�t=Şֽ V�=>�&�zɽ4n<&�}=n�Q��6��(�E��$�<���=��=���=�n>���>�3�=�m���rM�t="<�Y>�U=�N�����>ا�=̽h�>9A�:����W���>��=���?���=Z>�\���M�g�]�(�e�`\>�/>���= ٽ������۽�?>��-�3d-��립�3;��P=�->�Y�����;h�.>š��S�Y=�v��U�羲��B_��h>���=�S<r�����Q>�u����a����p�(>=@��J���;=��=�v�>�=���}I>��C>�~�=�]�=;bS����J^罆>	>s�=����s!�>��׽%�=���V|H=�5=���]��=�����G?:9|���������'��u��m��9��ؽ�Ց=�"��m�T�����T�5=\�� �=W��=Q?½����;���������<���;�%r�q���g�B>��s���=�)�$o7>�2�=y<c��qýc�V>9����$�=�4E>���>u=<�ey��q�;G
�>%t��QA�Z�s>)�^=�-~��K ��Q߽!P�=�Y3<zcY<<�����>Z���0Z��b,>�Ie=�3,��	�=�2�>4�*=oԘ=%C����={ԓ>��4>м:�0"5>q,?1�1������)��`1_>S��g�;�#�=NG���c5�� �����w�:>��->U��g��Υ9=���=�%=��=y�:���=D�R=�#�Xg>Ɣ��r��<�>������<9P�<�8�?��I�Ea-���罒=?��!����< ����%���=��	�@:=H�(�[����֍=����U�>���=k,������"�=6��=�;����=�
h�$?��>��ܽ/�����'��2= �+=�>�Fp=�P=݉��,;B>�:�=b>C>B�=U�>����=�>61�=P�I=�}j��%�<_J��6N%�bV!����=���=���<�沽�5�=�����A�����]Z�YL;U���*>��z����5p=��y���]�Y��=^��U�7=�Q�Z�y��M�=:$V�6@����C>,<�؊;WOB�|u߾�#��p>]Mi=�aJ��Qn���>h�R<B��= �����=�T���Н�%L>���=k~?��#��O��	��H� <|}�=�o�=���1S>���<2~�G�>�Fƽ�齔j�=��>�j�d��>��=)�t=��$�}�=F�|=P�
�W��_�ҽ?T��p���h����Ѽ��O���	���>�߼%�=������޽������k�;��>C�=~K1>��7��:-��^
�S�#>U;�=�j�>/�E�ҋ�=���=�O�=:/�~����s=��h��YR?����>�m�\�m�K<a�"�����>`>z�ҽ�I=�F��m�=%{��Y�R>�x��e��=\�j����/{>���D�S��;<Y���Gs��U{�=�V>c>_���V�����>3T��MS�=@����B�*�;CF�=��>��=v2?����=��= F�;�Ii>����������!>���4(�=�L ���Y�aCŽXT�,�4=ت���&�>�����V='��dJལ�� �=R�޼�v�� �D?���̸%���7=�F/�D�*���"V=�JQ>y��=��n��%�<h���p#����=@j�a��]�Z<�*�=G���o����r��X��77=R�B�Ը��Ў=���=��-<M��<m�=��a>fS����"=L`��e�t�]d�=��1��p;>>{>_-�½���>�T>x;�P�e>����Є=��۽-�=��B>�f>��h<�K=���=wt �O�ؽ�>H>�d�=�����"=ى�=��@x�
.���'�_"�T$����>t�=s��?��˽�u����6�Կ�>��F�ʍʽ��:�B�=e2"�<�L>2>�!�(7��f�>l}�= =l$V>�o>_���N�L�X> �Nn�����>=�?	u� œ=R�ԽƦ�=S5�;�Vվ`��4�>�K>��R�����S�<�=WΌ���	>�r=�ۇ佾�^>�}F>DýA�>��<E�����V|Լ��=@�U�B>M��<��1>�~>��ҿL!l<�TL>��I>G�޽�J>dM@�?��/PA���½s�>�����=G2=��)>����Bh���>����o(��6;�L��sԽG�c��0���8}�ш����\�J>t����p�;�'-��7�>w� �2Y�����< �#=����n>��=o��ާ���9�>�[�=�Xǽx��=��^<6�H=5�I��'3=b~4>ZWY> �K=̮K=CG>c� ����=&M=>;�9>��tѓ=K4`>�=L%=a��}�=�
�<[Ƚ����>+@��}�� �ޗ=�E"?�� >\8��;Hν91�
H�=`�=��>>R�Y>ʽ L���
����ҽv����x�=~M½�#>��[=�b\�t,>�����C|���>,��<�x=����#>0 ؼ��վ��&>��=����b��H=C�=���j�Q�=\:>K�Ļ��>l�=��>mv�<oY~=��N��q�>Z��g��o�=�Q����OeQ�m�>޽u�?\�6�bĵ==F�=ep[=�6��{�keb>���=��v=a饾n�7>o�j�<c�>�ҙ=��>�	���佚;=��=���=͞��Xc����������ފ<���=����$G>'��>�e�<*�O>͂Ͽ!�b�s8`>�fh�R]>U`;>R�/@�Z����q�"�1����>���=�
3�4�=��=��%�=���=����>d�<���=�{���6V��=)��I�=���=��½��(>!�p=mt���=%�h�f�K���:��+Y=��>�۹���>�Ǔ��>��+�<���>�b��S>isj��*=A'�=��N>�;s=s����g=r�=��A��^}��-+>0?/>�Ƚ� >�!>>>X��?y���<���D��S�?�>']!�?
���F������q:�E*�;7�&>P��<6䮾nS%>\�6=�R>-iսY>$n�=�9"��� >�%=5>�}��=P��\���mf�<��>1��<�^�=�����������$>�&I��	ҽ� _>$�ļ� ��յ=op;?#N���^�B�0�&�>*e^>��;¾-=q|���R�u��=_<eQ�=Hoh>��D=����(2��Q�jjнw<w7�;�����<�v�>��=46j>�����S�(>|��=C@�.�N>;:�>޷�������N�տ�<A">>��fQ�=���<�	=ɚ�I�=V{^��2���i���սP|
�M��>F ><�%�q���E4�<T��*<�	�=�͉=��H>Ѕ ����=c폽B`�qť=�ch�{�1>d�->��=!��=6�>�����Z`>�^�=���=�ۥ<���k><�B>��߽f� �L�R>���;=�b9�l��p��+7�=3�r���xs>L��>4>χ7>?��=�Ľ몝=�@��z�Լ�e3
�3K ?        ��<)���ӄ�C��*�!��z��3�=i�'�
�
�)�1��=Ɏ���;>͍=���������޽�De�����9�u���z/�6�� ��=��X>�����a�J��=V~�����'�>
�ܽ        T��>��7=#�]=د��r?�?=�������=�@��.��>��[>�r����>LVH?���I�>���p��n!��1������N���	�<���;o�>�Q4?�8���K�P�L>h$���SU?�X�       	�=w�@�