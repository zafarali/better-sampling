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
q0X   140471484311632q1X   cpuq2K@Ntq3QK K K�q4KK�q5tq6Rq7�q8Rq9��N�q:bX   biasq;h-h.((h/h0X   140471483601696q<h2K Ntq=QK K �q>K�q?tq@RqA�qBRqC��N�qDbuhh)RqEhh)RqFhh)RqGhh)RqHhh)RqIX   trainingqJ�X   in_featuresqKKX   out_featuresqLK ubX   linear_0qMh%)�qN}qO(hh	h
h)RqP(h,h-h.((h/h0X   140471484327120qQh2M NtqRQK K K �qSK K�qTtqURqV�qWRqX��N�qYbh;h-h.((h/h0X   140471483889328qZh2K Ntq[QK K �q\K�q]tq^Rq_�q`Rqa��N�qbbuhh)Rqchh)Rqdhh)Rqehh)Rqfhh)RqghJ�hKK hLK ubX   h_act_0_ReLUqh(h ctorch.nn.modules.activation
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
h)Rq{(h,h-h.((h/h0X   140471483997680q|h2M Ntq}QK K K �q~K K�qtq�Rq��q�Rq���N�q�bh;h-h.((h/h0X   140471483716544q�h2K Ntq�QK K �q�K�q�tq�Rq��q�Rq���N�q�buhh)Rq�hh)Rq�hh)Rq�hh)Rq�hh)Rq�hJ�hKK hLK ubX   h_act_1_ReLUq�hi)�q�}q�(hh	h
h)Rq�hh)Rq�hh)Rq�hh)Rq�hh)Rq�hh)Rq�hJ�huK hvK hw�ubX   Outq�h%)�q�}q�(hh	h
h)Rq�(h,h-h.((h/h0X   140471483562256q�h2K@Ntq�QK KK �q�K K�q�tq�Rq��q�Rq���N�q�bh;h-h.((h/h0X   140471484120864q�h2KNtq�QK K�q�K�q�tq�Rq��q�Rq���N�q�buhh)Rq�hh)Rq�hh)Rq�hh)Rq�hh)Rq�hJ�hKK hLKubuhJ�ubshJ�ub.�]q (X   140471483562256qX   140471483601696qX   140471483716544qX   140471483889328qX   140471483997680qX   140471484120864qX   140471484311632qX   140471484327120qe.@       ���_kԿ:=Q=�!��zw�?�Xm?�Eٽ E�>��:��qw��}+�>y�ɫ>��Ǿ�"O�[�>�>Tk�>�a�=E��>��ν���?�_��Z��nw_?G{����p>;L)�CE9>o��S��?$t��R��?pI����>��{�{a���=�~��wh�>���=��<�ȫ��E�=e���;�?�c�=��n�Y�>�ր�Õ>>��X�K$$>J1��Q��>�f>��u�y�>Lb�{	<@����3x�        ��q��E=������:��F	��KE��iO��Y2?������=w�>Jt�B#>���P�>K�
���|<ӏ�>BH�=�l�����d!�?�=���?xc>�k6�u��>�2?�A;?��>        ��3�Ot�=�;�=���;W����q��#
�w¼l��ٶ�m�> v5>�>��4=~�K=�fE>�"%<�n'��7>��8�S,0=�l��ШL<AA��[�=ןI=5�=vN�=��=P=���ꙃ�        &/^��;�F����b<�Y���]��"{��9=G�t==.��޼*�M=��ؼ�=
�4�P�a<��
>�f�;3��x�I�������ս�=e߽=+�:��E���7�=}Ǟ<y��G� �*�       ]�>z���u>Z�ټ ��J�Ͻ��G=�C>6���h�������	>�M;>oG>����	> ��;ibP��
=���=��4�'=���K�b=}K�9���=���>4l>n���8�=�ag��w>>L�=�v=�!�Yg�?&5o>D7>g���sԥ=\�a���@�$<��"%�>�[�>��<��W���?�$H�� Ⱦ.zž�K�=W�[�,��=p����J?�P��\�i�?��y>ɂF�-J�[�}����<�X��O;,7>����ee��j>	7A��T�=���=�м9�[>��*Si=1�ҽDb�e:�bc�=�9�od�����bbC=ˁ>ҵ��]���,>~�v�S�)(�r���y�=8��<Կ<�4�нw�ؼ��X=c���=
�9�S�=��>��52�v>��D��[s���D=��/�R������>6��x�T�J�����P��=l�����D�7>?&佉��#��kо��罵q�>,	[=ߏ!=�c�<-���_��6�>�Mg=�SZ<���A��=�y���\Ѿ�
�=v ��&R>�'�=����~�8��=�7�>#߾������>��[?�~���v����>�vd�F6a��'v?��=��̽6ǝ���Z��B	м����N��>��>�����7 ;o�<k>u읾q=��a�X�#>-��=�ʾ���y�>��>�>e�����%=� X?�ھ�Z��2�>����J�}��?j�D�j����=�c>Q%>�D����9?4��<�&u>�����<� ��"M��>���=�MK>tK>��$>��?J�m>���Z���U��'�;=/��<����R���=�&����?��>8|����=fE>7�>>sIg=B>�`�>�/�)鞽�= n�=��>U��������>@4|�i0��|f����k���>���r�^�56�Κ����=)�y��@"�� ���䥾�j��Q��n�B�>SS<>��X=5@:V�X=�M�
rk?�ӽ=���<��;>�mW�5�;y��f�u��=f��>y@7>���=��;?��H=*���Ҭ<�a7�����o���xf�����>��o�'C�Ij�?�Y�>����<�E����=α�34>E{�=` v?��p�>�;{6>�T)>�y�=�y<���=��m�G�s=�v>?o>�ĉ?`¥�x�;��M׾�u=)&=H�=�`�Z�?.��<:z>J?���<!���x�=T!�=�wQ=+���m�=$"Q=&�>D �Z�Q>�(	>�_-���=z��1>i.v>}{�>�qW>��8� ?���<����k���Ȑ�>�!>��=DӾ��>d��=(:��39L?d�;>Pƾ��=jv�=Z�j�FZ>��R�=����0N�=���hٽ��ڼ�>��м-=h;���=lȢ���Ƚ�>js*��.=���=M�.?~ۂ>��<>|�K���^?}�Q�8¼��[>�S����=��?�=���<���=`�b���H�B�!>��Z?�G*�p��=��3>E�W=��)>=�	>�ㄽ�x>
�9>�/�=?+�=Ǌn?�R�>G;߽���{�=�Z>��M:��7�'�-?�UB����a�|?S�x>T9�n��l�	�V�!>�.��ܜ=�F�UP��>d��>�������|r=�肽*�a�Ђe��	����=m9}=�G=_3�U92=Na�> ���_B�u�t=��>b|��+]!�K�=ξ"�>�|&?�9g��)��xd�:�~<]��=�^��Us?60��Q�=o����aX��h=,u�
��<��>6m��k�m�4?I���#��<�U���h��
�=#�۽�)&�9�>͘���Sؽ���?���>l�=Q�=��N������P<wB���cN?>�;ݽ<H:V�b��<��=2��=�Y���?>Ỏ>Lţ�z��[x)?��S>7(1�/	��e�Z��K��=-,�K��>��05(��I�?�=]x��� �$�w<s��=8�>\�>���ڕ�Z8X>��>�=u�h=�c<��r�j�x�S������I��=�->��<یؽ�_>q��>��0��+~��uf=�G4?;R������e�=��M�MT�>NSV?N<�3���&bt��V=\�=�?>�F>�k=��F��[���v���S9ӓ�CK�=c�Y=1-���'������?�p&��D�>�~�=��N�3�齇�G=�q=9ݬ=�z�=E�=�	>� s�1a����W�{ރ<���!���>9R���t�kd�>_���&��~��b2>4>:�<p�2�����̽>*� �?������;���=u�|>�X��?
 >C)=]�?P�g|?>!�b>$�c�S=JG�>qJ&�����y���tbY�3B�ě�<�@��}:��ڊ��&O=���!/�))�,_�E�>/r��?��=;`���m��G׽�O�|/�� ��<�Z=�����>�=.=z����H�=`{��u'������=�=�$���3�]H5>��c�I]P��2�j��>�tr>x�->ҋ�;R�>�p��K!/�G���_��'!>��=b�H�1m�Ǔ�=m�>y0��S��=n[�=X��>�d�<�'�<v��c=�縏>�t�>�>���#�&�.������ <a]��a'J�yD=�����-BN=3*:������͍=J[k��e	��:N�����d�O�Pʈ>6��žf����謼؋A�����$���$=>�n=�k�FJ�9My�`��=�o�>��">P�׽ժ��&<�9�����>�}�>�R���<m9=B2�<O��������M=u���"�>Ӷ��ycؾ(\�>*�)?�,�:�+��]�=܋m?�c>$�&��>�=�Œ�߸�>��??T >���6�=��;|>�I��]P?��1��I_>�*/=P(��]F��K��=�$��wǑ>��>���<����?�#�F>G;r���t[C���������5A�%V�>_�Ѿ
7p�={?*�R>���<��=}�����=��ν �)<���dI?�wȻ�A�=��<��
�K��=�<�>�YkA>�Ʀ>��n=7kW=@� ?��S-�i\ѾaŻ<��=����' �ju�>�ν��0��v_?�>�=�X���6ֽ��?�����z>�Q�=�;x����P�>�6�=�x=6�!?�<fCC=�1
�����6��g>R�3>�s���پ�^�>A�>����=�z�<H6e?��)���=��3>�|���=I2A?�&�����G2�w
U�cD뻦�>�톿Q��=�N���<�<�#>�<]Λ>/m���;Mv������G���Q2������Tɾ
��=�J�=��='�p�Ip,�M]<����Mᚿ=�H�x���gc;�����޽`Y�<5���SA����	���׾j�Y>���3�=��eX=Y/R=�*�V̀;�B׻3��Ep���ϕ��)j�d���55=�>F�+�ƽ��>&�>!�RО�ɐ�pe>�&$�Fe�A�����V;I9>�r�=��>�>�E�=�.�`�8>&�j>��>g�>P��V�=ԁV>�2�>7�=��?��>���hA�6��=�->$,v��8-�9�>��G�؁�8V?�<�>�OȾ�G��-yP=�-����17=���|�|����t����3�,�������S>�̷�v�׼wzH>��]>N�<N�ؾ����˻=Y�M=�4K>+��e[O�J���䘾�������<��%~��s�=�l�n�\��l�#_�=4^�;+�s��=�J
>f>���=�0U>K�_=q�S��)d=�>�ˠ=�y<8>_�)v <�.���|�='�;�
�����=?r�>�F�3�>x_=����]J�֠2>S_�=1x��  >�>��6=�F7���}��>p���N>����=��0�qՍ�n����ռ��=J�W>io���[��!J>dM
?=QM>���=���='$�?�ľGb�:�z>���Y>�ҏ?5D�=uR����ؽ       vV=~��=@       �s��'�m>f�b>� �`�)=��?��>����b�3<I^?��e�D��	m�>��u?X����0�?b��>Cb�?�-�����k/����?S��%/������߿���>0��?9�	>�����/�%��g�����������w@�+�=HTX����i�?���>EA���k�>���=ݻ��ѿ��/���g�n�� @v6�>�]޿���>��?,��=���R >�9���Ӿ~�?�>��?��ѻqvݿ       �=pڠ=R[L=�E�=o��yK��Z;���X�1.����;�=s���1��I�=k�=��=`LG�Z�_=d�m=��#=� W=�Լ�P>9&�;�0G�����������i�4>}�:�X�<a7��<X�������.i�=`5�=�0ɽώ=kP��_��<z��覘= s�=�f�j��V=<f >-�=�2$>`�3��<����=�3 ��p�=�D=�ʽ�C�=S��=#o>��˽����l@�h4?��>�Y1<��=��p��=�	Ͻ	D�<J��>~n�=�6�Yf�<c�i�<i>�u;����;��=f>fQ���<?����@���qX�t���mZ���=?�S�(><F���H/<��s�������4>��YD>:�w��(>륪=Z��UX=�M��'H�Gf�>��L>0��}7�= 9�=2M��޲��*=\�Ŀ���<�����d>��=-ࡽkkM>��+��D�>� �=��c���A��?g-v=k�C>Ah������	�<ݛ�>	u�K�~�"�˽�a`�G驺A�>�.�=�=��>�>~B>��G�!��Ӌ����}���{�>E��=]�W�"<�=����G>?�սJq
�z@����S?��=>%�
,=�x�yf�=���=�2D��@_�V>�i]�z�6>�e��9�;�q|>����]`>=�<Dzν�����y=�2>"F�=����%�=\��= =LYF>Y->7��<�^�=��?/��="E�=E==R�)>�F��	>�,ͽ�.��?����Z�]�5���ֽ�>�^��=�	����&>�/��I�=����4(�i�V����>���=<r�=T:���;�%4>�5X<�=e�d�g��=��D=��#����=^i��x.>�=2��=�ǽZm>;L>m���6�:�����=�#ٽ�J#��j&��)=I&�� �>�׿�X��px=�!#�y�ʽ�3�5��=]��=��b��Z >u�%�䇊�?��=���<f�νY��<�V)>/>�g�۾/�4�9�g|�=���H*b>�x�=���KOQ�D7m���b=w4&��6>��/�Kܽ��t��e�>����̼�`u=)/�=�Y}=�OO=?@���,���(?�!N������D�J�=�\>���<���=��>(#�<� Ƚ홷=Ho=�׼W�_b*�)a.>E��<�4�9���������=�-:�ս�>���P��+�M�8l�!�=<*\��kh�=�rO���=!
�=��>����>dc��t�=��=F2%='Y>:�U>�~�=�#=� (�3�5��^=(��9:����3�{��;�!2��J�?&m�=$E�򠳾ۜ�E���^B=�`>�|��d/>c*1>��ʽ)N|�vv>��>��־��#>7�.^	>��>��~�����vk�Ѵ�PE>�=�<t�,�֞�>��׽HH>z+=��_=�5��zݹ=Ȼt=�(h>у�=W��<F����1�=�5>��*�%;��>C��?�K�=�^L>�o��Q9�<ż]|>n�!�提���O=�{��މn���F>�7><wt=g��<���=h�2=��F!>kS���ro����<��>3��cGV�э�=u5�<PH�>+9W=�k ��c��2�?��=�'�>q�־?�=U����(<��7=ԏ#�h�+�I��;�*�s��>>��߶���9`48��#>�(����>V�"��8�<Ӏ�:��>>�i,�q�����&�0ͅ>���������>�J?w┼���Hyľ�W�=���=���=GO��M��{^׽1'z�����s2^>��>
3�=�<j� =��=żj�8R=1LM�y~Խ��l��y�>�cm< ���Z���~=!��>I����<�8۽��?�^���=�ݝ�l�>��=��U=v�ɽ�E��$?=k�=����
>u�$;/'�6�=9���F-���fY���$>�sz�0{�=���=]�#>��M��=8��c�;�������36/��c?�t̼��h���>��=#>|\⽼ku�nxc>�H>�]�=��=� � v7=5�=Gͽ��C�9㫻�s>�~�&}3?��ƕ=?�¾(��J�j>��R��Q��l2���e�=\��=�d>4{!�=�=޹�=��[�e4>	�=�g�=�?I��t�;AS;��>_�%����D>bbϽ��=[3�H������
2>(�0�5,���=br	>�n>`���v	�20�=�U�>��=�n�=r���w��>l���=>�땾��'<Z�&>7P�����붻3h�i&>CH=]�պ��5>
�=&G�=�>���>滽*P:����)>m�=�z�>!��<�/'���/>�QF���d=�u;=���<�`V<hpS>�j��S��<�*Q=DA=�t5>��<�����Ⱥ�5p>������?�<0!8>jЛ=��=/H?>xk=�9��IZ��"L��c	��5��Om�>2�=[7=;�,����=�~>��=��)>_E�=ZT�=>'�=.�l��o�>�/7�}`��=zS>��>VϪ>��Q<�p�=N�q=��=$(����=g9ͻ��ͯ���<9>4��=t0�>���=��޽���=��ļsې���M=�=L��=g��=Y->�1=���j��;��=4 P�2�/�=���=}�������4C>&S�+��=�7O�9�=�D�=L.>A�L=@O1>�v >�5��p 	<�����(�=�8�=�'o� ]ý�n>�g>"5����h=�ݙ�ɵ.�&7���=�!�=;u��N������yV>h�Ž��oO½ A_<��N���O=�H�<��l�r�O>�5>2x+>��d��\=Y�*��؞�4�>��+>�CO��P����=��|=Q!>�j-=yN�Z! �J�?�}�=�|o>������=S���z7�=����_Y��K|������[v���>L&�=>��;���=P�>���=~��< X0��lU�"��s��=�^O>��R>����>����>���B;���=�9?��>�ǒ;�J>�02=��S>�t��
P �g�W>]�>��;WS�������=7?#��FE����D�E�R�
>�! �	�.?���<!�DϾ�.
�E�<W�-��>V G��=�[��;/|=��5�9�S�h���8���~�=2��",V=��
�������R��������<�>�>�V���=���K>��Z=W%p<J�c��<�<�Y�����>ǱX>f��<�8�<`��=��y>�7��EN��@���x?����W�=6޼�w�=���7�6>�=�v}�o��=W6��8P/�c>L�*>�[�Ν�������0>�E��<=UO��u�=���a�@>�W$>N�k�*�=(k��N�>��=S�%�V�g$�?晕��X�=�U��K$�=ϣg�֞ =;�=蠇���T����I9��9'�>?���H%=�t|>a!3>F��<o1'�U+���9��V�t��f�=��>��C>��=X�μ��>��>(W���P�R���?ٚ=U��=B��P�C>�&<�KG>1י��&þ1n���F>d�:=��o>��=��#�wEj>���FOJ>�w��8}��mq�u�i=�7���@�>�M>��ʽ�2>�k�5W�>V��=Ƽ'ZS�á/?�8>�V�7����u�=��=�����=��s�]ɓ��V<M�E�c�J>�G=J�=�!`�`�'>����M�=i_�=� Ǿ{�ּ��ػAܖ>�ý	7���v�=��[wp�V?��Ӽ�Ҽ�b�>{��;�+�=U/>:�̽���=�+�=�
�k��>[��������=�a����G6>i�C~�=pk
�sb>y���ޒ�>���=�H�=:�j������y>w�Y���>����<c4��>�󽾵�9=�R��1�_>��9�'��;mV�=��:>���<~`��̔z�y�=�.�fG�=� ��D�B>�ޑ=�	�(2�<'Ҽ�0b>�#0>g�-�.�n>�>�?��uA�=E��=f��T���B޽��D�a!m�