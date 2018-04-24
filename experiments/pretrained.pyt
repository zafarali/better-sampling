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
q0X   140229188120576q1X   cpuq2K@Ntq3QK K K�q4KK�q5tq6Rq7�q8Rq9��N�q:bX   biasq;h-h.((h/h0X   140229185112944q<h2K Ntq=QK K �q>K�q?tq@RqA�qBRqC��N�qDbuhh)RqEhh)RqFhh)RqGhh)RqHhh)RqIX   trainingqJ�X   in_featuresqKKX   out_featuresqLK ubX   linear_0qMh%)�qN}qO(hh	h
h)RqP(h,h-h.((h/h0X   140229185043744qQh2M NtqRQK K K �qSK K�qTtqURqV�qWRqX��N�qYbh;h-h.((h/h0X   140229184967728qZh2K Ntq[QK K �q\K�q]tq^Rq_�q`Rqa��N�qbbuhh)Rqchh)Rqdhh)Rqehh)Rqfhh)RqghJ�hKK hLK ubX   h_act_0_ReLUqh(h ctorch.nn.modules.activation
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
h)Rq{(h,h-h.((h/h0X   140229183497152q|h2M Ntq}QK K K �q~K K�qtq�Rq��q�Rq���N�q�bh;h-h.((h/h0X   140229183443232q�h2K Ntq�QK K �q�K�q�tq�Rq��q�Rq���N�q�buhh)Rq�hh)Rq�hh)Rq�hh)Rq�hh)Rq�hJ�hKK hLK ubX   h_act_1_ReLUq�hi)�q�}q�(hh	h
h)Rq�hh)Rq�hh)Rq�hh)Rq�hh)Rq�hh)Rq�hJ�huK hvK hw�ubX   Outq�h%)�q�}q�(hh	h
h)Rq�(h,h-h.((h/h0X   140229183375984q�h2K@Ntq�QK KK �q�K K�q�tq�Rq��q�Rq���N�q�bh;h-h.((h/h0X   140229185856000q�h2KNtq�QK K�q�K�q�tq�Rq��q�Rq���N�q�buhh)Rq�hh)Rq�hh)Rq�hh)Rq�hh)Rq�hJ�hKK hLKubuhJ�ubshJ�ub.�]q (X   140229183375984qX   140229183443232qX   140229183497152qX   140229184967728qX   140229185043744qX   140229185112944qX   140229185856000qX   140229188120576qe.@       Va>Wݭ=��>��=W��o� ���<:.>���Am>|��@��<�	�>�>$�]>6F5���������`���;G���-�j>��Y:�+�>�L?���>ثv>�X[����=��>����q��3�=<rL����[�>-�x>h�>T\J��J>J{�	��>E:���V��}tM����kڽ�
E?C�>Q��>>��=��V>�>��6�d��ķ��hL��}�˼����=�?���M����Ȯ<        )�o��撼kC_��c��=��N�==�����0����wN{��V���Ef���˼w>G�����-����=�m�<@�r="C>ܑ׽4AL>���D⊼��=��>�-��Z�=��=��>}�t<G�Z�       �s�S2r=�S����K=e�.>��2>���k2>Q�>i�����@��%�=X�����%>=:>8�oݥ�CO��������;�g��������I����� �S��i5>��'=���$P&��՟=:��<G��=R��"� �0Ae=䋽��=ƳZ=꘽��L8	>�RY�vY�=�M�=��y8;=�-1�fBݽH��>�R����=RF>S;��q˭��KϽ�Cl���&>?��=�q=��
>�{	>`�ü~�$�%-1�S�">���=m�&�Jn&�/�>�N�=��K>�U�=If��%�Խ���
;��j��_��<^��=Kȍ������%�����h",�@|ν�a=#�>R��=#v�o�s=�쭽l�s��.=�[>Rf�����?����>��+����W]�=��=�����ȍ=~�ֽj��=�/���@Н��?����f���*�[z���½�~��I ���k=h���3�a������� Q����oM����_��=�$	��&�����sd="��>�?P<!4,>��>�/�=�K>�Ѝ>(7=�;���B ��i��5�/���>g凿��.��$��f^�V���J>�7�=4�ξ+�)�[�=>�`>dtV>g��)((��|�=�]��ʤ�k@>��v�-�%�g��=��ύ>��=|M=Ɯ
=�n�C�>V��@�?& �=uk����=?��[��m>�6>�K�=>���=�Av>X�U���=,�=M<v>Y�N>�~��<-�:�s�=�`�8N-�������E� >��=	�&2H�clT��	��1����E>r7$�r3@�
=NJ8= O�?Jc��X����M>
Y�=G����� >e�=��Q�޿��E��=B4#��'����@=<՘@>�B~?+��<Z� >{T={�r��H/>gL�ۛ>'�<&*�>'P�>X󅾝��� >�Z>+��=�12���?]x�9�,z�<_>���/�l������{�<���6��>���=y.�>��4>��>�:G��GE>(����� U#�3�x���>��a>%��S��v]��؈>��ƾ�^�?�x>IdP<�7�>|$�(�;>�P���0�\]�=���=�H<�-8���;=-�=|�=����H>�b彇��=��?*;�=�>�-�c��=��a=����2��4��=7R/>�<=ֶ�����6��%�>�ަ<�	ľ��`?��=���b�,=�pڽ5"�=bD
�G�<a�<<#o=9��<A-��r >�6>մ
������=����\Q����<]�¼ ��=�!��(��ג�O)�ȼT>?�<Q��?�N>�Q�&7/?�D�`����=$�>�f+������>�c�q�������=�1>��������$�=H����ɗʽ�ʁ�c�=ʏ�=��	��̿��g=a��ݠ>x�=f
�\s���� �;ʬ��h�6>-�����u��=�4>>;�@dq�@ ; ��i���L�<P��̯�~㈽`+J=����`u�@@�<����
y��d��)���++���D=����W=d�v��J��4=����\b=�D�9��@s%��Z�<�m�酕�`A��kS=wU!���d�ꡋ=/��2��=��x?�Ji=��>vyH>7��=�S>��4;hȔ=�霽�=>³�=A ��B�P蹿�U�ao=s��6�o?
��=>@4�9V�=a�<^K�<�
ٽv��z�7>^+>�@7��'>Gѽ�������=���Ӗ>��=2,��׎��P��;���>�4�_��=r�=��5����F����\p�p@�//>�{�4!�������>
�A�ф���YO=i�o����SA����>g����;��ֿ?0����v�=���=>瑽GwQ>ɝ�O�>�)��ls>���>�M=F�=p���5�ə�=�
>���?��;$�S���>��(>ɨ�=����Қ>#gJ>=�@�=L�=�>��޽&�m�v��=�������0���4侱@=(���>/z�����=�䞾O����>�+��f@	���_W�V�?����4�|=	�>2[�=�½ǖv���Z>OѾx\�=�)�'
c>�C�<Mѣ=��o��A^�x������;R|i�t?���{=H�νyu�=�$�<�D6>�����ֈ��ށ>�>[J@��>1I�=��v?����սW�q��~�hʚ�F��q,0>c�X�5ۥ=ZeE����>	�=U?>>��={o���N��g����⬾��n��E<��
����=(@�$��@e=(i���_�=5��<�<�?h(<��Ұ0?�2����;��i>�f@>��%���=y�d>M �4�B�b��zd>�+%>�e���,>�������M>�$H�f+O=���=Q7�=��>�$o���a=�ֿ=�a�="<�L=���?"b��e����
?Vʿq��=h}"� �׼]*"<��@��_->�l[��S=Q��α>t������0n=�G�<eT�H��=���l|�0�����V=�S#>��+>~D�ӖP=��m<1a>e����S�?��>���E��>�^�X� =�S>�3�=G���ue�=�) >�V
�Q(��c>q�w>�G>KA=#�������Ԡ�HiB>�E�鎾kH�=&�-�N�>����S�>�9�`�i<����W6�=��?��E>�����S?VK�	7�=B������BC=E4>��޽Y���t]�n��=�I�u�h��	A>AJ�^F]>�t?�o���K>�v�M�C>�R��X��$RĽ/Ҹ�RƯ>�`L>*��=�b�=���:�;t� ��7�<��?������a�}i�=�:<�� ���b��T >!߾h6�C-=�Tʿ����$c��,_)�5R�=�X�>b6>�ھ�'�>���:�>�>��o=��|=��l=�ş�"
�����*y����ȡ>O]1�k�<~���H9���=��)��*��� ��H<X��JΣ��4�<��0��P>���4@��=+wT>x�>�8��6��=ZJ����>-�>�v=���πĽ�<W�,�+�"���=�����
@��6=�Խ�'q:��I>y8��ƽI'6?���>��>�ft�Ȣ�-<�#֙��A>>��?�.m�N���v�=�>q�=PG�=��>�\ >����BI?7o��6�]�L��=)��C��=@�q�!�?�(�p��j�'��������Ʃ� �>�S۽zׄ��/~��a�=�g�=����ʷa>+��?���=]�Ǿ����`>��5>�xB�ui>ԙ��*� \=���<���=�(���d����$��=��?�0��艽3�=4{���8��l�=�Ӽ3���9��A���.�Ӽ/�����s����?J���>���=��=C� >P��=U�$����~>=�f=�$�=ܭz=����v�=&!�=]�)��!�?\�=!s�ڷ>�(<9�\=[�#=�9�>�`�=�
>=,�=���=	����ܵ=��S���|������9¾�A>z�O>%M�=.��<Y�@�2�9��VS��<*>�b?=#O)@GNʽc3�yb�?�:����=��==X�>����a=|Rw>���^�=�$�=�>��9>�����[=ŶV>�ג?�-ֽ��>S��>��?>��>1�=�g�HOo��ܢ=�E%>'��=��5��>��p��=��<�}!�9Y�?���$�ϣ�=�/4>w���0���!>g�_>t#>u���U�=T����N;;�㴼��ٻ}�=1d��==�|�=��������#>g����;ὲk�G�=���=`��3S���j�ŦM��H����:��<��=ƞ�=�������/=�[�=�q>���<)�>e�>I��<�
�=F5>��6����=>�a���=��;�}�7�r">��T��k�=��,���ҽk���pAI�˯����ܽ�:,>7��=�.�TϚ�]�>$DýѮ��n�;T%>w�ƽ��ѽ        ����o>s���h���'����<���=[�2=23l=Vb>�	=#c>��>�?������<e򻽁�Ž�����ͽØ�=b��9����=^�<=F	����=����R3̽�!>�a̼y��=       {��<�=@�*��;	:1�v� >�/�5���̫��x��Y�V=� =��]=��T�FB�AO�=͐�<���h�=O�<��!�tMG>��=�U7��+н��-<�[�M��@��`�N�]�O��� ='�=�u9>��<>W7ܽ�� ���=%��=�Uݽ�e�<DMӽ듼'K���=w)=�ջ�p�8Ͻ5%��$�=Gi2>�6i��-�u�%�#<z�=�B>�g=�M3>20�=I�#>�j=j�9�e=���<�>�Z���L/�Rr?=3u�1߀���'>Q	;6�=�o�Ŕ۽����_��$���->�5�x�<��!��I�����q*>"C*��������eD�<L�N�94�������hȼ�8�=��0>Z��xo=��=��=�����4��q�:E�?��ش�<�u?�h_>�B�<��L>&U�<�#>�0=���f5�!�'=������=���=\�U=��0>�=�01>2��=��;/�7<N]_�����:w>��)>n#�=W����=g7<c���E)5>Y4�;	f��Oмk�_>��=nW>����9A=kХ=�k6=�|6�dΏ<T�v����=�×�_	����Qi�<�:�=�K:A�H=a�r=�u_>M��=��>�Vμ�5�=~�����Խ�:>�ٍ����<.�`>N�=�B�D>͖�=��F>�P�k4�=ܩU�_w)�|dO��d
>Z܇�b�$>���د�>�U1�:��>D��;vz>
��=�8�Ї >C]>:��=�
�::�<#q>�} �nQ�=C=�Ѩ��罱uٽ�_.=='�aY�<9u��Rh=FY��Kx���>k<�y�}=Ӌ=y[:=�x=�=>%��=�s�=�e޽�m=�Q*>p��*?P��W�=R�Q��̽�#�{��kw>�z=��=%ׯ�<F�~�=d�c=ͽ:��Y�h�B�;��}ѽ�=����p�=,�˺�~Z>��,��
w�Μ=\/V=׬������h�R���*>@׽�4=RW9��ƽ�@(<.>��>� =m�	>�Ev:��׽�q'>#c�����<�����i�-=�x["��A=F�(�,�5>9#�=T�Ƽo@���^>�̽��v�^�1�4�d��&��W>��R<t[�<�">3�P>�!н` �=h�D>(,�����FN�L�	�Q��VAϽo�����e�>����^	���=�=��<N�9�����R}�����/=$Kj<�ӽeq�=�ky=^��<���(�1��m_�.p��u>�5ܽ"�ݽ�->��<]wH=���?�>>/[=@2����=�~��:��=��=���:e��=#'7;�_�=z!��%>3�I<ؽA{D���b�٢>�?;>c��e/��ǻ>��5�+�X����=:�<^UC���>t�X;԰V�Y�=>�=��P=�v��3e�|����=���=8¹�	���J=nN>�N$��rl��%=��>霄=�/>>r�ὯH?>h ���1E=W� :�=�:5�z=v�Ƀ<�W[�	�z�8��=��N���9�<:�]����;_���r=�ڽ�9>'#_<��~=�4=�x�����Oн�=*>�j��>T�v=�|���)=#��=5�>��Z>[�^�y��Q�=��>��=���9Q��=K�=)����ڃ�<��*��=���=��=�G`��Q��Gv/��Q����>��>��=�G[=�K�=�oc>	����>A�P�Z���j��ѫ�Kt����8=�h�=$iu�-#=����m �=��5>�0>�,e=U;�����=3%��c��=8	>Re�=�v�=�½�'=�i����=d�=S�$<a*>(� �.4'>Ȯ��Xi >�y�f˃��Μ�H��=�=
�Ὅ[>��=)��=|I�;�9*��R'>$>C�=�3=��F>G��;�<��!>&�2ᄽ��(��b�]=�2E�[ \>I��=Q�����>	���1>�j�=bf�=�H���'��=U�=x:$>��@>V�>-װ=�W�=�oǽ]�.�����)�\ao�������������N�J=*-�=��`>���<��>B�8�Z�B>�E�hW�������,4>����s�=��>��+>g�������,�=;��g�=Tu�����%�UqH��o7�衆�k�=�ͽ�3��E̽IJ�>�<ڡٽ�.(>Ӡ#>���E�Խ�e�ϰ=��p=}�j�n.>݂�=��M=���<�F�=��=�V	�Rҵ<9\��	��&�p>��4>ce�6�->Hy�=	4����=)v;���$��o��= ?;Dvb�z�;�	��9�=1�><��=�?	>�_&��:�">;����"�[�f�P>l��=������<�>�>�O> )@;򧽂���a;�qq�}f���i<����=py�=���1~>���;�C��m�h<��>��==T>0��vo�=T��.7>��'�$,>��>�IV>���=h/>2!���>��:�[�)��5�=����=�s�=��>��=�0�=����o�ͽ/E��*�6��½�i>h>K3u=�~>�`I>���=H�*=��>� �W�N��{�=P�< ��7c��䋽_��%d==��p�Y�0>�p$>�,R>W^�=�>�T�=,��=�h�>�N@==(m��
���=�+��[i�L���z�1<�8=<-�
>�=!����R>o�=(>�i���
>O� �JĪ=�|ؽ�C=c(A�.a>t)�=9b�=��<̓�jn9>�ۯ����;��=Z�I>,�P>9��=���?,>�LZ=1���>���=�;�c�7����<#>��W��P��D>�A����L6c�GFN����D���:�Q���>�����>uW�< 
w>9g >�g=;	>`�=�V6�6�x=t�����=��>>h�< ߍ������Y6���K=�`��v0=C�>|m����;��9=5ܬ<�*>�>��$��l;>��=\��=cz>Z7�+F^=7A�K�*��>7�>�}��T�}޽�V�<�f>�,=l�=�>�5S<�>)�R�+�S]q>2��=�t�r����$�=�*>$�	>4e= ��=vF1>�����<��,�Z��=�,�ڒҽK��j��<�S��\=���<��E���;"Z$�a���4>�`�=@�>��P<�t$>�=r����ܽ�:U=B�A�y�>~�=��λ=r�=O�ؽ���=6b�=��3=iE.���=c�%�)�9��9�ٮ=^��=:H>�e=��1>2ܽ�ˢ���>~M= -�>��=
�<�:ν�����;/>�
k���=}?>٣9;�]ռ����� �am�=��=>�z@��=2��]���0��� �= �X�&�k��J��r^�=o�=��&�` �;���� ��3����=�D={/<>;(�·=�{i��6 >3U=L��=�>{l�<����d�91}=����^Q�X��=2!�;�l��w=	��=�'�U�k7F=]�Y�u>���=�>�W�>��=��n=��;=:�=l��#8�DнiPN�,j��S�=�2�/���A9>��r�z������W'�J�=�,�ǽǁ���V�����ڽ�� ��%$��R1��5>;��=��>l>���/�k>��X>:�����=R�+<�_>�5�=t�*�c�½/w�=��=��?��I�<�^���>�*���H=J�����v���;!�=��(>����������J��Y�&Hʽk��=�e=~>"R��ӑS=q2�J�==��[>�O�=;`��*2�b>�>^�p�E7��N䖽�?4�Cw����i���;�u�=��>-�������>=Bt̽��1>����%�<<�����v��=��<��>sL}=��|Z��yF�*'�<a�v��Z�=���<.,����>=�*�0c-=����������=�=��p�ν�� ��+���@�����)�f�;��>��Ѽ�MM>���=�=GQ=8�"=�ü�ԃ=�$#>        �ڏ�����|-�>k�?���>{�/?��s�[Ԅ=�u1>���>�T�=n��=���>;���o�~��A�ͧ�>�t�����F�<��>�Qi���?��#?5@>�E?s�*;*z��m���b��Y��       ��=p80�@       aQ>����l�>�O�����pP��~B,?�ҿ�� �n�t=���>~�_?��ᾁ"˽|>)>l�?�>˄�?ʗ,�Q�!�t����ͧ�p��=���?;Ҡ>��?�i(�j���_��-��%ɾ.����S���m��p�>������"��?ZF����|�&00?x��?�žQ�꾲4�~r�?���>o����\�
�r?������uo ?��?��= U���ڔ�����Bd�>��ֿe�?�ȿ�b�� A�?