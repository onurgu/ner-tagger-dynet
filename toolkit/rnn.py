
import dynet
from dynet import BiRNNBuilder, CoupledLSTMBuilder, LSTMBuilder


# def ensure_freshness(a):
#     if a.cg_version != dynet.cg().version(): raise ValueError("Attempt to use a stale expression.")


class BiLSTMMultiLayeredWithShortcutConnections(BiRNNBuilder):

    def __init__(self, num_layers, input_dim, hidden_dim, model, rnn_builder_factory,
                 shortcut_connections):

        """
        
        This class implements a multilayered BiRNN with shortcut connections
        
        @param num_layers: depth of the BiRNN
        @param input_dim: size of the inputs
        @param hidden_dim: size of the outputs (and intermediate layer representations)
        @param model
        @param rnn_builder_factory: RNNBuilder subclass, e.g. LSTMBuilder
        """
        super(BiLSTMMultiLayeredWithShortcutConnections, self).__init__(num_layers, input_dim,
                                                                        hidden_dim, model,
                                                                        rnn_builder_factory)
        assert num_layers > 0
        assert hidden_dim % 2 == 0
        self.shortcut_connections = shortcut_connections
        self.builder_layers = [] # type: list[(LSTMBuilder, LSTMBuilder)]
        f = rnn_builder_factory(1, input_dim, hidden_dim/2, model)
        b = rnn_builder_factory(1, input_dim, hidden_dim/2, model)
        self.builder_layers.append((f,b))
        for _ in xrange(num_layers-1):
            if self.shortcut_connections:
                current_level_input_dim = input_dim+hidden_dim
            else:
                current_level_input_dim = hidden_dim
            f = rnn_builder_factory(1, current_level_input_dim, hidden_dim/2, model)
            b = rnn_builder_factory(1, current_level_input_dim, hidden_dim/2, model)
            self.builder_layers.append((f,b))

    def transduce(self, es):
        """
        returns the list of output Expressions obtained by adding the given inputs
        to the current state, one by one, to both the forward and backward RNNs, 
        and concatenating.

        @param es: a list of Expression

        see also add_inputs(xs)

        .transduce(xs) is different from .add_inputs(xs) in the following way:

            .add_inputs(xs) returns a list of RNNState pairs. RNNState objects can be
             queried in various ways. In particular, they allow access to the previous
             state, as well as to the state-vectors (h() and s() )

            .transduce(xs) returns a list of Expression. These are just the output
             expressions. For many cases, this suffices. 
             transduce is much more memory efficient than add_inputs. 
        """
        # for e in es:
        #     ensure_freshness(e)
        original_input = list(es)
        layer_outputs = []
        fs = self.builder_layers[0][0].initial_state().transduce(es)
        bs = self.builder_layers[0][1].initial_state().transduce(reversed(es))
        if self.shortcut_connections:
            es = [dynet.concatenate([original_input_item, f, b])
                  for original_input_item, f, b in zip(original_input, fs, reversed(bs))]
        else:
            es = [dynet.concatenate([f, b])
                  for f, b in zip(fs, reversed(bs))]
        layer_outputs.append(es)
        for (fb, bb) in self.builder_layers[1:]:
            fs = fb.initial_state().transduce(es)
            bs = bb.initial_state().transduce(reversed(es))
            if self.shortcut_connections:
                es = [dynet.concatenate([original_input_item, f, b])
                      for original_input_item, f, b in zip(original_input, fs, reversed(bs))]
            else:
                es = [dynet.concatenate([f, b])
                      for f, b in zip(fs, reversed(bs))]
            layer_outputs.append(es)
        return es, layer_outputs



