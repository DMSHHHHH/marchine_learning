import tensorflow as tf 
from config_gpt2 import GPT2Config
import os 


def get_initializer(initializer_range = 0.02):
    return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)

class TFConv1D(tf.keras.layers.Layer):
    def __init__(self, nf, nx, initializer_range=0.02, **kwargs):
        super().__init__(**kwargs)
        self.nf = nf 
        self.nx = nx 
        self.initializer_range = initializer_range
    
    def build(self, input_shape):
        print("building", input_shape)
        self.weight = self.add_weight(
            "weight", shape = [self.nx, self.nf] , initializer = get_initializer(self.initializer_range)
        )
        self.bias = self.add_weight("bias", shape=[1, self.nf], initializer = tf.zeros_initializer())

    def call(self, x):
        bz, sl = shape_list(x)[:2]

        x = tf.reshape(x, [-1, self.nx])
        x = tf.matmul(x, self.weight) + self.bias

        x = tf.reshape(x, [bz, sl, self.nf])

        return x


def shape_list(x):
    static  = x.shape.as_list()
    dynamic = tf.shape(x) 
    return [dynamic[i] if s is None else s  for i, s in enumerate(static)]

class PreTrainedModel(tf.keras.Model):
    config_class = None 

    def __init__(self, config, *input, **kwargs):
        super().__init__(*input, **kwargs)
        self.config = config
    

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # load config
        config = GPT2Config.from_pretrained(pretrained_model_name_or_path)

        # load model
        if os.path.isfile(pretrained_model_name_or_path): 
            model = cls(config, *model_args, **model_kwargs)

        model.load_weights(files_path, by_name = True)

        # check model
        # model(, training = False)


class TFSharedEmbeddings(tf.keras.layers.Layer):
    """Construct shared token embeddings.
    """

    def __init__(self, vocab_size, hidden_size, initializer_range=None, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.initializer_range = hidden_size ** -0.5 if initializer_range is None else initializer_range

    def build(self, input_shape):
        """Build shared token embedding layer
        Shared weights logic adapted from
            https://github.com/tensorflow/models/blob/a009f4fb9d2fc4949e32192a944688925ef78659/official/transformer/v2/embedding_layer.py#L24
        """
        self.weight = self.add_weight(
            "weight", shape=[self.vocab_size, self.hidden_size], initializer=get_initializer(self.initializer_range)
        )
        super().build(input_shape)

    def get_config(self):
        config = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "initializer_range": self.initializer_range,
        }
        base_config = super().get_config()

        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs, mode="embedding"):
        """Get token embeddings of inputs.
        Args:
            inputs: list of three int64 tensors with shape [batch_size, length]: (input_ids, position_ids, token_type_ids)
            mode: string, a valid value is one of "embedding" and "linear".
        Returns:
            outputs: (1) If mode == "embedding", output embedding tensor, float32 with
                shape [batch_size, length, embedding_size]; (2) mode == "linear", output
                linear tensor, float32 with shape [batch_size, length, vocab_size].
        Raises:
            ValueError: if mode is not valid.

        Shared weights logic adapted from
            https://github.com/tensorflow/models/blob/a009f4fb9d2fc4949e32192a944688925ef78659/official/transformer/v2/embedding_layer.py#L24
        """
        if mode == "embedding":
            return self._embedding(inputs)
        elif mode == "linear":
            return self._linear(inputs)
        else:
            raise ValueError("mode {} is not valid.".format(mode))

    def _embedding(self, input_ids):
        """Applies embedding based on inputs tensor."""
        return tf.gather(self.weight, input_ids)

    def _linear(self, inputs):
        """Computes logits by running inputs through a linear layer.
            Args:
                inputs: A float32 tensor with shape [..., hidden_size]
            Returns:
                float32 tensor with shape [..., vocab_size].
        """
        first_dims = shape_list(inputs)[:-1]

        x = tf.reshape(inputs, [-1, self.hidden_size])
        logits = tf.matmul(x, self.weight, transpose_b=True)

        return tf.reshape(logits, first_dims + [self.vocab_size])


def cast_bool_to_primitive(bool_variable, default_tensor_to_true=False):
    """Function arguments can be inserted as boolean tensor
        and bool variables to cope with keras serialization
        we need to cast `output_attentions` to correct bool
        if it is a tensor

    Args:
        default_tensor_to_true: bool, if tensor should default to True
        in case tensor has no numpy attribute
    """
    # if bool variable is tensor and has numpy value
    if tf.is_tensor(bool_variable):
        if hasattr(bool_variable, "numpy"):
            return bool(bool_variable.numpy())
        elif default_tensor_to_true:
            return True

    # else variable is bool
    return bool_variable
