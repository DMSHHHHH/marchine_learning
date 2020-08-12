from config_gpt2 import GPT2Config
import tensorflow as tf 
import numpy as np 

from model_utiles import (
    PreTrainedModel,
    TFConv1D,
    TFSharedEmbeddings,
    shape_list,
    cast_bool_to_primitive,
)
from tokenization_utils import BatchEncoding

def gelu(x):
    cdf = 0.5 * (1.0 + tf.tanh((np.sqrt(2/np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf 

# attention 层
class attention(tf.keras.layers.Layer):
    def __init__(self, nx, n_ctx, config, scale=False, **kwargs):
        super().__init__(**kwargs)
        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % config.n_head == 0
        self.n_ctx = n_ctx
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale

        self.c_attn = TFConv1D(n_state * 3, nx, initializer_range=config.initializer_range, name="c_attn")
        self.c_proj = TFConv1D(n_state, nx, initializer_range=config.initializer_range, name="c_proj")
        self.attn_dropout = tf.keras.layers.Dropout(config.attn_pdrop)
        self.resid_dropout = tf.keras.layers.Dropout(config.resid_pdrop)
        self.pruned_heads = set()

    
    @staticmethod
    def causal_attention_mask(nd, ns, dtype):
        i = tf.range(nd)[:, None]
        j = tf.range(ns)
        m = i >= j - ns + nd
        return tf.cast(m, dtype)

    def _attn(self, inputs, training = False):
        q, k, v, attention_mask, head_mask, output_attentions = inputs

        w = tf.matmul(q, k, transpose_b=True)

        if self.scale:
            dk = tf.cast(shape_list(k)[-1], tf.float32)
            w = w/tf.math.sqrt(dk)
        
        _,_,nd,ns = shape_list(w)
        b = self.causal_attention_mask(nd, ns, dtype = w.dtype)
        b = tf.reshape(b, [1, 1, nd, ns])
        w = w * b -1e4 * (1 - b)

        if attention_mask is not None:
            w = w + attention_mask 
        
        w = tf.nn.softmax(w, axis = -1)
        w = self.attn_dropout(w, training = training)

        if head_mask is not None:
            w = w * head_mask
        
        outputs = [tf.matmul(w, v)]

        if cast_bool_to_primitive(output_attentions) is True:
            outputs.append(w)
        return outputs

    def merge_head(self, x):
        x = tf.transpose(x, [0, 2, 1, 3])
        x_shape = shape_list(x)
        new_x_shape = x_shape[: -2] + [x_shape[-2] * x_shape[-1] ]
        return tf.reshape(x, new_x_shape)

    def split_heads(self, x):
        x_shape = shape_list(x)
        new_x_shape = x_shape[:-1] + [self.n_head, x_shape[-1] // self.n_head]
        x = tf.reshape(x, new_x_shape)
        return tf.transpose(x, (0, 2, 1, 3)) 

    def call(self, inputs, training = False):
        x, layer_past, attention_mask, head_mask, use_cache, output_attentions = inputs

        x = self.c_attn(x)

        query, key, value = tf.split(x, 3, axis=2)

        query = self.split_heads(query)
        key   = self.split_heads(key)
        value = self.split_heads(value)

        if layer_past is not None:
            past_key, past_value = tf.unstack(layer_past, axis=0)
            key   = tf.concat([past_key,   key],   axis=-2)
            value = tf.concat([past_value, value], axis=-2)

        if cast_bool_to_primitive(use_cache, True) is True:
            present = tf.stack([key, value], axis=0)
        else:
            present = (None,)

        attn_outputs = self._attn([query, key, value, attention_mask, head_mask, output_attentions], training=training)
        a = attn_outputs[0]

        a = self.merge_head(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a, training=training)

        outputs = [a, present] + attn_outputs[1:]
        return outputs

#
class mlp(tf.keras.layers.Layer):
    def __init__(self, n_state, config, **kwargs):
        super().__init__(**kwargs)
        nx = config.n_embd
        self.c_fc = TFConv1D(n_state, nx, initializer_range= config.initializer_range, name="c_fc")
        self.c_proj = TFConv1D(nx, n_state, initializer_range=config.initializer_range, name="c_proj")
        self.act = gelu
        self.dropout = tf.keras.layers.Dropout(config.resid_pdrop)
    
    def call(self, x, training = False):
        h  = self.act(self.c_fc(x)) 
        h2 = self.c_proj(h)
        h2 = self.dropout(h2, training = training)
        return h2 

# 
class block(tf.keras.layers.Layer):
    def __init__(self, n_ctx, config, scale=False, **kwargs):
        super().__init__(**kwargs)
        nx = config.n_embd
        self.ln_1 = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_epsilon, name="ln_1")
        self.attn = attention(nx, n_ctx, config, scale, name="attn")
        self.ln_2 = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_epsilon, name="ln_2")
        self.mlp  = mlp(4 * nx, config, name="mlp")

    def call(self, inputs, training = False):
        x, layer_past, attention_mask, head_mask, use_cache, output_attentions = inputs

        a = self.ln_1(x)
        output_attn = self.attn([a, layer_past, attention_mask, head_mask, use_cache, output_attentions], training = training)

        a = output_attn[0]
        x = x + a 

        m = self.ln_2(x)
        m = self.mlp(m, training = training)
        
        x = x + m 
        outputs = [x] + output_attn[1:]
        return outputs

#
class GTP2Mainlayer(tf.keras.layers.Layer):

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.use_cache = config.use_cache

        self.num_hidden_layers = config.n_layer 
        self.vocab_size = config.vocab_size
        self.n_embd = config.n_embd

        self.wte = TFSharedEmbeddings(
            config.vocab_size, config.hidden_size, initializer_range=config.initializer_range, name="wte"
        )

        self.wpe = tf.keras.layers.Embedding(
            config.n_positions,
            config.n_embd,
            name="wpe"
        )
        self.drop = tf.keras.layers.Dropout(config.embd_pdrop)
        self.h = [block(config.n_ctx, config, scale=True, name="h_.h{}".format(i)) for i in range(config.n_layer)]
        self.ln_f = tf.keras.layers.LayerNormalization(epsilon = config.layer_norm_epsilon, name="ln_f")
     
    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, value):
        self.wte.weight = value
        self.wte.vocab_size = self.wte.weight.shape[0]
    
    def _prune_heads(self, heads_to_prune):
        pass 


    def call(self, inputs, past = None, attention_mask = None, token_type_ids = None, position_ids = None, head_mask=None,
                    inputs_embeds=None, use_cache=None, output_attentions=None, output_hidden_states = None, training=False,):
        if isinstance(inputs, (tuple, list)):
            input_ids = inputs[0]
            past = inputs[1] if len(inputs) > 1 else past
            attention_mask = inputs[2] if len(inputs) > 2 else attention_mask
            token_type_ids = inputs[3] if len(inputs) > 3 else token_type_ids
            position_ids = inputs[4] if len(inputs) > 4 else position_ids
            head_mask = inputs[5] if len(inputs) > 5 else head_mask
            inputs_embeds = inputs[6] if len(inputs) > 6 else inputs_embeds
            use_cache = inputs[7] if len(inputs) > 7 else use_cache
            output_attentions = inputs[8] if len(inputs) > 7 else output_attentions
            output_hidden_states = inputs[9] if len(inputs) > 8 else output_hidden_states
            assert len(inputs) <= 10, "Too many inputs."
        elif isinstance(inputs, (dict, BatchEncoding)):
            input_ids = inputs.get("input_ids")
            past = inputs.get("past", past)
            attention_mask = inputs.get("attention_mask", attention_mask)
            token_type_ids = inputs.get("token_type_ids", token_type_ids)
            position_ids = inputs.get("position_ids", position_ids)
            head_mask = inputs.get("head_mask", head_mask)
            inputs_embeds = inputs.get("inputs_embeds", inputs_embeds)
            use_cache = inputs.get("use_cache", use_cache)
            output_attentions = inputs.get("output_attentions", output_attentions)
            output_hidden_states = inputs.get("output_hidden_states", output_hidden_states)
            assert len(inputs) <= 10, "Too many inputs."
        else:
            input_ids = inputs 
        
        output_attentions = output_attentions if output_attentions is not None else self.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.use_cache

        if input_ids is not None and inputs_embeds is not None:
            pass 
        elif input_ids is not None:
            input_shape = shape_list(input_ids)
            input_ids = tf.reshape(input_ids, [-1, input_shape[-1]])
        elif inputs_embeds is not None:
            input_shape = shape_list(inputs_embeds)[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")


        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = shape_list(past[0][0])[-2]

        if position_ids is  None:
            position_ids = tf.range(past_length, input_shape[-1] + past_length, dtype=tf.int32)[tf.newaxis, :]
        
        if attention_mask is not None:
            attention_mask = attention_mask[:, tf.newaxis, tf.newaxis, :]

            attention_mask = tf.cast(attention_mask, tf.float32 )

            attention_mask = (1 - attention_mask) * -10000.0
        else:
            attention_mask = None
        
        if head_mask is not None:
            pass 
        else:
            head_mask = [None] * self.num_hidden_layers
        
        position_ids = tf.reshape(position_ids, [-1, shape_list(position_ids)[-1]])

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids, mode = "embedding")

        position_embds = self.wpe(position_ids)

        if token_type_ids is not None:
            token_type_ids = tf.reshape(token_type_ids, [-1, shape_list(token_type_ids)[-1]])
            token_type_embeds = self.wte(token_type_ids, mode="embedding")
        else:
            token_type_embeds = 0
        
        hidden_states = inputs_embeds + position_embds + token_type_embeds 
        hidden_states = self.drop(hidden_states, training=training)

        output_shape = input_shape + [shape_list(hidden_states)[-1]]

        presents = ()
        all_attentions = []
        all_hidden_states = ()

        for i, (block, layer_past) in enumerate(zip(self.h, past)):
            if cast_bool_to_primitive(output_hidden_states) is True:
                all_hidden_states = all_hidden_states + (tf.reshape(hidden_states, output_shape))
            
            outputs = block(
                [hidden_states, layer_past, attention_mask, head_mask[i], use_cache, output_attentions],
                training = training
            )

            hidden_states, presents = outputs[:2]
            presents = presents + (presents,)

            if cast_bool_to_primitive(output_attentions) is True:
                all_attentions.append(outputs[2])
        
        hidden_states = self.ln_f(hidden_states)

        hidden_states = tf.reshape(hidden_states, output_shape)

        if cast_bool_to_primitive(output_hidden_states) is True:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        outputs = (hidden_states,)

        if use_cache is True:
            outputs = outputs + (presents,)
        if cast_bool_to_primitive(output_hidden_states) is True:
            outputs = outputs + (all_hidden_states,)
        if cast_bool_to_primitive(output_attentions) is True:
            pass 

class GPT2Pretrained(PreTrainedModel):
    config_class =  GPT2Config
    base_model_prefix = "transformers" 

class GPT2Model(GPT2Pretrained):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.transformer = GTP2Mainlayer(config, name="transformer")

    def call(self, inputs):
        outputs = self.transformer(inputs)
        return outputs

if __name__ == "__main__":
    DUMMY_INPUTS = [7, 6, 0, 0, 1]
    DUNMY_INPUTS_TEST = [[7, 6, 0, 0, 1, 2], [1, 2, 3, 0, 0, 3], [0, 0, 0, 4, 5, 2]]
    pretrained_model_name_or_path = "model_settings/gpt2-config.json"
    config = GPT2Config.from_pretrained(pretrained_model_name_or_path)
    model = GPT2Model(config)
    res = model({"input_ids": tf.constant(DUMMY_INPUTS)}, training=False)
    res = model({"input_ids": tf.constant(DUNMY_INPUTS_TEST)}, training=False)