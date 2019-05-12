# models/esim_modify
class MyModel(object):
    def __init__(self, seq_length, emb_dim, hidden_dim, embeddings, emb_train, total_relations):
        ## Define hyperparameters
        self.embedding_dim = emb_dim
        self.dim = hidden_dim
        self.sequence_length = seq_length 
        self.n = total_relations

        ## Define the placeholders
        self.premise_x = tf.placeholder(tf.int32, [None, self.sequence_length]) # change from (32,50) to (32*n,50)
        self.hypothesis_x = tf.placeholder(tf.int32, [None, self.sequence_length])# change from (32,50) to (32*n,50)
        self.y = tf.placeholder(tf.int32, [None])
        self.keep_rate_ph = tf.placeholder(tf.float32, [])
        
        ## Define parameters
        self.E = tf.Variable(embeddings, trainable=emb_train)
        
        self.W_mlp = tf.Variable(tf.random_normal([self.dim * 8, self.dim], stddev=0.1))
        self.b_mlp = tf.Variable(tf.random_normal([self.dim], stddev=0.1))

        self.W_cl = tf.Variable(tf.random_normal([self.dim, 3], stddev=0.1))
        self.b_cl = tf.Variable(tf.random_normal([3], stddev=0.1))
        
        ## Function for embedding lookup and dropout at embedding layer
        def emb_drop(x):
            emb = tf.nn.embedding_lookup(self.E, x)
            emb_drop = tf.nn.dropout(emb, self.keep_rate_ph)
            return emb_drop

        # Get lengths of unpadded sentences
        prem_seq_lengths, mask_prem = length(self.premise_x)
        hyp_seq_lengths, mask_hyp = length(self.hypothesis_x)


        ### First cbiLSTM layer ###
        premise_in = emb_drop(self.premise_x)
        hypothesis_in = emb_drop(self.hypothesis_x)# change from hypothesis_x[:,0,:] to hypothesis_x, [32 * 50] -> [(32 * n )* 50]

        hypothesis_outs, c2 = biLSTM(hypothesis_in, dim=self.dim, seq_len=hyp_seq_lengths, name='hypothesis')
        # calculate premise based on the condition of hypothesis
        with tf.variable_scope("conditional_first_premise_layer") as fstPremise_scope: # use average
            premise_outs, c1 = reader(premise_in, prem_seq_lengths, self.dim, c2, scope=fstPremise_scope)

#         (premise_out0, premise1) = premise_outs
#         paddings = tf.constant([[0, 0], [0, 0, ], [0, 300]])
#         premise_bi = tf.pad(premise_out0, paddings, "CONSTANT")
        premise_bi = tf.concat(premise_outs, axis=2) # (?, 50, 600)
        hypothesis_bi = tf.concat(hypothesis_outs, axis=2) # (?, 50, 600)

        premise_list = tf.unstack(premise_bi, axis=1) # premise_list[0] -> (?, 600)
        hypothesis_list = tf.unstack(hypothesis_bi, axis=1)


        ### Attention ###

        scores_all = []
        premise_attn = []
        alphas = []

        for i in range(self.sequence_length):

            scores_i_list = []
            for j in range(self.sequence_length):
                score_ij = tf.reduce_sum(tf.multiply(premise_list[i], hypothesis_list[j]), 1, keep_dims=True)
                scores_i_list.append(score_ij)
            
            scores_i = tf.stack(scores_i_list, axis=1)
            alpha_i = masked_softmax(scores_i, mask_hyp)
            a_tilde_i = tf.reduce_sum(tf.multiply(alpha_i, hypothesis_bi), 1)
            premise_attn.append(a_tilde_i)
            
            scores_all.append(scores_i)
            alphas.append(alpha_i)

        scores_stack = tf.stack(scores_all, axis=2)
        scores_list = tf.unstack(scores_stack, axis=1)

        hypothesis_attn = []
        betas = []
        for j in range(self.sequence_length):
            scores_j = scores_list[j]
            beta_j = masked_softmax(scores_j, mask_prem)
            b_tilde_j = tf.reduce_sum(tf.multiply(beta_j, premise_bi), 1)
            hypothesis_attn.append(b_tilde_j)

            betas.append(beta_j)

        # Make attention-weighted sentence representations into one tensor,
        premise_attns = tf.stack(premise_attn, axis=1) # (?, 50, 600)
        hypothesis_attns = tf.stack(hypothesis_attn, axis=1) # (?, 50, 600)

        # For making attention plots, 
        self.alpha_s = tf.stack(alphas, axis=2) # (?, 50, 50, 1)
        self.beta_s = tf.stack(betas, axis=2) # (?, 50, 50, 1)


        ### Subcomponent Inference ###

        prem_diff = tf.subtract(premise_bi, premise_attns)
        prem_mul = tf.multiply(premise_bi, premise_attns)
        hyp_diff = tf.subtract(hypothesis_bi, hypothesis_attns)
        hyp_mul = tf.multiply(hypothesis_bi, hypothesis_attns)

        m_a = tf.concat([premise_bi, premise_attns, prem_diff, prem_mul], 2) # (?,50, 2400)
        m_b = tf.concat([hypothesis_bi, hypothesis_attns, hyp_diff, hyp_mul], 2) # (?,50, 2400)


        ### Inference Composition ###

        v2_outs, c4 = biLSTM(m_b, dim=self.dim, seq_len=hyp_seq_lengths, name='v2') # hypothesis
        # same to hypothesis premise part, calculate v1 based on v2 during Inference Composition
        with tf.variable_scope("conditional_inference_composition-v1") as v1_scope:
            v1_outs, c3 = reader(m_a, prem_seq_lengths, self.dim, c4, scope=v1_scope) # premise

#         (v1_out0, v1_out1) = v1_outs # premise
#         v1_bi = tf.pad(v1_out0, paddings, "CONSTANT")
        v1_bi = tf.concat(v1_outs, axis=2) # (?, 50, 600)
        v2_bi = tf.concat(v2_outs, axis=2) # (?, 50, 600)


        ### Pooling Layer ###
        v_1_sum = tf.reduce_sum(v1_bi, 1) # 整列求和 (?, 600) 把每句话的50个单词省略了?
        v_1_ave = tf.div(v_1_sum, tf.expand_dims(tf.cast(prem_seq_lengths, tf.float32), -1)) # (?, 600)

        v_2_sum = tf.reduce_sum(v2_bi, 1) # 整列求和 (?, 600)
        v_2_ave = tf.div(v_2_sum, tf.expand_dims(tf.cast(hyp_seq_lengths, tf.float32), -1)) # (?, 600)

        v_1_max = tf.reduce_max(v1_bi, 1) # 整列求和 (?, 600)
        v_2_max = tf.reduce_max(v2_bi, 1) # 整列求和 (?, 600)

        v = tf.concat([v_1_ave, v_2_ave, v_1_max, v_2_max], 1) # (?, 2400)
        

        # MLP layer
        h_mlp = tf.nn.tanh(tf.matmul(v, self.W_mlp) + self.b_mlp) # (?, 300)

        # Dropout applied to classifier
        h_drop = tf.nn.dropout(h_mlp, self.keep_rate_ph) # (?, 300)

        # Get prediction
        self.logits = tf.matmul(h_drop, self.W_cl) + self.b_cl # (672, 3)
        
        # added, remove the results of 0s, majority vote for each hypothesis
        logit_list = tf.split(self.logits, num_or_size_splits = self.n, axis = 0)
        hypothesis_list = tf.split(self.hypothesis_x, num_or_size_splits = self.n, axis = 0)
        ave = []
        for i, logit in enumerate(logit_list):
          hyp = hypothesis_list[i]
          hyp_sum = tf.reduce_sum(tf.abs(hyp),0)
          mask = tf.greater(hyp_sum, 0)
          non_zero_logit = tf.boolean_mask(logit, mask) # 非0的都剩下, 接下来vote? 怎么vote,没法vote吧,不然下一步输入就不是三维的了,,干脆每列求和好了,这样entailment, contradict 和neutral都有平均值了
          hy_ave = tf.reduce_mean(non_zero_logit, 0)
          ave.append(hy_ave)
        self.logits_ave = tf.stack(ave) # (32,3)
        
        

        # Define the cost function
        self.total_cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits_ave)) # 一个数字
