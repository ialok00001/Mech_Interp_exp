# Embeddings

While we have input to the LLMs as texts, it cannot be passed directly to the model that can only understand numbers. For this, we convert these text into relevant numbers. Namely, we first convert a text sentence into tokens, where the set of tokens forms the set of full vocabulary for the model and each have a unique integer index assigned, then these tokens are replaced with the corresponding token (along with a token for BOS at the beginning of each sentence). Then, these set of indexes is replaced with the corresponding embedding for each token index.


Mathematically, if a sentence has say n tokens $i_1$, $i_2$, ... $i_n$, and the embedding dimension is $d_{model}$ with the model having a total of V tokens, then we use the embedding matrix $W_E \in \mathbb{R}^{V \times d_{model}}$ that assigns each token index $i_k$ a corresponding embedding vector $W_E\left[i_k, :\right]$. The resulting input from the full sentence results in a vector of shape $n \times d_{model}$. With batches of sentences, we even get a third dimension and the resulting shape becomes $B \times n \times d_{model}$.

This embedding matrix $W_E$ is learned by the model during training.

<br>

Along with the embedding matrix, we also have an umembedding matrix $W_U \in \mathbb{R}^{d_{model} \times V}$ that later maps the final processed vector from shape $d_model$ to V, making a next token prediction. Mathematically, we have something like:

$$\mathbf{h_i} = Norm_{final}(\mathbf{r}_i^L) \text{, }\hspace{1cm} \mathbf{z_i} = \mathbf{h_i}W_U + \mathbf{b_U} \text{, }\hspace{1cm} W_U \in \mathbb{R}^{d_{model} \times V}$$


<br>

Column j of $z_i$ corresponds to token $j$ in the vocabulary. This is obtained from the matrix multiplication by $z_{i, j} = \mathbf{h_i}W_U[:,\text{ }j] + b_{U, j}, and so moving $\mathbf{h_i}$ along that column of $W_U$ raises that token's score. If we try to compare the scores of two tokens $a$ and $b$, we can see that
$$z_{i, a} - z_{i, b} = \mathbf{h_i}(W_U[:, \text{ }a] - W_U[:, \text{ }b]) + (b_{U, a} - b_{U, b})$$

This equation can directly be used to compare the scores of other tokens having one token as base. [Direct logit]() attribution uses that geometry to measure how individual component writes align with an output preference, while the [logit lens]() applies the final readout to intermediate residual states.

At some places, we even set $W_U = W_E^T$ to work with a lower set of weights to tune. This is commonly known as **weight tying**.