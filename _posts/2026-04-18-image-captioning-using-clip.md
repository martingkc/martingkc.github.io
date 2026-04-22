---
layout: post
title: CLIP Models and Image Captioning using CLIP embeddings.
date: 2026-04-18
description: Exploring contrastive learning and Info-NCE loss and training a CLIP vision encoder, then building a GPT-2 image captioning model (ClipCap) trained on chest X-ray CLIP embeddings.
tags: []
categories: [VLA, VLM, CLIP, pytorch, GPT]
giscus_comments: true
---

The natural next step after exploring how ViT's work behind the scenes in my journey of mastering VLA's is learning how CLIP models work. First of all, some people, including some of my professors and friends, state that "CLIP models" is a wrong statement since CLIP is actually the training method used and the models themselves generally consist of ViT's trained alongside a text encoder. I think that's partly true, CLIP is a contrastive training objective, but it's also a specific model family and model, so I'd argue it's earned the right to be called its own thing at this point. But at the end of the day it's a bit of both, its both a training method and a model category. 


CLIP (Contrastive Language-Image Pre-training) models are models capable of associating visual concepts with textual descriptions by training a vision encoder and a text encoder jointly, pushing matching image-text pairs closer together in embedding space and pushing non-matching pairs apart. This shared embedding space is actually what makes CLIP relevant in VLA research. 

If your model understands that an image of a cup and the text "pick up the cup" live in the same space, you have the foundation for a robot that can reason about language and vision together.


In the past I worked on a project using CLIP embeddings to generate captions for chest X-rays, and in this post I'll walk through that project. At the end we'll also build a simple contrastive learning example from scratch using the ViT from the previous post paired with a pretrained BERT text encoder, using the same dataset. 


# How CLIP works? and Contrastive Learning

Contrastive Learning practically forces the model to push its produced embedding closer to the embedding space represented by the positive data points whilst pushing it away from the negative data points. Basically our objective is to maximise cosine similarity for positive matches and minimise it for the negative ones. 


In this example we're going to use Info-NCE.  

### Info NCE 

**Similarity matrix**

$$S_{ij} = \frac{v_i^\top u_j}{\tau \|v_i\| \|u_j\|}$$

**Image → text loss**

$$\mathcal{L}_{\text{i2t}} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(S_{ii})}{\sum_{j=1}^{N} \exp(S_{ij})}$$

**Text → image loss**

$$\mathcal{L}_{\text{t2i}} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(S_{ii})}{\sum_{j=1}^{N} \exp(S_{ji})}$$

**Final loss**

$$\mathcal{L}_{\text{InfoNCE}} = \frac{\mathcal{L}_{\text{i2t}} + \mathcal{L}_{\text{t2i}}}{2}$$

Where $v_i$ = image embedding, $u_j$ = text embedding, $\tau$ = temperature (0.07)

```python 
def info_nce_loss(image_emb, text_emb, temperature=0.07):
    image_emb = F.normalize(image_emb, dim=-1)  
    text_emb = F.normalize(text_emb, dim=-1)    

    # similarity matrix [N, N]
    logits = (image_emb @ text_emb.T) / temperature

    # labels are just 0,1,2...N-1 — the diagonal is always correct
    labels = torch.arange(len(logits), device=logits.device)

    # cross entropy in both directions
    loss_i = F.cross_entropy(logits, labels)        # image -> text
    loss_t = F.cross_entropy(logits.T, labels)      # text -> image

    return (loss_i + loss_t) / 2
```
## ViT implementation 

The ViT implementation I've used in this notebook is almost exactly the same as the one I've made in the [previous post](https://martingkc.github.io/blog/2026/learning-VIT/). The only difference is that the last MLP layer has been replaced with a linear projection and the embedding size has been changed to 512 from 768. 

```python 
class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=10, embed_dim=512, num_heads=8, depth=6, mlp_dim=1024, channels = 1):
        super().__init__()
        seq_len = (img_size // patch_size) ** 2 + 1 # Our seq len is going to be the num of patches + one CLS token
        self.patch_embedding = PatchEmbedding(img_size, patch_size, channels , embed_dim) # patch the image
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_dim, seq_len) for _ in range(depth)
        ]) # Our ViT is going to have a depth of 6.
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim)) # Assign the CLS token
        self.proj = nn.Linear(embed_dim, embed_dim) # <- this was an MLP layer
    def forward(self, x):
        B = x.size(0) # batch size
        x = self.patch_embedding(x)
        cls_tokens = self.cls_token.expand(B, -1, -1) # cls token is shaped [1, 1, 512] this expands it to [B, 1, 512]
        x = torch.cat((cls_tokens, x), dim=1) # concatenate the cls token
        for block in self.transformer_blocks:
            x = block(x) # run the input through the 6 transformer blocks
        return self.proj(x[:, 0]) # spit out the embedding
```

## Dataset
The dataset that I will use for this example is one that I had used previously for a uni challenge which is composed of B/W chest x-rays, a list of labels of the various pathologies and image and textual embeddings obtained through a pre trained CLIP model. The dataset is quite small so I suggest using another one, Flickr30K might be a good alternative if you want to train your model from scratch, but beware it doesn't contain the embeddings of the text so you would have to compute them using a pretrained text embedding model. 

In order to use our images with our ViT we need to normalize them into greyscale images with a size of 224x224, this loader here does exactly that. 

```python 
from transformers import CLIPTokenizer, CLIPTextModel

# Uncomment if you're using a dataset without embeddings of the text. 
#tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
#text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").cuda()

def get_embedding(caption):
    inputs = tokenizer(caption, return_tensors="pt", padding=True, truncation=True).to("cuda")
    with torch.no_grad():
        return text_model(**inputs).pooler_output.squeeze(0).cpu()

class DatasetPrepper(Dataset):
  """
  This is the dataset loader that we will use to "Normalize" our data.
  It basically ensures that the images are greyscale and that they are of size 224x224.
  If the text embeddings are missing they can be generated using the get_embedding function.
  """
  def __init__(self, df, embed_text=False, embed_model = None,  transform=None):
        self.df = df.reset_index(drop=True)
        self.embed_text = embed_text
        self.embed_model = embed_model
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])

  def __len__(self):
        return len(self.df)

  def __getitem__(self, idx):
          row = self.df.iloc[idx]
          img = self.transform(row['Image'])
          if not self.embed_text:
            text_emb = torch.tensor(row['Text Features'], dtype=torch.float32)
          else: 
            text_emb = get_embedding(row['Texts'])
          text_label = row['Texts']  
          return img, text_emb, text_label
```

## Training the ViT 

```python 
model = VisionTransformer(embed_dim=512).cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
train_dataset = DatasetPrepper(train_df)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

for i in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels, _ in train_loader:  # Iterate over train_loader
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = info_nce_loss(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{i+1}/{epochs}], Loss: {running_loss/len(train_loader)}")
```


## Evaluation 

We have two eval functions, one which compares our embedding with all possible labels present in our Dataset which are around 291 unique labels. !! this is actually a valid zero-shot retreival but it's extremely noisy because of the number of labels !!. And the second one pulls k-1 random labels with the correct label injected in the k-th position and does a cosine similarity comparison. 

```python 
import torch
import numpy as np
import random 
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

def cosine_similarity(a, b):
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return (a * b).sum(dim=-1)

def eval_model(prediction_embs, true_labels):
    embs = label_df["Text Features"].values
    text = label_df["Texts"].values
    results = []

    for e, l in zip(prediction_embs, true_labels):
        similarity = torch.stack([
            cosine_similarity(e, torch.tensor(embs[i]))
            for i in range(len(embs))
        ])
        pred_label = text[torch.argmax(similarity)]
        results.append(pred_label == l)

    return {
        "accuracy": np.mean(results),
        "total": len(results),
        "correct": sum(results)
    }

def eval_recall_at_k(prediction_embs, true_text_embs, k=10):
    prediction_embs = F.normalize(prediction_embs, dim=-1)
    true_text_embs = F.normalize(true_text_embs, dim=-1)
    
    correct = 0
    n = len(prediction_embs)
    
    for i in range(n):
        # pick 9 random indices that are NOT the correct one
        wrong_indices = random.sample([j for j in range(n) if j != i], k-1)
        candidate_indices = wrong_indices + [i]  # inject correct at the end
        
        candidates = true_text_embs[candidate_indices]  # [10, D]
        sim = prediction_embs[i] @ candidates.T         # [10]
        
        if sim.argmax().item() == k-1:  # correct is always at index k-1
            correct += 1
    
    return {f"accuracy@{k}": correct / n, "correct": correct, "total": n}


```

As expected with our limited dataset of approx 9 samples per label, the results leave a lot to be desired.

``` 
accuracy with 5 labels:{'accuracy@5': 0.31092436974789917, 'correct': 148, 'total': 476}
accuracy with 10 labels:{'accuracy@10': 0.16806722689075632, 'correct': 80, 'total': 476}
accuracy with 15 labels:{'accuracy@15': 0.14705882352941177, 'correct': 70, 'total': 476} 
```
the accuracy of 31% against a 20% random baseline with 5 labels suggests the model is learning **something** despite the small dataset.

I would like to try training it on the flickr30k dataset on a different occasion, but right now I don't have the compute necessary.

# Using CLIP embeddings to generate image captions with a GPT-2 model
 
So this is actually from an earlier experiment, where I used a decoder-only transformer to generate captions from CLIP embeddings as inputs. The way this works is through an MLP layer that projects the CLIP embeddings into GPT-2's embedding space, so they can be used as pseudo-tokens that the GPT model attends to when generating captions autoregressively.
 
This approach was explored in the paper [ClipCap: CLIP Prefix for Image Captioning](https://arxiv.org/pdf/2111.09734). The original paper uses the COCO-Captions dataset and explores two approaches: one trains the MLP alongside fine-tuning GPT-2, exploiting its pretrained linguistic abilities. The other trains a decoder-only model entirely from scratch to accept CLIP embeddings as input.
 
### Architecture
 
The core idea is simple: the MLP projects a CLIP image embedding into `prefix_length` pseudo-tokens in GPT-2's token embedding space. These get prepended to the actual caption tokens, and GPT-2 generates the caption autoregressively. The MLP is small enough that we can fine-tune it alongside GPT-2 without needing massive compute.
 
```python
class MLP(nn.Module):
    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)
 
    def forward(self, x):
        return self.model(x)
 
 
class ClipCaptionModel(nn.Module):
    def __init__(self, prefix_length: int, prefix_size: int = 512):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained("gpt2")
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        # For short prefixes use an MLP, for longer ones a linear layer saves memory
        if prefix_length > 10:
            self.clip_project = nn.Linear(
                prefix_size, self.gpt_embedding_size * prefix_length
            )
        else:
            self.clip_project = MLP((
                prefix_size,
                (self.gpt_embedding_size * prefix_length) // 2,
                self.gpt_embedding_size * prefix_length,
            ))
 
    def forward(self, tokens, prefix, mask=None, labels=None):
        embedding_text = self.gpt.transformer.wte(tokens)
        # Project CLIP embedding into prefix_length GPT-2 tokens
        prefix_projections = self.clip_project(prefix).view(
            -1, self.prefix_length, self.gpt_embedding_size
        )
        # Prepend projected prefix to the token embeddings
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = torch.zeros(tokens.shape[0], self.prefix_length,
                                      dtype=torch.int64, device=tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        return self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
```
 
### Preparing the data
 
ClipCap expects the data in a specific pickle format, a dict containing a stacked tensor of CLIP embeddings and a list of caption dicts, each pointing back to its embedding by index.
 
```python
EOS_TOKEN = '<|endoftext|>'
 
all_embeddings = []
caption_dicts  = []
 
for i, (_, row) in enumerate(train_df.iterrows()):
    all_embeddings.append(torch.as_tensor(row["Image Features"], dtype=torch.float32))
    caption_dicts.append({
        "caption":        row["Texts"] + EOS_TOKEN,
        "image_id":       row["Image Index"],
        "clip_embedding": i,
    })
 
embeddings_tensor = torch.stack(all_embeddings, dim=0)
 
with open("train_data.pkl", "wb") as f:
    pickle.dump({
        "clip_embedding": embeddings_tensor,
        "captions":       caption_dicts,
    }, f)
```
 
### Inference
 
At inference time we skip the token embeddings entirely — we just project the CLIP embedding and let GPT-2 generate from there. The function supports both greedy/nucleus sampling and beam search.
 
```python
PREFIX_LENGTH = 10
 
def generate_caption(
    model,
    clip_embedding,
    use_beam_search=False,
    beam_size=5,
    top_p=0.8,
    temperature=1.0,
    max_length=67
):
    clip_embedding = clip_embedding.to(DEVICE).to(torch.float32)
    prefix_embed = model.clip_project(clip_embedding)
    prefix_embed = prefix_embed.view(1, PREFIX_LENGTH, -1)
 
    if use_beam_search:
        return generate_beam(model, tokenizer, beam_size=beam_size,
                             embed=prefix_embed, entry_length=max_length,
                             stop_token=EOS_TOKEN, temperature=temperature)[0]
    else:
        return generate2(model, tokenizer, embed=prefix_embed,
                         entry_length=max_length, top_p=top_p,
                         stop_token=EOS_TOKEN, temperature=temperature)
```
 
### Evaluation
 
Since the captions are structured clinical descriptions (e.g. "Atelectasis, No Findings, PA view"), we evaluate on two axes: ROUGE for text overlap, and label level F1/precision/recall by checking whether each known pathology keyword appears in the generated text.
 
```python
label_cols = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax", "Consolidation",
    "Edema", "Emphysema", "Fibrosis", "Hernia", "Pleural Thickening",
    "No Finding", "AP", "PA"
]
 
def eval_model(pred_texts, base_texts):
    pred_labels, base_labels = [], []
    for p, b in zip(pred_texts, base_texts):
        p, b = p.lower(), b.lower()
        pred_labels.append([1 if l.lower() in p else 0 for l in label_cols])
        base_labels.append([1 if l.lower() in b else 0 for l in label_cols])
 
    rouge_scores = rouge.compute(predictions=pred_texts, references=base_texts)
    return {
        "f1":        f1_score(pred_labels, base_labels, average='macro', zero_division=0),
        "recall":    recall_score(pred_labels, base_labels, average='micro', zero_division=0),
        "precision": precision_score(pred_labels, base_labels, average='micro', zero_division=0),
        "accuracy":  accuracy_score(pred_labels, base_labels),
        "rouge":     rouge_scores,
    }
```
 
### Dealing with class imbalance
 
The dataset is heavily skewed toward "No Finding", which dominates training and causes the model to overpredict common labels. One approach is Focal Loss, which downweights easy examples by scaling the standard cross-entropy by $(1 - p_t)^\gamma$, forcing the model to focus on harder, underrepresented tokens.
 
```python
class TokenFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, ignore_index=0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # per-token weights of shape (vocab_size,)
        self.ignore_index = ignore_index
 
    def forward(self, logits, targets):
        B, L, V = logits.shape
        logits  = logits.reshape(-1, V)
        targets = targets.flatten()
        log_p = F.log_softmax(logits, dim=1)
        p     = torch.exp(log_p)
        ce    = F.nll_loss(log_p, targets, weight=self.alpha,
                           ignore_index=self.ignore_index, reduction='none')
        p_t   = p[torch.arange(p.size(0)), targets]
        return (ce * (1 - p_t) ** self.gamma).mean()
```
 
The per-token weights are computed from the training set token frequency distribution and clipped to avoid giving extreme boosts to very rare tokens:
 
```python
def get_token_weights(dataloader, tokenizer, max_alpha=10.0):
    counts = np.zeros(tokenizer.vocab_size, dtype=np.int64)
    for tokens, _, _ in dataloader:
        ids = tokens.cpu().numpy().ravel()
        ids = ids[ids < tokenizer.vocab_size]
        np.add.at(counts, ids, 1)
    counts[0] = 0  # ignore padding
    weights = np.ones_like(counts, dtype=np.float64)
    nonzero = counts > 0
    weights[nonzero] = counts[nonzero].max() / counts[nonzero]
    weights = np.clip(weights, 1.0, max_alpha)
    weights[0] = 0.0
    return torch.from_numpy(weights).float().to(DEVICE)
```
 
The results from both training runs are limited by the dataset size and domain specificity of the chest X-ray captions. That said, the focal loss variant does improve recall on minority pathology labels.

# Notebooks used

You can find the notebooks that contain the code and the datasets used in these examples here. 

- [Image Embedder](https://github.com/martingkc/NLP_Notebooks/blob/main/CLIP_Image_Encoder.ipynb)
- [Embedding Captioning](https://github.com/martingkc/NLP_Notebooks/blob/main/clip-caption-gpt.ipynb)
