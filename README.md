## Arctic-TILT Business Document Understanding at Sub-Billion Scale

This repository contains the implementation of the paper: [Arctic-TILT Business Document Understanding at Sub-Billion Scale]([Artic-tilt 2024.pdf](https://github.com/user-attachments/files/21500895/Artic-tilt.2024.pdf)). Note that, the authors have not released the original implementation of the paper.

Abstract: The vast portion of workloads employing LLMs involves answering questions grounded on PDF or scan content. We introduce the Arctic-TILT achieving accuracy on par with models 1000× its size on these use cases. It can be fine-tuned and deployed on a single 24GB GPU, lowering operational costs while processing Visually Rich Documents with up to 400k tokens. The model establishes state-of-the-art results on seven diverse Document Understand-ing benchmarks, as well as provides reliable confidence scores and quick inference, which are essential for processing files in large-scale or time-sensitive enterprise environments.


## Requirements
* See in the requirements.txt file


## Dataset
* I would be including the [FUNSD Dataset](https://guillaumejaume.github.io/FUNSD/).

## Modeling:
* The modeling part of the pipeline, basically is inspired from [HuggingFace's T5 implementation](https://huggingface.co/docs/transformers/model_doc/t5), and the initialization of the weights are being done from the same.
* About fusion operation, based on the paper, the text and image embeddings t, i ∈ ℝᵈ, we calculate the fused embedding using: Fuse(t,i) = O(V(t + i) ⊙ (1 + Rt)) + t where V, R, and O are trainable parameters of size ℝᵈ×ᵈ. Specifically, textual semantics, after passing through multi-head attention and the feed-forward layer, become the text embedding (pink block), which is then fused with the image embedding (yellow block), represented as t and i in the Fuse(t, i) formulation. These two blocks, after undergoing operations as defined in the formula, form the fused embedding (purple block), which is subsequently fed into the decoder.
<img width="1433" height="3474" alt="Image" src="https://github.com/user-attachments/assets/75140b31-4b76-4548-8a31-82a746576163" />
This image illustrates the procedure from the encoder block, combining Contextualized Vision and Textual Semantics through a Fusion operation, leading to the decoder, based on Figure 2 and Figure 3 in the paper.


## Examples:
* For finetuning Arctic-TiLT on FUNSD, the example along with the results are present [here](https://github.com/anthony-hung-do/arctic-tilt/blob/main/src/Part_3_Apply_chunked_processing.ipynb)


## My Results:
| Model Name      | Dataset Name | Number of Parameters | Overall Precision | Overall Recall | Overall F1 Score | Overall Accuracy |
|-----------------|--------------|----------------------|-------------------|----------------|------------------|------------------|
| Arctic-TILT     | FUNSD        |  225M                | 55.21             | 43.89          | 48.90            | 83.56            |
| TILT(Original)  | FUNSD        |  230M                | ---               | ---            | ---              | 95.25            |

Note, that in the case of my results on FUNSD, the model has not been pre-trained (the weights are intialized from the hugging face's implementation), and it has been trained for 50 epochs.

## Contributors:
- Anthony Hung Do from Swapbrain (swapbrain.com), Australia
- Hoang Xuan Linh from Da Nang University of Science and Technology, Vietnam
- Ho Phuc Hy from University of Aix-Marseille University, France
