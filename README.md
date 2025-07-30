## Arctic-TILT Business Document Understanding at Sub-Billion Scale

<img width="606" height="874" alt="Image" src="https://github.com/user-attachments/assets/2d682dcd-575c-41b0-bc32-536cda919719" />

This repository contains the implementation of the paper: [Arctic-TILT Business Document Understanding at Sub-Billion Scale]([Artic-tilt 2024.pdf](https://github.com/user-attachments/files/21500895/Artic-tilt.2024.pdf)). Note that, the authors have not released the original implementation of the paper.

Abstract: The vast portion of workloads employing LLMs involves answering questions grounded on PDF or scan content. We introduce the Arctic-TILT achieving accuracy on par with models 1000× its size on these use cases. It can be fine-tuned and deployed on a single 24GB GPU, lowering operational costs while processing Visually Rich Documents with up to 400k tokens. The model establishes state-of-the-art results on seven diverse Document Understand-ing benchmarks, as well as provides reliable confidence scores and quick inference, which are essential for processing files in large-scale or time-sensitive enterprise environments.


## Requirements
* See in the requirements.txt file


## Dataset
* I would be including the [FUNSD Dataset](https://guillaumejaume.github.io/FUNSD/).

## Modeling:
* The modeling part of the pipeline, basically is inspired from [HuggingFace's T5 implementation](https://huggingface.co/docs/transformers/model_doc/t5), and the initialization of the weights are being done from the same.
* About fusion operation, based on the paper, the text and image embeddings t, i ∈ ℝᵈ, we calculate the fused embedding using: Fuse(t,i) = O(V(t + i) ⊙ (1 + Rt)) + t where V, R, and O are trainable parameters of size ℝᵈ×ᵈ.
Specifically, Figure 2 illustrates the computation of the fused embedding using the two input parameters—image embedding and text embedding, corresponding to i and t respectively in the Fuse(t, i) formula. Figure 3 depicts the encoder block: the textual semantics, after being processed through multi-head attention and a feed-forward layer, are fused with contextualized visual, t and i as in the Fuse(t, i) formulation.
<img width="479" height="860" alt="Image" src="https://github.com/user-attachments/assets/3a2fb1a5-cd70-410a-a873-4215b84a3362" />
<img width="467" height="686" alt="Image" src="https://github.com/user-attachments/assets/de37f8f6-667b-4ce0-89e3-d13946b61436" />


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
