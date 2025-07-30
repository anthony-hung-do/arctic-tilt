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


## Examples:
* For finetuning Arctic-TiLT on FUNSD, the example along with the results are present [here](https://github.com/anthony-hung-do/arctic-tilt/blob/main/src/Part_3_Apply_chunked_processing.ipynb)


## My Results:
| Model Name      | Dataset Name | Number of Parameters | Overall Precision | Overall Recall | Overall F1 Score | Overall Accuracy |
|-----------------|--------------|----------------------|-------------------|----------------|------------------|------------------|
| Arctic-TILT     | FUNSD        |  225M                | 55.21             | 43.89          | 48.90            | 83.56            |

Note, that in the case of my results on FUNSD, the model has not been pre-trained (the weights are intialized from the hugging face's implementation), and it has been trained for 50 epochs.

## Contributors:
- Anthony Hung Do from Swapbrain (swapbrain.com), Australia
- Hoang Xuan Linh from Da Nang University of Science and Technology, Vietnam
- Ho Phuc Hy from University of Aix-Marseille University, France
