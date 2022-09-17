# M-Adapter

**This code is for Interspeech 2022 paper** [M-Adapter: Modality Adaptation for End-to-End Speech-to-Text Translation](https://arxiv.org/pdf/2207.00952.pdf)

### Environment and data reparation

This codebase is developed upon [UPC's repository](https://github.com/mt-upc/iwslt-2021). Please follow their instructions to set up the environment, preprocess data and download pretrained modules.

### Model Training

To train a speech translation model with 3-layer M-Adapter, run the following command.
```angular2html
# Step1
bash train_adapter_2steps.sh step1

# Step2
bash train_adapter_2steps.sh step2
```

### Inference

Run the following command for inference
```angular2html
bash adapt_generate.sh 
```

## Citation
Please cite if you use our code.

```bibtex
@article{zhao2022m,
  title={M-Adapter: Modality Adaptation for End-to-End Speech-to-Text Translation},
  author={Zhao, Jinming and Yang, Hao and Shareghi, Ehsan and Haffari, Gholamreza},
  journal={arXiv preprint arXiv:2207.00952},
  year={2022}
}
```