# Visual Recognition Project 2

## Directory Structure

```
VR_Project2_PraveenPeterJay_IMT2022064/
├── baseline_models/
│   ├── blip/
│   │   ├── images/
│   │   └── blip.py
│   ├── blip2/
│   │   ├── images/
│   │   ├── results/
│   │   └── blip2.ipynb
│   ├── clip/
│   │   ├── images/
│   │   ├── results/
│   │   ├── clip_2_and_5_percent.ipynb
│   │   └── clip_25_percent.ipynb
│   ├── llava/
│   │   ├── images/
│   │   ├── results/
│   │   └── llava.ipynb
│   └── vilt/
│       ├── images/
│       └── vilt.py
├── dataset_curation/
│   ├── files/
│       ├── balanced_dataset.csv
│       ├── images.csv
│       ├── vqa.csv
│   ├── images/
│   └── scripts/
│       ├── amazon_vqa_generator.py
│       ├── balance_dataset.py
│       └── vqa_generator_checkpoint.txt
├── finetuned_model/
│   └── blip
│       ├── images/
│       ├── model/
│       ├── results/
│       └── blip_lora.ipynb
├── inference_module/
│   └── IMT2022064
│   │   ├── model/
│   │   ├── inference.py
│   │   └── requirements.txt
├── readme.md
└── report.pdf
