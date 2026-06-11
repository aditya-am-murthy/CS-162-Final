**Script 12: Hard-to-learn stuff**

Run setup only:
python scripts/12_noise_detection_paper.py --no-wandb
Run end-to-end retraining + detector:

python scripts/12_noise_detection_paper.py --train --no-wandb
Outputs go to:
data/processed/noise_detection_paper/


**Script 8: Ambigious, easy-to-learn stuff**
To run the actual retraining sweep:

python scripts/08_role_easy_to_learn.py --train --no-wandb
For a quick smoke test with only a couple training runs:

python scripts/08_role_easy_to_learn.py \
  --train \
  --limit-training-runs 2 \
  --epochs 1 \
  --no-wandb
Important: the script expects a cartography with regions file, by default:
data/processed/cartography_with_regions.jsonl

So normally run these first, or point --input at an existing regions JSONL:
python scripts/01_collect_dynamics.py --input data/raw/epoch_predictions_toy.jsonl
python scripts/02_build_data_map.py
python scripts/08_role_easy_to_learn.py --no-wandb

For the paper WinoGrande version, you’ll want the input file to come from a WinoGrande cartography run, then:

python scripts/08_role_easy_to_learn.py \
  --input path/to/winogrande_cartography_with_regions.jsonl \
  --dataset winogrande \
  --preset roberta-base \
  --no-wandb