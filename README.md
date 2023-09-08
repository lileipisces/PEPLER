# PEPLER (PErsonalized Prompt Learning for Explainable Recommendation)

## Paper
- Lei Li, Yongfeng Zhang, Li Chen. [Personalized Prompt Learning for Explainable Recommendation](https://arxiv.org/abs/2202.07371). ACM Transactions on Information Systems (TOIS), 2023.

**A T5 version that can perform multiple recommendation tasks is available at [POD](https://github.com/lileipisces/POD)!**

**A small unpretrained Transformer version is available at [PETER](https://github.com/lileipisces/PETER)!**

**A small ecosystem for Recommender Systems-based Natural Language Generation is available at [NLG4RS](https://github.com/lileipisces/NLG4RS)!**

## Datasets to [download](https://lifehkbueduhk-my.sharepoint.com/:f:/g/personal/16484134_life_hkbu_edu_hk/Eln600lqZdVBslRwNcAJL5cBarq6Mt8WzDKpkq1YCqQjfQ?e=cISb1C)
- TripAdvisor Hong Kong
- Amazon Movies & TV
- Yelp 2019

For those who are interested in how to obtain (feature, opinion, template, sentiment) quadruples, please refer to [Sentires-Guide](https://github.com/lileipisces/Sentires-Guide).

## Usage
Below are examples of how to run PEPLER (continuous prompt, discrete prompt, MF regularization and MLP regularization).
```
python -u main.py \
--data_path ../TripAdvisor/reviews.pickle \
--index_dir ../TripAdvisor/1/ \
--cuda \
--checkpoint ./tripadvisor/ >> tripadvisor.log

python -u discrete.py \
--data_path ../TripAdvisor/reviews.pickle \
--index_dir ../TripAdvisor/1/ \
--cuda \
--checkpoint ./tripadvisord/ >> tripadvisord.log

python -u reg.py \
--data_path ../TripAdvisor/reviews.pickle \
--index_dir ../TripAdvisor/1/ \
--cuda \
--use_mf \
--checkpoint ./tripadvisormf/ >> tripadvisormf.log

python -u reg.py \
--data_path ../TripAdvisor/reviews.pickle \
--index_dir ../TripAdvisor/1/ \
--cuda \
--rating_reg 1 \
--checkpoint ./tripadvisormlp/ >> tripadvisormlp.log
```

## Code dependencies
- Python 3.6
- PyTorch 1.6
- transformers 4.18.0

## Code reference
- [mkultra: Prompt Tuning Toolkit for GPT-2](https://github.com/corolla-johnson/mkultra)

## Citation
```
@article{TOIS23-PEPLER,
	title={Personalized Prompt Learning for Explainable Recommendation},
	author={Li, Lei and Zhang, Yongfeng and Chen, Li},
	journal={ACM Transactions on Information Systems (TOIS)},
	year={2023}
}
```
