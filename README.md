# Human Liver Microsomal Stability

Prediction of human liver microsomal stability is key for the screening of drugs in the early stage of drug discovery. The liver is the main organ for metabolizing drugs in humans and testing its metabolic stability is essential for the early detection of viable drug compounds.

## Identifiers

* EOS model ID: `eos8osp`
* Slug: `human-liver-microsomal-stability`

## Characteristics

* Input: `Compound`
* Input Shape: `Single`
* Task: `Classification`
* Output: `Probability`
* Output Type: `Float`
* Output Shape: `Single`
* Interpretation: Probability of a compound being soluble at 10 μg/mL. (>0.5: Soluble), and probability of a compound being highly soluble (>52 μg/mL; >0.5: Soluble)

## References

* [Publication](https://slas-discovery.org/article/S2472-5552(22)06765-X/fulltext)
* [Source Code](https://github.com/ncats/ncats-adme)
* Ersilia contributor: [pauline-banye](https://github.com/pauline-banye)

## Citation

If you use this model, please cite the [original authors](https://slas-discovery.org/article/S2472-5552(22)06765-X/fulltext) of the model and the [Ersilia Model Hub](https://github.com/ersilia-os/ersilia/blob/master/CITATION.cff).

## License

This package is licensed under a GPL-3.0 license. The model contained within this package is licensed under a None license.

Notice: Ersilia grants access to these models 'as is' provided by the original authors, please refer to the original code repository and/or publication if you use the model in your research.

## About Us

The [Ersilia Open Source Initiative](https://ersilia.io) is a Non Profit Organization ([1192266](https://register-of-charities.charitycommission.gov.uk/charity-search/-/charity-details/5170657/full-print)) with the mission is to equip labs, universities and clinics in LMIC with AI/ML tools for infectious disease research.

[Help us](https://www.ersilia.io/donate) achieve our mission!