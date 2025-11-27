# Dataset

This folder is used to store the datasets for the **SignVision** project.

The datasets are **not included** in this repository because of their size. You need to download them from the official links below and place them inside this folder.

---

## Download Links

* **[9] ArSL21L – Arabic Sign Language Letters Dataset (Mendeley)**
  [https://data.mendeley.com/datasets/f63xhm286w/1](https://data.mendeley.com/datasets/f63xhm286w/1)

* **[10] AASL – RGB Arabic Alphabets Sign Language Dataset (arXiv)**
  [https://arxiv.org/abs/2301.11932](https://arxiv.org/abs/2301.11932)

---

## Suggested Folder Structure

After downloading the datasets, you can organize them like this:

```text
data/
├─ ArSL21L/
│  ├─ Train/
│  └─ Test/
└─ AASL/
   ├─ Train/
   └─ Test/
```

You can name the class folders (e.g. `A/`, `B/`, `C/` or `alif/`, `baa/`, `taa/`) any way you like, as long as they match what your training code expects.

---

## Notes

* This folder only contains the **structure and documentation** for the datasets.
* To train or retrain the models, make sure the data is downloaded and placed correctly before running the training scripts.
