# FACS Data Analysis App

&#x20;    &#x20;

---

## Overview

**FACS Data Analysis** is a simple web application built with **Streamlit** that allows you to:

- Upload Flow Cytometry Standard (.fcs) files
- Extract raw event data
- Preview and download the event data as a CSV file
- Inspect metadata embedded in FCS files

This app is especially useful for researchers and engineers who handle flow cytometry data and want a lightweight, browser-based way to inspect and convert data.

### ✨ [Try it Live](https://facsdataanalysis-zz29jykc7nfshywbujaj8q.streamlit.app/)

---

## Features

- ☑️ Upload `.fcs` files (Flow Cytometry Standard)
- ☑️ Display first 10 rows of the event data
- ☑️ Download full event table as CSV
- ☑️ Show metadata (optional)
- ☑️ Built-in metrics like event count, parameter count, and file size

---

## How to Use

1. Click the upload button to select your `.fcs` file.
2. The app will parse and preview the top 10 rows of the data.
3. Optional: check the box to inspect FCS metadata.
4. Click the **Download CSV** button to save the data locally.

---

## Requirements

This app requires the following Python packages:

```txt
streamlit
pandas
fcsparser
```

A `requirements.txt` file is included.

---

## Deployment

This app is deployed on [Streamlit Cloud](https://facsdataanalysis-zz29jykc7nfshywbujaj8q.streamlit.app/).

To run it locally:

```bash
git clone https://github.com/your-username/facs-data-analysis.git
cd facs-data-analysis
pip install -r requirements.txt
streamlit run app.py
```

---

## License

This project is licensed under the MIT License. See `LICENSE.txt` for details.

---

## Author

Developed by [Your Name / Organization]. Contributions welcome!

---

## Notes

- Large FCS files may take time to process.
- Only raw event data is extracted; no downstream gating or visualization is performed.
- Designed for simplicity and fast data inspection.

---

