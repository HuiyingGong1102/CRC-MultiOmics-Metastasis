# Proteomics demonstration data

`demo_proteomics.csv` contains 1000 protein rows and 334 sample columns.
The first column contains unique protein identifiers. Numeric values are
expected to be finite log2 abundances. `demo_metadata.csv` contains the
sample identifier (`Sample`) and categorical group (`Group`).
The current workflow selects 167 tumor samples: ColonT_NonMet (124),
ColonT_Liver (23), ColonT_Lung (12), and ColonT_Other (8).
Other metadata groups are not processed by this demonstration.

The demo is a subset of the original proteomics data used in the associated
article, containing 1000 proteins. These are real article-derived data,
provided as a compact example of the analysis workflow rather than the full
study dataset. The associated article citation and data accession will be
added with the final publication metadata.

The repository's MIT License applies to software, not to these research data.
Please cite the associated article and follow the data terms accompanying its
final publication. No patient-level clinical identifiers are required by the
software.
