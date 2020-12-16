# Similarity models
Similarity and data extraction

Source codes to accompany the following publications:
- https://ieeexplore.ieee.org/document/8892877 (older manuscript: https://arxiv.org/abs/1904.12577)
- TBD (older manuscript: https://arxiv.org/abs/2011.07964) 

Dataset is hosted under kaggle datasets:
- https://www.kaggle.com/martholi/anonymized-invoices

All codes and datasets are published under the LICENSE attached (GNU AFFERO GENERAL PUBLIC LICENSE).
Any derivative work or service should be published under the same license (for any other licensing options, feel free to reach out).

We kindly ask you to cite the abovementioned papers in Your reserach. Or your "Thank You" page 
together with the author's name (https://www.linkedin.com/in/martin-holecek/) and a link to https://rossum.ai/.

Testing baseline command (note the limit and n_epochs params):
```
experiments_ft.py --verbose=1 --sqlite_source="article_anon_a.sqlite" --neighbours=1 --debug=True
--cls_extract_types="['amount_total', 'amount_total_base', 'amount_total_tax', 'amount_rounding', 'amount_paid', 'amount_due', 'tax_detail_base', 'tax_detail_rate', 'tax_detail_tax', 'tax_detail_total', 'account_num', 'bank_num', 'iban', 'bic', 'const_sym', 'spec_sym', 'var_sym', 'invoice_id', 'order_id', 'customer_id', 'date_issue', 'date_uzp', 'date_due', 'terms', 'sender_ic', 'sender_dic', 'recipient_ic', 'recipient_dic', 'sender_name', 'sender_addrline', 'recipient_name', 'recipient_addrline', 'page_current', 'page_total', 'phone_num']"
--weights_separate --key_metric=custom --key_metric_mode=max --n_epochs=2 --limit=400
```
