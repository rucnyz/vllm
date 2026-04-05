## Extended Concurrency Results ($c=768, 1024$)

THETA+ (DP=4) vs disagg on 4×RTX PRO 6000 (Qwen3-8B, 2000 prompts). "−" denotes OOM or timeout. Throughput in tok/s; TTFT and TPOT in ms. Best per-row in **bold**.

|||THETA+|1P+3D|2P+2D|3P+1D|
|:---|:---|:---|:---|:---|:---|
|**$c=768$**||||||
|Prefill-heavy|Throughput|**51,809**|22,868|43,827|−|
|(1024:128)|TTFT|**2,333**|30,157|13,986|−|
||TPOT|110.8|**14.5**|23.5|−|
|Balanced|Throughput|**27,827**|25,991|−|−|
|(512:512)|TTFT|**1,205**|11,592|−|−|
||TPOT|**47.7**|29.5|−|−|
|Decode-heavy|Throughput|**21,242**|18,131|13,238|5,981|
|(128:1024)|TTFT|**551**|4,479|3,416|15,389|
||TPOT|**34.8**|37.2|54.0|116.5|
|**$c=1024$**||||||
|Prefill-heavy|Throughput|**52,426**|23,013|43,971|−|
|(1024:128)|TTFT|**3,876**|37,333|17,961|−|
||TPOT|141.1|**14.5**|23.2|−|
|Balanced|Throughput|**30,704**|27,127|−|−|
|(512:512)|TTFT|**2,062**|16,984|−|−|
||TPOT|47.9|**30.4**|−|−|
|Decode-heavy|Throughput|**23,160**|19,476|14,068|−|
|(128:1024)|TTFT|**1,217**|7,246|5,003|−|
||TPOT|44.2|**43.8**|67.8|−|
