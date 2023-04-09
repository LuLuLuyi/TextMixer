from datasets import load_dataset,load_from_disk
for eps in [1,2,3]:
    dataset_path = f'/root/contrastive_privacy/version_text/output_SanText_glove/ontonotes/eps_{eps}.00/sword_0.90_p_0.30/replaced_dataset'
    santext_dataset = load_from_disk(dataset_path)
    print(santext_dataset)
# dataset = load_dataset('ag_news')
pass