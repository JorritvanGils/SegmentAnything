from huggingface_hub import hf_hub_download


target_directory = "/media/jorrit/segment-anything/checkpoints"

chkpt_path = hf_hub_download("ybelkada/segment-anything", "checkpoints/sam_vit_b_01ec64.pth")
print('checkpoint path:', chkpt_path)
