from huggingface_hub import hf_hub_download
import shutil
import os

target_directory = "/media/jorrit/segment-anything/checkpoints"

chkpt_path = hf_hub_download("ybelkada/segment-anything", "checkpoints/sam_vit_b_01ec64.pth")
print('checkpoint path:', chkpt_path)

# # Extract the filename from the path
# filename = os.path.basename(chkpt_path)

# # Construct the destination path in the target directory
# destination_path = os.path.join(target_directory, filename)

# # Move or copy the file to the target directory
# shutil.move(chkpt_path, destination_path)  # Use shutil.copy() for copying instead of moving

# print('File moved to:', destination_path)

