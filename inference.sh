# VITON-HD
# accelerate launch inference.py \
#     --width 256 --height 256 --num_inference_steps 20 \
#     --output_dir "output/vtonhd" \
#     --unpaired \
#     --data_dir "./data/zalando-hd-resized" \
#     --seed 42 \
#     --test_batch_size 2 \
#     --guidance_scale 2.0




# DressCode
accelerate launch --cpu inference_dc.py \
    --width 256 --height 256 --num_inference_steps 20 \
    --output_dir "output/dresscode" \
    --unpaired \
    --data_dir "./data/archive" \
    --seed 42 \
    --test_batch_size 2 \
    --guidance_scale 2.0 \
    --category "upper_body"



# accelerate launch inference_dc.py \
#     --width 768 --height 1024 --num_inference_steps 30 \
#     --output_dir "output/dresscode" \
#     --unpaired \
#     --data_dir "./data/archive" \
#     --seed 42 
#     --test_batch_size 2
#     --guidance_scale 2.0
#     --category "upper_body"

#VITON-HD
##paired setting
# accelerate launch inference.py --pretrained_model_name_or_path "" \
#     --width 768 --height 1024 --num_inference_steps 30 \
#     --output_dir "result" --data_dir "./data/zalando-hd-resized" \
#     --seed 42 --test_batch_size 2 --guidance_scale 2.0


# ##unpaired setting
# accelerate launch inference.py --pretrained_model_name_or_path "./ckpt/vitonhd/VITONHD.ckpt" \
#     --width 768 --height 1024 --num_inference_steps 30 \
#     --output_dir "result" --unpaired --data_dir "./data/zalando-hd-resized" \
#     --seed 42 --test_batch_size 2 --guidance_scale 2.0



# #DressCode
# ##upper_body
# accelerate launch inference_dc.py --pretrained_model_name_or_path "yisol/IDM-VTON" \
#     --width 768 --height 1024 --num_inference_steps 30 \
#     --output_dir "result" --unpaired --data_dir "/home/omnious/workspace/yisol/DressCode" \
#     --seed 42 --test_batch_size 2 --guidance_scale 2.0 --category "upper_body"

# ##lower_body
# accelerate launch inference_dc.py --pretrained_model_name_or_path "yisol/IDM-VTON" \
#     --width 768 --height 1024 --num_inference_steps 30 \
#     --output_dir "result" --unpaired --data_dir "/home/omnious/workspace/yisol/DressCode" \
#     --seed 42 --test_batch_size 2 --guidance_scale 2.0 --category "lower_body"

# ##dresses
# accelerate launch inference_dc.py --pretrained_model_name_or_path "yisol/IDM-VTON" \
#     --width 768 --height 1024 --num_inference_steps 30 \
#     --output_dir "result" --unpaired --data_dir "/home/omnious/workspace/yisol/DressCode" \
#     --seed 42 --test_batch_size 2 --guidance_scale 2.0 --category "dresses"