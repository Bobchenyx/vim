python vim/main.py --eval \
    --resume hustvl/Vim-small-midclstok/vim_s_midclstok_80p5acc.pth \
    --model vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
    --data-path /home/imagenet

# python vim/main.py --eval \
#     --resume hustvl/Vim-tiny-midclstok/vim_t_midclstok_76p1acc.pth \
#     --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
#     --data-path /home/imagenet