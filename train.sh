export CXX="g++"
TID="./datasets/celebahq256x256/images"
TED="./datasets/celebahq256x256/edge_bdcn_awl"
TIL="./datasets/celebahq256x256/trainlist.txt"

VID="./datasets/celebahq256x256/images"
VED="./datasets/celebahq256x256/edge_bdcn_awl"
VMD="./datasets/celebahq256x256/val_masks"
VIL="./datasets/celebahq256x256/vallist.txt"

python train.py \
	--batchSize ${BSIZE} \
	--nThreads ${NWK} \
	--name debug \
	--not_om \
	--dataset_mode_train trainedge \
	--dataset_mode_val valedge \
	--train_image_dir $TID \
	--train_edge_dir $TED \
	--train_image_list $TIL \
	--train_image_postfix '.jpg' \
	--train_mask_postfix '.png' \
	--val_image_dir $VID \
	--val_edge_dir $VED \
	--val_mask_dir $VMD \
	--val_image_list $VIL \
	--val_image_postfix '.jpg' \
	--val_mask_postfix '.png' \
	--load_size 256 \
	--crop_size 256 \
	--cjit 0.1 \
	--model inpaintc \
	--netG deepfillc \
	--netD deepfillc \
	--preprocess_mode scale_shortside_and_crop \
	--validation_freq 10000 \
	--niter 300 \
	--use_cam \
	${EXTRA} \
