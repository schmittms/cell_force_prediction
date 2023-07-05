#!/bin/bash


#SBATCH --output=./out/output.out
#SBATCH --account=pi-vitelli
#SBATCH --partition=vitelli-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --exclude=midway3-0294

verbose=0

while getopts v: flag
do
	case "${flag}" in 
		v) verbose=${OPTARG};;
	esac
done


module load python
source activate /project/vitelli/ml_venv
cd /project/vitelli/matthew/cell_force_prediction_public

##### Data args ######
directory=/project/vitelli/cell_stress/TractionData_All_16kpa_new_HighForce
in_channels=6 # Channel of image to use as input. 4 = mask, 6 = zyxin, 7 = actin 
testtrain_split=bycell # How to do test/train split. This will reserve some entire cells for testing.
test_cell="pax_cell_5,pax_cell_2,myo_cell_6,myo_cell_1,08_cell_1,11_cell_1,myo_cell_4,17_cell_3,myo_cell_5,11_cell_4,10_cell_4,11_cell_2,10_cell_1,myo_cell_3,17_cell_4"
out_channels=2,3 # Channel of image to use as output. [2,3] are x,y components of traction force
frames_to_keep=1000 # Frames to use during training. More is better, but takes longer
crop_size=256 # Typically used 512

magnitude_only=0 # whether to only try to predict force magnitude (a scalar)
angmag=1 # whether to predict [magnitude, angle] or [x component, y component]
perturb_input=none
perturb_output=none
add_noise=none,none

# Dataset normalization:
input_baseline_normalization=outside_inside # will normalize zyxin signals for each cell as described in Methods.
output_baseline_normalization=mean_dataset # will normalize forces by the average force in the entire dataset
normalization_output="rescale,0.25/threshold,0.4" # additional normalization factor used across all cells. This essentially converts them back to kPa. 

##### Training args ######
batch_size=8
num_workers=4
epochs=301
width=$((epochs+10))
save_model_freq=10

n=0
loss_hparams=""
for x in {"0.0","1.0"}" "{100,}" "{30,}" "{0.,0.1}; 
do
	ar=($x)
	N=${#ar[@]}
	if [[ "$N" == "4" ]]; then
		expweight=${ar[0]}
		#width=${ar[1]}
		e_crit=${ar[2]}
		strain_reg=${ar[3]}
		loss_hparams+="loss_type,am:exp_weight,${expweight}:exp_schedule,type>linear,width>${width},e_crit>${e_crit}:strainenergy_regularization,${strain_reg}:reg_schedule,type>linear,width>${width},e_crit>${e_crit}:loss_kwargs,max_force>8./"
	fi
done
loss_hparams="${loss_hparams::-1}"

log_dir=$(date +"./tensorboard_logs/%y%m%d_%H%M")
logger_params="log_dir,${log_dir}:image_epoch_freq,10:image_callbacks,vectorfield,hists:save_model_freq,20"
optim_hparams="LR,0.001:schedule_rate,0.99"

##### Model args ######

act1="gelu"
n_lyr=3
ds_krnl=4 #downsample kernel
nch=4

n_blocks=4

prepend_struct="1,${nch},${nch},${n_blocks}"  # start_channel, resnet_channel, end_channel, N_blocks
prepend_layer_args="7,1,4,1,0.1,${act1},1" # kernel, stride, inv_bottleneck_factor, dilation, dropout_rate, cctivation, batchnomr
# 'kernel','stride','inv_bottleneck_factor','dilation','dropout_rate','activation','batchnorm'
encoder_struct="${nch},${n_lyr},${n_blocks},${n_blocks}" # n_ch, n_layers, n_node_blocks, n_skip_blocks
encoder_dwnsmpl_args="${ds_krnl},${act1},1" # kernel, activation, batchnorm 
encoder_intrlyr_args="7,1,4,1,0.1,${act1},1" #,1,0/2,3,1,0.2,${act2},1,0" # String with 7 csvs: 'stride','kernel','dilation','dropout_rate','activation_function','batchnorm','split' 
decoder_struct="${n_lyr},${n_blocks},${ds_krnl}" # n_layers, n_node_blocks, upsample_kernel 
decoder_layer_args="7,1,4,1,0.1,${act1},1"
append_struct="${nch},${nch},2,${n_blocks}" # start, resnet_ch, end_ch, N_blocks
append_layer_args="7,1,8,1,0.1,${act1},1"



python train_predictor.py \
	--directory $directory \
	--testtrain_split $testtrain_split \
	--in_channels $in_channels \
	--out_channels $out_channels \
	--test_cell $test_cell \
	--frames_to_keep $frames_to_keep \
	--crop_size $crop_size \
	--magnitude_only $magnitude_only \
	--angmag $angmag \
	--perturb_input $perturb_input \
	--perturb_output $perturb_output \
	--add_noise $add_noise \
	--input_baseline_normalization $input_baseline_normalization \
	--output_baseline_normalization $output_baseline_normalization \
	--normalization_output $normalization_output \
	--batch_size $batch_size \
	--num_workers $num_workers \
	--epochs $epochs \
	--loss_hparams $loss_hparams \
	--logger_params $logger_params \
	--optim_hparams $optim_hparams \
	--prepend_struct $prepend_struct \
	--prepend_layer_args $prepend_layer_args \
	--encoder_struct $encoder_struct \
	--encoder_dwnsmpl_args $encoder_dwnsmpl_args \
	--encoder_intrlyr_args $encoder_intrlyr_args \
	--decoder_struct $decoder_struct \
	--decoder_layer_args $decoder_layer_args \
	--append_struct $append_struct \
	--append_layer_args $append_layer_args \
	--save_model_freq $save_model_freq \
	--verbose $verbose \


