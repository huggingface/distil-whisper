#!/bin/bash

# This script is adapted from https://github.com/peregilk/ttconnect#ttconnect

zone="us-central2-b"  # TPU v4's always are in us-central2-b. Update if using TPU v2/v3's
name=$1

echo "Connecting to $name";

## Some basic checks if the input is valid
output=$(gcloud compute tpus describe $name --zone $zone 2>/dev/null)
if [ $? != 0 ]; then
	echo "Could not find a tpu-v4 with this name in the zone $zone. Exiting."
	exit 1
fi

tputype=$(echo $output | awk '{print $2}')
tpusize=$(echo $tputype| cut -c4-)
size="$(($tpusize / 8))"

if (( $size < 1 )); then
	echo "This is reported as a $tputype with $size tpu(s). This is not a valid tpu-v4 resource. Exiting."
	exit 1
fi


# Check if the session exists, if not create it
# If there already is a session with this name, it will just attach

tmux has-session -t $name 2>/dev/null


if [ $? != 0 ]; then
	tmux new-session -d -s $name
	tmux select-layout main-vertical

	for i in $(seq $(($size-1))); do
		tmux split-window -v -d -t $name
		# Making sure there is space to split
		tmux select-layout main-horizontal
	done

	for i in $(seq $(($size))); do
		worker=$(($i -1))
                command="gcloud alpha compute tpus tpu-vm ssh $name --zone $zone --worker $worker"
		tmux select-pane -t $name:0.$worker
		tmux send-keys -t $name "$command" Enter

        done

	# Select the final layout
	if ((size >= 16));then
		tmux select-layout tiled
	else
		tmux select-layout tiled
		tmux select-layout main-vertical
	fi

	# Enable mouse control - for changing pane size
	# Disabled for now since it makes copying more difficult
	# tmux set-mouse on

	# Move cursor to worker 0
	tmux select-pane -t $name:0.0

	# Resize the left window
        tmux resize-pane -L 50



	# Set pane synchronization
	tmux set-window-option -t $name:0 synchronize-panes on

	# Set pane border format
	tmux set-option -t $name pane-border-status top
	tmux set-option -t $name pane-border-format "worker #{pane_index} "


fi

# Attach to the session
tmux attach -t $name
