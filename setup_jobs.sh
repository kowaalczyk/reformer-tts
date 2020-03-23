#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

function dirs() {
    data_dir="/scidatasm/$USER/"
    if [[ -e "$data_dir" ]]; then
        echo "using existing $data_dir"
    else
        echo "creating $data_dir"
        mkdir "$data_dir"
    fi

    results_dir="/results/$USER"
    if [[ -e "$results_dir" ]]; then
        echo "using existing $results_dir"
    else
        echo "creating $results_dir"
        mkdir "$results_dir"
    fi
}

function sync() {
    echo "copying directories to /scidatasm/$USER/"
    # rsync is used so that existing files are updated only if hash is different
    rsync -avr "$(pwd)/data" "/scidatasm/$USER/"
    rsync -avr "$(pwd)/nv_wavenet" "/scidatasm/$USER/"

    echo "sync complete at $(date), data will be available on compute nodes in 10 minutes"
}

function clean_users() {
    # we use single quotes on purpose, as we want to change usernames to $USER
    # (not to evaluate $USER variable as expression)
    echo 'changing all usernames in clean_users/*.sh to $USER'
    sed -i -e 's/kk385830/$USER/' jobs/*.sh
    sed -i -e 's/mo382777/$USER/' jobs/*.sh
    sed -i -e 's/tm385898/$USER/' jobs/*.sh
}

function check() {
    active_python=$(which python)
    if [[ "$active_python" == "/scidatasm/"* ]]; then
        echo "python location ok"
    else
        echo ""
        echo "ERROR"
        echo "active python expected to be under /scidatasm/ partition, yours is at $active_python"
        echo "possibly, your virtual environment has not been activated"
        return 1
    fi

    python_version=$(python --version)
    if [[ "$python_version" == *"3.8"* ]]; then
        echo "python version ok"
    else
        echo ""
        echo "ERROR"
        echo "active python version expected to be 3.8, yours is $python_version"
        echo "possibly, your virtual environment has not been activated"
        return 1
    fi

    usernames=$(grep -e kk385830 -e mo382777 -e tm385898 -e "$USER" -- jobs/*.sh || true)
    if [[ -z "$usernames" ]]; then
        echo "usernames ok"
    else
        echo ""
        echo "ERROR"
        echo "found usernames for users:"
        echo "$usernames"
        echo ""
        echo "change them to a generic \$USER before proceeding:"
        echo "$0 clean_users"
        return 1
    fi

    results_dir="/results/$USER"
    data_dir="/scidatasm/$USER"
    if [[ -d "$results_dir" ]] && [[ -d "$data_dir" ]]; then
        echo "directories ok"
    else
        echo "ERROR"
        echo "missing data or results directory"
        echo ""
        echo "to create them & synchronize with current directory:"
        echo "$0 all"
        return 1
    fi

    # setting +e to preserve output, but only display it in case pip fails
    set +e
    pip_out=$(pip install -r requirements.txt)
    if [[ $? -eq 0 ]]; then
        echo "python packages ok"
    else
        echo "error installing requirements"
        echo "$pip_out"
        return 1
    fi
    set -e
    
    echo "all checks complete"
}

function run() {
    echo "starting job $1..."

    # create output file based on job name
    job_filename=$(basename -- "$1")
    job_name="${job_filename%.*}"
    out_filename="/results/$USER/${job_name}_$(date +"%Y-%m-%d_%H-%M-%S").out"

    # start the job
    sbatch -o "$out_filename" "$1"

    echo "results will be saved to:"
    echo "$out_filename"
}

function help() {
    echo "Setup tasks:"
    echo "$0 dirs - make directories necessary to run the jobs"
    echo "$0 sync - sync all necessary data to /scidatasm/$USER/ partiion"
    echo "$0 clean_users - change usernames in job files to a generic \$USER"
    echo "$0 all - perform all of the setup tasks in sequence"
    echo ""
    echo "Running jobs:"
    echo "$0 check - checks scripts for common errors"
    echo "$0 run [job_file] performs checks and runs the job using sbatch"
}

# argument checks
if [[ $# -eq 0 ]]; then
    help
    exit 2
fi

# additional check for run command
if [[ "$1" == "run" ]] && [[ $# -ne 2 ]]; then
    echo "Usage:"
    echo "$0 run [job_file]"
    exit 2
fi

# call function based on 1st argument or display help message
case "$1" in
    dirs)
        dirs
        ;;

    sync)
        sync
        ;;
    
    clean_users)
        clean_users
        ;;

    check)
        check
        ;;

    run)
        check
        run "$2"
        ;;

    all)
        dirs
        sync
        clean_users
        ;;

    *)
        help
        exit 2
        ;;
esac
