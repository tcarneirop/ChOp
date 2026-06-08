#source ~/.config.sh
#reserveNodes 2

source elliot.sh
setSSHServers

buildChpl

cd ~/12042021_fsp
source export.sh
./compile.sh
./bin/fsp.out --instance=30 --coordinated=false --pgas=true --lower_bound="simple" -nl 2
