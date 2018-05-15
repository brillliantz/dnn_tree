fn=$1
log_fn="log_$fn.txt"
echo "Will run python $fn background. And redicrect log to $log_fn"
nohup python -u $fn >> $log_fn 2>&1 &
