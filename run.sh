set -e 
for i in $(seq $1); do
     echo "Run $i"
    ./build/app 
     echo "exit $?"
done
