#Converts all .pth models in current directory into c++ header files. Model must bee created using training.py for this script to work.
for i in out/*.pth; do
    [ -f "$i" ] || break
    python3 create_header.py $i
done