if ! [ "$(basename $PWD)" = "face_alignment" ]
then
	echo "Please run from the base directory of this repo."
	exit 1
fi

docker build -t ibug-face_alignment -f ./demo/Dockerfile .

prefix=""
if [ $(uname | grep -iE "(mingw|cygwin)") ]
then
	prefix="winpty"
fi

echo $prefix
$prefix docker run -it --rm -p 8888:8888 ibug-face_alignment
