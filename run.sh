#!/bin/sh

# Fonction d'aide
show_help() {
    echo "Usage: $0 -l [cpp|cuda] -f input_file -o output_file"
    echo "  -l  Specify the library to use ('cpp' or 'cuda')"
    echo "  -f  Specify the input file"
    echo "  -o  Specify the output file name"
    echo "  -h  Display this help"
}

# Analyse des arguments de ligne de commande
while getopts "hl:f:o:" opt; do
    case $opt in
        h) 
            show_help
            exit 0
            ;;
        l) 
            LIB_TYPE=$OPTARG
            ;;
        f) 
            INPUT_FILE=$OPTARG
            ;;
        o) 
            OUTPUT_FILE=$OPTARG
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            show_help
            exit 1
            ;;
    esac
done

# Vérification des paramètres requis
if [ -z "$LIB_TYPE" ] || [ -z "$INPUT_FILE" ] || [ -z "$OUTPUT_FILE" ]; then
    echo "Missing required arguments"
    show_help
    exit 1
fi

# Construction du projet
cmake --build build
export GST_PLUGIN_PATH=$(pwd)

# Suppression du lien symbolique précédent, s'il existe
rm -f libgstcudafilter.so

# Création du lien symbolique en fonction du type de bibliothèque spécifié
if [ "$LIB_TYPE" = "cpp" ]; then
    ln -s ./build/libgstcudafilter-cpp.so libgstcudafilter.so
else
    ln -s ./build/libgstcudafilter-cu.so libgstcudafilter.so
fi

# Lancement de la chaîne GStreamer
gst-launch-1.0 uridecodebin uri=file://$(pwd)/$INPUT_FILE ! videoconvert ! "video/x-raw, format=(string)RGB" ! cudafilter ! videoconvert ! video/x-raw, format=I420 ! x264enc ! mp4mux ! filesink location=$OUTPUT_FILE
