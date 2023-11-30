#!/bin/sh

# Fonction d'aide
show_help() {
    echo "Usage: $0 -f input_folder [-o output_folder] -l [cpp|cuda|all]" --fps
    echo "  -f  Specify the input folder"
    echo "  -o  Specify the output folder name"
    echo "  -l  [cpp|cuda|all]"
    echo "  --fps  Display the FPS of the output video"
    echo "  -h  Display this help"
}

FPS=false

# Analyse des arguments de ligne de commande
while getopts "hl:f:o:-:" opt; do
    case $opt in
        h) 
            show_help
            exit 0
            ;;
        l) 
            LIB_FILE=$OPTARG
            ;;
        f) 
            INPUT_FILE=$OPTARG
            ;;
        o) 
            OUTPUT_FILE=$OPTARG
            ;;
        -)
            case "${OPTARG}" in
                fps)
                    FPS=true
                    ;;
                *)
                    echo "Invalid option: --$OPTARG" >&2
                    show_help
                    exit 1
                    ;;
            esac
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            show_help
            exit 1
            ;;
    esac
done

if [ -z "$LIB_FILE" ] || [ -z "$INPUT_FILE" ] || [ -z "$OUTPUT_FILE" ]; then
    echo "Missing required arguments"
    show_help
    exit 1
fi

cmake --build build
export GST_PLUGIN_PATH=$(pwd)

if [ "$LIB_FILE" = "cuda" ] || [ "$LIB_FILE" = "all" ]; then
    

rm -f libgstcudafilter.so

# premèrement en C++
ln -s ./build/libgstcudafilter-cu.so libgstcudafilter.so

OUTPUT_RESULT=""
regex="[0-9:]+\.[0-9]+"
regex_fps='average: ([0-9]+\.[0-9]+)'


echo "Benchmarking $LIB_FILE"
echo "CUDA"
# Boucle sur tous les fichiers vidéo dans le répertoire "test_video"
for INPUT_FILE in test_video/*; do
    echo "$INPUT_FILE"
    # Vérifie si le fichier est un fichier vidéo
    if [ -f "$INPUT_FILE" ]; then
        # Construit le chemin de sortie pour chaque fichier
        OUTPUT_FILE="output/$(basename "$INPUT_FILE")_output.mp4"
            
        # Construction de la commande GStreamer pour le benchmark
        GST_COMMAND="gst-launch-1.0 uridecodebin uri=file://$(pwd)/$INPUT_FILE ! videoconvert ! \"video/x-raw, format=(string)RGB\" ! cudafilter ! videoconvert ! video/x-raw, format=I420 ! x264enc ! mp4mux ! filesink location=$OUTPUT_FILE"
        
        GST_COMMAND_FPS="gst-launch-1.0 -v uridecodebin uri=file://$(pwd)/$INPUT_FILE ! videoconvert ! \"video/x-raw, format=(string)RGB\" ! cudafilter ! videoconvert ! fpsdisplaysink video-sink=fakesink sync=false"

        output=""
        
        output=$(eval "$GST_COMMAND")
        
        echo "$output"
        echo
        regex_str=$(echo "$output" | grep -oE "$regex")

        OUTPUT_RESULT="${OUTPUT_RESULT}Temps pour ${INPUT_FILE} :${regex_str}\\n"

        if [ "$FPS" = true ]; then
            output=$(eval "$GST_COMMAND_FPS")
            FPS_STR=$(echo "$output" | grep -oE "$regex_fps" | tail -n 1)
            OUTPUT_RESULT="${OUTPUT_RESULT}FPS pour ${INPUT_FILE} ${FPS_STR}\\n"
        fi
    fi
done

echo -e "$OUTPUT_RESULT"

fi

################ CPP ################

if [ "$LIB_FILE" = "cpp" ] || [ "$LIB_FILE" = "all" ]; then
    

rm -f libgstcudafilter.so

# premèrement en C++
ln -s ./build/libgstcudafilter-cpp.so libgstcudafilter.so

OUTPUT_RESULT=""
regex="[0-9:]+\.[0-9]+"


echo "Benchmarking $LIB_FILE"
echo "CUDA"
# Boucle sur tous les fichiers vidéo dans le répertoire "test_video"
for INPUT_FILE in test_video/*; do
    echo "$INPUT_FILE"
    # Vérifie si le fichier est un fichier vidéo
    if [ -f "$INPUT_FILE" ]; then
        # Construit le chemin de sortie pour chaque fichier
        OUTPUT_FILE="output/$(basename "$INPUT_FILE")_output.mp4"
            
        # Construction de la commande GStreamer pour le benchmark
        GST_COMMAND="gst-launch-1.0 uridecodebin uri=file://$(pwd)/$INPUT_FILE ! videoconvert ! \"video/x-raw, format=(string)RGB\" ! cudafilter ! videoconvert ! video/x-raw, format=I420 ! x264enc ! mp4mux ! filesink location=$OUTPUT_FILE"
        
        GST_COMMAND_FPS="gst-launch-1.0 -v uridecodebin uri=file://$(pwd)/$INPUT_FILE ! videoconvert ! \"video/x-raw, format=(string)RGB\" ! cudafilter ! videoconvert ! fpsdisplaysink video-sink=fakesink sync=false"

        output=""
        
        output=$(eval "$GST_COMMAND")
        
        echo "$output"
        echo
        regex_str=$(echo "$output" | grep -oE "$regex")
        OUTPUT_RESULT="${OUTPUT_RESULT}Temps pour ${INPUT_FILE} :${regex_str}\\n"

        if [ "$FPS" = true ]; then
            output=$(eval "$GST_COMMAND_FPS")
            FPS_STR=$(echo "$output" | grep -oE "$regex_fps" | tail -n 1)
            OUTPUT_RESULT="${OUTPUT_RESULT}FPS pour ${INPUT_FILE} ${FPS_STR}\\n"
        fi
    fi
done

echo -e "$OUTPUT_RESULT"

fi