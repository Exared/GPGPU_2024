#!/bin/sh

# Fonction d'aide
show_help() {
    echo "Usage: $0 -l [cpp|cuda] -f input_file [-o output_file] [--local] [--benchmark]"
    echo "  -l  Specify the library to use ('cpp' or 'cuda')"
    echo "  -f  Specify the input file"
    echo "  -o  Specify the output file name"
    echo "  --savefile  Execute the GStreamer pipeline and save the output to a file"
    echo "  --benchmark  Run the benchmark pipeline"
    echo "  -h  Display this help"
}

# Variables pour les nouvelles options
LOCAL_EXECUTION=false
BENCHMARK_MODE=false
NSIGHT=false

# Analyse des arguments de ligne de commande
while getopts "hl:f:o:-:" opt; do
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
        -) 
            case "${OPTARG}" in
                savefile)
                    LOCAL_EXECUTION=true
                    ;;
                benchmark)
                    BENCHMARK_MODE=true
                    ;;
                nsight)
                    NSIGHT=true
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


if [ -z "$LIB_TYPE" ] || [ -z "$INPUT_FILE" ]; then
    echo "Missing required arguments"
    show_help
    exit 1
fi

cmake --build build
export GST_PLUGIN_PATH=$(pwd)

rm -f libgstcudafilter.so

if [ "$LIB_TYPE" = "cpp" ]; then
    ln -s ./build/libgstcudafilter-cpp.so libgstcudafilter.so
else
    ln -s ./build/libgstcudafilter-cu.so libgstcudafilter.so
fi

if [ "$LOCAL_EXECUTION" = true ]; then
    gst-launch-1.0 uridecodebin uri=file://$(pwd)/$INPUT_FILE ! videoconvert ! "video/x-raw, format=(string)RGB" ! cudafilter ! videoconvert ! video/x-raw, format=I420 ! x264enc ! mp4mux ! filesink location=$OUTPUT_FILE
elif [ "$BENCHMARK_MODE" = true ]; then
    gst-launch-1.0 -e -v uridecodebin uri=file://$(pwd)/$INPUT_FILE ! videoconvert ! "video/x-raw, format=(string)RGB" ! cudafilter ! videoconvert ! fpsdisplaysink video-sink=fakesink sync=false
elif [ "$NSIGHT" = true ]; then
    nvprof gst-launch-1.0 uridecodebin uri=file://$(pwd)/$INPUT_FILE ! videoconvert ! "video/x-raw, format=(string)RGB" ! cudafilter ! videoconvert ! video/x-raw, format=I420 ! x264enc ! mp4mux ! filesink location=$OUTPUT_FILE
fi
