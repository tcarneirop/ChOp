#!/bin/bash


while [[ $# -gt 0 ]]; do
    case "$1" in
        --binary)
            BINARY="$2"
            shift 2
            ;;
        --maxsize)
            MAX_BOARD_SIZE="$2"
            shift 2
            ;;
        --depths)
            DEPTHS_STR="$2"
            shift 2
            ;;
        --blocks)
            BLOCKS_STR="$2"
            shift 2
            ;;
        --runs)
            RUNS="$2"
            shift 2
            ;;
        --gpu)
            MACHINE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done


USES_BLOCK=false

##there is a block size for both cuda and hip
case "$BINARY" in
    *cuda*|*hip*)
        USES_BLOCK=true
        ;;
esac


CSV="${BINARY}.csv"

echo "MAX_BOARD=$MAX_BOARD_SIZE"
echo "DEPTHS_STR=$DEPTHS_STR"
echo "BLOCKS_STR=$BLOCKS_STR"
echo "RUNS=$RUNS"
echo "GPU=$MACHINE"

IFS=',' read -ra DEPTHS <<< "$DEPTHS_STR"
IFS=',' read -ra BLOCKS <<< "$BLOCKS_STR"

echo "Depths: ${DEPTHS[*]}"
echo "Blocks: ${BLOCKS[*]}"

if [ ! -x "$BINARY" ]; then
    echo "Error: executable '$BINARY' not found."
    exit 1
fi

if [ ! -f "$CSV" ]; then
    echo "machine,binary,board,depth,block_size,run,time" > "$CSV"
fi

echo
echo "############################################"
echo "#              STARTING                    #"
echo "############################################"
echo

for ((board=15; board<=MAX_BOARD_SIZE; board++)); do
    for depth in "${DEPTHS[@]}"; do
        for block in "${BLOCKS[@]}"; do
            for ((run=1; run<=RUNS; run++)); do


                ##our exes that have block -- cuda and hip, mainly
                if $USES_BLOCK; then
                    csv_block="$block"
                else
                    csv_block=""
                fi

                key="$MACHINE,$BINARY,$board,$depth,$csv_block,$run,"

                if grep -Fq "$key" "$CSV"; then
                    if [[ "$BINARY" == *cuda* ]]; then
                        echo "[SKIP] Board=$board Depth=$depth Block=$block Run=$run"
                    else
                        echo "[SKIP] Board=$board Depth=$depth Run=$run"
                    fi
                    continue
                fi

                if [[ "$BINARY" == *cuda* ]]; then
                    echo "[RUN ] Board=$board Depth=$depth Block=$block Run=$run"
                else
                    echo "[RUN ] Board=$board Depth=$depth Run=$run"
                fi

                OUTPUT=$(mktemp)

                if [[ "$BINARY" == *cuda* ]]; then
                    CUDA_VISIBLE_DEVICES=0 ./"$BINARY" \
                        "$board" "$depth" "$block" \
                        > "$OUTPUT" 2>&1
                else
                    CUDA_VISIBLE_DEVICES=0 ./"$BINARY" \
                        "$board" "$depth" \
                        > "$OUTPUT" 2>&1
                fi

                elapsed=$(awk '/Elapsed total:/ {sub(/s$/, "", $NF); print $NF}' "$OUTPUT")

                if [ -z "$elapsed" ]; then
                    echo "ERROR: Failed to extract elapsed time."
                    cat "$OUTPUT"
                    rm "$OUTPUT"
                    exit 1
                fi

                echo "$MACHINE,$BINARY,$board,$depth,$csv_block,$run,$elapsed" >> "$CSV"

                rm "$OUTPUT"

            done
        done
    done
done

echo
echo "Results written to $CSV"
