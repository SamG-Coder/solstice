#!/bin/sh
# Builds up to a stage, renders the named views, and tiles them into one sheet to look at.
#   sh tools/blender/gt3r/go.sh STAGE "view,view,..." [columns] [width] [extra make.py flags]
stage=$1; views=$2; columns=${3:-2}; width=${4:-760}; shift 4 2>/dev/null
/d/Blender/blender.exe --background --factory-startup --python-exit-code 1 --python tools/blender/gt3r/make.py -- target/gt3r/gt3r.glb --until "$stage" --no-export --views target/gt3r/views "$views" "$@" 2>&1 | grep -E "make:|glazing:|audit:|  top|half-width|worst|Error|Traceback|line [0-9]+|render:|rror" | grep -v "^  top at -" | head -40
python tools/blender/gt3r/sheet.py target/gt3r/views target/gt3r/views/sheet.png $(echo "$views" | tr ',' ' ') --columns "$columns" --width "$width"
